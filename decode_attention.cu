#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

/*
Tensor layout:

Q:       [B, QH, D]
K cache: [B, S, KVH, D]
V cache: [B, S, KVH, D]
scores:  [B, QH, S]
probs:   [B, QH, S]
Out:     [B, QH, D]

B   : batch size
S   : 当前 KV Cache 长度，即已经缓存的历史 token 数
QH  : Query head 数
KVH : Key / Value head 数
D   : 每个 attention head 的维度

Decode 阶段一次只处理当前生成 token 的 Query，
所以 Q 不带 query sequence length 维度。

GQA 映射：
    group_size = QH / KVH
    kvh = qh / group_size

例如：
    QH=32, KVH=8
    group_size=4

则：
    qh 0~3   -> kvh 0
    qh 4~7   -> kvh 1
    ...
    qh 28~31 -> kvh 7
*/

namespace {

// ============================================================
// Kernel 配置常量
// ============================================================

// 三段式 QK / PV GEMV kernel 每个 block 的线程数。
constexpr int kGemmThreadsPerBlock = 256;

// 单独 Softmax kernel 每个 block 的线程数。
constexpr int kSoftmaxThreadsPerBlock = 256;

// NVIDIA GPU warp 固定为 32 个线程。
constexpr int kWarpSize = 32;

// Softmax block 中 warp 数。
// 256 threads / 32 threads per warp = 8 warps。
constexpr int kSoftmaxWarpCount =
    kSoftmaxThreadsPerBlock / kWarpSize;

// ------------------------------------------------------------
// Fused Online Softmax / Split-KV Partial 配置
//
// 一个 fused / partial block 使用 128 个线程。
// 一个 tile 同时处理 32 个 KV token。
// 每个 token 的 QK 点积由 4 个线程协作完成。
//
// 因此：
//     128 threads = 32 tokens * 4 lanes/token
// ------------------------------------------------------------

constexpr int kFusedThreadsPerBlock = 128;
constexpr int kFusedTokensPerBlock = 32;
constexpr int kFusedLanesPerToken = 4;

// ------------------------------------------------------------
// Split-KV 配置
//
// 每个 split 最多处理 512 个 token。
// S=8192 时：
//     num_splits = ceil(8192 / 512) = 16
//
// 一个 (b, qh, split_idx) 对应一个 partial block。
// ------------------------------------------------------------

// constexpr int kSplitTokens = 512;

// ============================================================
// 参数检查函数
// ============================================================

void check_gqa_decode_shape(
    int B,
    int S,
    int QH,
    int KVH,
    int D
) {
    if (B <= 0 || S <= 0 || QH <= 0 || KVH <= 0 || D <= 0) {
        throw std::invalid_argument(
            "B、S、QH、KVH、D 必须全部为正数"
        );
    }

    // GQA 要求每个 KV head 服务整数个 Query head。
    if (QH % KVH != 0) {
        throw std::invalid_argument(
            "GQA 要求 QH 必须能被 KVH 整除"
        );
    }
}

void check_not_null(
    const void* ptr,
    const char* name
) {
    if (ptr == nullptr) {
        throw std::invalid_argument(
            std::string(name) + " 不能为空"
        );
    }
}

// 这里只能检查 kernel launch 本身是否成功，例如：
// - grid/block 配置非法
// - 参数配置错误
//
// 它不能保证 kernel 内部所有异步错误都已经出现。
// 真正运行期错误通常会在 cudaDeviceSynchronize 或 cudaMemcpy 时暴露。
void check_launch(
    const char* kernel_name
) {
    const cudaError_t status = cudaGetLastError();

    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(kernel_name) +
            " launch failed: " +
            cudaGetErrorString(status)
        );
    }
}

// ============================================================
// Block-level reduction helpers
// ============================================================

// 一个 block 对所有线程的 value 求最大值。
//
// 实现分两步：
//
// 1. 每个 warp 内用 shuffle 求最大值。
// 2. 每个 warp 的 lane 0 将结果写到 warp_smem。
// 3. warp 0 再对所有 warp 的结果求最大值。
// 4. warp_smem[0] 保存最终 block 最大值。
//
// BLOCK_THREADS 必须和实际 launch 的 block.x 一致。
template <int BLOCK_THREADS>
__device__ float block_reduce_max(
    float value,
    float* warp_smem
) {
    const int lane_id = threadIdx.x % kWarpSize;
    const int warp_id = threadIdx.x / kWarpSize;

    constexpr int kWarpCount =
        (BLOCK_THREADS + kWarpSize - 1) / kWarpSize;

    // 第一步：warp 内 reduction。
    for (
        int offset = kWarpSize / 2;
        offset > 0;
        offset /= 2
    ) {
        value = fmaxf(
            value,
            __shfl_down_sync(
                0xffffffff,
                value,
                offset
            )
        );
    }

    // 每个 warp 的 lane 0 保存该 warp 的最大值。
    if (lane_id == 0) {
        warp_smem[warp_id] = value;
    }

    __syncthreads();

    // 第二步：warp 0 合并所有 warp 的结果。
    if (warp_id == 0) {
        value =
            lane_id < kWarpCount
                ? warp_smem[lane_id]
                : -FLT_MAX;

        for (
            int offset = kWarpSize / 2;
            offset > 0;
            offset /= 2
        ) {
            value = fmaxf(
                value,
                __shfl_down_sync(
                    0xffffffff,
                    value,
                    offset
                )
            );
        }

        if (lane_id == 0) {
            warp_smem[0] = value;
        }
    }

    __syncthreads();

    return warp_smem[0];
}

// 一个 block 对所有线程的 value 求和。
//
// 实现流程与 block_reduce_max 相同，只是运算由 max 改成 sum。
template <int BLOCK_THREADS>
__device__ float block_reduce_sum(
    float value,
    float* warp_smem
) {
    const int lane_id = threadIdx.x % kWarpSize;
    const int warp_id = threadIdx.x / kWarpSize;

    constexpr int kWarpCount =
        (BLOCK_THREADS + kWarpSize - 1) / kWarpSize;

    // 第一步：warp 内求和。
    for (
        int offset = kWarpSize / 2;
        offset > 0;
        offset /= 2
    ) {
        value += __shfl_down_sync(
            0xffffffff,
            value,
            offset
        );
    }

    // 每个 warp 的 lane 0 写出 warp 局部和。
    if (lane_id == 0) {
        warp_smem[warp_id] = value;
    }

    __syncthreads();

    // 第二步：warp 0 合并所有 warp 的局部和。
    if (warp_id == 0) {
        value =
            lane_id < kWarpCount
                ? warp_smem[lane_id]
                : 0.0f;

        for (
            int offset = kWarpSize / 2;
            offset > 0;
            offset /= 2
        ) {
            value += __shfl_down_sync(
                0xffffffff,
                value,
                offset
            );
        }

        if (lane_id == 0) {
            warp_smem[0] = value;
        }
    }

    __syncthreads();

    return warp_smem[0];
}

// ============================================================
// Subgroup reduction helper
// ============================================================
//
// 作用：
//   在一个小线程组内部做求和。
//
// 当前 fused / Split-KV QK 阶段使用 4-lane subgroup：
//
//     threads 0~3      -> tile token 0
//     threads 4~7      -> tile token 1
//     ...
//     threads 124~127  -> tile token 31
//
// 一个 token 的 QK 点积：
//
//     score = sum_d Q[d] * K[token, d]
//
// D=128 时，4 个线程分摊 D 维：
//
//     subgroup lane 0: d = 0, 4, 8, ...
//     subgroup lane 1: d = 1, 5, 9, ...
//     subgroup lane 2: d = 2, 6, 10, ...
//     subgroup lane 3: d = 3, 7, 11, ...
//
// 每个线程先得到 local_dot，随后仅在 4 个线程内部求和。
// 这避免了“每个 token 都进行一次 128-thread full-block reduction”。
template <int SUBGROUP_SIZE>
__device__ __forceinline__ float subgroup_reduce_sum(
    float value
) {
    #pragma unroll
    for (
        int offset = SUBGROUP_SIZE / 2;
        offset > 0;
        offset /= 2
    ) {
        value += __shfl_down_sync(
            0xffffffff,
            value,
            offset,
            SUBGROUP_SIZE
        );
    }

    // 只有 subgroup lane 0 一定得到完整和。
    // 当前后续代码也只让 lane 0 写 score_smem。
    return value;
}

}  // namespace

// ============================================================
// CPU Reference
//
// 功能：
//   以最直接的方式实现完整 GQA Decode Attention。
//   用于验证 CUDA kernel 结果正确性。
//
// 流程：
//   1. QK^T 得到 scores
//   2. stable softmax 得到 probs
//   3. probs * V 得到 Out
// ============================================================

void decode_attention_gqa_cpu(
    const float* q,
    const float* k_cache,
    const float* v_cache,
    float* scores,
    float* probs,
    float* out,
    int B,
    int S,
    int QH,
    int KVH,
    int D
) {
    check_not_null(q, "q");
    check_not_null(k_cache, "k_cache");
    check_not_null(v_cache, "v_cache");
    check_not_null(scores, "scores");
    check_not_null(probs, "probs");
    check_not_null(out, "out");

    check_gqa_decode_shape(B, S, QH, KVH, D);

    // 每个 KV head 对应多少个 Query head。
    const int group_size = QH / KVH;

    // Attention 缩放项：1 / sqrt(D)。
    const float scale =
        1.0f / std::sqrt(static_cast<float>(D));

    for (int b = 0; b < B; ++b) {
        for (int qh = 0; qh < QH; ++qh) {
            // 当前 qh 对应的 K/V head。
            const int kvh = qh / group_size;

            // Q[b, qh, 0] 的展平偏移。
            const std::size_t q_offset =
                (static_cast<std::size_t>(b) * QH + qh) * D;

            // scores[b, qh, 0] 的展平偏移。
            const std::size_t score_offset =
                (static_cast<std::size_t>(b) * QH + qh) * S;

            float max_score =
                -std::numeric_limits<float>::infinity();

            // ------------------------------------------------
            // Stage 1: QK^T
            //
            // 对每个历史 token s：
            //
            // score[s] =
            //   dot(Q[b, qh, :], K[b, s, kvh, :]) / sqrt(D)
            // ------------------------------------------------

            for (int s = 0; s < S; ++s) {
                // K[b, s, kvh, 0] 的展平偏移。
                const std::size_t kv_offset =
                    (
                        (
                            static_cast<std::size_t>(b) * S + s
                        ) * KVH + kvh
                    ) * D;

                float dot = 0.0f;

                for (int d = 0; d < D; ++d) {
                    dot +=
                        q[q_offset + d] *
                        k_cache[kv_offset + d];
                }

                const float score = dot * scale;

                scores[score_offset + s] = score;
                max_score = std::max(max_score, score);
            }

            // ------------------------------------------------
            // Stage 2: Stable Softmax
            //
            // probs[s] =
            //   exp(score[s] - max_score) /
            //   sum_j exp(score[j] - max_score)
            // ------------------------------------------------

            float denominator = 0.0f;

            for (int s = 0; s < S; ++s) {
                const float value = std::exp(
                    scores[score_offset + s] - max_score
                );

                probs[score_offset + s] = value;
                denominator += value;
            }

            const float inv_denominator =
                denominator > 0.0f
                    ? 1.0f / denominator
                    : 0.0f;

            for (int s = 0; s < S; ++s) {
                probs[score_offset + s] *= inv_denominator;
            }

            // ------------------------------------------------
            // Stage 3: P * V
            //
            // Out[b, qh, d] =
            //   sum_s probs[b, qh, s] * V[b, s, kvh, d]
            // ------------------------------------------------

            for (int d = 0; d < D; ++d) {
                float accumulator = 0.0f;

                for (int s = 0; s < S; ++s) {
                    // V[b, s, kvh, 0] 的展平偏移。
                    const std::size_t v_offset =
                        (
                            (
                                static_cast<std::size_t>(b) * S + s
                            ) * KVH + kvh
                        ) * D;

                    accumulator +=
                        probs[score_offset + s] *
                        v_cache[v_offset + d];
                }

                out[q_offset + d] = accumulator;
            }
        }
    }
}

// ============================================================
// Kernel 1: QK GEMV Baseline
//
// scores[b, qh, s] =
//   dot(Q[b, qh, :], K_cache[b, s, kvh, :]) * scale
//
// 线程映射：
//   一个线程负责一个 score[b, qh, s]。
//
// 对该 score，线程内部沿 D 维循环完成 dot product。
// ============================================================

__global__ void gqa_decode_qk_gemv_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    float* __restrict__ scores,
    int B,
    int S,
    int QH,
    int KVH,
    int D,
    float scale
) {
    // 当前线程对应 scores 的一维全局索引。
    const std::size_t global_tid =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    const std::size_t total_scores =
        static_cast<std::size_t>(B) * QH * S;

    if (global_tid >= total_scores) {
        return;
    }

    // scores layout:
    //   [B, QH, S]
    //
    // 展平：
    //   global_tid = ((b * QH + qh) * S + s)
    //
    // 反解：
    //   s = global_tid % S
    //   q_head_linear_idx = global_tid / S
    //   qh = q_head_linear_idx % QH
    //   b = q_head_linear_idx / QH

    const int s =
        static_cast<int>(global_tid % S);

    const std::size_t q_head_linear_idx =
        global_tid / S;

    const int qh =
        static_cast<int>(q_head_linear_idx % QH);

    const int b =
        static_cast<int>(q_head_linear_idx / QH);

    // GQA：当前 Query head 对应的 KV head。
    const int group_size = QH / KVH;
    const int kvh = qh / group_size;

    // Q[b, qh, 0] 的偏移。
    const std::size_t q_offset =
        (static_cast<std::size_t>(b) * QH + qh) * D;

    // K[b, s, kvh, 0] 的偏移。
    const std::size_t k_offset =
        (
            (
                static_cast<std::size_t>(b) * S + s
            ) * KVH + kvh
        ) * D;

    float dot = 0.0f;

    for (int d = 0; d < D; ++d) {
        dot +=
            q[q_offset + d] *
            k_cache[k_offset + d];
    }

    // scores 的展平布局与 global_tid 完全一致。
    scores[global_tid] = dot * scale;
}

// ============================================================
// Kernel 2: Softmax
//
// probs[b, qh, :] = softmax(scores[b, qh, :])
//
// 线程映射：
//   一个 CUDA block 负责一个 (b, qh) row。
//   block 内线程沿 S 维分片扫描。
//
// Decode 下 K/V cache 中只放有效历史 token，
// 所以这里不再需要 causal mask。
// ============================================================

__global__ void gqa_decode_softmax_kernel(
    const float* __restrict__ scores,
    float* __restrict__ probs,
    int B,
    int S,
    int QH
) {
    // 每个 block 对应一个 scores[b, qh, :] row。
    const int row = blockIdx.x;

    const int total_rows = B * QH;

    if (row >= total_rows) {
        return;
    }

    const int tid = threadIdx.x;

    // row_offset 指向 scores[row, 0]。
    const std::size_t row_offset =
        static_cast<std::size_t>(row) * S;

    const float* score_row = scores + row_offset;
    float* prob_row = probs + row_offset;

    // 256 threads = 8 warps，因此需要 8 个 float。
    __shared__ float warp_smem[kSoftmaxWarpCount];

    // --------------------------------------------------------
    // Phase 1: 求 row max
    // --------------------------------------------------------

    float local_max = -FLT_MAX;

    // 每个线程处理：
    // tid, tid + blockDim.x, tid + 2 * blockDim.x, ...
    for (int s = tid; s < S; s += blockDim.x) {
        local_max = fmaxf(local_max, score_row[s]);
    }

    const float row_max =
        block_reduce_max<kSoftmaxThreadsPerBlock>(
            local_max,
            warp_smem
        );

    // --------------------------------------------------------
    // Phase 2: 计算 exp(score - row_max)，同时求和
    // --------------------------------------------------------

    float local_sum = 0.0f;

    for (int s = tid; s < S; s += blockDim.x) {
        const float value = __expf(
            score_row[s] - row_max
        );

        // 临时写入未归一化概率。
        prob_row[s] = value;

        local_sum += value;
    }

    const float row_sum =
        block_reduce_sum<kSoftmaxThreadsPerBlock>(
            local_sum,
            warp_smem
        );

    const float inv_sum =
        row_sum > 0.0f
            ? 1.0f / row_sum
            : 0.0f;

    // --------------------------------------------------------
    // Phase 3: 归一化
    // --------------------------------------------------------

    for (int s = tid; s < S; s += blockDim.x) {
        prob_row[s] *= inv_sum;
    }
}

// ============================================================
// Kernel 3: PV GEMV Baseline
//
// Out[b, qh, d] =
//   sum_s probs[b, qh, s] * V_cache[b, s, kvh, d]
//
// 线程映射：
//   一个线程负责一个 Out[b, qh, d]。
//
// 每个线程沿 S 维串行扫描。
// ============================================================

__global__ void gqa_decode_pv_gemv_kernel(
    const float* __restrict__ probs,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    int B,
    int S,
    int QH,
    int KVH,
    int D
) {
    // 当前线程对应 Out 的一维全局索引。
    const std::size_t global_tid =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    const std::size_t total_outputs =
        static_cast<std::size_t>(B) * QH * D;

    if (global_tid >= total_outputs) {
        return;
    }

    // Out layout:
    //   [B, QH, D]
    //
    // global_tid = ((b * QH + qh) * D + d)

    const int d =
        static_cast<int>(global_tid % D);

    const std::size_t q_head_linear_idx =
        global_tid / D;

    const int qh =
        static_cast<int>(q_head_linear_idx % QH);

    const int b =
        static_cast<int>(q_head_linear_idx / QH);

    const int group_size = QH / KVH;
    const int kvh = qh / group_size;

    // probs[b, qh, 0] 的偏移。
    const std::size_t prob_offset =
        (static_cast<std::size_t>(b) * QH + qh) * S;

    float accumulator = 0.0f;

    for (int s = 0; s < S; ++s) {
        // V[b, s, kvh, 0] 的偏移。
        const std::size_t v_offset =
            (
                (
                    static_cast<std::size_t>(b) * S + s
                ) * KVH + kvh
            ) * D;

        accumulator +=
            probs[prob_offset + s] *
            v_cache[v_offset + d];
    }

    out[global_tid] = accumulator;
}

// ============================================================
// Split-KV Partial Kernel
//
// 目标：
//   把长序列 S 按 split 切分，增加 CTA 数量。
//
// 普通 fused online softmax：
//   一个 block 处理一个 (b, qh)。
//   grid.x = B * QH。
//
// Split-KV：
//   一个 block 处理一个 (b, qh, split_idx)。
//   grid.x = B * QH * num_splits。
//
// 每个 split 独立输出：
//
//   partial_max[b, qh, split]
//      当前 split 内最大 score。
//
//   partial_sum[b, qh, split]
//      Σ exp(score - partial_max)。
//
//   partial_acc[b, qh, split, d]
//      Σ exp(score - partial_max) * V[token, d]。
//
// Merge kernel 后续将这些 partial 状态合并成最终 Out。
// ============================================================

template <int BLOCK_THREADS, int BLOCK_N>
__global__ void gqa_decode_split_kv_partial_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_acc,
    int B,
    int S,
    int QH,
    int KVH,
    int D,
    int num_splits,
    int split_tokens,
    float scale
) {
    // 约束：
    // 128 threads = 32 tokens * 4 lanes/token。
    static_assert(
        BLOCK_THREADS ==
        BLOCK_N * kFusedLanesPerToken,
        "BLOCK_THREADS 必须等于 BLOCK_N * kFusedLanesPerToken"
    );

    // --------------------------------------------------------
    // blockIdx.x <=> (b, qh, split_idx)
    //
    // 展平：
    // global_partial_idx =
    //   ((b * QH + qh) * num_splits + split_idx)
    // --------------------------------------------------------

    const std::size_t global_partial_idx = blockIdx.x;

    const std::size_t total_partial_blocks =
        static_cast<std::size_t>(B) *
        QH *
        num_splits;

    if (global_partial_idx >= total_partial_blocks) {
        return;
    }

    const int split_idx =
        static_cast<int>(
            global_partial_idx % num_splits
        );

    const std::size_t q_head_linear_idx =
        global_partial_idx / num_splits;

    const int batch_idx =
        static_cast<int>(q_head_linear_idx / QH);

    const int q_head_idx =
        static_cast<int>(q_head_linear_idx % QH);

    // GQA 映射。
    const int group_size = QH / KVH;

    const int kv_head_idx =
        q_head_idx / group_size;

    // --------------------------------------------------------
    // 线程角色
    //
    // dim_idx:
    //   在 PV 累加阶段，threadIdx.x 对应输出 d。
    //
    // tile_token_idx:
    //   当前线程所在 4-lane subgroup 对应 tile 内哪个 token。
    //
    // subgroup_lane_idx:
    //   当前线程是 subgroup 内第几个 lane。
    // --------------------------------------------------------

    const int dim_idx = threadIdx.x;

    const int tile_token_idx =
        threadIdx.x / kFusedLanesPerToken;

    const int subgroup_lane_idx =
        threadIdx.x % kFusedLanesPerToken;

    // --------------------------------------------------------
    // 当前 split 的 token 范围：
    //
    // split 0: [0, split_tokens)
    // split 1: [split_tokens, 2 * split_tokens)
    // ...
    //
    // 最后一个 split 可能不足 split_tokens。
    // --------------------------------------------------------

    const int split_start =
        split_idx * split_tokens;

    const int split_end =
        split_start + split_tokens < S
            ? split_start + split_tokens
            : S;

    // Q[batch_idx, q_head_idx, 0] 的偏移。
    const std::size_t q_offset =
        (static_cast<std::size_t>(batch_idx) * QH +
         q_head_idx) * D;

         

    // partial_acc 的逻辑布局：
    // [B, QH, num_splits, D]
    //
    // global_partial_idx 对应前三维。
    const std::size_t partial_acc_offset =
        global_partial_idx * D;

    // 当前线程负责的 d 维局部输出累加器。
    //
    // 最终代表：
    // partial_acc[b, qh, split_idx, d]。
    float output_acc = 0.0f;

    // --------------------------------------------------------
    // Shared Memory
    // --------------------------------------------------------

    // q_smem:
    // 当前 Q[b, qh, :] 向量。
    //
    // 当前 block 处理的所有 token 都复用同一个 Q，
    // 因此从 global memory 读一次后放入 shared memory。
    __shared__ float q_smem[BLOCK_THREADS];

    // score_smem:
    // 当前 tile 的临时数组，长度为 BLOCK_N=32。
    //
    // 第一阶段存 score：
    //   score_smem[i] = score_i
    //
    // 第二阶段复用为未归一化概率：
    //   score_smem[i] = exp(score_i - online_max)
    __shared__ float score_smem[BLOCK_N];

    // block reduction 临时空间。
    // 128 threads = 4 warps，因此需要 4 个 float。
    __shared__ float warp_smem[
        (BLOCK_THREADS + kWarpSize - 1) / kWarpSize
    ];

    // 当前 split 内的 online softmax 状态。
    __shared__ float online_max;

    // 当前 split 内未归一化 softmax 分母：
    // Σ exp(score - online_max)。
    __shared__ float online_sum;

    // 当 online_max 更新时，旧状态需要乘的缩放因子：
    // exp(old_max - new_max)。
    __shared__ float rescale_factor;

    // 前 D 个线程加载当前 Q 向量。
    if (dim_idx < D) {
        q_smem[dim_idx] =
            q[q_offset + dim_idx];
    }

    // 只有 thread 0 初始化 block 公共状态。
    if (dim_idx == 0) {
        online_max = -FLT_MAX;
        online_sum = 0.0f;
        rescale_factor = 0.0f;
    }

    __syncthreads();

    // --------------------------------------------------------
    // 扫描当前 split 内的 token。
    //
    // 每一轮处理 BLOCK_N=32 个 token。
    // --------------------------------------------------------

    for (
        int token_block_start = split_start;
        token_block_start < split_end;
        token_block_start += BLOCK_N
    ) {
        // 当前 subgroup 对应的真实 KV Cache token。
        const int cache_token_idx =
            token_block_start + tile_token_idx;

        // ----------------------------------------------------
        // QK：4-lane subgroup 计算一个 token 的 score。
        // ----------------------------------------------------

        float local_dot = 0.0f;

        if (cache_token_idx < split_end) {
            // K[batch_idx, cache_token_idx, kv_head_idx, 0]。
            const std::size_t k_offset =
                (
                    (
                        static_cast<std::size_t>(batch_idx) * S +
                        cache_token_idx
                    ) * KVH + kv_head_idx
                ) * D;

            // 每个 subgroup lane 只计算 D 的一部分。
            for (
                int d = subgroup_lane_idx;
                d < D;
                d += kFusedLanesPerToken
            ) {
                local_dot +=
                    q_smem[d] *
                    k_cache[k_offset + d];
            }
        }

        // 仅在当前 4 个线程中规约。
        //
        // lane 0 得到完整 QK 点积。
        const float score =
            subgroup_reduce_sum<kFusedLanesPerToken>(
                local_dot
            );

        // 每个 subgroup 只有 lane 0 写出对应 token 的 score。
        if (subgroup_lane_idx == 0) {
            score_smem[tile_token_idx] =
                cache_token_idx < split_end
                    ? score * scale
                    : -FLT_MAX;
        }

        __syncthreads();

        // ----------------------------------------------------
        // 求当前 tile 最大 score。
        // ----------------------------------------------------

        float local_tile_max = -FLT_MAX;

        // 前 32 个线程各读取一个 score。
        if (dim_idx < BLOCK_N) {
            local_tile_max = score_smem[dim_idx];
        }

        const float tile_max =
            block_reduce_max<BLOCK_THREADS>(
                local_tile_max,
                warp_smem
            );

        // ----------------------------------------------------
        // Online softmax：更新 max 和缩放因子。
        //
        // previous_max:
        //   之前已处理 token 的最大 score。
        //
        // next_max:
        //   max(previous_max, tile_max)。
        //
        // 旧状态从 previous_max 坐标系转到 next_max 坐标系：
        //
        // exp(score - next_max)
        // = exp(score - previous_max)
        //   * exp(previous_max - next_max)
        // ----------------------------------------------------

        if (dim_idx == 0) {
            const float previous_max = online_max;

            const float next_max =
                fmaxf(previous_max, tile_max);

            rescale_factor =
                previous_max == -FLT_MAX
                    ? 0.0f
                    : __expf(previous_max - next_max);

            online_max = next_max;
        }

        __syncthreads();

        // ----------------------------------------------------
        // 计算当前 tile 的未归一化概率。
        //
        // p_i = exp(score_i - online_max)
        // ----------------------------------------------------

        float local_probability_sum = 0.0f;

        if (dim_idx < BLOCK_N) {
            const int probability_token_idx =
                token_block_start + dim_idx;

            float probability = 0.0f;

            if (probability_token_idx < split_end) {
                probability = __expf(
                    score_smem[dim_idx] - online_max
                );
            }

            // score_smem 从 score 复用为 p_i。
            score_smem[dim_idx] = probability;

            local_probability_sum = probability;
        }

        const float tile_probability_sum =
            block_reduce_sum<BLOCK_THREADS>(
                local_probability_sum,
                warp_smem
            );

        // 更新当前 split 的 softmax 分母。
        //
        // new_sum =
        //   rescale_factor * old_sum
        //   + current_tile_probability_sum。
        if (dim_idx == 0) {
            online_sum =
                rescale_factor * online_sum +
                tile_probability_sum;
        }

        __syncthreads();

        // ----------------------------------------------------
        // PV：更新当前 split 的 partial_acc。
        //
        // 每个线程负责一个输出维度 d。
        //
        // output_acc[d] =
        //   Σ exp(score - online_max) * V[token, d]
        // ----------------------------------------------------

        if (dim_idx < D) {
            // old output_acc 也要重标定到新的 online_max。
            output_acc *= rescale_factor;

            #pragma unroll
            for (
                int output_tile_token_idx = 0;
                output_tile_token_idx < BLOCK_N;
                ++output_tile_token_idx
            ) {
                const int output_cache_token_idx =
                    token_block_start +
                    output_tile_token_idx;

                if (output_cache_token_idx < split_end) {
                    // V[batch_idx, output_cache_token_idx,
                    //   kv_head_idx, dim_idx]。
                    const std::size_t v_offset =
                        (
                            (
                                static_cast<std::size_t>(batch_idx) * S +
                                output_cache_token_idx
                            ) * KVH + kv_head_idx
                        ) * D;

                    output_acc +=
                        score_smem[output_tile_token_idx] *
                        v_cache[v_offset + dim_idx];
                }
            }
        }

        // 当前 tile 使用结束；下一个 tile 会覆写 score_smem。
        __syncthreads();
    }

    // --------------------------------------------------------
    // 当前 split 已处理完毕，写回 partial state。
    // --------------------------------------------------------

    if (dim_idx == 0) {
        partial_max[global_partial_idx] = online_max;
        partial_sum[global_partial_idx] = online_sum;
    }

    if (dim_idx < D) {
        partial_acc[partial_acc_offset + dim_idx] =
            output_acc;
    }
}

// ============================================================
// Split-KV Merge Kernel
//
// 一个 block 负责一个 (b, qh)。
//
// 输入：
//   partial_max[i]      = m_i
//   partial_sum[i]      = l_i
//   partial_acc[i, d]   = acc_i[d]
//
// 合并公式：
//
//   merged_max = max_i(m_i)
//
//   weight_i = exp(m_i - merged_max)
//
//   merged_sum = Σ weight_i * l_i
//
//   merged_acc[d] = Σ weight_i * acc_i[d]
//
//   Out[d] = merged_acc[d] / merged_sum
// ============================================================

template <int BLOCK_THREADS>
__global__ void gqa_decode_split_kv_merge_kernel(
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const float* __restrict__ partial_acc,
    float* __restrict__ out,
    int B,
    int QH,
    int D,
    int num_splits
) {
    // 一个 block 对应一个 (b, qh)。
    const int global_q_head_idx = blockIdx.x;

    const int total_q_heads = B * QH;

    if (global_q_head_idx >= total_q_heads) {
        return;
    }

    // 当前线程对应输出维度 d。
    const int dim_idx = threadIdx.x;

    // partial_max / partial_sum 逻辑布局：
    // [B, QH, num_splits]
    //
    // partial_base 指向当前 (b, qh) 的 split 0。
    const std::size_t partial_base =
        static_cast<std::size_t>(global_q_head_idx) *
        num_splits;

    // Out 逻辑布局：
    // [B, QH, D]
    //
    // out_offset 指向 Out[b, qh, 0]。
    const std::size_t out_offset =
        static_cast<std::size_t>(global_q_head_idx) *
        D;

    // Dynamic shared memory 布局：
    //
    // merge_smem[0 : num_splits)
    //   当前 (b, qh) 下每个 split 的 weight_i。
    //
    // merge_smem[num_splits : num_splits + warp_count)
    //   block_reduce_max / block_reduce_sum 的 warp 临时空间。
    extern __shared__ float merge_smem[];

    float* split_weight_smem = merge_smem;

    float* warp_smem =
        merge_smem + num_splits;

    // --------------------------------------------------------
    // Phase 1: merged_max = max(partial_max[i])
    // --------------------------------------------------------

    float local_max = -FLT_MAX;

    for (
        int split_idx = dim_idx;
        split_idx < num_splits;
        split_idx += BLOCK_THREADS
    ) {
        local_max = fmaxf(
            local_max,
            partial_max[partial_base + split_idx]
        );
    }

    const float merged_max =
        block_reduce_max<BLOCK_THREADS>(
            local_max,
            warp_smem
        );

    // --------------------------------------------------------
    // Phase 2: 计算每个 split 的重标定权重与 merged_sum。
    // --------------------------------------------------------

    float local_sum = 0.0f;

    for (
        int split_idx = dim_idx;
        split_idx < num_splits;
        split_idx += BLOCK_THREADS
    ) {
        const float weight = __expf(
            partial_max[partial_base + split_idx] -
            merged_max
        );

        // 后面合并 partial_acc 时需要再次使用这个 weight。
        split_weight_smem[split_idx] = weight;

        local_sum +=
            weight *
            partial_sum[partial_base + split_idx];
    }

    // 保证所有 split weight 都已写入 shared memory。
    __syncthreads();

    const float merged_sum =
        block_reduce_sum<BLOCK_THREADS>(
            local_sum,
            warp_smem
        );

    // --------------------------------------------------------
    // Phase 3: 合并 partial_acc。
    //
    // 每个线程处理一个输出维度 d。
    // --------------------------------------------------------

    if (dim_idx < D) {
        float merged_acc = 0.0f;

        for (
            int split_idx = 0;
            split_idx < num_splits;
            ++split_idx
        ) {
            // partial_acc layout:
            // [B, QH, num_splits, D]
            const std::size_t partial_acc_offset =
                (partial_base + split_idx) * D;

            merged_acc +=
                split_weight_smem[split_idx] *
                partial_acc[
                    partial_acc_offset + dim_idx
                ];
        }

        out[out_offset + dim_idx] =
            merged_sum > 0.0f
                ? merged_acc / merged_sum
                : 0.0f;
    }
}

// ============================================================
// Fused Online Softmax Kernel
//
// 一个 block 负责一个 (b, qh)。
//
// 与三段式 baseline 相比：
//   - 不写 scores 到 global memory。
//   - 不读取 scores。
//   - 不写 probs 到 global memory。
//   - 不读取 probs。
//   - 只 launch 一个 kernel。
//
// 线程映射：
//   QK 阶段：
//     每 4 个线程协作计算一个 token score。
//     128 threads 同时处理 32 个 token。
//
//   PV 阶段：
//     threadIdx.x 对应输出维度 d。
// ============================================================

template <int BLOCK_THREADS, int BLOCK_N>
__global__ void gqa_decode_fused_online_softmax_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    int B,
    int S,
    int QH,
    int KVH,
    int D,
    float scale
) {
    static_assert(
        BLOCK_THREADS ==
        BLOCK_N * kFusedLanesPerToken,
        "BLOCK_THREADS 必须等于 BLOCK_N * kFusedLanesPerToken"
    );

    // 一个 block 对应一个 (b, qh)。
    const int global_q_head_idx = blockIdx.x;

    const int total_q_heads = B * QH;

    if (global_q_head_idx >= total_q_heads) {
        return;
    }

    const int batch_idx = global_q_head_idx / QH;
    const int q_head_idx = global_q_head_idx % QH;

    const int group_size = QH / KVH;
    const int kv_head_idx = q_head_idx / group_size;

    // dim_idx 在 PV 阶段对应输出维度 d。
    const int dim_idx = threadIdx.x;

    // 当前线程的 subgroup 对应 tile 中哪个 token。
    const int tile_token_idx =
        threadIdx.x / kFusedLanesPerToken;

    // 当前线程在 4-lane subgroup 中的位置。
    const int subgroup_lane_idx =
        threadIdx.x % kFusedLanesPerToken;

    // Q[batch_idx, q_head_idx, 0] 偏移。
    const std::size_t q_offset =
        (static_cast<std::size_t>(batch_idx) * QH +
         q_head_idx) * D;

    // 当前线程负责的输出维度 d 的未归一化累加器。
    float output_acc = 0.0f;

    // 当前 Q head 向量。
    __shared__ float q_smem[BLOCK_THREADS];

    // 当前 tile 临时 score / probability。
    __shared__ float score_smem[BLOCK_N];

    // block reduction 临时空间。
    __shared__ float warp_smem[
        (BLOCK_THREADS + kWarpSize - 1) / kWarpSize
    ];

    // 当前完整 KV Cache 前缀的 online softmax 状态。
    __shared__ float online_max;

    // Σ exp(score - online_max)。
    __shared__ float online_sum;

    // old state 转换到 new max 的缩放系数。
    __shared__ float rescale_factor;

    // 加载 Q[b, qh, :] 到 shared memory。
    if (dim_idx < D) {
        q_smem[dim_idx] = q[q_offset + dim_idx];
    }

    // 初始化 online softmax 状态。
    if (dim_idx == 0) {
        online_max = -FLT_MAX;
        online_sum = 0.0f;
        rescale_factor = 0.0f;
    }

    __syncthreads();

    // --------------------------------------------------------
    // 遍历完整 KV Cache，每轮处理 BLOCK_N=32 个 token。
    // --------------------------------------------------------

    for (
        int token_block_start = 0;
        token_block_start < S;
        token_block_start += BLOCK_N
    ) {
        const int cache_token_idx =
            token_block_start + tile_token_idx;

        // ----------------------------------------------------
        // QK：每 4 个线程合作计算一个 token score。
        // ----------------------------------------------------

        float local_dot = 0.0f;

        if (cache_token_idx < S) {
            // K[batch_idx, cache_token_idx, kv_head_idx, 0]。
            const std::size_t k_offset =
                (
                    (
                        static_cast<std::size_t>(batch_idx) * S +
                        cache_token_idx
                    ) * KVH + kv_head_idx
                ) * D;

            for (
                int d = subgroup_lane_idx;
                d < D;
                d += kFusedLanesPerToken
            ) {
                local_dot +=
                    q_smem[d] *
                    k_cache[k_offset + d];
            }
        }

        // 当前 subgroup lane 0 获得完整 score。
        const float score =
            subgroup_reduce_sum<kFusedLanesPerToken>(
                local_dot
            );

        // lane 0 写入每个 token 对应的 score。
        if (subgroup_lane_idx == 0) {
            score_smem[tile_token_idx] =
                cache_token_idx < S
                    ? score * scale
                    : -FLT_MAX;
        }

        __syncthreads();

        // ----------------------------------------------------
        // 当前 tile 最大 score。
        // ----------------------------------------------------

        float local_tile_max = -FLT_MAX;

        if (dim_idx < BLOCK_N) {
            local_tile_max = score_smem[dim_idx];
        }

        const float tile_max =
            block_reduce_max<BLOCK_THREADS>(
                local_tile_max,
                warp_smem
            );

        // ----------------------------------------------------
        // 更新 online max。
        // ----------------------------------------------------

        if (dim_idx == 0) {
            const float previous_max = online_max;

            const float next_max =
                fmaxf(previous_max, tile_max);

            rescale_factor =
                previous_max == -FLT_MAX
                    ? 0.0f
                    : __expf(previous_max - next_max);

            online_max = next_max;
        }

        __syncthreads();

        // ----------------------------------------------------
        // 当前 tile 的未归一化 softmax 概率。
        // ----------------------------------------------------

        float local_probability_sum = 0.0f;

        if (dim_idx < BLOCK_N) {
            const int probability_token_idx =
                token_block_start + dim_idx;

            float probability = 0.0f;

            if (probability_token_idx < S) {
                probability = __expf(
                    score_smem[dim_idx] - online_max
                );
            }

            score_smem[dim_idx] = probability;
            local_probability_sum = probability;
        }

        const float tile_probability_sum =
            block_reduce_sum<BLOCK_THREADS>(
                local_probability_sum,
                warp_smem
            );

        // 更新完整前缀的 softmax 分母。
        if (dim_idx == 0) {
            online_sum =
                rescale_factor * online_sum +
                tile_probability_sum;
        }

        __syncthreads();

        // ----------------------------------------------------
        // PV：每个线程处理一个输出维度 d。
        // ----------------------------------------------------

        if (dim_idx < D) {
            // 旧累积值重标定。
            output_acc *= rescale_factor;

            #pragma unroll
            for (
                int output_tile_token_idx = 0;
                output_tile_token_idx < BLOCK_N;
                ++output_tile_token_idx
            ) {
                const int output_cache_token_idx =
                    token_block_start +
                    output_tile_token_idx;

                if (output_cache_token_idx < S) {
                    // V[batch_idx, output_cache_token_idx,
                    //   kv_head_idx, dim_idx]。
                    const std::size_t v_offset =
                        (
                            (
                                static_cast<std::size_t>(batch_idx) * S +
                                output_cache_token_idx
                            ) * KVH + kv_head_idx
                        ) * D;

                    output_acc +=
                        score_smem[output_tile_token_idx] *
                        v_cache[v_offset + dim_idx];
                }
            }
        }

        __syncthreads();
    }

    // 最终归一化：
    //
    // Out[d] =
    //   Σ exp(score - max) * V[d] /
    //   Σ exp(score - max)
    if (dim_idx < D) {
        out[q_offset + dim_idx] =
            online_sum > 0.0f
                ? output_acc / online_sum
                : 0.0f;
    }
}

// ============================================================
// 返回当前 S 需要多少个 Split-KV split。
// ============================================================

// ============================================================
// 根据当前序列长度 S 和指定 split_tokens，计算 Split-KV 的 split 数量。
//
// num_splits = ceil(S / split_tokens)
//
// 例如：
// S=8192, split_tokens=512
// num_splits=16
// ============================================================

int get_decode_attention_split_kv_num_splits(
    int S,
    int split_tokens
) {
    if (S <= 0) {
        throw std::invalid_argument(
            "S 必须为正数"
        );
    }

    if (split_tokens <= 0) {
        throw std::invalid_argument(
            "split_tokens 必须为正数"
        );
    }

    return 1 + (S - 1) / split_tokens;
}

// ============================================================
// Launchers
// ============================================================

// 启动 QK GEMV baseline。
void decode_qk_gemv_launcher(
    const float* d_q,
    const float* d_k_cache,
    float* d_scores,
    int B,
    int S,
    int QH,
    int KVH,
    int D
) {
    check_not_null(d_q, "d_q");
    check_not_null(d_k_cache, "d_k_cache");
    check_not_null(d_scores, "d_scores");

    check_gqa_decode_shape(B, S, QH, KVH, D);

    const std::size_t total_scores =
        static_cast<std::size_t>(B) * QH * S;

    const std::size_t blocks =
        (total_scores + kGemmThreadsPerBlock - 1) /
        kGemmThreadsPerBlock;

    if (
        blocks >
        static_cast<std::size_t>(
            std::numeric_limits<unsigned int>::max()
        )
    ) {
        throw std::invalid_argument(
            "QK grid.x 超出范围"
        );
    }

    const float scale =
        1.0f / std::sqrt(static_cast<float>(D));

    gqa_decode_qk_gemv_kernel<<<
        static_cast<unsigned int>(blocks),
        kGemmThreadsPerBlock
    >>>(
        d_q,
        d_k_cache,
        d_scores,
        B,
        S,
        QH,
        KVH,
        D,
        scale
    );

    check_launch("gqa_decode_qk_gemv_kernel");
}

// 启动 Softmax baseline。
void decode_softmax_launcher(
    const float* d_scores,
    float* d_probs,
    int B,
    int S,
    int QH
) {
    check_not_null(d_scores, "d_scores");
    check_not_null(d_probs, "d_probs");

    if (B <= 0 || S <= 0 || QH <= 0) {
        throw std::invalid_argument(
            "B、S、QH 必须全部为正数"
        );
    }

    // 一个 block 对应一个 (b, qh)。
    const std::size_t total_rows =
        static_cast<std::size_t>(B) * QH;

    if (
        total_rows >
        static_cast<std::size_t>(
            std::numeric_limits<unsigned int>::max()
        )
    ) {
        throw std::invalid_argument(
            "Softmax grid.x 超出范围"
        );
    }

    gqa_decode_softmax_kernel<<<
        static_cast<unsigned int>(total_rows),
        kSoftmaxThreadsPerBlock
    >>>(
        d_scores,
        d_probs,
        B,
        S,
        QH
    );

    check_launch("gqa_decode_softmax_kernel");
}

// 启动 PV GEMV baseline。
void decode_pv_gemv_launcher(
    const float* d_probs,
    const float* d_v_cache,
    float* d_out,
    int B,
    int S,
    int QH,
    int KVH,
    int D
) {
    check_not_null(d_probs, "d_probs");
    check_not_null(d_v_cache, "d_v_cache");
    check_not_null(d_out, "d_out");

    check_gqa_decode_shape(B, S, QH, KVH, D);

    const std::size_t total_outputs =
        static_cast<std::size_t>(B) * QH * D;

    const std::size_t blocks =
        (total_outputs + kGemmThreadsPerBlock - 1) /
        kGemmThreadsPerBlock;

    if (
        blocks >
        static_cast<std::size_t>(
            std::numeric_limits<unsigned int>::max()
        )
    ) {
        throw std::invalid_argument(
            "PV grid.x 超出范围"
        );
    }

    gqa_decode_pv_gemv_kernel<<<
        static_cast<unsigned int>(blocks),
        kGemmThreadsPerBlock
    >>>(
        d_probs,
        d_v_cache,
        d_out,
        B,
        S,
        QH,
        KVH,
        D
    );

    check_launch("gqa_decode_pv_gemv_kernel");
}

// 三段式 baseline：
// QK GEMV -> Softmax -> PV GEMV。
void launch_decode_attention_three_stage(
    const float* d_q,
    const float* d_k_cache,
    const float* d_v_cache,
    float* d_scores,
    float* d_probs,
    float* d_out,
    int B,
    int S,
    int QH,
    int KVH,
    int D
) {
    decode_qk_gemv_launcher(
        d_q,
        d_k_cache,
        d_scores,
        B,
        S,
        QH,
        KVH,
        D
    );

    decode_softmax_launcher(
        d_scores,
        d_probs,
        B,
        S,
        QH
    );

    decode_pv_gemv_launcher(
        d_probs,
        d_v_cache,
        d_out,
        B,
        S,
        QH,
        KVH,
        D
    );
}

// 启动 Fused Online Softmax v1。
void launch_decode_attention_fused_online_softmax(
    const float* d_q,
    const float* d_k_cache,
    const float* d_v_cache,
    float* d_out,
    int B,
    int S,
    int QH,
    int KVH,
    int D
) {
    check_not_null(d_q, "d_q");
    check_not_null(d_k_cache, "d_k_cache");
    check_not_null(d_v_cache, "d_v_cache");
    check_not_null(d_out, "d_out");

    check_gqa_decode_shape(B, S, QH, KVH, D);

    // 当前实现中：
    // threadIdx.x 负责 output d，
    // 因此暂时只支持 D <= 128。
    if (D > kFusedThreadsPerBlock) {
        throw std::invalid_argument(
            "fused online softmax 当前只支持 D <= 128"
        );
    }

    // 一个 block 对应一个 (b, qh)。
    const std::size_t total_q_heads =
        static_cast<std::size_t>(B) * QH;

    if (
        total_q_heads >
        static_cast<std::size_t>(
            std::numeric_limits<unsigned int>::max()
        )
    ) {
        throw std::invalid_argument(
            "Fused grid.x 超出范围"
        );
    }

    const float scale =
        1.0f / std::sqrt(static_cast<float>(D));

    gqa_decode_fused_online_softmax_kernel<
        kFusedThreadsPerBlock,
        kFusedTokensPerBlock
    ><<<
        static_cast<unsigned int>(total_q_heads),
        kFusedThreadsPerBlock
    >>>(
        d_q,
        d_k_cache,
        d_v_cache,
        d_out,
        B,
        S,
        QH,
        KVH,
        D,
        scale
    );

    check_launch(
        "gqa_decode_fused_online_softmax_kernel"
    );
}

// ============================================================
// 启动 Split-KV Decode Attention。
//
// Stage 1:
// partial kernel 输出 partial_max / partial_sum / partial_acc。
//
// Stage 2:
// merge kernel 合并全部 split，得到最终 Out。
//
// split_tokens 是运行时参数，可以用于 auto-tuning。
// ============================================================

void launch_decode_attention_split_kv(
    const float* d_q,
    const float* d_k_cache,
    const float* d_v_cache,
    float* d_partial_max,
    float* d_partial_sum,
    float* d_partial_acc,
    float* d_out,
    int B,
    int S,
    int QH,
    int KVH,
    int D,
    int split_tokens
) {
    check_not_null(d_q, "d_q");
    check_not_null(d_k_cache, "d_k_cache");
    check_not_null(d_v_cache, "d_v_cache");
    check_not_null(d_partial_max, "d_partial_max");
    check_not_null(d_partial_sum, "d_partial_sum");
    check_not_null(d_partial_acc, "d_partial_acc");
    check_not_null(d_out, "d_out");

    check_gqa_decode_shape(B, S, QH, KVH, D);

    if (D > kFusedThreadsPerBlock) {
        throw std::invalid_argument(
            "Split-KV 当前只支持 D <= 128"
        );
    }

    if (split_tokens <= 0) {
        throw std::invalid_argument(
            "split_tokens 必须为正数"
        );
    }

    const int num_splits =
        get_decode_attention_split_kv_num_splits(
            S,
            split_tokens
        );

    const std::size_t total_q_heads =
        static_cast<std::size_t>(B) * QH;

    // 一个 partial block 对应一个 (b, qh, split_idx)。
    const std::size_t total_partial_blocks =
        total_q_heads * num_splits;

    if (
        total_q_heads >
        static_cast<std::size_t>(
            std::numeric_limits<unsigned int>::max()
        )
    ) {
        throw std::invalid_argument(
            "Split-KV merge grid.x 超出范围"
        );
    }

    if (
        total_partial_blocks >
        static_cast<std::size_t>(
            std::numeric_limits<unsigned int>::max()
        )
    ) {
        throw std::invalid_argument(
            "Split-KV partial grid.x 超出范围"
        );
    }

    const float scale =
        1.0f / std::sqrt(static_cast<float>(D));

    // --------------------------------------------------------
    // Stage 1:
    //
    // grid.x = B * QH * num_splits
    //
    // 每个 block 处理一个：
    // (batch_idx, q_head_idx, split_idx)
    // --------------------------------------------------------

    gqa_decode_split_kv_partial_kernel<
        kFusedThreadsPerBlock,
        kFusedTokensPerBlock
    ><<<
        static_cast<unsigned int>(total_partial_blocks),
        kFusedThreadsPerBlock
    >>>(
        d_q,
        d_k_cache,
        d_v_cache,
        d_partial_max,
        d_partial_sum,
        d_partial_acc,
        B,
        S,
        QH,
        KVH,
        D,
        num_splits,
        split_tokens,
        scale
    );

    check_launch(
        "gqa_decode_split_kv_partial_kernel"
    );

    // --------------------------------------------------------
    // Stage 2:
    //
    // 每个 (b, qh) 一个 merge block。
    //
    // Dynamic shared memory:
    //
    // num_splits 个 float：
    //   split_weight_smem
    //
    // kMergeWarpCount 个 float：
    //   reduction 使用的 warp_smem
    // --------------------------------------------------------

    constexpr int kMergeWarpCount =
        (kFusedThreadsPerBlock + kWarpSize - 1) /
        kWarpSize;

    const std::size_t merge_smem_bytes =
        static_cast<std::size_t>(
            num_splits + kMergeWarpCount
        ) * sizeof(float);

    gqa_decode_split_kv_merge_kernel<
        kFusedThreadsPerBlock
    ><<<
        static_cast<unsigned int>(total_q_heads),
        kFusedThreadsPerBlock,
        merge_smem_bytes
    >>>(
        d_partial_max,
        d_partial_sum,
        d_partial_acc,
        d_out,
        B,
        QH,
        D,
        num_splits
    );

    check_launch(
        "gqa_decode_split_kv_merge_kernel"
    );
}