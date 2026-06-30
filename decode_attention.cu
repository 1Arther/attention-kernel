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
Q:       [B, QH, D]
K cache: [B, S, KVH, D]
V cache: [B, S, KVH, D]
scores:  [B, QH, S]
probs:   [B, QH, S]
Out:     [B, QH, D]
B   : batch size
S   : 当前 KV Cache 长度，也就是历史 token 数
QH  : Query head 数
KVH : Key / Value head 数
D   : 每个 head 的维度
*/

namespace {

constexpr int kGemmThreadsPerBlock = 256;
constexpr int kSoftmaxThreadsPerBlock = 256;
constexpr int kWarpSize = 32;
constexpr int kSoftmaxWarpCount =
    kSoftmaxThreadsPerBlock / kWarpSize;

constexpr int kFusedThreadsPerBlock = 128;  //一个 fused block：128 线程
constexpr int kFusedTokensPerBlock = 32;    //一次处理：32 个 KV token
constexpr int kFusedLanesPerToken = 4;      //每个 token 的 QK 点积：4 个线程协作

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

    if (QH % KVH != 0) {  //GQA 的核心是多个 Q head 共享一组 K/V head。
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

//一个block处理一行，得到一行的最大值
template <int BLOCK_THREADS>
__device__ float block_reduce_max(
    float value,
    float* warp_smem
) {
    const int lane_id = threadIdx.x % kWarpSize;
    const int warp_id = threadIdx.x / kWarpSize;
    constexpr int kWarpCount =
        (BLOCK_THREADS + kWarpSize - 1) / kWarpSize;

    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(
            value,
            __shfl_down_sync(0xffffffff, value, offset)
        );
    }

    if (lane_id == 0) {
        warp_smem[warp_id] = value;
    }

    __syncthreads();

    if (warp_id == 0) {
        value =
            lane_id < kWarpCount
                ? warp_smem[lane_id]
                : -FLT_MAX;

        for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
            value = fmaxf(
                value,
                __shfl_down_sync(0xffffffff, value, offset)
            );
        }

        if (lane_id == 0) {
            warp_smem[0] = value;
        }
    }

    __syncthreads();

    return warp_smem[0];
}

//一个block处理一行，得到总和
template <int BLOCK_THREADS>
__device__ float block_reduce_sum(
    float value,
    float* warp_smem
) {
    const int lane_id = threadIdx.x % kWarpSize;
    const int warp_id = threadIdx.x / kWarpSize;
    constexpr int kWarpCount =
        (BLOCK_THREADS + kWarpSize - 1) / kWarpSize;

    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }

    if (lane_id == 0) {
        warp_smem[warp_id] = value;
    }

    __syncthreads();

    if (warp_id == 0) {
        value =
            lane_id < kWarpCount
                ? warp_smem[lane_id]
                : 0.0f;

        for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }

        if (lane_id == 0) {
            warp_smem[0] = value;
        }
    }

    __syncthreads();

    return warp_smem[0];
}

}  // namespace

// ============================================================
// GQA Decode Attention layout:
//
// Q:       [B, QH, D]
// K cache: [B, S, KVH, D]
// V cache: [B, S, KVH, D]
// scores:  [B, QH, S]
// probs:   [B, QH, S]
// O:       [B, QH, D]
//
// K/V cache 默认已经包含写入 cache 的历史 token。
// 若模型使用 RoPE，则 K cache 应当是 RoPE 后的 K。
// ============================================================

// ============================================================
// CPU Reference
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

    const int group_size = QH / KVH;  //多个 Q head 共享一组 K/V head
    const float scale =
        1.0f / std::sqrt(static_cast<float>(D));

    for (int b = 0; b < B; ++b) {
        for (int qh = 0; qh < QH; ++qh) {
            const int kvh = qh / group_size;  //共享

            //score[s] = dot(Q[b, qh, :], K_cache[b, s, kvh, :]) / sqrt(D)
            const std::size_t q_offset =
                (static_cast<std::size_t>(b) * QH + qh) * D;

            const std::size_t score_offset =
                (static_cast<std::size_t>(b) * QH + qh) * S;

            float max_score = -std::numeric_limits<float>::infinity();

            // Kernel 1 对应：Q × K^T。 当成[B,QH,D]*[B,S,D]T
            for (int s = 0; s < S; ++s) {
                const std::size_t kv_offset =
                    (
                        (
                            static_cast<std::size_t>(b) * S + s
                        ) * KVH + kvh
                    ) * D;

                float dot = 0.0f;

                for (int d = 0; d < D; ++d) {
                    dot += q[q_offset + d] *
                           k_cache[kv_offset + d];
                }

                const float score = dot * scale;

                scores[score_offset + s] = score;
                max_score = std::max(max_score, score);
            }

            // Kernel 2 对应：stable softmax。
            //prob[s] = softmax(score)[s]
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

            // Kernel 3 对应：P × V。
            /*
            Out[b, qh, d] =
            sum over s:
            prob[s] * V_cache[b, s, kvh, d]
            */
            //[B QH S]*[B S D]
            for (int d = 0; d < D; ++d) {
                float accumulator = 0.0f;

                for (int s = 0; s < S; ++s) {
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
// Kernel 1:
// scores[b, qh, s] = dot(Q[b, qh], K_cache[b, s, kvh]) * scale
//
// 一个线程负责一个 score。
// 这是最清晰的 GEMV baseline；同一个 Q 会被多个 token 重复读取。
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
    const std::size_t flat_score_id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    const std::size_t total_scores =
        static_cast<std::size_t>(B) * QH * S;

    if (flat_score_id >= total_scores) {
        return;
    }

    //  scores:  [B, QH, S]
    const int s =
        static_cast<int>(flat_score_id % S);

    const std::size_t q_head_linear_id =
        flat_score_id / S;

    const int qh =
        static_cast<int>(q_head_linear_id % QH);

    const int b =
        static_cast<int>(q_head_linear_id / QH);

    const int group_size = QH / KVH;
    const int kvh = qh / group_size;

    const std::size_t q_offset =
        (static_cast<std::size_t>(b) * QH + qh) * D;

    const std::size_t k_offset =
        (
            (
                static_cast<std::size_t>(b) * S + s
            ) * KVH + kvh
        ) * D;

    float dot = 0.0f;

    for (int d = 0; d < D; ++d) {
        dot += q[q_offset + d] *
               k_cache[k_offset + d];
    }

    scores[flat_score_id] = dot * scale;
}

// ============================================================
// Kernel 2:
// probs[b, qh, :] = softmax(scores[b, qh, :])
//
// 一个 block 负责一条 score row，即一个 (b, qh)。
// Decode 下 KV cache 中只有有效历史 token，无需 causal mask。
// ============================================================

__global__ void gqa_decode_softmax_kernel(
    const float* __restrict__ scores,
    float* __restrict__ probs,
    int B,
    int S,
    int QH
) {
    const int row = blockIdx.x;
    const int total_rows =  B * QH;

    if (row >= total_rows) {
        return;
    }

    const int tid = threadIdx.x;
    
    const std::size_t row_offset =
        static_cast<std::size_t>(row) * S;

    const float* score_row = scores + row_offset;
    float* prob_row = probs + row_offset;

    __shared__ float warp_smem[
        kSoftmaxWarpCount
    ];

    float local_max = -FLT_MAX;

    for (int s = tid; s < S; s += blockDim.x) {
        local_max = fmaxf(local_max, score_row[s]);
    }

    const float row_max =
        block_reduce_max<kSoftmaxThreadsPerBlock>(
            local_max,
            warp_smem
        );

    float local_sum = 0.0f;

    for (int s = tid; s < S; s += blockDim.x) {
        const float value = __expf(score_row[s] - row_max);

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

    for (int s = tid; s < S; s += blockDim.x) {
        prob_row[s] *= inv_sum;
    }
}

// ============================================================
// Kernel 3:
// O[b, qh, d] = sum_s probs[b, qh, s] * V_cache[b, s, kvh, d]
//
// 一个线程负责一个输出维度 d。
// 这是沿 S 维归约的 GEMV。
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
    const std::size_t flat_out_id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    const std::size_t total_outputs =
        static_cast<std::size_t>(B) * QH * D;

    if (flat_out_id >= total_outputs) {
        return;
    }

    const int d =
        static_cast<int>(flat_out_id % D);

    const std::size_t q_head_linear_id =
        flat_out_id / D;

    const int qh =
        static_cast<int>(q_head_linear_id % QH);

    const int b =
        static_cast<int>(q_head_linear_id / QH);

    const int group_size = QH / KVH;
    const int kvh = qh / group_size;

    const std::size_t prob_offset =
        (static_cast<std::size_t>(b) * QH + qh) * S;

    float accumulator = 0.0f;

    for (int s = 0; s < S; ++s) {
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

    out[flat_out_id] = accumulator;
}

//online_softmax
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
    const int global_q_head_idx = blockIdx.x;
    const int total_q_heads = B * QH;

    if (global_q_head_idx >= total_q_heads) {
        return;
    }

    const int batch_idx = global_q_head_idx / QH;
    const int q_head_idx = global_q_head_idx % QH;

    const int group_size = QH / KVH;
    const int kv_head_idx = q_head_idx / group_size;

    const int dim_idx = threadIdx.x;  //D维

    const std::size_t q_offset =
        (static_cast<std::size_t>(batch_idx) * QH + q_head_idx) * D;

    float q_value = 0.0f;  //每个线程维护的寄存器，把自己的Q元素读到寄存器

    if (dim_idx < D) {
        q_value = q[q_offset + dim_idx];
    }

    float output_acc = 0.0f;  //输出累加器

    __shared__ float score_smem[BLOCK_N];  //长度32

    __shared__ float warp_smem[
        (BLOCK_THREADS + kWarpSize - 1) / kWarpSize
    ];

    __shared__ float online_max;  //当前已处理 token 的最大 score
    __shared__ float online_sum;  //当前已处理 token 的最大 score
    __shared__ float rescale_factor;  //旧状态切换到新最大值时的缩放系数

    ////thread0写入，避免重复写
    if (dim_idx == 0) {
        online_max = -FLT_MAX;
        online_sum = 0.0f;
        rescale_factor = 0.0f;
    }

    __syncthreads();

    for (
        int token_block_start = 0;
        token_block_start < S;
        token_block_start += BLOCK_N
    ) {
        #pragma unroll
        for (
            int tile_token_idx = 0;
            tile_token_idx < BLOCK_N;
            ++tile_token_idx
        ) {
            const int cache_token_idx =
                token_block_start + tile_token_idx;

            float local_dot = 0.0f;

            if (dim_idx < D && cache_token_idx < S) {
                const std::size_t k_offset =
                    (
                        (
                            static_cast<std::size_t>(batch_idx) * S +
                            cache_token_idx
                        ) * KVH + kv_head_idx
                    ) * D;

                local_dot =
                    q_value *
                    k_cache[k_offset + dim_idx];
            }

            //局部乘积求和
            const float score =
                block_reduce_sum<BLOCK_THREADS>(
                    local_dot,
                    warp_smem
                );

            //thread0写入，避免重复写
            if (dim_idx == 0) {
                score_smem[tile_token_idx] =
                    cache_token_idx < S
                        ? score * scale
                        : -FLT_MAX;
            }
        }

        __syncthreads();

        float local_tile_max = -FLT_MAX;

        if (dim_idx < BLOCK_N) {
            local_tile_max = score_smem[dim_idx];
        }

        //当前tile最大值
        const float tile_max =
            block_reduce_max<BLOCK_THREADS>(
                local_tile_max,
                warp_smem
            );

        //翻倍
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

        //当前tile计算临时概率
        float local_probability_sum = 0.0f;

        for (
            int tile_token_idx = dim_idx;
            tile_token_idx < BLOCK_N;
            tile_token_idx += BLOCK_THREADS
        ) {
            const int cache_token_idx =
                token_block_start + tile_token_idx;

            float probability = 0.0f;

            if (cache_token_idx < S) {
                probability = __expf(
                    score_smem[tile_token_idx] - online_max
                );
            }

            score_smem[tile_token_idx] = probability;
            local_probability_sum += probability;
        }

        //更新分母
        const float tile_probability_sum =
            block_reduce_sum<BLOCK_THREADS>(
                local_probability_sum,
                warp_smem
            );

        if (dim_idx == 0) {
            online_sum =
                rescale_factor * online_sum +
                tile_probability_sum;
        }

        __syncthreads();

        if (dim_idx < D) {
            output_acc *= rescale_factor;

            #pragma unroll
            for (
                int tile_token_idx = 0;
                tile_token_idx < BLOCK_N;
                ++tile_token_idx
            ) {
                const int cache_token_idx =
                    token_block_start + tile_token_idx;
                //更新输出结果
                if (cache_token_idx < S) {
                    const std::size_t v_offset =
                        (
                            (
                                static_cast<std::size_t>(batch_idx) * S +
                                cache_token_idx
                            ) * KVH + kv_head_idx
                        ) * D;

                    output_acc +=
                        score_smem[tile_token_idx] *
                        v_cache[v_offset + dim_idx];
                }
            }
        }

        __syncthreads();
    }
    //归一化
    if (dim_idx < D) {
        out[q_offset + dim_idx] =
            online_sum > 0.0f
                ? output_acc / online_sum
                : 0.0f;
    }
}

// ============================================================
// Launchers
// ============================================================

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

    if (blocks >
        static_cast<std::size_t>(
            std::numeric_limits<unsigned int>::max()
        )) {
        throw std::invalid_argument("QK grid.x 超出范围");
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

    const std::size_t total_rows =
        static_cast<std::size_t>(B) * QH;

    if (total_rows >
        static_cast<std::size_t>(
            std::numeric_limits<unsigned int>::max()
        )) {
        throw std::invalid_argument("Softmax grid.x 超出范围");
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

    if (blocks >
        static_cast<std::size_t>(
            std::numeric_limits<unsigned int>::max()
        )) {
        throw std::invalid_argument("PV grid.x 超出范围");
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

    if (D > kFusedThreadsPerBlock) {
        throw std::invalid_argument(
            "fused online softmax 当前只支持 D <= 128"
        );
    }

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