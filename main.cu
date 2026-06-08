// main.cu
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// ============================================================
// functions from attention_kernel.cu
// ============================================================

void launch_qk_matmul(
    const float* d_Q,
    const float* d_K,
    float* d_scores,
    int BH,
    int S,
    int D
);

void launch_qk_matmul_tiled(
    const float* d_Q,
    const float* d_K,
    float* d_scores,
    int BH,
    int S,
    int D
);

void launch_scaled_causal_softmax(
    const float* d_input,
    float* d_output,
    int M,
    int N,
    int query_len,
    float scale,
    int block_size
);

void launch_pv_matmul(
    const float* d_probs,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

void launch_pv_matmul_tiled(
    const float* d_probs,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

void launch_attention_forward(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_scores,
    float* d_probs,
    float* d_O,
    int BH,
    int S,
    int D
);

void launch_attention_forward_tiled(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_scores,
    float* d_probs,
    float* d_O,
    int BH,
    int S,
    int D
);

void launch_attention_forward_fused_row(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

void launch_flash_attention_v1(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

void launch_flash_attention_causal_tile_skipping(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

void launch_flash_attention_causal_tile_skipping_vec4(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

// old smem accumulator
void launch_flash_attention_skip_bm4_smem(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

void launch_flash_attention_skip_bm8_smem(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

void launch_flash_attention_skip_bm16_smem(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

// regacc scalar load
void launch_flash_attention_skip_bm4_regacc(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

void launch_flash_attention_skip_bm8_regacc(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

void launch_flash_attention_skip_bm16_regacc(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

// regacc + float4 load
void launch_flash_attention_skip_bm4_regacc_vec4(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

void launch_flash_attention_skip_bm8_regacc_vec4(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

void launch_flash_attention_skip_bm16_regacc_vec4(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

// noscore v0: BM16 only
void launch_flash_attention_skip_bm16_regacc_vec4_noscore(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

// pcache: BM16 only
void launch_flash_attention_skip_bm16_regacc_vec4_pcache(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

void attention_cpu_reference(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    std::vector<float>& O,
    int BH,
    int S,
    int D
);

// ============================================================
// configs / helpers
// ============================================================

struct AttnConfig {
    int B;
    int H;
    int S;
    int D;
    int warmup;
    int repeat;
};

using AttentionLauncher = void (*)(
    const float*,
    const float*,
    const float*,
    float*,
    int,
    int,
    int
);

float max_abs_error(
    const std::vector<float>& ref,
    const std::vector<float>& out
) {
    float err = 0.0f;

    for (size_t i = 0; i < ref.size(); ++i) {
        err = std::max(err, std::fabs(ref[i] - out[i]));
    }

    return err;
}

float check_attention_launcher_correctness(
    AttentionLauncher launcher,
    const std::vector<float>& h_ref,
    std::vector<float>& h_out,
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    size_t qkv_bytes,
    int BH,
    int S,
    int D
) {
    launcher(
        d_Q,
        d_K,
        d_V,
        d_O,
        BH,
        S,
        D
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(
        h_out.data(),
        d_O,
        qkv_bytes,
        cudaMemcpyDeviceToHost
    ));

    return max_abs_error(h_ref, h_out);
}

// ============================================================
// benchmark helpers
// ============================================================

float benchmark_qk(
    bool use_tiled,
    const float* d_Q,
    const float* d_K,
    float* d_scores,
    int BH,
    int S,
    int D,
    int warmup,
    int repeat
) {
    for (int i = 0; i < warmup; ++i) {
        if (use_tiled) {
            launch_qk_matmul_tiled(d_Q, d_K, d_scores, BH, S, D);
        } else {
            launch_qk_matmul(d_Q, d_K, d_scores, BH, S, D);
        }
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        if (use_tiled) {
            launch_qk_matmul_tiled(d_Q, d_K, d_scores, BH, S, D);
        } else {
            launch_qk_matmul(d_Q, d_K, d_scores, BH, S, D);
        }
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    return total_ms / static_cast<float>(repeat);
}

float benchmark_causal_softmax(
    const float* d_scores,
    float* d_probs,
    int BH,
    int S,
    int D,
    int warmup,
    int repeat
) {
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    int M_softmax = BH * S;
    int N_softmax = S;
    int query_len = S;
    int block_size = 256;

    for (int i = 0; i < warmup; ++i) {
        launch_scaled_causal_softmax(
            d_scores,
            d_probs,
            M_softmax,
            N_softmax,
            query_len,
            scale,
            block_size
        );
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        launch_scaled_causal_softmax(
            d_scores,
            d_probs,
            M_softmax,
            N_softmax,
            query_len,
            scale,
            block_size
        );
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    return total_ms / static_cast<float>(repeat);
}

float benchmark_pv(
    bool use_tiled,
    const float* d_probs,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D,
    int warmup,
    int repeat
) {
    for (int i = 0; i < warmup; ++i) {
        if (use_tiled) {
            launch_pv_matmul_tiled(d_probs, d_V, d_O, BH, S, D);
        } else {
            launch_pv_matmul(d_probs, d_V, d_O, BH, S, D);
        }
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        if (use_tiled) {
            launch_pv_matmul_tiled(d_probs, d_V, d_O, BH, S, D);
        } else {
            launch_pv_matmul(d_probs, d_V, d_O, BH, S, D);
        }
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    return total_ms / static_cast<float>(repeat);
}

float benchmark_attention_total(
    bool use_tiled,
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_scores,
    float* d_probs,
    float* d_O,
    int BH,
    int S,
    int D,
    int warmup,
    int repeat
) {
    for (int i = 0; i < warmup; ++i) {
        if (use_tiled) {
            launch_attention_forward_tiled(
                d_Q,
                d_K,
                d_V,
                d_scores,
                d_probs,
                d_O,
                BH,
                S,
                D
            );
        } else {
            launch_attention_forward(
                d_Q,
                d_K,
                d_V,
                d_scores,
                d_probs,
                d_O,
                BH,
                S,
                D
            );
        }
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        if (use_tiled) {
            launch_attention_forward_tiled(
                d_Q,
                d_K,
                d_V,
                d_scores,
                d_probs,
                d_O,
                BH,
                S,
                D
            );
        } else {
            launch_attention_forward(
                d_Q,
                d_K,
                d_V,
                d_scores,
                d_probs,
                d_O,
                BH,
                S,
                D
            );
        }
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    return total_ms / static_cast<float>(repeat);
}

float benchmark_attention_launcher(
    AttentionLauncher launcher,
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D,
    int warmup,
    int repeat
) {
    for (int i = 0; i < warmup; ++i) {
        launcher(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D
        );
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        launcher(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D
        );
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    return total_ms / static_cast<float>(repeat);
}

// ============================================================
// run one config
// ============================================================

void run_one_config(const AttnConfig& cfg) {
    int B = cfg.B;
    int H = cfg.H;
    int BH = B * H;
    int S = cfg.S;
    int D = cfg.D;

    size_t qkv_elems = static_cast<size_t>(BH) * S * D;
    size_t score_elems = static_cast<size_t>(BH) * S * S;

    size_t qkv_bytes = qkv_elems * sizeof(float);
    size_t score_bytes = score_elems * sizeof(float);

    std::vector<float> h_Q(qkv_elems);
    std::vector<float> h_K(qkv_elems);
    std::vector<float> h_V(qkv_elems);

    std::vector<float> h_ref(qkv_elems);
    std::vector<float> h_out(qkv_elems);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float& x : h_Q) {
        x = dist(gen);
    }

    for (float& x : h_K) {
        x = dist(gen);
    }

    for (float& x : h_V) {
        x = dist(gen);
    }

    attention_cpu_reference(
        h_Q,
        h_K,
        h_V,
        h_ref,
        BH,
        S,
        D
    );

    float* d_Q = nullptr;
    float* d_K = nullptr;
    float* d_V = nullptr;
    float* d_scores = nullptr;
    float* d_probs = nullptr;
    float* d_O = nullptr;

    CHECK_CUDA(cudaMalloc(&d_Q, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_K, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_V, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_scores, score_bytes));
    CHECK_CUDA(cudaMalloc(&d_probs, score_bytes));
    CHECK_CUDA(cudaMalloc(&d_O, qkv_bytes));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), qkv_bytes, cudaMemcpyHostToDevice));

    // ============================================================
    // correctness
    // ============================================================

    launch_attention_forward(
        d_Q,
        d_K,
        d_V,
        d_scores,
        d_probs,
        d_O,
        BH,
        S,
        D
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_O, qkv_bytes, cudaMemcpyDeviceToHost));
    float naive_err = max_abs_error(h_ref, h_out);

    launch_attention_forward_tiled(
        d_Q,
        d_K,
        d_V,
        d_scores,
        d_probs,
        d_O,
        BH,
        S,
        D
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_O, qkv_bytes, cudaMemcpyDeviceToHost));
    float tiled_err = max_abs_error(h_ref, h_out);

    float fused_err = check_attention_launcher_correctness(
        launch_attention_forward_fused_row,
        h_ref,
        h_out,
        d_Q,
        d_K,
        d_V,
        d_O,
        qkv_bytes,
        BH,
        S,
        D
    );

    float flash_err = check_attention_launcher_correctness(
        launch_flash_attention_v1,
        h_ref,
        h_out,
        d_Q,
        d_K,
        d_V,
        d_O,
        qkv_bytes,
        BH,
        S,
        D
    );

    float skip_err = check_attention_launcher_correctness(
        launch_flash_attention_causal_tile_skipping,
        h_ref,
        h_out,
        d_Q,
        d_K,
        d_V,
        d_O,
        qkv_bytes,
        BH,
        S,
        D
    );

    float skip_vec4_err = check_attention_launcher_correctness(
        launch_flash_attention_causal_tile_skipping_vec4,
        h_ref,
        h_out,
        d_Q,
        d_K,
        d_V,
        d_O,
        qkv_bytes,
        BH,
        S,
        D
    );

    float bm4_smem_err = check_attention_launcher_correctness(
        launch_flash_attention_skip_bm4_smem,
        h_ref,
        h_out,
        d_Q,
        d_K,
        d_V,
        d_O,
        qkv_bytes,
        BH,
        S,
        D
    );

    float bm8_smem_err = check_attention_launcher_correctness(
        launch_flash_attention_skip_bm8_smem,
        h_ref,
        h_out,
        d_Q,
        d_K,
        d_V,
        d_O,
        qkv_bytes,
        BH,
        S,
        D
    );

    float bm16_smem_err = check_attention_launcher_correctness(
        launch_flash_attention_skip_bm16_smem,
        h_ref,
        h_out,
        d_Q,
        d_K,
        d_V,
        d_O,
        qkv_bytes,
        BH,
        S,
        D
    );

    float bm4_reg_err = check_attention_launcher_correctness(
        launch_flash_attention_skip_bm4_regacc,
        h_ref,
        h_out,
        d_Q,
        d_K,
        d_V,
        d_O,
        qkv_bytes,
        BH,
        S,
        D
    );

    float bm8_reg_err = check_attention_launcher_correctness(
        launch_flash_attention_skip_bm8_regacc,
        h_ref,
        h_out,
        d_Q,
        d_K,
        d_V,
        d_O,
        qkv_bytes,
        BH,
        S,
        D
    );

    float bm16_reg_err = check_attention_launcher_correctness(
        launch_flash_attention_skip_bm16_regacc,
        h_ref,
        h_out,
        d_Q,
        d_K,
        d_V,
        d_O,
        qkv_bytes,
        BH,
        S,
        D
    );

    float bm4_vec4_err = check_attention_launcher_correctness(
        launch_flash_attention_skip_bm4_regacc_vec4,
        h_ref,
        h_out,
        d_Q,
        d_K,
        d_V,
        d_O,
        qkv_bytes,
        BH,
        S,
        D
    );

    float bm8_vec4_err = check_attention_launcher_correctness(
        launch_flash_attention_skip_bm8_regacc_vec4,
        h_ref,
        h_out,
        d_Q,
        d_K,
        d_V,
        d_O,
        qkv_bytes,
        BH,
        S,
        D
    );

    float bm16_vec4_err = check_attention_launcher_correctness(
        launch_flash_attention_skip_bm16_regacc_vec4,
        h_ref,
        h_out,
        d_Q,
        d_K,
        d_V,
        d_O,
        qkv_bytes,
        BH,
        S,
        D
    );

    float bm16_noscore_err = check_attention_launcher_correctness(
        launch_flash_attention_skip_bm16_regacc_vec4_noscore,
        h_ref,
        h_out,
        d_Q,
        d_K,
        d_V,
        d_O,
        qkv_bytes,
        BH,
        S,
        D
    );

    float bm16_pcache_err = check_attention_launcher_correctness(
        launch_flash_attention_skip_bm16_regacc_vec4_pcache,
        h_ref,
        h_out,
        d_Q,
        d_K,
        d_V,
        d_O,
        qkv_bytes,
        BH,
        S,
        D
    );

    // ============================================================
    // stage-wise timing
    // ============================================================

    float qk_naive_ms = benchmark_qk(
        false,
        d_Q,
        d_K,
        d_scores,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    float qk_tiled_ms = benchmark_qk(
        true,
        d_Q,
        d_K,
        d_scores,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    launch_qk_matmul_tiled(d_Q, d_K, d_scores, BH, S, D);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    float softmax_ms = benchmark_causal_softmax(
        d_scores,
        d_probs,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    launch_scaled_causal_softmax(
        d_scores,
        d_probs,
        BH * S,
        S,
        S,
        scale,
        256
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    float pv_naive_ms = benchmark_pv(
        false,
        d_probs,
        d_V,
        d_O,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    float pv_tiled_ms = benchmark_pv(
        true,
        d_probs,
        d_V,
        d_O,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    // ============================================================
    // total timing
    // ============================================================

    float total_naive_ms = benchmark_attention_total(
        false,
        d_Q,
        d_K,
        d_V,
        d_scores,
        d_probs,
        d_O,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    float total_tiled_ms = benchmark_attention_total(
        true,
        d_Q,
        d_K,
        d_V,
        d_scores,
        d_probs,
        d_O,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    float total_fused_ms = benchmark_attention_launcher(
        launch_attention_forward_fused_row,
        d_Q,
        d_K,
        d_V,
        d_O,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    float total_flash_ms = benchmark_attention_launcher(
        launch_flash_attention_v1,
        d_Q,
        d_K,
        d_V,
        d_O,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    float total_skip_ms = benchmark_attention_launcher(
        launch_flash_attention_causal_tile_skipping,
        d_Q,
        d_K,
        d_V,
        d_O,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    float total_skip_vec4_ms = benchmark_attention_launcher(
        launch_flash_attention_causal_tile_skipping_vec4,
        d_Q,
        d_K,
        d_V,
        d_O,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    float bm4_smem_ms = benchmark_attention_launcher(
        launch_flash_attention_skip_bm4_smem,
        d_Q,
        d_K,
        d_V,
        d_O,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    float bm8_smem_ms = benchmark_attention_launcher(
        launch_flash_attention_skip_bm8_smem,
        d_Q,
        d_K,
        d_V,
        d_O,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    float bm16_smem_ms = benchmark_attention_launcher(
        launch_flash_attention_skip_bm16_smem,
        d_Q,
        d_K,
        d_V,
        d_O,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    float bm4_reg_ms = benchmark_attention_launcher(
        launch_flash_attention_skip_bm4_regacc,
        d_Q,
        d_K,
        d_V,
        d_O,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    float bm8_reg_ms = benchmark_attention_launcher(
        launch_flash_attention_skip_bm8_regacc,
        d_Q,
        d_K,
        d_V,
        d_O,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    float bm16_reg_ms = benchmark_attention_launcher(
        launch_flash_attention_skip_bm16_regacc,
        d_Q,
        d_K,
        d_V,
        d_O,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    float bm4_vec4_ms = benchmark_attention_launcher(
        launch_flash_attention_skip_bm4_regacc_vec4,
        d_Q,
        d_K,
        d_V,
        d_O,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    float bm8_vec4_ms = benchmark_attention_launcher(
        launch_flash_attention_skip_bm8_regacc_vec4,
        d_Q,
        d_K,
        d_V,
        d_O,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    float bm16_vec4_ms = benchmark_attention_launcher(
        launch_flash_attention_skip_bm16_regacc_vec4,
        d_Q,
        d_K,
        d_V,
        d_O,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    float bm16_noscore_ms = benchmark_attention_launcher(
        launch_flash_attention_skip_bm16_regacc_vec4_noscore,
        d_Q,
        d_K,
        d_V,
        d_O,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    float bm16_pcache_ms = benchmark_attention_launcher(
        launch_flash_attention_skip_bm16_regacc_vec4_pcache,
        d_Q,
        d_K,
        d_V,
        d_O,
        BH,
        S,
        D,
        cfg.warmup,
        cfg.repeat
    );

    // ============================================================
    // stats
    // ============================================================

    double qk_speedup = qk_naive_ms / qk_tiled_ms;
    double pv_speedup = pv_naive_ms / pv_tiled_ms;

    double tiled_speedup = total_naive_ms / total_tiled_ms;
    double flash_vs_naive = total_naive_ms / total_flash_ms;
    double skip_vs_flash = total_flash_ms / total_skip_ms;
    double skip_vec4_vs_flash = total_flash_ms / total_skip_vec4_ms;

    double reg4_vs_smem = bm4_smem_ms / bm4_reg_ms;
    double reg8_vs_smem = bm8_smem_ms / bm8_reg_ms;
    double reg16_vs_smem = bm16_smem_ms / bm16_reg_ms;

    double vec4_4_vs_reg = bm4_reg_ms / bm4_vec4_ms;
    double vec4_8_vs_reg = bm8_reg_ms / bm8_vec4_ms;
    double vec4_16_vs_reg = bm16_reg_ms / bm16_vec4_ms;

    double noscore_vs_vec4 = bm16_vec4_ms / bm16_noscore_ms;
    double pcache_vs_vec4 = bm16_vec4_ms / bm16_pcache_ms;

    double score_mb = static_cast<double>(score_bytes) / 1024.0 / 1024.0;
    double qkv_mb = static_cast<double>(qkv_bytes) / 1024.0 / 1024.0;

    std::cout << std::left
              << std::setw(6) << B
              << std::setw(6) << H
              << std::setw(8) << BH
              << std::setw(8) << S
              << std::setw(8) << D

              << std::setw(12) << std::fixed << std::setprecision(4) << qk_naive_ms
              << std::setw(12) << std::fixed << std::setprecision(4) << qk_tiled_ms
              << std::setw(10) << std::fixed << std::setprecision(3) << qk_speedup

              << std::setw(12) << std::fixed << std::setprecision(4) << softmax_ms

              << std::setw(12) << std::fixed << std::setprecision(4) << pv_naive_ms
              << std::setw(12) << std::fixed << std::setprecision(4) << pv_tiled_ms
              << std::setw(10) << std::fixed << std::setprecision(3) << pv_speedup

              << std::setw(14) << std::fixed << std::setprecision(4) << total_naive_ms
              << std::setw(14) << std::fixed << std::setprecision(4) << total_tiled_ms
              << std::setw(14) << std::fixed << std::setprecision(4) << total_fused_ms
              << std::setw(14) << std::fixed << std::setprecision(4) << total_flash_ms
              << std::setw(14) << std::fixed << std::setprecision(4) << total_skip_ms
              << std::setw(14) << std::fixed << std::setprecision(4) << total_skip_vec4_ms

              << std::setw(12) << std::fixed << std::setprecision(4) << bm4_smem_ms
              << std::setw(12) << std::fixed << std::setprecision(4) << bm8_smem_ms
              << std::setw(12) << std::fixed << std::setprecision(4) << bm16_smem_ms

              << std::setw(12) << std::fixed << std::setprecision(4) << bm4_reg_ms
              << std::setw(12) << std::fixed << std::setprecision(4) << bm8_reg_ms
              << std::setw(12) << std::fixed << std::setprecision(4) << bm16_reg_ms

              << std::setw(12) << std::fixed << std::setprecision(4) << bm4_vec4_ms
              << std::setw(12) << std::fixed << std::setprecision(4) << bm8_vec4_ms
              << std::setw(12) << std::fixed << std::setprecision(4) << bm16_vec4_ms

              << std::setw(14) << std::fixed << std::setprecision(4) << bm16_noscore_ms
              << std::setw(14) << std::fixed << std::setprecision(4) << bm16_pcache_ms

              << std::setw(12) << std::fixed << std::setprecision(3) << tiled_speedup
              << std::setw(12) << std::fixed << std::setprecision(3) << flash_vs_naive
              << std::setw(12) << std::fixed << std::setprecision(3) << skip_vs_flash
              << std::setw(12) << std::fixed << std::setprecision(3) << skip_vec4_vs_flash

              << std::setw(12) << std::fixed << std::setprecision(3) << reg4_vs_smem
              << std::setw(12) << std::fixed << std::setprecision(3) << reg8_vs_smem
              << std::setw(12) << std::fixed << std::setprecision(3) << reg16_vs_smem

              << std::setw(12) << std::fixed << std::setprecision(3) << vec4_4_vs_reg
              << std::setw(12) << std::fixed << std::setprecision(3) << vec4_8_vs_reg
              << std::setw(12) << std::fixed << std::setprecision(3) << vec4_16_vs_reg

              << std::setw(14) << std::fixed << std::setprecision(3) << noscore_vs_vec4
              << std::setw(14) << std::fixed << std::setprecision(3) << pcache_vs_vec4

              << std::setw(10) << std::fixed << std::setprecision(2) << score_mb
              << std::setw(10) << std::fixed << std::setprecision(2) << qkv_mb

              << std::setw(12) << std::scientific << std::setprecision(2) << naive_err
              << std::setw(12) << std::scientific << std::setprecision(2) << tiled_err
              << std::setw(12) << std::scientific << std::setprecision(2) << fused_err
              << std::setw(12) << std::scientific << std::setprecision(2) << flash_err
              << std::setw(12) << std::scientific << std::setprecision(2) << skip_err
              << std::setw(12) << std::scientific << std::setprecision(2) << skip_vec4_err

              << std::setw(12) << std::scientific << std::setprecision(2) << bm4_smem_err
              << std::setw(12) << std::scientific << std::setprecision(2) << bm8_smem_err
              << std::setw(12) << std::scientific << std::setprecision(2) << bm16_smem_err

              << std::setw(12) << std::scientific << std::setprecision(2) << bm4_reg_err
              << std::setw(12) << std::scientific << std::setprecision(2) << bm8_reg_err
              << std::setw(12) << std::scientific << std::setprecision(2) << bm16_reg_err

              << std::setw(12) << std::scientific << std::setprecision(2) << bm4_vec4_err
              << std::setw(12) << std::scientific << std::setprecision(2) << bm8_vec4_err
              << std::setw(12) << std::scientific << std::setprecision(2) << bm16_vec4_err

              << std::setw(12) << std::scientific << std::setprecision(2) << bm16_noscore_err
              << std::setw(12) << std::scientific << std::setprecision(2) << bm16_pcache_err
              << "\n";

    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_scores));
    CHECK_CUDA(cudaFree(d_probs));
    CHECK_CUDA(cudaFree(d_O));
}

// ============================================================
// main
// ============================================================

int main() {
    int device = 0;
    CHECK_CUDA(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n\n";

    std::cout << "=== Causal Attention Forward: smem vs regacc vs vec4 vs BM16 noscore/pcache ===\n";

    std::cout << std::left
              << std::setw(6) << "B"
              << std::setw(6) << "H"
              << std::setw(8) << "BH"
              << std::setw(8) << "S"
              << std::setw(8) << "D"

              << std::setw(12) << "qk_naive"
              << std::setw(12) << "qk_tiled"
              << std::setw(10) << "qk_spd"

              << std::setw(12) << "softmax"

              << std::setw(12) << "pv_naive"
              << std::setw(12) << "pv_tiled"
              << std::setw(10) << "pv_spd"

              << std::setw(14) << "total_naive"
              << std::setw(14) << "total_tiled"
              << std::setw(14) << "total_fused"
              << std::setw(14) << "total_flash"
              << std::setw(14) << "total_skip"
              << std::setw(14) << "skip_vec4"

              << std::setw(12) << "bm4_smem"
              << std::setw(12) << "bm8_smem"
              << std::setw(12) << "bm16_smem"

              << std::setw(12) << "bm4_reg"
              << std::setw(12) << "bm8_reg"
              << std::setw(12) << "bm16_reg"

              << std::setw(12) << "bm4_vec4"
              << std::setw(12) << "bm8_vec4"
              << std::setw(12) << "bm16_vec4"

              << std::setw(14) << "bm16_noscore"
              << std::setw(14) << "bm16_pcache"

              << std::setw(12) << "tiled_spd"
              << std::setw(12) << "flash_spd"
              << std::setw(12) << "skip/flash"
              << std::setw(12) << "v4/flash"

              << std::setw(12) << "reg4/smem"
              << std::setw(12) << "reg8/smem"
              << std::setw(12) << "reg16/smem"

              << std::setw(12) << "v4/reg4"
              << std::setw(12) << "v4/reg8"
              << std::setw(12) << "v4/reg16"

              << std::setw(14) << "noscore/v4"
              << std::setw(14) << "pcache/v4"

              << std::setw(10) << "score_MB"
              << std::setw(10) << "qkv_MB"

              << std::setw(12) << "naive_err"
              << std::setw(12) << "tiled_err"
              << std::setw(12) << "fused_err"
              << std::setw(12) << "flash_err"
              << std::setw(12) << "skip_err"
              << std::setw(12) << "v4skip_err"

              << std::setw(12) << "bm4s_err"
              << std::setw(12) << "bm8s_err"
              << std::setw(12) << "bm16s_err"

              << std::setw(12) << "bm4r_err"
              << std::setw(12) << "bm8r_err"
              << std::setw(12) << "bm16r_err"

              << std::setw(12) << "bm4v_err"
              << std::setw(12) << "bm8v_err"
              << std::setw(12) << "bm16v_err"

              << std::setw(12) << "bm16n_err"
              << std::setw(12) << "bm16p_err"
              << "\n";

    std::vector<AttnConfig> configs = {
        {1, 1, 64,  64, 20, 100},
        {1, 1, 128, 64, 20, 100},
        {1, 8, 128, 64, 20, 100},
        {1, 8, 256, 64, 10, 50},
        {1, 8, 512, 64, 10, 30}
    };

    for (const auto& cfg : configs) {
        run_one_config(cfg);
    }

    return 0;
}