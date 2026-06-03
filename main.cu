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

void attention_cpu_reference(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    std::vector<float>& O,
    int BH,
    int S,
    int D
);

struct AttnConfig {
    int B;
    int H;
    int S;
    int D;
    int warmup;
    int repeat;
};

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

    CHECK_CUDA(cudaMemcpy(
        d_Q,
        h_Q.data(),
        qkv_bytes,
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA(cudaMemcpy(
        d_K,
        h_K.data(),
        qkv_bytes,
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA(cudaMemcpy(
        d_V,
        h_V.data(),
        qkv_bytes,
        cudaMemcpyHostToDevice
    ));

    // correctness: naive total
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

    CHECK_CUDA(cudaMemcpy(
        h_out.data(),
        d_O,
        qkv_bytes,
        cudaMemcpyDeviceToHost
    ));

    float naive_err = max_abs_error(h_ref, h_out);

    // correctness: tiled total
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

    CHECK_CUDA(cudaMemcpy(
        h_out.data(),
        d_O,
        qkv_bytes,
        cudaMemcpyDeviceToHost
    ));

    float tiled_err = max_abs_error(h_ref, h_out);

    // stage-wise timing
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

    // prepare scores for softmax benchmark
    launch_qk_matmul_tiled(
        d_Q,
        d_K,
        d_scores,
        BH,
        S,
        D
    );

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

    // prepare probs for PV benchmark
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

    double qk_speedup = qk_naive_ms / qk_tiled_ms;
    double pv_speedup = pv_naive_ms / pv_tiled_ms;
    double total_speedup = total_naive_ms / total_tiled_ms;

    double score_mb = static_cast<double>(score_bytes) / 1024.0 / 1024.0;
    double qkv_mb = static_cast<double>(qkv_bytes) / 1024.0 / 1024.0;

    std::cout << std::left
              << std::setw(6) << B
              << std::setw(6) << H
              << std::setw(8) << BH
              << std::setw(8) << S
              << std::setw(8) << D

              << std::setw(14) << std::fixed << std::setprecision(4) << qk_naive_ms
              << std::setw(14) << std::fixed << std::setprecision(4) << qk_tiled_ms
              << std::setw(12) << std::fixed << std::setprecision(3) << qk_speedup

              << std::setw(14) << std::fixed << std::setprecision(4) << softmax_ms

              << std::setw(14) << std::fixed << std::setprecision(4) << pv_naive_ms
              << std::setw(14) << std::fixed << std::setprecision(4) << pv_tiled_ms
              << std::setw(12) << std::fixed << std::setprecision(3) << pv_speedup

              << std::setw(16) << std::fixed << std::setprecision(4) << total_naive_ms
              << std::setw(16) << std::fixed << std::setprecision(4) << total_tiled_ms
              << std::setw(14) << std::fixed << std::setprecision(3) << total_speedup

              << std::setw(12) << std::fixed << std::setprecision(2) << score_mb
              << std::setw(12) << std::fixed << std::setprecision(2) << qkv_mb

              << std::setw(14) << std::scientific << std::setprecision(2) << naive_err
              << std::setw(14) << std::scientific << std::setprecision(2) << tiled_err
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

    std::cout << "=== Unfused Causal Attention Forward: Naive vs Tiled ===\n";

    std::cout << std::left
              << std::setw(6) << "B"
              << std::setw(6) << "H"
              << std::setw(8) << "BH"
              << std::setw(8) << "S"
              << std::setw(8) << "D"

              << std::setw(14) << "qk_naive"
              << std::setw(14) << "qk_tiled"
              << std::setw(12) << "qk_spd"

              << std::setw(14) << "softmax"

              << std::setw(14) << "pv_naive"
              << std::setw(14) << "pv_tiled"
              << std::setw(12) << "pv_spd"

              << std::setw(16) << "total_naive"
              << std::setw(16) << "total_tiled"
              << std::setw(14) << "total_spd"

              << std::setw(12) << "score_MB"
              << std::setw(12) << "qkv_MB"

              << std::setw(14) << "naive_err"
              << std::setw(14) << "tiled_err"
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