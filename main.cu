// main.cu
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
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
// launcher declarations from attention_kernel.cu
// ============================================================

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

// regacc + float4 baseline
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

// regacc + float4 + pcache
void launch_flash_attention_skip_bm4_regacc_vec4_pcache(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

void launch_flash_attention_skip_bm8_regacc_vec4_pcache(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

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
// types
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

// ============================================================
// helper functions
// ============================================================

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

float check_correctness(
    AttentionLauncher launcher,
    const std::vector<float>& h_ref,
    std::vector<float>& h_out,
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    size_t bytes,
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
        bytes,
        cudaMemcpyDeviceToHost
    ));

    return max_abs_error(h_ref, h_out);
}

float benchmark_launcher(
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
    size_t qkv_bytes = qkv_elems * sizeof(float);

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
    float* d_O = nullptr;

    CHECK_CUDA(cudaMalloc(&d_Q, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_K, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_V, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_O, qkv_bytes));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), qkv_bytes, cudaMemcpyHostToDevice));

    // ============================================================
    // correctness
    // ============================================================

    float flash_err = check_correctness(
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

    float dispatch_err = check_correctness(
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

    float dispatch_vec4_err = check_correctness(
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

    float bm4_vec4_err = check_correctness(
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

    float bm8_vec4_err = check_correctness(
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

    float bm16_vec4_err = check_correctness(
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

    float bm4_pcache_err = check_correctness(
        launch_flash_attention_skip_bm4_regacc_vec4_pcache,
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

    float bm8_pcache_err = check_correctness(
        launch_flash_attention_skip_bm8_regacc_vec4_pcache,
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

    float bm16_pcache_err = check_correctness(
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
    // timing
    // ============================================================

    float flash_ms = benchmark_launcher(
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

    float dispatch_ms = benchmark_launcher(
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

    float dispatch_vec4_ms = benchmark_launcher(
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

    float bm4_vec4_ms = benchmark_launcher(
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

    float bm8_vec4_ms = benchmark_launcher(
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

    float bm16_vec4_ms = benchmark_launcher(
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

    float bm4_pcache_ms = benchmark_launcher(
        launch_flash_attention_skip_bm4_regacc_vec4_pcache,
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

    float bm8_pcache_ms = benchmark_launcher(
        launch_flash_attention_skip_bm8_regacc_vec4_pcache,
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

    float bm16_pcache_ms = benchmark_launcher(
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

    double dispatch_vs_flash = flash_ms / dispatch_ms;
    double vec4_dispatch_vs_flash = flash_ms / dispatch_vec4_ms;

    double pc4_vs_v4 = bm4_vec4_ms / bm4_pcache_ms;
    double pc8_vs_v4 = bm8_vec4_ms / bm8_pcache_ms;
    double pc16_vs_v4 = bm16_vec4_ms / bm16_pcache_ms;

    double bm4_pc_vs_flash = flash_ms / bm4_pcache_ms;
    double bm8_pc_vs_flash = flash_ms / bm8_pcache_ms;
    double bm16_pc_vs_flash = flash_ms / bm16_pcache_ms;

    std::string best_name = "bm4_pcache";
    float best_ms = bm4_pcache_ms;

    if (bm8_pcache_ms < best_ms) {
        best_ms = bm8_pcache_ms;
        best_name = "bm8_pcache";
    }

    if (bm16_pcache_ms < best_ms) {
        best_ms = bm16_pcache_ms;
        best_name = "bm16_pcache";
    }

    double best_vs_flash = flash_ms / best_ms;

    // ============================================================
    // print
    // ============================================================

    std::cout << std::left
              << std::setw(5) << B
              << std::setw(5) << H
              << std::setw(6) << BH
              << std::setw(6) << S
              << std::setw(5) << D

              << std::setw(11) << std::fixed << std::setprecision(4) << flash_ms
              << std::setw(11) << std::fixed << std::setprecision(4) << dispatch_ms
              << std::setw(11) << std::fixed << std::setprecision(4) << dispatch_vec4_ms

              << std::setw(11) << std::fixed << std::setprecision(4) << bm4_vec4_ms
              << std::setw(11) << std::fixed << std::setprecision(4) << bm8_vec4_ms
              << std::setw(11) << std::fixed << std::setprecision(4) << bm16_vec4_ms

              << std::setw(12) << std::fixed << std::setprecision(4) << bm4_pcache_ms
              << std::setw(12) << std::fixed << std::setprecision(4) << bm8_pcache_ms
              << std::setw(12) << std::fixed << std::setprecision(4) << bm16_pcache_ms

              << std::setw(10) << std::fixed << std::setprecision(3) << dispatch_vs_flash
              << std::setw(10) << std::fixed << std::setprecision(3) << vec4_dispatch_vs_flash

              << std::setw(9) << std::fixed << std::setprecision(3) << pc4_vs_v4
              << std::setw(9) << std::fixed << std::setprecision(3) << pc8_vs_v4
              << std::setw(10) << std::fixed << std::setprecision(3) << pc16_vs_v4

              << std::setw(10) << std::fixed << std::setprecision(3) << bm4_pc_vs_flash
              << std::setw(10) << std::fixed << std::setprecision(3) << bm8_pc_vs_flash
              << std::setw(11) << std::fixed << std::setprecision(3) << bm16_pc_vs_flash

              << std::setw(14) << best_name
              << std::setw(10) << std::fixed << std::setprecision(4) << best_ms
              << std::setw(11) << std::fixed << std::setprecision(3) << best_vs_flash

              << std::setw(11) << std::scientific << std::setprecision(2) << flash_err
              << std::setw(11) << std::scientific << std::setprecision(2) << dispatch_err
              << std::setw(11) << std::scientific << std::setprecision(2) << dispatch_vec4_err

              << std::setw(11) << std::scientific << std::setprecision(2) << bm4_vec4_err
              << std::setw(11) << std::scientific << std::setprecision(2) << bm8_vec4_err
              << std::setw(11) << std::scientific << std::setprecision(2) << bm16_vec4_err

              << std::setw(11) << std::scientific << std::setprecision(2) << bm4_pcache_err
              << std::setw(11) << std::scientific << std::setprecision(2) << bm8_pcache_err
              << std::setw(11) << std::scientific << std::setprecision(2) << bm16_pcache_err
              << "\n";

    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
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

    std::cout << "=== Causal Attention Forward: BM4/BM8/BM16 pcache dispatch benchmark ===\n";

    std::cout << std::left
              << std::setw(5) << "B"
              << std::setw(5) << "H"
              << std::setw(6) << "BH"
              << std::setw(6) << "S"
              << std::setw(5) << "D"

              << std::setw(11) << "flash"
              << std::setw(11) << "dispatch"
              << std::setw(11) << "disp_v4"

              << std::setw(11) << "bm4_v4"
              << std::setw(11) << "bm8_v4"
              << std::setw(11) << "bm16_v4"

              << std::setw(12) << "bm4_pc"
              << std::setw(12) << "bm8_pc"
              << std::setw(12) << "bm16_pc"

              << std::setw(10) << "disp/fl"
              << std::setw(10) << "v4/fl"

              << std::setw(9) << "pc4/v4"
              << std::setw(9) << "pc8/v4"
              << std::setw(10) << "pc16/v4"

              << std::setw(10) << "pc4/fl"
              << std::setw(10) << "pc8/fl"
              << std::setw(11) << "pc16/fl"

              << std::setw(14) << "best"
              << std::setw(10) << "best_ms"
              << std::setw(11) << "best/fl"

              << std::setw(11) << "flash_err"
              << std::setw(11) << "disp_err"
              << std::setw(11) << "v4d_err"

              << std::setw(11) << "bm4v_err"
              << std::setw(11) << "bm8v_err"
              << std::setw(11) << "bm16v_err"

              << std::setw(11) << "bm4p_err"
              << std::setw(11) << "bm8p_err"
              << std::setw(11) << "bm16p_err"
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