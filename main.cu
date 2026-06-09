// main.cu
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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

// unfused: QK + softmax + PV
// 注意：这个版本需要 d_scores / d_probs workspace
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

// tiled QK/PV + softmax
// 注意：这个版本也需要 d_scores / d_probs workspace
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

// fused row attention
void launch_attention_forward_fused_row(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

// FlashAttention v1 baseline
void launch_flash_attention_v1(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

// final dispatch：你现在应该让它内部走 padding pcache
void launch_flash_attention_causal_tile_skipping(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

// padding dispatch：用于确认和 final 是否一致
void launch_flash_attention_causal_tile_skipping_pcache_padding(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

// BM4/BM8/BM16 padding pcache
void launch_flash_attention_skip_bm4_regacc_vec4_pcache_padding(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

void launch_flash_attention_skip_bm8_regacc_vec4_pcache_padding(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

void launch_flash_attention_skip_bm16_regacc_vec4_pcache_padding(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
);

// ============================================================
// config / kernel enum
// ============================================================

struct AttnConfig {
    int B;
    int H;
    int S;
    int D;
    int warmup;
    int repeat;
};

enum class KernelKind {
    Unfused,
    Tiled,
    FusedRow,
    FlashV1,
    Final,
    PadDispatch,
    BM4Pad,
    BM8Pad,
    BM16Pad
};

struct KernelSpec {
    std::string name;
    KernelKind kind;
};

struct KernelResult {
    std::string name;
    float ms;
    float err;
};

// ============================================================
// CPU reference
// layout:
// Q/K/V/O: [BH, S, D]
// causal attention:
// O[q] = softmax(Q[q] K[0:q]^T / sqrt(D)) V[0:q]
// ============================================================

void attention_cpu_reference_local(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    std::vector<float>& O,
    int BH,
    int S,
    int D
) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::fill(O.begin(), O.end(), 0.0f);

    std::vector<float> scores(S, 0.0f);

    for (int bh = 0; bh < BH; ++bh) {
        const int base = bh * S * D;

        for (int q = 0; q < S; ++q) {
            float max_score = -std::numeric_limits<float>::infinity();

            for (int k = 0; k <= q; ++k) {
                float dot = 0.0f;

                for (int d = 0; d < D; ++d) {
                    dot += Q[base + q * D + d] * K[base + k * D + d];
                }

                float score = dot * scale;
                scores[k] = score;
                max_score = std::max(max_score, score);
            }

            float denom = 0.0f;

            for (int k = 0; k <= q; ++k) {
                scores[k] = std::exp(scores[k] - max_score);
                denom += scores[k];
            }

            for (int d = 0; d < D; ++d) {
                float acc = 0.0f;

                for (int k = 0; k <= q; ++k) {
                    float p = scores[k] / denom;
                    acc += p * V[base + k * D + d];
                }

                O[base + q * D + d] = acc;
            }
        }
    }
}

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

void call_kernel(
    KernelKind kind,
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_scores,
    float* d_probs,
    float* d_O,
    int BH,
    int S,
    int D
) {
    switch (kind) {
        case KernelKind::Unfused:
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
            break;

        case KernelKind::Tiled:
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
            break;

        case KernelKind::FusedRow:
            launch_attention_forward_fused_row(
                d_Q,
                d_K,
                d_V,
                d_O,
                BH,
                S,
                D
            );
            break;

        case KernelKind::FlashV1:
            launch_flash_attention_v1(
                d_Q,
                d_K,
                d_V,
                d_O,
                BH,
                S,
                D
            );
            break;

        case KernelKind::Final:
            launch_flash_attention_causal_tile_skipping(
                d_Q,
                d_K,
                d_V,
                d_O,
                BH,
                S,
                D
            );
            break;

        case KernelKind::PadDispatch:
            launch_flash_attention_causal_tile_skipping_pcache_padding(
                d_Q,
                d_K,
                d_V,
                d_O,
                BH,
                S,
                D
            );
            break;

        case KernelKind::BM4Pad:
            launch_flash_attention_skip_bm4_regacc_vec4_pcache_padding(
                d_Q,
                d_K,
                d_V,
                d_O,
                BH,
                S,
                D
            );
            break;

        case KernelKind::BM8Pad:
            launch_flash_attention_skip_bm8_regacc_vec4_pcache_padding(
                d_Q,
                d_K,
                d_V,
                d_O,
                BH,
                S,
                D
            );
            break;

        case KernelKind::BM16Pad:
            launch_flash_attention_skip_bm16_regacc_vec4_pcache_padding(
                d_Q,
                d_K,
                d_V,
                d_O,
                BH,
                S,
                D
            );
            break;
    }
}

float check_correctness(
    const KernelSpec& spec,
    const std::vector<float>& h_ref,
    std::vector<float>& h_out,
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_scores,
    float* d_probs,
    float* d_O,
    size_t out_bytes,
    size_t scores_bytes,
    int BH,
    int S,
    int D
) {
    CHECK_CUDA(cudaMemset(d_O, 0, out_bytes));

    if (d_scores != nullptr) {
        CHECK_CUDA(cudaMemset(d_scores, 0, scores_bytes));
    }

    if (d_probs != nullptr) {
        CHECK_CUDA(cudaMemset(d_probs, 0, scores_bytes));
    }

    call_kernel(
        spec.kind,
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

    cudaError_t kernel_err = cudaGetLastError();

    if (kernel_err != cudaSuccess) {
        std::cerr << "Kernel launch failed for " << spec.name
                  << ": " << cudaGetErrorString(kernel_err) << std::endl;
        return std::numeric_limits<float>::infinity();
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(
        h_out.data(),
        d_O,
        out_bytes,
        cudaMemcpyDeviceToHost
    ));

    return max_abs_error(h_ref, h_out);
}

float benchmark_kernel(
    const KernelSpec& spec,
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
        call_kernel(
            spec.kind,
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

    cudaError_t kernel_err = cudaGetLastError();

    if (kernel_err != cudaSuccess) {
        std::cerr << "Warmup failed for " << spec.name
                  << ": " << cudaGetErrorString(kernel_err) << std::endl;
        return std::numeric_limits<float>::infinity();
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        call_kernel(
            spec.kind,
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

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    kernel_err = cudaGetLastError();

    if (kernel_err != cudaSuccess) {
        std::cerr << "Benchmark failed for " << spec.name
                  << ": " << cudaGetErrorString(kernel_err) << std::endl;
        return std::numeric_limits<float>::infinity();
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    return total_ms / static_cast<float>(repeat);
}

double speedup(float baseline_ms, float target_ms) {
    if (!std::isfinite(baseline_ms) ||
        !std::isfinite(target_ms) ||
        target_ms <= 0.0f) {
        return 0.0;
    }

    return static_cast<double>(baseline_ms) / static_cast<double>(target_ms);
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

    size_t score_elems = static_cast<size_t>(BH) * S * S;
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

    attention_cpu_reference_local(
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

    float* d_scores = nullptr;
    float* d_probs = nullptr;

    CHECK_CUDA(cudaMalloc(&d_Q, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_K, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_V, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_O, qkv_bytes));

    CHECK_CUDA(cudaMalloc(&d_scores, score_bytes));
    CHECK_CUDA(cudaMalloc(&d_probs, score_bytes));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), qkv_bytes, cudaMemcpyHostToDevice));

    std::vector<KernelSpec> kernels = {
        {"unfused",      KernelKind::Unfused},
        {"tiled",        KernelKind::Tiled},
        {"fused_row",    KernelKind::FusedRow},
        {"flash_v1",     KernelKind::FlashV1},
        {"final",        KernelKind::Final},
        {"pad_dispatch", KernelKind::PadDispatch},
        {"bm4_pad",      KernelKind::BM4Pad},
        {"bm8_pad",      KernelKind::BM8Pad},
        {"bm16_pad",     KernelKind::BM16Pad}
    };

    std::vector<KernelResult> results;
    results.reserve(kernels.size());

    for (const auto& spec : kernels) {
        float err = check_correctness(
            spec,
            h_ref,
            h_out,
            d_Q,
            d_K,
            d_V,
            d_scores,
            d_probs,
            d_O,
            qkv_bytes,
            score_bytes,
            BH,
            S,
            D
        );

        float ms = benchmark_kernel(
            spec,
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

        results.push_back({spec.name, ms, err});
    }

    auto get_ms = [&](const std::string& name) -> float {
        for (const auto& r : results) {
            if (r.name == name) {
                return r.ms;
            }
        }

        return std::numeric_limits<float>::infinity();
    };

    auto get_err = [&](const std::string& name) -> float {
        for (const auto& r : results) {
            if (r.name == name) {
                return r.err;
            }
        }

        return std::numeric_limits<float>::infinity();
    };

    float unfused_ms = get_ms("unfused");
    float tiled_ms = get_ms("tiled");
    float fused_row_ms = get_ms("fused_row");
    float flash_ms = get_ms("flash_v1");
    float final_ms = get_ms("final");
    float pad_dispatch_ms = get_ms("pad_dispatch");
    float bm4_pad_ms = get_ms("bm4_pad");
    float bm8_pad_ms = get_ms("bm8_pad");
    float bm16_pad_ms = get_ms("bm16_pad");

    float unfused_err = get_err("unfused");
    float tiled_err = get_err("tiled");
    float fused_row_err = get_err("fused_row");
    float flash_err = get_err("flash_v1");
    float final_err = get_err("final");
    float pad_dispatch_err = get_err("pad_dispatch");
    float bm4_pad_err = get_err("bm4_pad");
    float bm8_pad_err = get_err("bm8_pad");
    float bm16_pad_err = get_err("bm16_pad");

    KernelResult best = results[0];

    for (const auto& r : results) {
        if (r.ms < best.ms) {
            best = r;
        }
    }

    std::cout << std::left
              << std::setw(5) << B
              << std::setw(5) << H
              << std::setw(6) << BH
              << std::setw(6) << S
              << std::setw(5) << D

              << std::setw(11) << std::fixed << std::setprecision(4) << unfused_ms
              << std::setw(11) << std::fixed << std::setprecision(4) << tiled_ms
              << std::setw(12) << std::fixed << std::setprecision(4) << fused_row_ms
              << std::setw(11) << std::fixed << std::setprecision(4) << flash_ms
              << std::setw(11) << std::fixed << std::setprecision(4) << final_ms
              << std::setw(13) << std::fixed << std::setprecision(4) << pad_dispatch_ms

              << std::setw(11) << std::fixed << std::setprecision(4) << bm4_pad_ms
              << std::setw(11) << std::fixed << std::setprecision(4) << bm8_pad_ms
              << std::setw(11) << std::fixed << std::setprecision(4) << bm16_pad_ms

              << std::setw(12) << std::fixed << std::setprecision(3) << speedup(unfused_ms, final_ms)
              << std::setw(11) << std::fixed << std::setprecision(3) << speedup(flash_ms, final_ms)
              << std::setw(13) << std::fixed << std::setprecision(3) << speedup(tiled_ms, final_ms)
              << std::setw(13) << std::fixed << std::setprecision(3) << speedup(fused_row_ms, final_ms)

              << std::setw(13) << best.name
              << std::setw(10) << std::fixed << std::setprecision(4) << best.ms

              << std::setw(12) << std::scientific << std::setprecision(2) << unfused_err
              << std::setw(11) << std::scientific << std::setprecision(2) << tiled_err
              << std::setw(12) << std::scientific << std::setprecision(2) << fused_row_err
              << std::setw(11) << std::scientific << std::setprecision(2) << flash_err
              << std::setw(11) << std::scientific << std::setprecision(2) << final_err
              << std::setw(13) << std::scientific << std::setprecision(2) << pad_dispatch_err
              << std::setw(12) << std::scientific << std::setprecision(2) << bm4_pad_err
              << std::setw(12) << std::scientific << std::setprecision(2) << bm8_pad_err
              << std::setw(13) << std::scientific << std::setprecision(2) << bm16_pad_err
              << "\n";

    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));

    CHECK_CUDA(cudaFree(d_scores));
    CHECK_CUDA(cudaFree(d_probs));
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

    std::cout << "=== Causal Attention Forward: unfused vs optimized padding dispatch ===\n";

    std::cout << std::left
              << std::setw(5) << "B"
              << std::setw(5) << "H"
              << std::setw(6) << "BH"
              << std::setw(6) << "S"
              << std::setw(5) << "D"

              << std::setw(11) << "unfused"
              << std::setw(11) << "tiled"
              << std::setw(12) << "fused_row"
              << std::setw(11) << "flash_v1"
              << std::setw(11) << "final"
              << std::setw(13) << "pad_dispatch"

              << std::setw(11) << "bm4_pad"
              << std::setw(11) << "bm8_pad"
              << std::setw(11) << "bm16_pad"

              << std::setw(12) << "unf/final"
              << std::setw(11) << "fl/final"
              << std::setw(13) << "tiled/final"
              << std::setw(13) << "fused/final"

              << std::setw(13) << "best"
              << std::setw(10) << "best_ms"

              << std::setw(12) << "unf_err"
              << std::setw(11) << "tiled_err"
              << std::setw(12) << "fused_err"
              << std::setw(11) << "flash_err"
              << std::setw(11) << "final_err"
              << std::setw(13) << "pad_disp_err"
              << std::setw(12) << "bm4pad_err"
              << std::setw(12) << "bm8pad_err"
              << std::setw(13) << "bm16pad_err"
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