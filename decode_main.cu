#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <limits>

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
);

void decode_qk_gemv_launcher(
    const float* d_q,
    const float* d_k_cache,
    float* d_scores,
    int B,
    int S,
    int QH,
    int KVH,
    int D
);

void decode_softmax_launcher(
    const float* d_scores,
    float* d_probs,
    int B,
    int S,
    int QH
);

void decode_pv_gemv_launcher(
    const float* d_probs,
    const float* d_v_cache,
    float* d_out,
    int B,
    int S,
    int QH,
    int KVH,
    int D
);

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
);

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
);

int get_decode_attention_split_kv_num_splits(
    int S,
    int split_tokens
);

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
);

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        const cudaError_t status = (call);                                 \
        if (status != cudaSuccess) {                                       \
            throw std::runtime_error(                                      \
                std::string("CUDA error at ") + __FILE__ + ":" +          \
                std::to_string(__LINE__) + " -> " +                        \
                cudaGetErrorString(status)                                 \
            );                                                             \
        }                                                                  \
    } while (false)

namespace {
    // 默认用于普通正确性测试。
constexpr int kDefaultSplitTokens = 512;

// 用于 Split-KV auto-tuning 的候选参数。
// benchmark 时会逐个测试。
constexpr int kSplitTokenCandidates[] = {
    128,
    256,
    512,
    1024
};
}
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count)
        : count_(count) {
        if (count_ == 0) {
            throw std::invalid_argument(
                "DeviceBuffer count 必须大于 0"
            );
        }

        CUDA_CHECK(
            cudaMalloc(
                reinterpret_cast<void**>(&ptr_),
                count_ * sizeof(T)
            )
        );
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() {
        return ptr_;
    }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

class CudaEvent {
public:
    CudaEvent() {
        CUDA_CHECK(cudaEventCreate(&event_));
    }

    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const {
        return event_;
    }

private:
    cudaEvent_t event_ = nullptr;
};

std::vector<float> make_random_vector(
    std::size_t count,
    unsigned int seed
) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(
        -0.5f,
        0.5f
    );

    std::vector<float> values(count);

    for (float& value : values) {
        value = distribution(generator);
    }

    return values;
}

float max_abs_error(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    std::size_t* max_index
) {
    if (actual.size() != expected.size()) {
        throw std::invalid_argument("tensor size 不一致");
    }

    float max_error = 0.0f;
    std::size_t index = 0;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float error =
            std::fabs(actual[i] - expected[i]);

        if (error > max_error) {
            max_error = error;
            index = i;
        }
    }

    if (max_index != nullptr) {
        *max_index = index;
    }

    return max_error;
}

bool check_tensor_close(
    const std::string& name,
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tolerance = 2e-4f
) {
    std::size_t index = 0;

    const float error = max_abs_error(
        actual,
        expected,
        &index
    );

    if (error > tolerance) {
        std::cerr << name
                  << " failed: max_abs_error = "
                  << std::scientific << error
                  << " at index " << index
                  << ", actual = " << actual[index]
                  << ", expected = " << expected[index]
                  << "\n";
        return false;
    }

    std::cout << name
              << " passed! max_abs_error = "
              << std::scientific << error
              << " at index " << index
              << "\n";

    return true;
}

bool check_probability_rows(
    const std::vector<float>& probs,
    int B,
    int S,
    int QH
) {
    bool ok = true;

    for (int b = 0; b < B; ++b) {
        for (int qh = 0; qh < QH; ++qh) {
            const std::size_t offset =
                (static_cast<std::size_t>(b) * QH + qh) * S;

            float sum = 0.0f;

            for (int s = 0; s < S; ++s) {
                const float p = probs[offset + s];

                if (p < -1e-6f) {
                    std::cerr << "Softmax probability < 0"
                              << " at b=" << b
                              << ", qh=" << qh
                              << ", s=" << s
                              << "\n";
                    return false;
                }

                sum += p;
            }

            if (std::fabs(sum - 1.0f) > 2e-4f) {
                std::cerr << "Softmax row sum failed"
                          << " at b=" << b
                          << ", qh=" << qh
                          << ", sum=" << sum
                          << "\n";
                ok = false;
            }
        }
    }

    if (ok) {
        std::cout << "Softmax row-sum check passed!\n";
    }

    return ok;
}

template <typename LaunchFn>
float benchmark_cuda_event(
    LaunchFn launch,
    int warmup,
    int repeat
) {
    for (int i = 0; i < warmup; ++i) {
        launch();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CudaEvent start;
    CudaEvent stop;

    CUDA_CHECK(cudaEventRecord(start.get()));

    for (int i = 0; i < repeat; ++i) {
        launch();
    }

    CUDA_CHECK(cudaEventRecord(stop.get()));
    CUDA_CHECK(cudaEventSynchronize(stop.get()));

    float elapsed_ms = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &elapsed_ms,
            start.get(),
            stop.get()
        )
    );

    return elapsed_ms / static_cast<float>(repeat);
}

template <typename LaunchFn>
float benchmark_host_sync(
    LaunchFn launch,
    int warmup,
    int repeat
) {
    for (int i = 0; i < warmup; ++i) {
        launch();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    const auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; ++i) {
        launch();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    const auto stop = std::chrono::steady_clock::now();

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(
            stop - start
        ).count();

    return static_cast<float>(
        elapsed_ms / static_cast<double>(repeat)
    );
}

bool test_gqa_decode_attention() {
    std::cout << "\n========================================\n";
    std::cout << "Test: GQA Decode Attention Correctness\n";
    std::cout << "========================================\n";

    constexpr int B = 2;
    constexpr int S = 17;
    constexpr int QH = 4;
    constexpr int KVH = 2;
    constexpr int D = 32;

    const std::size_t q_elements =
        static_cast<std::size_t>(B) * QH * D;

    const std::size_t kv_elements =
        static_cast<std::size_t>(B) * S * KVH * D;

    const std::size_t score_elements =
        static_cast<std::size_t>(B) * QH * S;

    const std::vector<float> q =
        make_random_vector(q_elements, 20260629);

    const std::vector<float> k_cache =
        make_random_vector(kv_elements, 20260630);

    const std::vector<float> v_cache =
        make_random_vector(kv_elements, 20260631);

    std::vector<float> scores_cpu(score_elements);
    std::vector<float> probs_cpu(score_elements);
    std::vector<float> out_cpu(q_elements);

    decode_attention_gqa_cpu(
        q.data(),
        k_cache.data(),
        v_cache.data(),
        scores_cpu.data(),
        probs_cpu.data(),
        out_cpu.data(),
        B,
        S,
        QH,
        KVH,
        D
    );

    DeviceBuffer<float> d_q(q_elements);
    DeviceBuffer<float> d_k_cache(kv_elements);
    DeviceBuffer<float> d_v_cache(kv_elements);
    DeviceBuffer<float> d_scores(score_elements);
    DeviceBuffer<float> d_probs(score_elements);
    DeviceBuffer<float> d_out(q_elements);

    CUDA_CHECK(cudaMemcpy(
        d_q.get(),
        q.data(),
        q.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_k_cache.get(),
        k_cache.data(),
        k_cache.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_v_cache.get(),
        v_cache.data(),
        v_cache.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    launch_decode_attention_three_stage(
        d_q.get(),
        d_k_cache.get(),
        d_v_cache.get(),
        d_scores.get(),
        d_probs.get(),
        d_out.get(),
        B,
        S,
        QH,
        KVH,
        D
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> scores_gpu(score_elements);
    std::vector<float> probs_gpu(score_elements);
    std::vector<float> out_gpu(q_elements);

    CUDA_CHECK(cudaMemcpy(
        scores_gpu.data(),
        d_scores.get(),
        scores_gpu.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        probs_gpu.data(),
        d_probs.get(),
        probs_gpu.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        out_gpu.data(),
        d_out.get(),
        out_gpu.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    bool all_ok = true;

    all_ok &= check_tensor_close(
        "QK GEMV scores vs CPU",
        scores_gpu,
        scores_cpu
    );

    all_ok &= check_tensor_close(
        "Softmax probs vs CPU",
        probs_gpu,
        probs_cpu
    );

    all_ok &= check_tensor_close(
        "PV GEMV output vs CPU",
        out_gpu,
        out_cpu
    );

    all_ok &= check_probability_rows(
        probs_gpu,
        B,
        S,
        QH
    );

    launch_decode_attention_fused_online_softmax(
        d_q.get(),
        d_k_cache.get(),
        d_v_cache.get(),
        d_out.get(),
        B,
        S,
        QH,
        KVH,
        D
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> fused_out_gpu(q_elements);

    CUDA_CHECK(cudaMemcpy(
        fused_out_gpu.data(),
        d_out.get(),
        fused_out_gpu.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    all_ok &= check_tensor_close(
        "Fused online softmax output vs CPU",
        fused_out_gpu,
        out_cpu
    );

        // --------------------------------------------------------
    // Split-KV 正确性测试。
    //
    // 普通 correctness test 固定用 split_tokens=512。
    // benchmark 阶段会再扫描多个 split_tokens。
    // --------------------------------------------------------

    constexpr int split_tokens =
        kDefaultSplitTokens;

    const int split_num =
        get_decode_attention_split_kv_num_splits(
            S,
            split_tokens
        );

    const std::size_t partial_elements =
        static_cast<std::size_t>(B) *
        QH *
        split_num;

    const std::size_t partial_acc_elements =
        partial_elements * D;

    DeviceBuffer<float> d_partial_max(
        partial_elements
    );

    DeviceBuffer<float> d_partial_sum(
        partial_elements
    );

    DeviceBuffer<float> d_partial_acc(
        partial_acc_elements
    );

    launch_decode_attention_split_kv(
        d_q.get(),
        d_k_cache.get(),
        d_v_cache.get(),
        d_partial_max.get(),
        d_partial_sum.get(),
        d_partial_acc.get(),
        d_out.get(),
        B,
        S,
        QH,
        KVH,
        D,
        split_tokens
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> split_kv_out_gpu(q_elements);

    CUDA_CHECK(cudaMemcpy(
        split_kv_out_gpu.data(),
        d_out.get(),
        split_kv_out_gpu.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    all_ok &= check_tensor_close(
        "Split-KV output vs CPU",
        split_kv_out_gpu,
        out_cpu
    );

    return all_ok;
}

struct BenchmarkConfig {
    const char* name;
    int B;
    int S;
    int QH;
    int KVH;
    int D;
    int event_repeat;
    int host_repeat;
};

void benchmark_decode_case(
    const BenchmarkConfig& config
) {
    constexpr int kWarmup = 10;

    // --------------------------------------------------------
    // 计算各张量的元素数量。
    // --------------------------------------------------------

    const std::size_t q_elements =
        static_cast<std::size_t>(config.B) *
        config.QH *
        config.D;

    const std::size_t kv_elements =
        static_cast<std::size_t>(config.B) *
        config.S *
        config.KVH *
        config.D;

    const std::size_t score_elements =
        static_cast<std::size_t>(config.B) *
        config.QH *
        config.S;

    // --------------------------------------------------------
    // 为当前 S 生成真正有效的 split candidate。
    //
    // 例如：
    // S=128 时：
    // 原候选 [128,256,512,1024]
    // 实际只保留 [128]
    //
    // S=512 时：
    // 实际保留 [128,256,512]
    //
    // 避免多个候选都退化为同一个 num_splits=1。
    // --------------------------------------------------------

    std::vector<int> split_token_candidates;

    for (const int candidate : kSplitTokenCandidates) {
        const int effective_split_tokens =
            std::min(candidate, config.S);

        const bool already_exists =
            std::find(
                split_token_candidates.begin(),
                split_token_candidates.end(),
                effective_split_tokens
            ) != split_token_candidates.end();

        if (!already_exists) {
            split_token_candidates.push_back(
                effective_split_tokens
            );
        }
    }

    // --------------------------------------------------------
    // 构造确定性随机输入。
    // --------------------------------------------------------

    const std::vector<float> q =
        make_random_vector(
            q_elements,
            1000U + static_cast<unsigned int>(config.S)
        );

    const std::vector<float> k_cache =
        make_random_vector(
            kv_elements,
            2000U + static_cast<unsigned int>(config.S)
        );

    const std::vector<float> v_cache =
        make_random_vector(
            kv_elements,
            3000U + static_cast<unsigned int>(config.S)
        );

    // --------------------------------------------------------
    // CPU reference。
    // --------------------------------------------------------

    std::vector<float> scores_cpu(score_elements);
    std::vector<float> probs_cpu(score_elements);
    std::vector<float> out_cpu(q_elements);

    decode_attention_gqa_cpu(
        q.data(),
        k_cache.data(),
        v_cache.data(),
        scores_cpu.data(),
        probs_cpu.data(),
        out_cpu.data(),
        config.B,
        config.S,
        config.QH,
        config.KVH,
        config.D
    );

    // --------------------------------------------------------
    // GPU 输入 / 输出 buffer。
    // --------------------------------------------------------

    DeviceBuffer<float> d_q(q_elements);
    DeviceBuffer<float> d_k_cache(kv_elements);
    DeviceBuffer<float> d_v_cache(kv_elements);
    DeviceBuffer<float> d_scores(score_elements);
    DeviceBuffer<float> d_probs(score_elements);
    DeviceBuffer<float> d_out(q_elements);

    CUDA_CHECK(cudaMemcpy(
        d_q.get(),
        q.data(),
        q.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_k_cache.get(),
        k_cache.data(),
        k_cache.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_v_cache.get(),
        v_cache.data(),
        v_cache.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    const std::string check_name =
        std::string("Benchmark correctness: ") +
        config.name;

    std::vector<float> out_gpu(q_elements);

    // --------------------------------------------------------
    // 三段式 baseline 正确性。
    // --------------------------------------------------------

    launch_decode_attention_three_stage(
        d_q.get(),
        d_k_cache.get(),
        d_v_cache.get(),
        d_scores.get(),
        d_probs.get(),
        d_out.get(),
        config.B,
        config.S,
        config.QH,
        config.KVH,
        config.D
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        out_gpu.data(),
        d_out.get(),
        out_gpu.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    if (!check_tensor_close(
            check_name + " three-stage output",
            out_gpu,
            out_cpu
        )) {
        throw std::runtime_error(
            "Three-stage benchmark correctness check failed"
        );
    }

    // --------------------------------------------------------
    // Fused Online Softmax v1 正确性。
    // --------------------------------------------------------

    launch_decode_attention_fused_online_softmax(
        d_q.get(),
        d_k_cache.get(),
        d_v_cache.get(),
        d_out.get(),
        config.B,
        config.S,
        config.QH,
        config.KVH,
        config.D
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        out_gpu.data(),
        d_out.get(),
        out_gpu.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    if (!check_tensor_close(
            check_name + " fused v1 output",
            out_gpu,
            out_cpu
        )) {
        throw std::runtime_error(
            "Fused v1 benchmark correctness check failed"
        );
    }

    // --------------------------------------------------------
    // 封装 baseline / fused 的 launch。
    // --------------------------------------------------------

    const auto launch_qk = [&]() {
        decode_qk_gemv_launcher(
            d_q.get(),
            d_k_cache.get(),
            d_scores.get(),
            config.B,
            config.S,
            config.QH,
            config.KVH,
            config.D
        );
    };

    const auto launch_softmax = [&]() {
        decode_softmax_launcher(
            d_scores.get(),
            d_probs.get(),
            config.B,
            config.S,
            config.QH
        );
    };

    const auto launch_pv = [&]() {
        decode_pv_gemv_launcher(
            d_probs.get(),
            d_v_cache.get(),
            d_out.get(),
            config.B,
            config.S,
            config.QH,
            config.KVH,
            config.D
        );
    };

    const auto launch_three_stage = [&]() {
        launch_decode_attention_three_stage(
            d_q.get(),
            d_k_cache.get(),
            d_v_cache.get(),
            d_scores.get(),
            d_probs.get(),
            d_out.get(),
            config.B,
            config.S,
            config.QH,
            config.KVH,
            config.D
        );
    };

    const auto launch_fused_v1 = [&]() {
        launch_decode_attention_fused_online_softmax(
            d_q.get(),
            d_k_cache.get(),
            d_v_cache.get(),
            d_out.get(),
            config.B,
            config.S,
            config.QH,
            config.KVH,
            config.D
        );
    };

    // --------------------------------------------------------
    // baseline / fused CUDA Event 计时。
    // --------------------------------------------------------

    const float qk_ms = benchmark_cuda_event(
        launch_qk,
        kWarmup,
        config.event_repeat
    );

    const float softmax_ms = benchmark_cuda_event(
        launch_softmax,
        kWarmup,
        config.event_repeat
    );

    const float pv_ms = benchmark_cuda_event(
        launch_pv,
        kWarmup,
        config.event_repeat
    );

    const float three_stage_ms =
        benchmark_cuda_event(
            launch_three_stage,
            kWarmup,
            config.event_repeat
        );

    const float fused_v1_ms =
        benchmark_cuda_event(
            launch_fused_v1,
            kWarmup,
            config.event_repeat
        );

    float three_stage_host_sync_ms = 0.0f;
    float fused_v1_host_sync_ms = 0.0f;

    if (config.host_repeat > 0) {
        three_stage_host_sync_ms =
            benchmark_host_sync(
                launch_three_stage,
                kWarmup,
                config.host_repeat
            );

        fused_v1_host_sync_ms =
            benchmark_host_sync(
                launch_fused_v1,
                kWarmup,
                config.host_repeat
            );
    }

    // 三段式 workspace：
    // scores + probs。
    const double three_stage_workspace_mb =
        static_cast<double>(
            2 * score_elements * sizeof(float)
        ) /
        (1024.0 * 1024.0);

    // --------------------------------------------------------
    // 输出当前 shape 的 baseline / fused 结果。
    // --------------------------------------------------------

    std::cout << "\n---- " << config.name << " ----\n";

    std::cout << "shape: B=" << config.B
              << ", S=" << config.S
              << ", QH=" << config.QH
              << ", KVH=" << config.KVH
              << ", D=" << config.D
              << "\n";

    std::cout << "GQA group size: "
              << config.QH / config.KVH
              << "\n";

    std::cout << "scores + probs workspace: "
              << std::fixed << std::setprecision(3)
              << three_stage_workspace_mb
              << " MiB\n";

    std::cout << "CUDA Event QK GEMV:  "
              << qk_ms * 1000.0f
              << " us\n";

    std::cout << "CUDA Event Softmax:  "
              << softmax_ms * 1000.0f
              << " us\n";

    std::cout << "CUDA Event PV GEMV:  "
              << pv_ms * 1000.0f
              << " us\n";

    std::cout << "CUDA Event stage sum:"
              << (qk_ms + softmax_ms + pv_ms) * 1000.0f
              << " us\n";

    std::cout << "CUDA Event three-stage: "
              << three_stage_ms * 1000.0f
              << " us\n";

    std::cout << "CUDA Event fused v1:    "
              << fused_v1_ms * 1000.0f
              << " us\n";

    std::cout << "Three-stage / fused v1: "
              << three_stage_ms / fused_v1_ms
              << "x\n";

    if (config.host_repeat > 0) {
        std::cout << "Host Sync three-stage: "
                  << three_stage_host_sync_ms * 1000.0f
                  << " us\n";

        std::cout << "Host Sync fused v1:    "
                  << fused_v1_host_sync_ms * 1000.0f
                  << " us\n";
    }

    // --------------------------------------------------------
    // Split-KV auto-tuning。
    //
    // 对每个 split_tokens：
    //
    // 1. 分配对应 workspace。
    // 2. 做 CPU reference 正确性验证。
    // 3. CUDA Event benchmark。
    // 4. Host Sync benchmark。
    // 5. 记录最快候选。
    // --------------------------------------------------------

    float best_split_kv_ms =
        std::numeric_limits<float>::infinity();

    float best_split_kv_host_sync_ms = 0.0f;

    int best_split_tokens = 0;
    int best_num_splits = 0;

    std::cout << "Split-KV candidate scan:\n";

    for (const int split_tokens : split_token_candidates) {
        const int num_splits =
            get_decode_attention_split_kv_num_splits(
                config.S,
                split_tokens
            );

        // partial_max / partial_sum:
        // [B, QH, num_splits]
        const std::size_t partial_elements =
            static_cast<std::size_t>(config.B) *
            config.QH *
            num_splits;

        // partial_acc:
        // [B, QH, num_splits, D]
        const std::size_t partial_acc_elements =
            partial_elements *
            config.D;

        DeviceBuffer<float> d_partial_max(
            partial_elements
        );

        DeviceBuffer<float> d_partial_sum(
            partial_elements
        );

        DeviceBuffer<float> d_partial_acc(
            partial_acc_elements
        );

        const auto launch_split_kv = [&]() {
            launch_decode_attention_split_kv(
                d_q.get(),
                d_k_cache.get(),
                d_v_cache.get(),
                d_partial_max.get(),
                d_partial_sum.get(),
                d_partial_acc.get(),
                d_out.get(),
                config.B,
                config.S,
                config.QH,
                config.KVH,
                config.D,
                split_tokens
            );
        };

        // ----------------------------------------------------
        // 当前 split_tokens 的正确性。
        // ----------------------------------------------------

        launch_split_kv();

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(
            out_gpu.data(),
            d_out.get(),
            out_gpu.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        if (!check_tensor_close(
                check_name +
                " Split-KV split=" +
                std::to_string(split_tokens),
                out_gpu,
                out_cpu
            )) {
            throw std::runtime_error(
                "Split-KV benchmark correctness check failed"
            );
        }

        // ----------------------------------------------------
        // 当前 split_tokens 的性能。
        // ----------------------------------------------------

        const float split_kv_ms =
            benchmark_cuda_event(
                launch_split_kv,
                kWarmup,
                config.event_repeat
            );

        float split_kv_host_sync_ms = 0.0f;

        if (config.host_repeat > 0) {
            split_kv_host_sync_ms =
                benchmark_host_sync(
                    launch_split_kv,
                    kWarmup,
                    config.host_repeat
                );
        }

        const double split_workspace_mb =
            static_cast<double>(
                (
                    partial_elements +
                    partial_elements +
                    partial_acc_elements
                ) * sizeof(float)
            ) /
            (1024.0 * 1024.0);

        std::cout
            << "  split_tokens="
            << std::setw(4)
            << split_tokens
            << ", num_splits="
            << std::setw(2)
            << num_splits
            << ", workspace="
            << std::fixed << std::setprecision(3)
            << split_workspace_mb
            << " MiB"
            << ", CUDA Event="
            << split_kv_ms * 1000.0f
            << " us"
            << ", Three-stage / Split-KV="
            << three_stage_ms / split_kv_ms
            << "x"
            << ", Fused v1 / Split-KV="
            << fused_v1_ms / split_kv_ms
            << "x";

        if (config.host_repeat > 0) {
            std::cout
                << ", Host Sync="
                << split_kv_host_sync_ms * 1000.0f
                << " us";
        }

        std::cout << "\n";

        if (split_kv_ms < best_split_kv_ms) {
            best_split_kv_ms = split_kv_ms;
            best_split_kv_host_sync_ms =
                split_kv_host_sync_ms;

            best_split_tokens = split_tokens;
            best_num_splits = num_splits;
        }
    }

    std::cout
        << "Best Split-KV candidate: "
        << "split_tokens="
        << best_split_tokens
        << ", num_splits="
        << best_num_splits
        << ", CUDA Event="
        << best_split_kv_ms * 1000.0f
        << " us"
        << ", Three-stage / best Split-KV="
        << three_stage_ms / best_split_kv_ms
        << "x"
        << ", Fused v1 / best Split-KV="
        << fused_v1_ms / best_split_kv_ms
        << "x";

    if (config.host_repeat > 0) {
        std::cout
            << ", Host Sync="
            << best_split_kv_host_sync_ms * 1000.0f
            << " us";
    }

    std::cout << "\n";

    // --------------------------------------------------------
    // 给当前 shape 输出最快路径。
    // --------------------------------------------------------

    if (
        fused_v1_ms <= three_stage_ms &&
        fused_v1_ms <= best_split_kv_ms
    ) {
        std::cout
            << "Recommended path: Fused v1\n";
    } else if (
        three_stage_ms <= fused_v1_ms &&
        three_stage_ms <= best_split_kv_ms
    ) {
        std::cout
            << "Recommended path: Three-stage\n";
    } else {
        std::cout
            << "Recommended path: Split-KV"
            << " (split_tokens="
            << best_split_tokens
            << ")\n";
    }
}

void run_benchmarks() {
    std::cout << "\n========================================\n";
    std::cout << "GQA Decode Attention Auto-Tuning Benchmark\n";
    std::cout << "========================================\n";

    std::cout
        << "Implementations: "
        << "Three-stage / Fused v1 / Split-KV\n";

    std::cout
        << "Split-KV candidates: "
        << "128 / 256 / 512 / 1024\n";

    std::cout
        << "Not included: "
        << "cudaMalloc, H2D, D2H, input generation\n";

    const BenchmarkConfig configs[] = {
        {
            "Decode S=1",
            1, 1, 32, 8, 128,
            100000, 10000
        },
        {
            "Decode S=128",
            1, 128, 32, 8, 128,
            10000, 3000
        },
        {
            "Decode S=256",
            1, 256, 32, 8, 128,
            6000, 2000
        },
        {
            "Decode S=384",
            1, 384, 32, 8, 128,
            4000, 1500
        },
        {
            "Decode S=512",
            1, 512, 32, 8, 128,
            3000, 1000
        },
        {
            "Decode S=768",
            1, 768, 32, 8, 128,
            2000, 800
        },
        {
            "Decode S=1024",
            1, 1024, 32, 8, 128,
            1500, 600
        },
        {
            "Decode S=1536",
            1, 1536, 32, 8, 128,
            1000, 400
        },
        {
            "Decode S=2048",
            1, 2048, 32, 8, 128,
            800, 300
        },
        {
            "Decode S=4096",
            1, 4096, 32, 8, 128,
            400, 150
        },
        {
            "Decode S=8192",
            1, 8192, 32, 8, 128,
            200, 80
        }
    };

    for (const BenchmarkConfig& config : configs) {
        benchmark_decode_case(config);
    }
}

int main(int argc, char** argv) {
    try {
        bool run_benchmark = false;

        if (argc == 2) {
            const std::string arg = argv[1];

            if (arg == "--bench") {
                run_benchmark = true;
            } else {
                std::cerr
                    << "Usage: ./decode_attention_bench [--bench]\n";
                return EXIT_FAILURE;
            }
        } else if (argc > 2) {
            std::cerr
                << "Usage: ./decode_attention_bench [--bench]\n";
            return EXIT_FAILURE;
        }

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));

        if (device_count == 0) {
            std::cerr << "No CUDA device found.\n";
            return EXIT_FAILURE;
        }

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

        std::cout << "CUDA GQA Decode Attention\n";
        std::cout << "GPU: " << prop.name << "\n";
        std::cout << "Q layout:       [B, QH, D]\n";
        std::cout << "K/V cache:      [B, S, KVH, D]\n";
        std::cout << "scores / probs: [B, QH, S]\n";
        std::cout << "O layout:       [B, QH, D]\n";
        std::cout << "Pipeline: QK GEMV -> Softmax -> PV GEMV\n";

        if (!test_gqa_decode_attention()) {
            std::cerr << "GQA Decode Attention test failed.\n";
            return EXIT_FAILURE;
        }

        std::cout << "\nAll GQA Decode Attention tests passed!\n";

        if (run_benchmark) {
            run_benchmarks();
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Exception: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}