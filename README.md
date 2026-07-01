# GQA Decode Attention CUDA Kernel Optimization

面向 Decoder 推理阶段的 FP32 GQA Decode Attention CUDA 算子优化项目。

项目从最基础的三段式 Attention baseline 出发，逐步实现：

- 三段式 `QK GEMV -> Softmax -> PV GEMV`
- Fused Online Softmax
- 4-lane subgroup QK 点积协作
- Split-KV Partial + Merge
- Split-KV `split_tokens` 参数扫描与路径推荐

测试平台为 NVIDIA A40，重点研究长 KV Cache 下的 Decode Attention 并行度和中间 workspace 开销。

---

## 1. 项目背景

Decode 阶段通常一次只生成一个 token，因此 Query 的形状为：

```text
Q: [B, QH, D]
```

历史 Key / Value 存放在 KV Cache 中：

```text
K cache: [B, S, KVH, D]
V cache: [B, S, KVH, D]
```

其中：

```text
B   : batch size
S   : KV Cache 中的历史 token 数
QH  : Query head 数
KVH : Key / Value head 数
D   : 每个 head 的维度
```

本项目使用 GQA（Grouped Query Attention）：

```text
group_size = QH / KVH
kv_head_idx = q_head_idx / group_size
```

例如：

```text
QH = 32
KVH = 8
group_size = 4
```

则每 4 个 Query head 共享一个 Key / Value head。

---

## 2. Attention 计算流程

对于每个 `(b, qh)`：

```text
score[s] = dot(Q[b, qh, :], K[b, s, kvh, :]) / sqrt(D)

prob[s] = softmax(score)[s]

Out[b, qh, d] =
    sum_s prob[s] * V[b, s, kvh, d]
```

三段式实现对应：

```text
QK GEMV
    ↓
scores [B, QH, S]
    ↓
Softmax
    ↓
probs [B, QH, S]
    ↓
PV GEMV
    ↓
Out [B, QH, D]
```

---

## 3. 实现路径

### 3.1 Three-stage Baseline

基础版本拆成三个 kernel：

```text
1. QK GEMV
2. Softmax
3. PV GEMV
```

优点：

- 实现清晰
- 便于逐阶段验证
- 方便作为性能与正确性 baseline

不足：

- 需要额外写入和读取 `scores`
- 需要额外写入和读取 `probs`
- 需要三次 kernel launch
- 长序列时中间 workspace 随 S 线性增长

---

### 3.2 Fused Online Softmax

将 QK、Softmax、PV 合并到一个 kernel 中。

核心优化：

```text
不再写 scores 到 global memory
不再写 probs 到 global memory
一个 kernel 完成 QK + Online Softmax + PV
```

对于每个 `(b, qh)`，一个 CUDA block 扫描完整 KV Cache。

Online Softmax 维护：

```text
m = 当前处理 token 的最大 score
l = sum(exp(score - m))
acc[d] = sum(exp(score - m) * V[d])
```

最终输出：

```text
Out[d] = acc[d] / l
```

短序列下，Fused 版本能明显减少 kernel launch 与中间访存。

---

### 3.3 4-lane Subgroup QK

在 Fused 和 Split-KV 的 QK 阶段中，采用 4-lane subgroup 协作计算一个 token 的 QK 点积。

```text
128 threads = 32 tokens × 4 threads/token
```

对于一个 token：

```text
lane 0: d = 0, 4, 8, ...
lane 1: d = 1, 5, 9, ...
lane 2: d = 2, 6, 10, ...
lane 3: d = 3, 7, 11, ...
```

每个线程先计算部分 `local_dot`，随后只在 4 个线程内部通过 warp shuffle 求和。

相比“一个 token 使用完整 block reduction”，该方式降低了 QK 点积规约开销。

---

### 3.4 Split-KV

普通 Fused Online Softmax 的 grid 为：

```text
grid.x = B × QH
```

在本项目测试配置中：

```text
B = 1
QH = 32
```

因此普通 Fused kernel 只有 32 个 CTA。

当 `S` 很大时，每个 CTA 都要串行扫描完整 KV Cache，GPU 并行度不足。

Split-KV 沿 S 维切分 KV Cache：

```text
grid.x = B × QH × num_splits
```

例如：

```text
S = 8192
split_tokens = 128
num_splits = 64

partial CTA 数量：
1 × 32 × 64 = 2048
```

每个 partial CTA 只处理一个 `(b, qh, split_idx)`。

Partial kernel 输出：

```text
partial_max[b, qh, split]
partial_sum[b, qh, split]
partial_acc[b, qh, split, d]
```

其中：

```text
m_i      = partial_max[i]
l_i      = partial_sum[i]
acc_i[d] = partial_acc[i, d]
```

Merge kernel 使用稳定 softmax 合并：

```text
M = max_i(m_i)

weight_i = exp(m_i - M)

merged_sum = sum_i(weight_i * l_i)

merged_acc[d] = sum_i(weight_i * acc_i[d])

Out[d] = merged_acc[d] / merged_sum
```

---

## 4. 编译与运行

### 编译

A40 对应 `sm_86`：

```bash
nvcc -O3 -std=c++17 -arch=sm_86 \
  decode_main.cu decode_attention.cu \
  -o decode_attention_bench
```

### 正确性测试

```bash
./decode_attention_bench
```

测试内容：

```text
QK GEMV scores vs CPU
Softmax probs vs CPU
PV GEMV output vs CPU
Softmax row-sum
Fused Online Softmax output vs CPU
Split-KV output vs CPU
```

### Benchmark 与 Split-KV 参数扫描

```bash
./decode_attention_bench --bench
```

当前扫描候选：

```text
split_tokens = 128 / 256 / 512 / 1024
```

---

## 5. Benchmark 设置

```text
GPU: NVIDIA A40
Precision: FP32

B   = 1
QH  = 32
KVH = 8
D   = 128

S = 1 / 128 / 256 / 384 / 512 / 768 /
    1024 / 1536 / 2048 / 4096 / 8192
```

计时方式：

```text
CUDA Event:
只统计 GPU 时间

Host Sync:
包含 kernel launch、CPU 等待与 GPU 执行时间
```

以下主表使用 CUDA Event 平均延迟。

不包含：

```text
cudaMalloc
H2D
D2H
随机输入生成
```

---

## 6. Benchmark 结果

### 最优路径

| S | 最优路径 | 最优 Split Tokens | 最优延迟 | Three-stage 延迟 | 相对 Three-stage |
|---:|---|---:|---:|---:|---:|
| 1 | Fused v1 | - | 5.211 us | 12.719 us | 2.44x |
| 128 | Fused v1 | - | 16.176 us | 31.810 us | 1.97x |
| 256 | Split-KV | 128 | 23.142 us | 36.541 us | 1.58x |
| 384 | Split-KV | 128 | 25.331 us | 41.490 us | 1.64x |
| 512 | Split-KV | 128 | 25.183 us | 46.123 us | 1.83x |
| 768 | Split-KV | 128 | 34.104 us | 92.116 us | 2.70x |
| 1024 | Split-KV | 128 | 39.079 us | 124.641 us | 3.19x |
| 1536 | Split-KV | 128 | 43.721 us | 184.134 us | 4.21x |
| 2048 | Split-KV | 128 | 48.486 us | 243.075 us | 5.01x |
| 4096 | Split-KV | 256 | 88.422 us | 460.168 us | 5.20x |
| 8192 | Split-KV | 128 | 163.953 us | 910.607 us | 5.55x |

### 长上下文重点结果

```text
GPU: NVIDIA A40
B=1, QH=32, KVH=8, D=128, S=8192

Three-stage: 910.607 us
Fused v1:    1538.831 us
Split-KV:     163.953 us
```

对应加速比：

```text
Split-KV vs Three-stage: 5.55x
Split-KV vs Fused v1:    9.39x
```

---

## 7. 性能分析

### 短序列

在 `S=1` 和 `S=128` 下，Fused v1 最优。

原因：

```text
减少两次 kernel launch
消除 scores / probs 的 global memory 中间读写
```

### 中长序列

普通 Fused v1 在长序列下变慢。

原因：

```text
grid.x = B × QH = 32
```

CTA 数量过少，无法充分利用 A40 的并行能力；每个 CTA 还需要串行扫描完整 S。

### Split-KV

Split-KV 将 S 维切成多个 split：

```text
grid.x = B × QH × num_splits
```

通过增加 partial CTA 数量获得更高并行度。

例如 `S=8192, split_tokens=128`：

```text
num_splits = 64
partial CTA = 1 × 32 × 64 = 2048
```

因此 Split-KV 在长上下文下显著优于单 CTA Fused。

---

## 8. Workspace 对比

三段式 baseline 需要：

```text
scores + probs
```

当 `S=8192`：

```text
Three-stage workspace: 2.000 MiB
```

Split-KV workspace 随 split 数增加：

| S=8192 配置 | Split-KV Workspace | CUDA Event |
|---|---:|---:|
| split_tokens=128 | 1.016 MiB | 163.953 us |
| split_tokens=256 | 0.508 MiB | 189.875 us |
| split_tokens=512 | 0.254 MiB | 167.757 us |
| split_tokens=1024 | 0.127 MiB | 270.259 us |

结论：

```text
split_tokens=128：
最低延迟优先

split_tokens=512：
延迟接近最低，但 workspace 更低
```

因此 Split-KV 不只是单纯追求速度，也存在延迟与 workspace 的工程权衡。

---

## 9. 当前实验性 Dispatch 建议

针对当前测试环境：

```text
GPU: A40
FP32
B=1, QH=32, KVH=8, D=128
```

可以采用：

```cpp
if (S <= 128) {
    use_fused_online_softmax();
} else if (S <= 2048) {
    use_split_kv(128);
} else if (S <= 4096) {
    use_split_kv(256);
} else {
    use_split_kv(128);
}
```

注意：

```text
该表仅基于当前 A40、当前 shape 与当前实现测试得到。
不同 GPU、batch size、head 数、head dimension 下需要重新 benchmark。
```

当前代码实现的是 benchmark 侧的 candidate scan 与推荐输出，尚未把该 dispatch 逻辑封装为实际 runtime API。

---

## 10. 正确性验证

所有测试 shape 均通过 CPU reference 对齐。

典型误差：

```text
max_abs_error ≈ 1e-8 ~ 1e-7
```

不同实现的浮点累加顺序不同，因此不要求 bitwise 完全一致。

---

## 11. 项目结构

```text
attention_kernel/
├── decode_attention.cu
│   ├── CPU reference
│   ├── QK GEMV kernel
│   ├── Softmax kernel
│   ├── PV GEMV kernel
│   ├── Fused Online Softmax kernel
│   ├── Split-KV Partial kernel
│   ├── Split-KV Merge kernel
│   └── kernel launchers
│
├── decode_main.cu
│   ├── correctness test
│   ├── CUDA Event benchmark
│   ├── Host Sync benchmark
│   ├── Split-KV candidate scan
│   └── path recommendation
│
└── README.md
```

---

## 12. 当前限制

当前实现是面向学习和算子优化验证的 FP32 GQA Decode Attention kernel，不是生产级推理引擎实现。

限制包括：

```text
FP32 only
D <= 128
固定 128-thread block 配置
连续 KV Cache layout
不支持 PagedAttention
未使用 FP16 / BF16 / Tensor Core
未使用 vectorized load
未接入 vLLM / TensorRT-LLM / SGLang
未实现真实 runtime dispatch
```

---

## 13. 后续可扩展方向

```text
1. 支持 FP16 / BF16
2. 支持 half2 / vectorized load
3. 支持更多 D，例如 64 / 80 / 96 / 128 / 256
4. 为不同 D、S、QH、KVH 建立更完整的 dispatch table
5. 支持 PagedAttention KV Cache
6. 对比 FlashDecoding 类方案
7. 引入 Nsight Compute 分析 occupancy、memory throughput、warp stall
8. 集成到简化推理引擎或 PyTorch extension
```

---

## 14. 项目总结

本项目完成了从 baseline 到并行度优化的完整 CUDA 算子优化闭环：

```text
Three-stage baseline
    ↓
Fused Online Softmax
    ↓
4-lane subgroup QK
    ↓
Split-KV Partial + Merge
    ↓
split_tokens candidate scan
    ↓
长上下文下获得显著加速
```

在 NVIDIA A40、FP32、`B=1,QH=32,KVH=8,D=128,S=8192` 下：

```text
Split-KV: 163.953 us
Three-stage: 910.607 us

Speedup: 5.55x
```
