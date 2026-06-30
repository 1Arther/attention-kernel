# CUDA Attention Kernels

基于 C++ / CUDA 实现并优化 Attention 核心算子。当前仓库包含两类实验：

- Prefill Causal Attention / FlashAttention
- GQA Decode Attention

本分支聚焦 **GQA Decode Attention**：从三段式 baseline 出发，完成 online softmax 融合，并将 QK 阶段从“逐 token 全 block 规约”优化为“4-lane subgroup 并行规约”。

---

## 1. 项目结构

```text
attention_kernel/
├── attention_kernel.cu
├── main.cu
├── decode_attention.cu
├── decode_main.cu
└── README.md
attention_kernel.cu / main.cu
    Prefill Causal Attention / FlashAttention 实验

decode_attention.cu / decode_main.cu
    GQA Decode Attention：
    - 三段式 QK GEMV -> Softmax -> PV GEMV baseline
    - Fused Online Softmax v1
2. GQA Decode Attention

Decode 阶段一次只处理当前生成 token 的 Query：

Q:       [B, QH, D]
K cache: [B, S, KVH, D]
V cache: [B, S, KVH, D]

scores:  [B, QH, S]
probs:   [B, QH, S]
Out:     [B, QH, D]

其中：

B   : batch size
S   : 当前 KV Cache 长度，即历史 token 数
QH  : Query head 数
KVH : Key / Value head 数
D   : 每个 head 的维度

GQA 中多个 Query head 共享一组 Key / Value head：

group_size = QH / KVH
kvh = qh / group_size

例如：

QH = 32
KVH = 8
group_size = 4

则：

Q head 0~3   -> KV head 0
Q head 4~7   -> KV head 1
Q head 8~11  -> KV head 2
...
Q head 28~31 -> KV head 7

对于固定的 (b, qh)：

score[s] =
dot(Q[b, qh, :], K_cache[b, s, kvh, :]) / sqrt(D)

prob[s] = softmax(score)[s]

Out[b, qh, d] =
sum_s prob[s] * V_cache[b, s, kvh, d]
3. 三段式 Baseline
3.1 QK GEMV
scores[b, qh, s] =
dot(Q[b, qh, :], K_cache[b, s, kvh, :]) / sqrt(D)

线程映射：

一个线程负责一个 scores[b, qh, s]
一个线程内部沿 D 维完成点积
3.2 Softmax
probs[b, qh, :] =
softmax(scores[b, qh, :])

线程映射：

一个 block 负责一条 score row，即一个 (b, qh)

使用 warp shuffle 与 shared memory 完成 block-level max / sum reduction。

3.3 PV GEMV
Out[b, qh, d] =
sum_s probs[b, qh, s] * V_cache[b, s, kvh, d]

线程映射：

一个线程负责一个 Out[b, qh, d]
线程沿 S 维扫描并累积
4. Fused Online Softmax

Fused 版本将：

QK GEMV -> Softmax -> PV GEMV

合并成一个 kernel，避免中间张量：

scores: [B, QH, S]
probs:  [B, QH, S]

写回和重新读取 global memory。

一个 fused block 负责一个 (b, qh)：

grid.x  = B * QH
block.x = 128

对于常用测试形状：

B=1, QH=32, KVH=8, D=128

每个 block 对应一个 Query head，并维护该 head 的：

online_max
online_sum
output_acc[d]

在线 softmax 状态更新：

m_new = max(m_old, tile_max)

alpha = exp(m_old - m_new)

l_new =
alpha * l_old +
sum(exp(score_tile - m_new))

acc_new[d] =
alpha * acc_old[d] +
sum(exp(score_tile - m_new) * V_tile[:, d])

Out[d] = acc[d] / l
5. Fused v0：逐 Token Full-Block Reduction

初版 fused kernel 将 KV Cache 按 BLOCK_N=32 切分。

但每一个 tile 内的 32 个 token 使用如下方式计算 QK：

token 0  -> 128-thread block reduction
token 1  -> 128-thread block reduction
...
token 31 -> 128-thread block reduction

即每个 tile 需要 32 次 full-block reduction，导致 token score 计算接近串行化。

该版本数值正确，但长序列性能较差。

6. Fused v1：4-Lane Subgroup QK

当前 fused kernel 使用：

BLOCK_THREADS = 128
BLOCK_N = 32
LANES_PER_TOKEN = 4

满足：

128 threads = 32 tokens * 4 lanes/token

线程映射：

threads 0~3     -> tile token 0
threads 4~7     -> tile token 1
...
threads 124~127 -> tile token 31

每个 token 的 QK 点积由 4 个线程合作：

lane 0: d = 0, 4, 8, ...
lane 1: d = 1, 5, 9, ...
lane 2: d = 2, 6, 10, ...
lane 3: d = 3, 7, 11, ...

随后仅在 4-lane subgroup 内做 shuffle reduction。

因此一个 tile 中的 32 个 QK score 可以并行生成，而不是逐 token 执行 32 次 full-block reduction。

PV 阶段仍采用：

threadIdx.x = d

即：

thread 0   -> Out[..., 0]
thread 1   -> Out[..., 1]
...
thread 127 -> Out[..., 127]
7. Correctness

正确性测试配置：

B=2
S=17
QH=4
KVH=2
D=32

测试结果：

QK GEMV scores vs CPU passed!
max_abs_error = 3.725290e-08

Softmax probs vs CPU passed!
max_abs_error = 1.117587e-08

PV GEMV output vs CPU passed!
max_abs_error = 3.725290e-08

Fused online softmax output vs CPU passed!
max_abs_error = 4.470348e-08

三段式 baseline 与 fused v1 均和 CPU reference 对齐，误差保持在 1e-7 以下。

8. Benchmark Environment
GPU: NVIDIA A40
Precision: FP32

B   = 1
QH  = 32
KVH = 8
D   = 128
S   = 1 / 128 / 512 / 2048 / 8192

计时不包含：

cudaMalloc
Host-to-Device copy
Device-to-Host copy
输入生成
9. Benchmark Results
9.1 Three-Stage Baseline vs Fused v1
S	QK GEMV	Softmax	PV GEMV	Three-stage	Fused v1	Three-stage / Fused
1	7.949 us	2.910 us	2.935 us	12.630 us	5.259 us	2.401x
128	22.278 us	2.419 us	7.161 us	31.863 us	16.185 us	1.969x
512	22.357 us	2.738 us	21.065 us	46.202 us	58.255 us	0.793x
2048	82.629 us	4.481 us	154.776 us	243.067 us	388.403 us	0.626x
8192	282.332 us	11.494 us	606.945 us	909.896 us	1539.809 us	0.591x

Three-stage / Fused > 1 表示 fused 更快。

9.2 Fused v0 到 Fused v1
S	Fused v0	Fused v1	v0 / v1
1	12.950 us	5.259 us	2.46x
128	65.060 us	16.185 us	4.02x
512	253.877 us	58.255 us	4.36x
2048	1447.347 us	388.403 us	3.73x
8192	5775.191 us	1539.809 us	3.75x

4-lane subgroup QK 显著减少了 fused kernel 中 QK score 计算的同步与规约开销。

10. Performance Analysis
短序列

S=1 和 S=128 时，fused v1 分别达到：

2.401x
1.969x

优势来自：

三次 kernel launch -> 一次 kernel launch
不写 scores workspace
不读 scores workspace
不写 probs workspace
不读 probs workspace
Q 向量在 block 内复用
长序列

S=512 之后，fused v1 开始落后于三段式 baseline。

根因不再是逐 token full-block reduction，而是：

fused grid.x = B * QH = 32 blocks

每个 block 需要串行扫描完整 KV Cache。

当：

S = 8192
BLOCK_N = 32

每个 block 需要处理：

8192 / 32 = 256 个 tile

而三段式 QK kernel 的 block 数会随 S 增长，能提供更高的 GPU 并行度。

因此，fused v1 的结论是：

短序列：融合明显有效
长序列：需要进一步提升 CTA 数量
11. Next Step

下一步实现 Split-KV Decode Attention：

Split-KV partial kernel
-> 生成 partial_max / partial_sum / partial_acc
-> merge kernel 合并各 split 的 online-softmax 状态
-> 输出最终 Out

对于：

B=1, QH=32, S=8192, split_size=512

可将 block 数从：

B * QH = 32

提升到：

B * QH * ceil(S / split_size)
= 1 * 32 * 16
= 512

目标是解决长 KV Cache 下 fused kernel 的并行度不足问题。

12. Build and Run

编译：

nvcc -O3 -std=c++17 -arch=sm_86 \
  decode_main.cu decode_attention.cu \
  -o decode_attention_bench

正确性测试：

./decode_attention_bench

Benchmark：

./decode_attention_bench --bench

查看寄存器和 shared memory 使用：

nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas -v \
  decode_main.cu decode_attention.cu \
  -o decode_attention_bench