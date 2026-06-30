# CUDA GQA Decode Attention Kernel

基于 C++ / CUDA 实现 GQA（Grouped Query Attention）场景下的 Decode Attention 前向算子，并完成以下两个版本：

1. 三段式 baseline：`QK GEMV -> Softmax -> PV GEMV`
2. Fused Online Softmax：在单个 kernel 内完成 `QK -> online softmax -> PV`

项目重点关注 LLM 单 token Decode 阶段的 KV Cache 访问、GQA head 映射、在线 softmax，以及不同线程映射对性能的影响。

---

## 1. Decode Attention 与 GQA

Decode 阶段一次只处理当前生成 token 的 Query，因此输入输出布局为：

```text
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
D   : head dimension

对于固定的 (b, qh)：

group_size = QH / KVH
kvh = qh / group_size

例如：

QH = 32
KVH = 8
group_size = 4

则每 4 个 Query head 共享一组 Key / Value head：

Q head 0~3   -> KV head 0
Q head 4~7   -> KV head 1
Q head 8~11  -> KV head 2
...
Q head 28~31 -> KV head 7

单个 Query head 的计算为：

score[s] =
dot(Q[b, qh, :], K_cache[b, s, kvh, :]) / sqrt(D)

prob[s] = softmax(score)[s]

Out[b, qh, d] =
sum_s prob[s] * V_cache[b, s, kvh, d]
2. 三段式 GQA Decode Attention Baseline
2.1 QK GEMV
scores[b, qh, s] =
dot(Q[b, qh, :], K_cache[b, s, kvh, :]) / sqrt(D)

线程映射：

一个线程负责一个 scores[b, qh, s]

即每个线程对 D 维执行点积。

2.2 Softmax
probs[b, qh, :] =
softmax(scores[b, qh, :])

线程映射：

一个 CUDA block 负责一个 (b, qh) score row

使用 warp shuffle + shared memory 完成 block-level max / sum reduction。

2.3 PV GEMV
Out[b, qh, d] =
sum_s probs[b, qh, s] * V_cache[b, s, kvh, d]

线程映射：

一个线程负责一个 Out[b, qh, d]

每个线程沿 KV Cache 长度 S 做串行归约。

3. Fused Online Softmax Decode Attention

Fused 版本将：

QK GEMV -> Softmax -> PV GEMV

合并为一次 kernel launch。

目标是避免中间张量：

scores: [B, QH, S]
probs:  [B, QH, S]

的全局内存读写。

当前 fused kernel 的映射：

一个 CUDA block 负责一个 (b, qh)
一个线程对应一个输出维度 d

对 KV Cache 按 tile 遍历：

每个 tile 处理 32 个 token

并维护 online softmax 状态：

m_new = max(m_old, tile_max)

l_new =
    exp(m_old - m_new) * l_old
    + sum(exp(score_tile - m_new))

acc_new[d] =
    exp(m_old - m_new) * acc_old[d]
    + sum(exp(score_tile - m_new) * V_tile[:, d])

最终：

Out[d] = acc[d] / l
4. Numerical Correctness

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

三段式与 fused 版本均与 CPU reference 对齐，误差保持在 1e-8 量级。

5. Benchmark Environment
GPU: NVIDIA A40
Precision: FP32

B   = 1
QH  = 32
KVH = 8
D   = 128

测试序列长度：

S = 1, 128, 512, 2048, 8192

计时不包含：

cudaMalloc
Host-to-Device copy
Device-to-Host copy
输入数据生成
6. Benchmark Results
S	QK GEMV	Softmax	PV GEMV	Three-stage Total	Fused Online Softmax	Three-stage / Fused
1	8.010 us	2.912 us	2.945 us	12.717 us	12.950 us	0.982x
128	22.280 us	2.437 us	7.166 us	31.867 us	65.060 us	0.490x
512	22.359 us	2.755 us	21.072 us	46.212 us	253.877 us	0.182x
2048	82.627 us	4.512 us	154.874 us	243.130 us	1447.347 us	0.168x
8192	282.435 us	11.540 us	607.360 us	909.978 us	5775.191 us	0.158x

三段式 baseline 的 Host Sync 总耗时：

S	Host Sync Total
1	18.378 us
128	37.549 us
512	51.924 us
2048	248.740 us
8192	915.670 us
7. Performance Analysis
7.1 三段式 baseline 的主要瓶颈

长序列下，PV GEMV 成为主要瓶颈。

以 S=8192 为例：

QK GEMV  = 282.435 us
Softmax  = 11.540 us
PV GEMV  = 607.360 us
Total    = 909.978 us

PV GEMV 占总耗时约三分之二。

原因是当前 PV kernel 的总线程数为：

B * QH * D
= 1 * 32 * 128
= 4096 threads

对应：

4096 / 256 = 16 blocks

A40 有 84 个 SM，16 个 block 无法提供足够并行度。

同时，每个线程需要沿 S 扫描完整 KV Cache：

for s in [0, S):
    acc += prob[s] * V[s, d]

因此随着 S 增长，PV 延迟近似线性上升。

7.2 当前 fused 版本为什么更慢

当前 fused online softmax 已经消除了：

scores workspace
probs workspace
三段式之间的中间全局内存读写
两次额外 kernel launch

但性能仍明显落后于三段式 baseline。

原因是当前 fused QK 阶段采用：

一个 token 的 QK 点积
-> 使用整个 block 对 D 维做 reduction

一个 tile 有 32 个 token，因此每个 tile 内会发生：

32 次 full-block reduction

这使 tile 内 token score 的计算接近串行化，reduction 与同步成本远大于融合带来的访存收益。

此外，fused kernel 的 grid 为：

B * QH
= 1 * 32
= 32 blocks

仍不足以填满 A40 的 84 个 SM。

因此当前 fused 版本的定位是：

数值正确的 online softmax 融合原型

而不是最终高性能实现。

8. Next Steps

后续优化路线：

Three-stage baseline
-> Fused online softmax prototype
-> 4-lane subgroup QK reduction
-> 32-token parallel QK tile
-> Split-KV Decode Attention

重点优化方向：

4 个线程协作计算一个 token 的 QK score。
128 个线程同时计算 32 个 token 的 score，而不是逐 token 做 full-block reduction。
对长 KV Cache 使用 Split-KV，提高 Decode 阶段 block 数量。
为不同 S、D、QH/KVH 配置设计 dispatch 策略。
使用 Nsight Compute 分析 global load efficiency、occupancy、warp stall 与 memory throughput。
9. Build and Run

编译：

nvcc -O3 -std=c++17 -arch=sm_86 \
  decode_main.cu decode_attention.cu \
  -o decode_attention_bench

运行正确性测试：

./decode_attention_bench

运行 benchmark：

./decode_attention_bench --bench

查看寄存器和 shared memory 使用：

nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas -v \
  decode_main.cu decode_attention.cu \
  -o decode_attention_bench
10. Project Structure
attention_kernel/
├── attention_kernel.cu
├── main.cu
├── decode_attention.cu
├── decode_main.cu
└── README.md

其中：

attention_kernel.cu / main.cu
    Prefill Causal Attention / FlashAttention 优化实验

decode_attention.cu / decode_main.cu
    GQA Decode Attention 三段式 baseline
    + Fused Online Softmax 原型