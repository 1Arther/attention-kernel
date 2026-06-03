# CUDA Attention Kernel Lab

本项目从零实现并测试了 **CUDA Causal Attention Forward** 的多个版本，重点覆盖：

- Naive QK^T
- Scaled Causal Softmax
- Naive P @ V
- Shared-memory tiled QK^T
- Shared-memory tiled P @ V
- Naive vs Tiled stage-wise benchmark

当前实现是 **unfused causal attention baseline**，不是 FlashAttention。它仍然显式保存 `scores` 和 `probs` 两个中间矩阵，用于验证 Attention forward 的正确性，并作为后续 fused attention / online softmax / FlashAttention-style 实现的基础。

---

## 1. Tensor Layout

当前所有张量都使用 row-major layout：

```text
Q:      [BH, S, D]
K:      [BH, S, D]
V:      [BH, S, D]

scores: [BH, S, S]
probs:  [BH, S, S]
O:      [BH, S, D]

其中：

BH = B * H
B  = batch size
H  = number of heads
S  = sequence length
D  = head dimension

Attention forward 计算公式：

scores = Q @ K^T
probs  = causal_softmax(scores / sqrt(D))
O      = probs @ V
2. 项目文件
.
├── attention_kernel.cu   # CUDA kernels, launchers, CPU reference
├── main.cu               # correctness check and benchmark
├── Makefile              # optional build script
└── README.md
3. 已实现 Kernel
Kernel	说明
qk_matmul_kernel	Naive QK^T，一个线程计算一个 scores[bh, q, k]
qk_matmul_tiled_kernel	Shared-memory tiled QK^T，复用 Q/K tile
scaled_causal_softmax_warp_kernel	Row-wise scaled causal softmax
pv_matmul_kernel	Naive P @ V，一个线程计算一个 O[bh, q, d]
pv_matmul_tiled_kernel	Shared-memory tiled P @ V，复用 probs/V tile
launch_attention_forward	Naive unfused attention 总流程
launch_attention_forward_tiled	Tiled unfused attention 总流程
4. Naive Causal Attention

Naive 总流程分为三步：

1. scores = Q @ K^T
2. probs = scaled_causal_softmax(scores)
3. O = probs @ V
4.1 Naive QK^T

qk_matmul_kernel 中，一个线程负责一个 score：

scores[bh, q, k] = sum_d Q[bh, q, d] * K[bh, k, d]

这种实现结构清晰，但没有数据复用。对于同一个 query 向量 Q[q, :]，它会被不同 key 重复读取；对于同一个 key 向量 K[k, :]，也会被不同 query 重复读取。

4.2 Scaled Causal Softmax

Causal softmax 用于 decoder self-attention，保证当前位置不能看未来 token：

valid iff key_index <= query_index

实现中将 scores 视为：

scores: [BH * S, S]

对于第 row 行：

int q = row % query_len;
bool valid = i <= q;

softmax 计算时会跳过 future token，并将 future token 的输出置为 0。

4.3 Naive P @ V

pv_matmul_kernel 中，一个线程负责一个输出元素：

O[bh, q, d] = sum_k probs[bh, q, k] * V[bh, k, d]

这里的归约维度是 S，也就是 key/token 方向。

5. Shared-memory Tiled QK^T

tiled QK 使用一个 block 计算一个 scores tile：

scores tile: [BLOCK_Q, BLOCK_K]

当前设置：

BLOCK_Q = 16
BLOCK_K = 16
BLOCK_D = 16

每个 block 内部加载：

Q tile: [BLOCK_Q, BLOCK_D]
K tile: [BLOCK_D, BLOCK_K]

注意：global memory 中的 K 仍然是 [BH, S, D]，没有真正预转置。这里只是在 shared memory 中把 K tile 临时转成 [D, K]，使计算形式接近 GEMM：

scores_tile = Q_tile [Q, D] × K_tile^T [D, K]

核心计算：

sum += q_smem[ty][d] * k_smem[d][tx];

这对应：

Q[bh, q_pos, d] * K[bh, k_pos, d]
6. Shared-memory Tiled P @ V

tiled PV 使用一个 block 计算一个 O tile：

O tile: [BLOCK_P, BLOCK_V]

当前设置：

BLOCK_P = 16
BLOCK_V = 16
BLOCK_S = 16

每个 block 内部加载：

probs tile: [BLOCK_P, BLOCK_S]
V tile:     [BLOCK_S, BLOCK_V]

核心计算：

sum += probs_smem[ty][k] * v_smem[k][tx];

对应：

O[bh, q, d] += probs[bh, q, k] * V[bh, k, d]

注意：

QK 沿 D 维归约
PV 沿 S 维归约

这是实现 QK 和 PV 时最容易混淆的地方。

7. 编译与运行

A40 使用：

nvcc -O3 -std=c++17 -arch=sm_86 main.cu attention_kernel.cu -o attention_bench
./attention_bench

RTX 4090 / L40 等 Ada 架构 GPU 可以使用：

nvcc -O3 -std=c++17 -arch=sm_89 main.cu attention_kernel.cu -o attention_bench
./attention_bench

如果使用 Makefile：

make
./attention_bench
8. Benchmark 配置

测试配置：

B = 1
H = 1 / 8
S = 64 / 128 / 256 / 512
D = 64

测试 GPU：

NVIDIA A40
SM count: 84

Benchmark 输出包括：

qk_naive
qk_tiled
qk_spd

softmax

pv_naive
pv_tiled
pv_spd

total_naive
total_tiled
total_spd

naive_err
tiled_err

其中：

qk_spd    = qk_naive / qk_tiled
pv_spd    = pv_naive / pv_tiled
total_spd = total_naive / total_tiled
9. Benchmark Results
=== Unfused Causal Attention Forward: Naive vs Tiled ===
B     H     BH      S       D       qk_naive      qk_tiled      qk_spd      softmax       pv_naive      pv_tiled      pv_spd      total_naive     total_tiled     total_spd     score_MB    qkv_MB      naive_err     tiled_err
1     1     1       64      64      0.0123        0.0038        3.272       0.0031        0.0043        0.0036        1.216       0.0196          0.0103          1.901         0.02        0.02        1.49e-07      1.49e-07
1     1     1       128     64      0.0124        0.0038        3.252       0.0033        0.0065        0.0054        1.210       0.0223          0.0125          1.783         0.06        0.03        1.49e-07      1.49e-07
1     8     8       128     64      0.0722        0.0104        6.949       0.0069        0.0122        0.0106        1.150       0.0915          0.0279          3.280         0.50        0.25        1.79e-07      1.79e-07
1     8     8       256     64      0.2532        0.0314        8.064       0.0113        0.0373        0.0304        1.228       0.2993          0.0726          4.120         2.00        0.50        1.79e-07      1.79e-07
1     8     8       512     64      0.9755        0.1152        8.468       0.0283        0.1355        0.1052        1.288       1.1381          0.2479          4.590         8.00        1.00        1.79e-07      1.79e-07
10. 结果分析
10.1 Tiled QK 是主要优化来源

对于较大序列长度，QK 是 naive attention 的主要瓶颈。

在 B=1, H=8, S=512, D=64 下：

qk_naive = 0.9755 ms
qk_tiled = 0.1152 ms
speedup  = 8.468x

说明 shared memory tile 有效减少了 Q/K 的重复 global memory 读取。

10.2 Tiled PV 也有收益，但没有 QK 明显

在 B=1, H=8, S=512, D=64 下：

pv_naive = 0.1355 ms
pv_tiled = 0.1052 ms
speedup  = 1.288x

PV 的提升较小，因为当前 naive PV 相比 naive QK 原本耗时就更低，并且输出维度是 [BH, S, D]，不像 QK 需要生成 [BH, S, S] 的 score matrix。

10.3 总体 Attention 加速明显

在 B=1, H=8, S=512, D=64 下：

total_naive = 1.1381 ms
total_tiled = 0.2479 ms
speedup     = 4.590x

同时：

naive_err = 1.79e-07
tiled_err = 1.79e-07

说明 tiled 版本在保持数值正确性的前提下显著提升了性能。

11. 当前实现的局限

当前版本仍然是 unfused attention：

QK^T -> causal softmax -> P @ V

它仍然显式保存：

scores: [BH, S, S]
probs:  [BH, S, S]

因此随着 S 增大，中间矩阵的显存开销会快速增长：

scores memory = BH * S * S * sizeof(float)
probs memory  = BH * S * S * sizeof(float)

例如：

B=1, H=8, S=512
scores = 8 MB
probs  = 8 MB

这也是 FlashAttention 要解决的核心问题之一。

12. 面试讲法

可以这样介绍：

我实现了一个 CUDA causal attention forward benchmark，首先实现了 unfused baseline，
将 attention 拆成 QK^T、scaled causal softmax 和 P@V 三个阶段。

在 naive 版本中，一个线程计算一个 score 或一个 output 元素，结构清晰但缺少数据复用。
随后我实现了 shared-memory tiled QK 和 tiled PV。

对于 QK，一个 block 计算 16x16 的 scores tile，并沿 D 维分块加载 Q/K。
其中 K 在 global memory 中仍然是 [BH, S, D]，但在 shared memory 中临时转成 [D, K]，
这样计算形式接近 GEMM：Q_tile [Q,D] × K_tile^T [D,K]。

对于 PV，一个 block 计算 16x16 的 O tile，并沿 S 维分块加载 probs/V。
实验显示，在 B=1,H=8,S=512,D=64 时，QK 从 0.9755 ms 降到 0.1152 ms，
提升 8.47x；整体 attention 从 1.1381 ms 降到 0.2479 ms，提升 4.59x。
同时 max error 保持在 1e-7 量级，说明 tiled 版本和 CPU reference 对齐。

当前版本仍然显式保存 scores 和 probs，下一步可以继续实现 fused attention 和 online softmax，
逐步过渡到 FlashAttention 的思想。
13. 后续方向

后续可以继续实现：

1. Fused row-wise causal attention
   - 不保存 probs
   - 一个 block 处理一个 query row

2. Online softmax attention
   - 分块遍历 K/V
   - 维护 running max 和 running sum

3. Simplified FlashAttention
   - tile Q/K/V
   - 在 tile 内完成 score、online softmax 和 O 累加
   - 避免显式落地完整 scores/probs

