# CUDA Causal Attention Kernel Lab

本项目从零实现并测试了多个 CUDA Causal Attention Forward 算子，用来理解 Attention 算子从 naive 实现、shared memory tiled 优化、fused row attention，再到 FlashAttention-style online softmax 的演进过程。

当前已实现：

- Naive QK^T
- Scaled Causal Softmax
- Naive P @ V
- Shared-memory Tiled QK^T
- Shared-memory Tiled P @ V
- Fused Row-wise Causal Attention
- FlashAttention v1，基于 Online Softmax

---

## 1. 张量布局

所有张量均采用 row-major 布局：

```text
Q:      [BH, S, D]
K:      [BH, S, D]
V:      [BH, S, D]
O:      [BH, S, D]

scores: [BH, S, S]
probs:  [BH, S, S]

其中：

BH = B * H
B  = batch size
H  = number of heads
S  = sequence length
D  = head dimension

Causal Attention Forward 的计算公式为：

scores = Q @ K^T / sqrt(D)
probs  = causal_softmax(scores)
O      = probs @ V

代码中的一维地址访问方式为：

Q[bh, q, d] = Q[bh * S * D + q * D + d]
K[bh, k, d] = K[bh * S * D + k * D + d]
V[bh, k, d] = V[bh * S * D + k * D + d]
O[bh, q, d] = O[bh * S * D + q * D + d]
2. 项目结构
.
├── attention_kernel.cu   # CUDA kernels、launcher、CPU reference
├── main.cu               # 正确性测试和 benchmark
├── Makefile              # 编译脚本
└── README.md
3. 已实现算子
算子	说明
qk_matmul_kernel	Naive QK^T，一个线程计算一个 score 元素
qk_matmul_tiled_kernel	Shared-memory tiled QK^T
scaled_causal_softmax_warp_kernel	行级 scaled causal softmax
pv_matmul_kernel	Naive P @ V，一个线程计算一个输出元素
pv_matmul_tiled_kernel	Shared-memory tiled P @ V
fused_causal_attention_row_kernel	一个 block 计算一个 query row，不落地 global scores/probs
flash_attention_v1_kernel	FlashAttention-style tiled attention，使用 online softmax
4. Naive Unfused Attention

Naive Attention 拆成三个阶段：

1. scores = Q @ K^T
2. probs  = scaled_causal_softmax(scores)
3. O      = probs @ V

这个版本结构清晰，便于验证正确性，但性能较差。

它会显式保存两个中间矩阵：

scores: [BH, S, S]
probs:  [BH, S, S]

当序列长度 S 增大时，中间矩阵的显存占用和读写开销会快速增加。

5. Shared-memory Tiled Unfused Attention

Tiled 版本仍然保持三阶段流程：

tiled QK -> causal softmax -> tiled PV

但是 QK 和 PV 使用 shared memory tile 来减少 global memory 重复读取。

5.1 Tiled QK

QK 计算：

scores[bh, q, k] = sum_d Q[bh, q, d] * K[bh, k, d]

当前 tile 配置：

BLOCK_Q = 16
BLOCK_K = 16
BLOCK_D = 16

一个 CUDA block 计算一个 score tile：

scores tile: [BLOCK_Q, BLOCK_K]

shared memory 中保存：

Q tile: [BLOCK_Q, BLOCK_D]
K tile: [BLOCK_D, BLOCK_K]

注意：global memory 中的 K 仍然是 [BH, S, D] 布局，并没有真正预转置。这里只是在 shared memory 中临时将 K tile 组织成 [D, K]，方便计算：

Q tile [Q, D] × K tile [D, K]
5.2 Tiled PV

PV 计算：

O[bh, q, d] = sum_k probs[bh, q, k] * V[bh, k, d]

当前 tile 配置：

BLOCK_P = 16
BLOCK_V = 16
BLOCK_S = 16

shared memory 中保存：

probs tile: [BLOCK_P, BLOCK_S]
V tile:     [BLOCK_S, BLOCK_V]

这里最容易混淆的是：

QK 沿 D 维归约
PV 沿 S 维归约
6. Fused Row-wise Causal Attention

Fused row 版本中，一个 CUDA block 负责一个 attention row：

one block -> one (bh, q)

也就是一个 block 处理一个 query。

在一个 kernel 内部完成：

1. score[k] = Q[q] · K[k] / sqrt(D)
2. 对 k <= q 做 causal softmax
3. O[q, d] = sum_k prob[k] * V[k, d]

这个版本不再显式保存 global memory 中的：

scores: [BH, S, S]
probs:  [BH, S, S]

但是它的缺点是跨 query 的 K/V 复用较差。每个 query row 都会重新读取 K/V，因此在大 S、多 head 场景下，不一定比 tiled unfused 版本更快。

7. FlashAttention v1

FlashAttention v1 是一个教学版 FlashAttention-style 实现，重点是验证：

tiled QK + online softmax + tiled PV accumulation

它不是工业级高性能版本，没有使用 Tensor Core / MMA，也没有做寄存器级 accumulator 优化。

当前参数：

BLOCK_M = 4
BLOCK_N = 32
MAX_D   = 128

含义：

BLOCK_M: 一个 block 处理多少个 query
BLOCK_N: 每轮处理多少个 key/value
MAX_D:   kernel 支持的最大 head dimension，用于静态 shared memory 分配

一个 block 内部处理：

Q tile:     [BLOCK_M, D]
K tile:     [BLOCK_N, D]
V tile:     [BLOCK_N, D]
score tile: [BLOCK_M, BLOCK_N]
acc tile:   [BLOCK_M, D]

它不保存完整的：

scores: [BH, S, S]
probs:  [BH, S, S]

而是循环遍历 K/V tile，并使用 online softmax 维护每个 query 的 running max、running sum 和 output accumulator。

8. Online Softmax 原理

普通 softmax 需要先看到完整一行 score：

softmax(score[q, :])

但 FlashAttention 是分块遍历 K/V，因此不能一次性得到完整 score row。

所以对每个 query 维护：

m   = 当前已经看过的 score 最大值
l   = 当前 softmax 分母
acc = 当前输出累加器

每处理一个 K/V tile，先计算：

score_tile = Q_tile @ K_tile^T / sqrt(D)

然后更新：

m_new = max(m_old, max(score_tile))

l_new =
    exp(m_old - m_new) * l_old
    + sum(exp(score_tile - m_new))

acc_new =
    exp(m_old - m_new) * acc_old
    + exp(score_tile - m_new) @ V_tile

所有 K/V tile 处理完后：

O = acc / l

这就是 FlashAttention 的核心思想：只保存局部 tile，不保存完整 attention matrix，同时保持 softmax 数值稳定。

9. 编译运行

A40：

nvcc -O3 -std=c++17 -arch=sm_86 main.cu attention_kernel.cu -o attention_bench
./attention_bench

RTX 4090 / Ada GPU：

nvcc -O3 -std=c++17 -arch=sm_89 main.cu attention_kernel.cu -o attention_bench
./attention_bench

为了避免编译失败后误运行旧二进制，建议：

rm -f attention_bench
nvcc -O3 -std=c++17 -arch=sm_86 main.cu attention_kernel.cu -o attention_bench
./attention_bench
10. Benchmark 环境

测试 GPU：

NVIDIA A40
SM count: 84

测试 shape：

B = 1
H = 1 / 8
S = 64 / 128 / 256 / 512
D = 64
11. Benchmark 结果
=== Causal Attention Forward: Naive vs Tiled vs Fused Row vs FlashAttention v1 ===
B     H     BH      S       D       qk_naive      qk_tiled      qk_spd      softmax       pv_naive      pv_tiled      pv_spd      total_naive     total_tiled     total_fused     total_flash     tiled_spd     fused/naive   flash/naive   flash/tiled   flash/fused   score_MB    qkv_MB      naive_err     tiled_err     fused_err     flash_err
1     1     1       64      64      0.0122        0.0038        3.240       0.0031        0.0043        0.0035        1.227       0.0195          0.0103          0.0075          0.0185          1.884         2.579         1.050         0.557         0.407         0.02        0.02        1.49e-07      1.49e-07      1.49e-07      1.49e-07
1     1     1       128     64      0.0124        0.0039        3.210       0.0033        0.0066        0.0054        1.212       0.0222          0.0125          0.0123          0.0344          1.777         1.796         0.646         0.363         0.359         0.06        0.03        1.49e-07      1.49e-07      1.49e-07      1.79e-07
1     8     8       128     64      0.0717        0.0104        6.927       0.0069        0.0122        0.0106        1.148       0.0909          0.0278          0.0505          0.0786          3.268         1.800         1.156         0.354         0.642         0.50        0.25        1.79e-07      1.79e-07      1.79e-07      1.86e-07
1     8     8       256     64      0.2510        0.0312        8.053       0.0112        0.0369        0.0302        1.223       0.2994          0.0727          0.1600          0.2606          4.120         1.871         1.149         0.279         0.614         2.00        0.50        1.79e-07      1.79e-07      1.79e-07      1.86e-07
1     8     8       512     64      0.9755        0.1150        8.480       0.0283        0.1353        0.1051        1.288       1.1379          0.2476          0.5641          0.9309          4.596         2.017         1.222         0.266         0.606         8.00        1.00        1.79e-07      1.79e-07      1.79e-07      1.94e-07
12. 结果分析
12.1 当前最快的是 tiled unfused attention

在 B=1,H=8,S=512,D=64 下：

total_naive = 1.1379 ms
total_tiled = 0.2476 ms
speedup     = 4.596x

主要收益来自 tiled QK：

qk_naive = 0.9755 ms
qk_tiled = 0.1150 ms
speedup  = 8.480x

说明 shared memory tile 对 Q/K 数据复用非常有效。

12.2 fused row attention 正确，但大 S 下慢于 tiled

在 B=1,H=8,S=512,D=64 下：

total_fused = 0.5641 ms
fused_err   = 1.79e-07

fused row 版本不保存 global scores/probs，正确性通过。

但由于一个 block 只处理一个 query row，K/V 跨 query 复用不足，因此在大 S 下比 tiled unfused 慢。

12.3 FlashAttention v1 正确实现了 online softmax

在 B=1,H=8,S=512,D=64 下：

total_flash = 0.9309 ms
flash_err   = 1.94e-07

flash_err 在 1e-7 量级，说明 online softmax 更新逻辑是正确的。

但是该版本是教学版，性能还没有超过 tiled unfused。主要原因：

1. QK 使用普通 FP32 for-loop，没有使用 Tensor Core / MMA。
2. online softmax 的 max/sum 更新部分比较串行。
3. acc_smem 放在 shared memory 中，没有寄存器化。
4. BLOCK_M / BLOCK_N 比较保守。
5. 没有使用 float4 向量化加载。
6. 没有做 warp-level 高效 softmax 优化。
13. 版本对比
版本	是否保存 scores/probs	是否使用 shared memory	是否使用 online softmax	当前性能
Naive unfused	是	否	否	最慢
Tiled unfused	是	是	否	当前最快
Fused row	否	是	否	正确，中等速度
FlashAttention v1	否	是	是	正确，教学版
14. 项目展示了什么

本项目展示了 causal attention kernel 的完整演进路线：

naive attention
-> shared-memory tiled attention
-> fused row attention
-> FlashAttention-style online softmax

当前 FlashAttention v1 已经验证了 FlashAttention 的核心算法：

tile-based QK
online softmax
tile-based PV accumulation
no global scores/probs materialization

虽然还不是高性能工业实现，但已经完整体现了 FlashAttention 的关键思想。

15. 后续优化方向

后续可以继续做：

1. Causal tile skipping
   如果当前 K/V tile 全部在未来位置，直接跳过。

2. 调整 tile 参数
   测试 BLOCK_M=8, BLOCK_N=32
   测试 BLOCK_M=8, BLOCK_N=64

3. 减少 shared memory 使用
   尝试去掉 score_smem，边算边更新局部统计量。

4. accumulator 寄存器化
   减少 acc_smem 的 shared memory 读写。

5. float4 向量化加载
   当 D % 4 == 0 时使用 float4 load/store。

6. Tensor Core / WMMA 版本
   用 MMA 加速 QK 和 P@V tile 计算。

7. 和 PyTorch / cuDNN / FlashAttention 官方库做对比。
