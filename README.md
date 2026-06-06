# CUDA Causal Attention Kernel Lab

本项目从零实现并测试了多个 CUDA Causal Attention Forward 算子，用于理解 Attention 从 naive 实现、shared-memory tiled 优化、fused attention，再到 FlashAttention-style online softmax 的演进过程。

当前版本在 `flash-attention-v1` 的基础上，进一步加入了：

- Causal tile skipping
- BM4 / BM8 / BM16 多种 query tile size 测试
- 基于序列长度的简单 dispatch
- FlashAttention v1 与 tile skipping 版本的性能对比

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

Causal Attention Forward 计算公式为：

scores = Q @ K^T / sqrt(D)
probs  = causal_softmax(scores)
O      = probs @ V

代码中的一维访问方式为：

Q[bh, q, d] = Q[bh * S * D + q * D + d]
K[bh, k, d] = K[bh * S * D + k * D + d]
V[bh, k, d] = V[bh * S * D + k * D + d]
O[bh, q, d] = O[bh * S * D + q * D + d]
2. 已实现版本
版本	说明
Naive Unfused Attention	QK、softmax、PV 三阶段分开实现
Shared-memory Tiled Attention	对 QK 和 PV 使用 shared memory tiling
Fused Row Attention	一个 block 处理一个 query row，不落地 global scores/probs
FlashAttention v1	一个 block 处理多个 query，使用 online softmax
FlashAttention + Tile Skipping	在 v1 基础上跳过 causal attention 中无效的未来 K/V tile
BM4 / BM8 / BM16 Dispatch	根据序列长度选择不同 BLOCK_M 配置
3. Naive Unfused Attention

Naive Attention 被拆成三个 kernel：

1. scores = Q @ K^T
2. probs  = scaled_causal_softmax(scores)
3. O      = probs @ V

该版本结构简单，但会显式保存：

scores: [BH, S, S]
probs:  [BH, S, S]

随着序列长度 S 增大，中间矩阵的显存占用和访存开销会快速上升。

4. Shared-memory Tiled Attention

Tiled 版本仍然保持三阶段流程：

tiled QK -> causal softmax -> tiled PV

其中 QK 和 PV 使用 shared memory tile 来减少 global memory 重复读取。

QK 计算：

scores[bh, q, k] = sum_d Q[bh, q, d] * K[bh, k, d]

PV 计算：

O[bh, q, d] = sum_k probs[bh, q, k] * V[bh, k, d]

需要注意：

QK 沿 D 维归约
PV 沿 S 维归约

当前实验中，tiled unfused 版本仍然是性能最好的 baseline。

5. Fused Row Attention

Fused row 版本中，一个 CUDA block 负责一个 attention row：

one block -> one (bh, q)

在一个 kernel 内完成：

1. score[k] = Q[q] · K[k] / sqrt(D)
2. 对 k <= q 做 causal softmax
3. O[q, d] = sum_k prob[k] * V[k, d]

该版本不再保存 global scores/probs，但由于一个 block 只处理一个 query row，K/V 跨 query 复用不足，因此大序列下不一定比 tiled unfused 更快。

6. FlashAttention v1

FlashAttention v1 使用 tile-based attention 和 online softmax。

当前基础配置：

BLOCK_M = 4
BLOCK_N = 32
MAX_D   = 128

一个 block 处理：

Q tile:     [BLOCK_M, D]
K tile:     [BLOCK_N, D]
V tile:     [BLOCK_N, D]
score tile: [BLOCK_M, BLOCK_N]
acc tile:   [BLOCK_M, D]

它不保存完整的：

scores: [BH, S, S]
probs:  [BH, S, S]

而是分块遍历 K/V tile，并对每个 query 维护：

m   = running max score
l   = running softmax denominator
acc = running output accumulator

每处理一个 K/V tile，执行：

score_tile = Q_tile @ K_tile^T / sqrt(D)

m_new = max(m_old, max(score_tile))

l_new =
    exp(m_old - m_new) * l_old
    + sum(exp(score_tile - m_new))

acc_new =
    exp(m_old - m_new) * acc_old
    + exp(score_tile - m_new) @ V_tile

最后：

O = acc / l

这就是 FlashAttention 的核心思想：只保存局部 tile，不保存完整 attention matrix，同时保持 softmax 数值稳定。

7. Causal Tile Skipping

在 causal attention 中，第 q 个 query 只能看：

k <= q

对于一个 query block：

q_start, q_start + 1, ..., q_start + BLOCK_M - 1

它能看到的最大 key 位置是：

q_end = min(q_start + BLOCK_M - 1, S - 1)

如果当前 K/V tile 的起点满足：

k_start > q_end

说明这个 K/V tile 以及后续 K/V tile 全部都是未来 token，可以直接跳过。

因此 tile skipping 版本在 K/V tile 主循环中加入：

if (k_start > q_end) {
    break;
}

这个优化对长序列更有效，因为长序列中可跳过的未来 tile 更多。

8. BM4 / BM8 / BM16 Dispatch

为了测试不同 query tile size 的影响，本项目实现了三个 launcher：

launch_flash_attention_skip_bm4
launch_flash_attention_skip_bm8
launch_flash_attention_skip_bm16

对应：

BM4:  BLOCK_M = 4,  BLOCK_N = 32
BM8:  BLOCK_M = 8,  BLOCK_N = 32
BM16: BLOCK_M = 16, BLOCK_N = 32

其中 BM16 使用：

MAX_D = 64

原因是 BLOCK_M=16, MAX_D=128 会导致 shared memory 超过默认单 block 48KB 限制。

最终 dispatch 规则：

if (S >= 512 && D <= 64) {
    use BM16;
} else if (S >= 256) {
    use BM8;
} else {
    use BM4;
}

实验表明：

短序列下 BM4 更稳
S=256 时 BM8 略优
S=512 时 BM16 最优
9. 编译运行

A40：

rm -f attention_bench
nvcc -O3 -std=c++17 -arch=sm_86 main.cu attention_kernel.cu -o attention_bench
./attention_bench

RTX 4090 / Ada GPU：

rm -f attention_bench
nvcc -O3 -std=c++17 -arch=sm_89 main.cu attention_kernel.cu -o attention_bench
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
=== Causal Attention Forward: Naive vs Tiled vs Fused Row vs FlashAttention v1 vs Tile Skipping ===
B     H     BH      S       D       total_naive     total_tiled     total_fused     total_flash     total_skip      bm4_ms          bm8_ms          bm16_ms
1     1     1       64      64      0.0196          0.0103          0.0076          0.0186          0.0186          0.0186          0.0271          0.0426
1     1     1       128     64      0.0224          0.0125          0.0124          0.0346          0.0346          0.0346          0.0509          0.0825
1     8     8       128     64      0.0915          0.0279          0.0511          0.0790          0.0659          0.0658          0.0679          0.0831
1     8     8       256     64      0.2993          0.0726          0.1594          0.2607          0.1979          0.1998          0.1975          0.2216
1     8     8       512     64      1.1381          0.2479          0.5671          0.9301          0.5821          0.6553          0.5958          0.5837

所有版本误差均保持在 1e-7 量级。

12. 结果分析
12.1 Tiled unfused 仍然最快

在 B=1,H=8,S=512,D=64 下：

total_naive = 1.1381 ms
total_tiled = 0.2479 ms

tiled unfused 通过 shared memory tile 复用 Q/K/V，在当前实现中仍然是最快版本。

12.2 FlashAttention v1 正确但不够快

在 B=1,H=8,S=512,D=64 下：

total_flash = 0.9301 ms

它比 naive 快，但明显慢于 tiled unfused。主要原因是当前版本是教学实现：

1. QK 使用普通 FP32 for-loop，没有 Tensor Core / MMA。
2. online softmax 的 max/sum 更新比较串行。
3. acc_smem 放在 shared memory 中，没有寄存器化。
4. 没有 float4 向量化加载。
5. 没有 warp-level 高效 softmax 优化。
12.3 Tile skipping 有明显收益

在 B=1,H=8,S=512,D=64 下：

total_flash = 0.9301 ms
total_skip  = 0.5821 ms

相对原始 FlashAttention v1：

speedup = 0.9301 / 0.5821 ≈ 1.60x

说明 causal tile skipping 对长序列有效。

12.4 BM4 / BM8 / BM16 的适用范围不同

实验结果显示：

S=64 / S=128：BM4 更快
S=256：BM8 略快
S=512：BM16 更快

因此使用基于 S 的简单 dispatch 更合理。

13. 当前结论

本项目展示了 causal attention kernel 的演进路径：

naive attention
-> shared-memory tiled attention
-> fused row attention
-> FlashAttention-style online softmax
-> causal tile skipping + tile-size dispatch

当前版本已经验证：

1. FlashAttention online softmax 正确性
2. causal tile skipping 的有效性
3. 不同 BLOCK_M 对不同序列长度的性能影响
4. 基于 shape 的 dispatch 是有必要的
14. 后续优化方向

后续可以继续优化：

1. 将 acc_smem 寄存器化，减少 shared memory 读写。
2. 去掉 score_smem，边算 score 边更新 tile max / tile sum。
3. 使用 float4 向量化加载 Q/K/V。
4. 使用 warp-level reduction 优化 online softmax。
5. 使用 Tensor Core / WMMA 加速 QK 和 PV。
6. 增加 D=128、S=1024/2048 的测试。
7. 与 PyTorch / cuDNN / 官方 FlashAttention 实现对比。
15. 面试讲法

可以这样讲：

我实现了一个 CUDA causal attention benchmark，先从 naive 三阶段 attention 开始，然后实现 shared-memory tiled QK 和 tiled PV。之后实现 fused row attention，在一个 kernel 中完成 QK、causal softmax 和 PV，不再落地 global scores/probs。

在此基础上，我实现了 FlashAttention-style v1。一个 block 处理多个 query，并按 K/V tile 分块遍历；每个 query 维护 running max、running softmax sum 和 output accumulator，通过 online softmax 避免保存完整 attention matrix。

随后我进一步加入 causal tile skipping。对于 causal attention，如果当前 K/V tile 全部位于 query block 的未来位置，就直接跳过。实验中，在 B=1,H=8,S=512,D=64 下，FlashAttention v1 从 0.9301 ms 优化到 0.5821 ms，提升约 1.60 倍，误差仍保持在 1e-7 量级。

我还测试了 BM4、BM8、BM16 三种 query tile size，发现短序列下 BM4 更稳，S=256 时 BM8 略优，S=512 时 BM16 最优，因此实现了基于序列长度的 dispatch。

