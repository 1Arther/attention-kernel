# CUDA Causal Attention Kernel Optimization

本项目从零实现并逐步优化 Causal Attention Forward，用于学习和验证 Transformer Attention 中常见 CUDA kernel 优化方法。

Attention 计算形式为：

```text
scores = Q @ K^T / sqrt(D)
probs  = causal_softmax(scores)
O      = probs @ V

其中：

B：batch size
H：attention head 数
BH = B * H
S：sequence length
D：head dimension
Q/K/V/O shape 均为 [BH, S, D]
scores/probs shape 为 [BH, S, S]

当前主要针对 float32、causal attention、D=64 场景进行优化和 benchmark。

1. 实现版本

项目中包含以下几个版本：

1.1 Naive Unfused Attention

最基础的三阶段实现：

QK MatMul -> Causal Softmax -> PV MatMul

特点：

每个阶段单独 kernel
中间显式写出 scores 和 probs
方便验证正确性和作为 baseline

缺点：

需要读写完整 scores/probs
kernel launch 次数多
global memory traffic 较大
1.2 Tiled QK / PV

对 QK 和 PV 分别实现 shared memory tiled matmul。

优化点：

Q/K/V tile 加载到 shared memory
减少 global memory 重复访问
QK 和 PV 相比 naive 均有明显加速
1.3 Fused Row Attention

实现一行 query 对应一个 CUDA block 的 fused attention：

一个 block 负责一个 query row
QK + softmax + PV 在一个 kernel 内完成

优化点：

不再显式写出完整 scores/probs
减少 kernel launch
对小序列场景有一定优势

局限：

每个 block 只处理一个 query row
并行度和数据复用有限
对较大 BH/S 场景不一定优于 tiled unfused
1.4 FlashAttention v1

实现 FlashAttention v1 风格的 online softmax。

核心思想：

一个 block 处理 BLOCK_M 个 query
每轮处理 BLOCK_N 个 key/value
使用 online softmax 维护每个 query row 的 running max m 和 running sum l
避免显式保存完整 S x S attention matrix

online softmax 更新公式：

m_new = max(m_old, tile_max)
l_new = exp(m_old - m_new) * l_old + sum(exp(score - m_new))
acc_new = exp(m_old - m_new) * acc_old + sum(exp(score - m_new) * V)
1.5 Causal Tile Skipping

在 causal attention 中，当当前 K/V tile 完全位于 query block 右侧时，该 tile 对所有 query 都不可见，可以直接跳过。

优化点：

if (k_start > q_end) {
    break;
}

作用：

减少无效 K/V tile 计算
对长序列 causal attention 有明显收益
1.6 Register Accumulator

原始版本使用 shared memory 保存 accumulator：

acc_smem[BLOCK_M][MAX_D]

register accumulator 版本将每个线程负责的 accumulator 保存在寄存器中，减少 shared memory 读写。

实验结论：

register accumulator 并不是所有 shape 都更快
短序列下可能因为寄存器压力和额外控制逻辑变慢
长序列下，尤其是 S=512,D=64，BM16 regacc 有明显收益
1.7 Float4 Vectorized Load

在 regacc 版本基础上，进一步使用 float4 向量化加载 Q/K/V：

float4 q_vec = *reinterpret_cast<const float4*>(q_gmem);
float4 k_vec = *reinterpret_cast<const float4*>(k_gmem);
float4 v_vec = *reinterpret_cast<const float4*>(v_gmem);

条件：

D % 4 == 0

当前 benchmark 主要使用 D=64，因此满足 float4 对齐和连续访问条件。

优化点：

Q/K/V 从 global memory 到 shared memory 的加载更高效
不改变 attention 计算逻辑
正确性保持在 1e-7 量级
2. Noscore 实验

在 regacc + float4 基础上，尝试去掉：

score_smem[BLOCK_M][BLOCK_N]

即不再保存 score tile，而是通过重复计算 QK 来完成：

1. 第一次计算 score，用于求 tile_max
2. 第二次计算 score，用于求 tile_sum
3. 第三次计算 score，用于更新 acc

该版本命名为 noscore v0。

实验结论

noscore v0 正确性通过，但性能严重退化。

以 B=1,H=8,S=512,D=64 为例：

BM16 vec4      = 0.4627 ms
BM16 noscore   = 2.8765 ms
noscore / vec4 = 0.161x

也就是说 noscore v0 比 BM16 vec4 慢约：

2.8765 / 0.4627 ≈ 6.2x

原因：

虽然减少了 score_smem 读写
但是 QK dot 被重复计算过多
对 D=64 场景，重复计算成本远大于 shared memory 读写成本

因此，noscore v0 仅作为失败实验保留，不进入默认 dispatch。

3. Probability Cache Optimization

noscore 实验说明，直接去掉 score_smem 不划算。进一步分析发现，原始 vec4 版本中存在大量重复 expf：

for each (qi, d):
    for each kj:
        p = expf(score_smem[qi][kj] - m_new)
        acc += p * V[kj][d]

同一个 p(qi,kj) 会被不同输出维度 d 重复计算多次。

因此实现 probability cache 优化：

1. score_smem 先保存 score
2. 求出 m_new 后，将 score_smem 原地覆盖为 p = exp(score - m_new)
3. acc update 阶段直接读取 p，不再重复 expf

即：

score_smem[qi][kj] = p;

此时 score_smem 变成了临时的 probability cache。

4. Benchmark Results

测试环境：

GPU: NVIDIA A40
SM count: 84
CUDA arch: sm_86
Data type: float32

编译命令：

nvcc -O3 -std=c++17 -arch=sm_86 main.cu attention_kernel.cu -o attention_bench

运行命令：

./attention_bench
4.1 关键结果：B=1,H=8,S=512,D=64
Version	Time
Naive unfused	1.1378 ms
Tiled unfused	0.2477 ms
FlashAttention v1	0.9306 ms
Regacc + float4 dispatch	0.4626 ms
BM16 noscore v0	2.8765 ms
BM16 pcache	0.3344 ms

相较 FlashAttention v1：

FlashAttention v1 = 0.9306 ms
BM16 pcache       = 0.3344 ms
speedup           = 2.78x

相较 BM16 vec4：

BM16 vec4   = 0.4627 ms
BM16 pcache = 0.3344 ms
speedup     = 1.38x
4.2 关键结果：B=1,H=8,S=256,D=64
Version	Time
FlashAttention v1	0.2606 ms
BM8 vec4 dispatch	0.1550 ms
BM16 pcache	0.1316 ms

相较当前 vec4 dispatch：

BM8 vec4 dispatch = 0.1550 ms
BM16 pcache       = 0.1316 ms
speedup           = 1.18x
5. Final Dispatch Strategy

当前最终默认 dispatch 策略：

if (D <= 64 && S >= 256) {
    use BM16 regacc vec4 pcache;
} else {
    use regacc vec4 dispatch;
}

其中 vec4 dispatch 内部为：

if (S >= 512 && D <= 64) {
    use BM16 regacc vec4;
} else if (S >= 256) {
    use BM8 regacc vec4;
} else {
    use BM4 regacc vec4;
}

这样可以避免小序列下 BM16 pcache 控制逻辑过重，同时保留中长序列下 pcache 的收益。

6. Correctness

所有 CUDA kernel 都与 CPU reference 对齐，最大误差保持在：

1e-7 ~ 1e-6

典型结果：

flash_err   ≈ 1.9e-7
v4_err      ≈ 1.9e-7
noscore_err ≈ 1.9e-7
pcache_err  ≈ 1.9e-7
7. Lessons Learned
7.1 Noscore 不一定更快

去掉 shared memory 并不一定会提升性能。对于当前实现，score_smem 的读写成本远小于重复 QK dot 的计算成本。

7.2 减少 expf 比减少 score_smem 更有效

pcache 版本保留 score_smem，但将其复用为 probability cache，避免在不同输出维度上重复计算 expf，收益明显。

7.3 Dispatch 需要按 shape 选择

不同 S/BH/D 下最优 kernel 不同。小序列更适合轻量 BM4/BM8 vec4；中长序列更适合 BM16 pcache。

8. File Structure
attention_kernel.cu   CUDA kernels and launchers
main.cu               Benchmark and correctness test
README.md             Project documentation
9. Build and Run
rm -f attention_bench

nvcc -O3 -std=c++17 -arch=sm_86 main.cu attention_kernel.cu -o attention_bench

./attention_bench

For RTX 4090, use:

nvcc -O3 -std=c++17 -arch=sm_89 main.cu attention_kernel.cu -o attention_bench
10. Current Best Result

当前最佳版本为：

BM16 regacc + float4 load + probability cache

在 B=1,H=8,S=512,D=64 下：

FlashAttention v1 = 0.9306 ms
Final pcache      = 0.3344 ms
Speedup           = 2.78x

