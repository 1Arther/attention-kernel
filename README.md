# CUDA Causal Attention Kernel Optimization

本项目基于 C++ / CUDA 实现并优化 causal attention forward kernel，目标是在保持数值正确性的前提下，逐步优化传统 attention 计算流程，并与 unfused baseline、tiled baseline、fused row kernel 和基础 FlashAttention v1 进行性能对比。

项目主要关注 LLM 推理中常见的 causal self-attention 前向计算：

```text
scores = QK^T / sqrt(D)
probs  = causal_softmax(scores)
O      = probs V
```

数据布局为：

```text
Q, K, V, O : [BH, S, D]
scores    : [BH, S, S]
probs     : [BH, S, S]
```

其中：

```text
BH = B * H
B  = batch size
H  = number of heads
S  = sequence length
D  = head dimension
```

---

## 1. Implemented Kernels

本项目实现了多种 causal attention forward kernel：

### 1.1 Unfused Baseline

传统三阶段 attention 实现：

```text
QK^T -> causal softmax -> PV
```

对应 kernel：

```cpp
launch_qk_matmul
launch_scaled_causal_softmax
launch_pv_matmul
launch_attention_forward
```

该版本结构清晰，但需要读写完整的 `scores` 和 `probs` 矩阵，访存开销较大，并且包含多次 kernel launch。

---

### 1.2 Tiled Baseline

对 `QK^T` 和 `PV` 引入 shared memory tiling：

```cpp
launch_qk_matmul_tiled
launch_pv_matmul_tiled
launch_attention_forward_tiled
```

该版本相比 unfused baseline 明显减少了全局访存，在中小规模 shape 下表现较好。

---

### 1.3 Fused Row Attention

一个 CUDA block 负责一行 query，融合：

```text
QK -> softmax -> PV
```

对应接口：

```cpp
launch_attention_forward_fused_row
```

该版本避免了中间 `scores/probs` 的全局内存写回，在小序列长度下性能较好。

---

### 1.4 FlashAttention v1

实现基于 online softmax 的 FlashAttention v1：

```cpp
launch_flash_attention_v1
```

该版本不再显式保存完整 attention matrix，而是在 tile 级别进行 online softmax 更新，减少中间访存。

---

### 1.5 Optimized Causal FlashAttention

最终优化版本基于 FlashAttention 思路，加入了以下优化：

```text
causal tile skipping
+ register accumulator
+ float4 vectorized global load
+ probability cache
+ shared memory padding
+ __expf fast exponential
+ branchless accumulator update
+ BLOCK_N loop unroll
```

最终 dispatch 接口：

```cpp
launch_flash_attention_causal_tile_skipping
```

---

## 2. Optimization Timeline

### 2.1 Causal Tile Skipping

由于 causal attention 中 query `q` 不能访问未来 key `k > q`，因此右上角 tile 不需要计算。

对于每个 query block：

```cpp
int max_k_for_block = min(S, q_start + BLOCK_M);
```

只遍历满足 causal 条件的 key tile：

```text
k_start < max_k_for_block
```

这样可以跳过无效的 future tile，减少冗余计算。

---

### 2.2 Register Accumulator

原始 FlashAttention v1 中，每轮 tile 更新后会频繁访问 shared memory / global memory。

优化后，将每个线程负责的 output partial result 保存在寄存器中：

```cpp
float acc_reg[ITEMS_PER_THREAD];
```

在每个 K/V tile 上更新：

```cpp
acc = acc * alpha + sum(p * V)
```

最终再写回 global memory。

这样减少了中间结果的访存开销。

---

### 2.3 float4 Vectorized Global Load

对于 `D % 4 == 0` 的情况，使用 `float4` 对 Q/K/V 进行向量化加载：

```cpp
float4 q_vec = *reinterpret_cast<const float4*>(q_gmem);
float4 k_vec = *reinterpret_cast<const float4*>(k_gmem);
float4 v_vec = *reinterpret_cast<const float4*>(v_gmem);
```

这样可以提升 global memory load efficiency。

实验中尝试过将 float4 load 改回 scalar load，以降低 shared store bank conflict，但结果变慢：

```text
float4 + padding final : 约 0.2288 ms
scalar load final     : 约 0.2576 ms
```

因此最终保留 float4 vectorized global load。

---

### 2.4 Probability Cache

在 online softmax 中，每个 tile 会先计算 score：

```cpp
score = dot(Q, K) * scale
```

随后计算：

```cpp
p = exp(score - m_new)
```

为了避免在 accumulator update 阶段重复计算 `exp`，将 `score_smem` 复用为 probability cache：

```cpp
score_smem[qi][kj] = p;
```

之后 accumulator update 直接读取 probability：

```cpp
float p = score_smem[qi][kj];
acc += p * v_smem[kj][d];
```

该优化减少了重复 `expf` 计算。

---

### 2.5 Shared Memory Padding

使用 Nsight Compute 分析发现，原始 shared memory 布局存在明显 bank conflict。

当 `D = 64` 时，如果 shared memory 的 leading dimension 也是 64，那么按列访问 K/V tile 时容易出现 shared memory bank conflict。

因此将 Q/K/V shared memory 的 leading dimension 从 `MAX_D` 改为 `MAX_D + 1`：

```cpp
constexpr int SMEM_D = MAX_D + 1;

__shared__ float q_smem[BLOCK_M][SMEM_D];
__shared__ float k_smem[BLOCK_N][SMEM_D];
__shared__ float v_smem[BLOCK_N][SMEM_D];
```

padding 后，shared load bank conflict 明显下降。

NCU 对比：

| Metric                     | Before Padding | After Final Optimization |
| -------------------------- | -------------: | -----------------------: |
| shared load bank conflict  |        5.6-way |                  1.2-way |
| shared store bank conflict |        7.8-way |                  4.4-way |

shared store conflict 仍然存在，但继续尝试对 `score_smem` 加 padding 后性能退化，因此最终只对 Q/K/V shared memory 做 padding。

失败实验：

```cpp
__shared__ float score_smem[BLOCK_M][BLOCK_N + 1];
```

该实验使 `S=512` 下 final 从约 `0.2288 ms` 退化到约 `0.2538 ms`，因此不保留。

---

### 2.6 Fast Exponential with `__expf`

将部分 `expf` 替换为 CUDA fast math intrinsic：

```cpp
float alpha = (m_old == -FLT_MAX) ? 0.0f : __expf(m_old - m_new);
```

以及：

```cpp
p = __expf(s - m_new);
```

该优化将 `S=512` 下 final 从约：

```text
0.2288 ms
```

优化到：

```text
0.2253 ms
```

误差仍保持在 `1e-7` 量级，因此保留。

---

### 2.7 Branchless Accumulator Update

原始 accumulator update 中存在分支：

```cpp
if (p != 0.0f) {
    acc += p * v_smem[kj][d];
}
```

由于无效位置的 `p` 已经被写成 `0.0f`，因此可以去掉分支：

```cpp
#pragma unroll
for (int kj = 0; kj < BLOCK_N; ++kj) {
    float p = score_smem[qi][kj];
    acc += p * v_smem[kj][d];
}
```

该优化减少了分支判断和 predication，使 `S=512` 下 final 从约：

```text
0.2253 ms
```

进一步优化到：

```text
0.2201 ms
```

因此保留。

---

### 2.8 BLOCK_N Loop Unroll

对固定长度 `BLOCK_N = 32` 的循环加入显式展开：

```cpp
#pragma unroll
for (int kj = 0; kj < BLOCK_N; ++kj) {
    ...
}
```

主要应用在：

```text
tile_max reduction
tile_sum computation
accumulator update
```

加入 unroll 后，最终 `S=512` 下 final 约为：

```text
0.2199 ms
```

编译信息显示寄存器使用为 40 registers/thread，无 spill stores / spill loads，因此该优化没有引入明显寄存器压力。

---

## 3. Final Dispatch Strategy

最终 dispatch 逻辑根据 sequence length 和 head dimension 选择不同 BLOCK_M：

```cpp
if (S >= 512 && D <= 64) {
    launch_flash_attention_skip_bm16_regacc_vec4_pcache_padding(...);
} else if (S >= 256 && D <= 128) {
    launch_flash_attention_skip_bm8_regacc_vec4_pcache_padding(...);
} else if (S >= 128 && BH >= 4 && D <= 128) {
    launch_flash_attention_skip_bm8_regacc_vec4_pcache_padding(...);
} else if (D <= 128) {
    launch_flash_attention_skip_bm4_regacc_vec4_pcache_padding(...);
} else {
    launch_flash_attention_causal_tile_skipping_vec4(...);
}
```

其中：

```text
BM4  : small sequence / small workload
BM8  : medium sequence
BM16 : long sequence, D <= 64
```

---

## 4. Final Benchmark Results

测试设备：

```text
GPU: NVIDIA A40
SM count: 84
Precision: FP32
Task: causal attention forward
```

测试 shape：

```cpp
{1, 1, 64,  64}
{1, 1, 128, 64}
{1, 8, 128, 64}
{1, 8, 256, 64}
{1, 8, 512, 64}
```

最终性能结果：

|  B |  H |   S |  D |   unfused |         tiled |     fused_row | FlashAttention v1 |         final | Best      |
| -: | -: | --: | -: | --------: | ------------: | ------------: | ----------------: | ------------: | --------- |
|  1 |  1 |  64 | 64 | 0.0196 ms |     0.0103 ms | **0.0076 ms** |         0.0186 ms |     0.0104 ms | fused_row |
|  1 |  1 | 128 | 64 | 0.0222 ms |     0.0125 ms | **0.0124 ms** |         0.0346 ms |     0.0184 ms | fused_row |
|  1 |  8 | 128 | 64 | 0.0916 ms | **0.0280 ms** |     0.0511 ms |         0.0788 ms |     0.0295 ms | tiled     |
|  1 |  8 | 256 | 64 | 0.3018 ms | **0.0732 ms** |     0.1612 ms |         0.2622 ms |     0.0798 ms | tiled     |
|  1 |  8 | 512 | 64 | 1.1392 ms |     0.2483 ms |     0.5641 ms |         0.9288 ms | **0.2199 ms** | final     |

---

## 5. Speedup Summary

在长序列场景 `B=1,H=8,S=512,D=64` 下：

```text
unfused  = 1.1392 ms
tiled    = 0.2483 ms
flash_v1 = 0.9288 ms
final    = 0.2199 ms
```

最终版本加速比：

| Baseline             | Speedup |
| -------------------- | ------: |
| vs unfused           |   5.18x |
| vs FlashAttention v1 |   4.22x |
| vs tiled             |   1.13x |

可以看到，最终优化版本在长序列场景下表现最好。对于小序列场景，`fused_row` 或 `tiled` 仍然可能更快，因为 FlashAttention 风格 kernel 的 block 调度、shared memory 和同步开销在小 shape 下不容易被摊薄。

---

## 6. Numerical Correctness

所有 kernel 输出均与 CPU reference 对齐，最大误差保持在 `1e-7 ~ 1e-6` 量级。

最终版本在 `B=1,H=8,S=512,D=64` 下误差为：

```text
final_err = 1.86e-07
```

因此，`__expf` 和 branchless accumulator update 没有破坏数值正确性。

---

## 7. Nsight Compute Observations

最终版本的 NCU 关键指标：

```text
Grid Size                 = 256
Block Size                = 128
Registers Per Thread      = 40
Static Shared Memory      = 23.10 KB / block
Theoretical Occupancy     = 33.33%
Achieved Occupancy        = 18.48%
```

shared memory bank conflict：

```text
shared load  bank conflict = 1.2-way
shared store bank conflict = 4.4-way
```

说明 shared memory padding 对 shared load bank conflict 有明显效果；shared store conflict 仍然存在，但进一步修改 `score_smem` padding、scalar load 等实验均导致性能退化，因此最终没有保留。

---

## 8. Failed Experiments

本项目也记录了一些失败优化，避免只展示成功结果。

### 8.1 Scalar Load Instead of float4

尝试去掉 float4 global load，改为 scalar load/store，希望降低 shared store conflict。

结果：

```text
float4 + padding : 约 0.2288 ms
scalar load      : 约 0.2576 ms
```

结论：float4 global load 的收益大于 shared store conflict 的损失，因此不保留 scalar load。

---

### 8.2 score_smem Padding

尝试将：

```cpp
__shared__ float score_smem[BLOCK_M][BLOCK_N];
```

改为：

```cpp
__shared__ float score_smem[BLOCK_M][BLOCK_N + 1];
```

结果：

```text
0.2288 ms -> 0.2538 ms
```

性能退化，因此不保留。

---

### 8.3 alpha_smem

尝试将每个 query 行的：

```cpp
alpha = exp(m_old - m_new)
```

缓存到 shared memory，避免在 accumulator update 中重复计算。

结果：

```text
0.2288 ms -> 0.2301 ms
```

收益不足且略有退化，因此不保留。

---

### 8.4 No-score Version

尝试完全去掉 `score_smem`，在 accumulator update 阶段重新计算 score/probability。

该版本数值正确，但性能明显退化，因为重复计算 QK 和 exp 的开销过大。

结论：保留 probability cache 更合理。

---

## 9. Build and Run

编译：

```bash
nvcc -O3 -std=c++17 -arch=sm_86 main.cu attention_kernel.cu -o attention_bench
```

运行：

```bash
./attention_bench
```

带 ptxas 编译信息：

```bash
nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas -v \
  main.cu attention_kernel.cu \
  -o attention_bench
```

Nsight Compute profiling：

```bash
sudo -E ncu \
  --set full \
  --kernel-name-base demangled \
  --kernel-name regex:pcache_padding \
  --launch-count 1 \
  --force-overwrite \
  -o ncu_bm16_padding_exp_branchless_unroll_s512_full \
  ./attention_bench
```

导出 profile 文本：

```bash
ncu --import ncu_bm16_padding_exp_branchless_unroll_s512_full.ncu-rep \
  --page details > ncu_bm16_padding_exp_branchless_unroll_s512_full.txt
```

查看关键指标：

```bash
grep -n "Memory Workload Analysis Tables" -A 30 ncu_bm16_padding_exp_branchless_unroll_s512_full.txt
grep -n "Launch Statistics" -A 30 ncu_bm16_padding_exp_branchless_unroll_s512_full.txt
grep -n "Occupancy" -A 40 ncu_bm16_padding_exp_branchless_unroll_s512_full.txt
```

---

## 10. Conclusion

本项目从传统 unfused causal attention 出发，逐步实现 tiled baseline、fused row attention、FlashAttention v1 以及最终优化版 FlashAttention kernel。

最终版本通过：

```text
causal tile skipping
register accumulator
float4 vectorized global load
probability cache
shared memory padding
__expf
branchless accumulator update
#pragma unroll
```

在 NVIDIA A40 上，`B=1,H=8,S=512,D=64` 场景下达到：

```text
0.2199 ms
```

相比 unfused baseline 加速约：

```text
5.18x
```

相比基础 FlashAttention v1 加速约：

```text
4.22x
```

相比 tiled baseline 加速约：

```text
1.13x
```

该项目体现了一个完整的 CUDA kernel 优化流程：先实现正确 baseline，再逐步融合计算、减少中间访存、利用 shared memory、使用 Nsight Compute 定位 bank conflict，并通过 benchmark 验证每一步优化是否真正有效。
