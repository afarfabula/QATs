# DeiT-Tiny depth-wise ternary QAT 结果记录

日期：2026-11-12

## 1. 实验范围

本记录比较 three settings，全部使用同一训练协议：

- DeiT-Tiny distilled，ImageNet-1K。
- 20 epochs，4 x RTX3090，local batch 64，global effective batch 512。
- CE(cls, GT) + KL(dist, FP teacher)，T=1.0。
- QKR type 0，StatsQ weights，LSQ activations。
- patch embedding 和两个 classifier heads 保持 W8A8。
- no MixUp/CutMix，seed 42。
- 每个 full-validation 点使用 50,000 samples。

三值 quantizer 使用真实 `{-1, 0, 1}` levels；softmax 的 nonnegative
three-level quantizer 使用 `{0, s, 2s}`。三值的 entropy-equivalent bit 数记为：

```text
log2(3) = 1.5849625 bit
```

该记录中的 average bit 是 Transformer-block 平均；patch/head W8A8 在三个
profile 中相同。

## 2. 实验配置

| 名称 | Block profile | Effective mean | 状态 |
|---|---|---:|---|
| W2A2 baseline | `[2,2,2,2,2,2,2,2,2,2,2,2]` | 2.0000 | 20 epoch 完成 |
| 3/T H-L-H | `[3,3,T,T,T,T,T,T,T,T,3,3]` | 2.0566 | 20 epoch 完成 |
| 3/2/T H-L-H | `[3,2,T,T,T,T,T,T,T,T,2,3]` | 1.8900 | 进行中，至少 7 个 validation 点 |

与 W2A2 相比：

```text
3/T    effective-bit overhead = +2.83%
3/2/T  effective-bit overhead = -5.50%
```

## 3. 已完成 experiment 的最终结果

### 3.1 W2A2 baseline

| 指标 | 数值 |
|---|---:|
| Final Loss | 2.0427 |
| Final Top-1 | 53.0040 |
| Final Top-5 | 77.1540 |
| Mean Top-1 over 20 epochs | 41.2448 |
| Mean Top-5 over 20 epochs | 65.4046 |
| Last-5 mean Top-1 | 51.9340 |

### 3.2 `[3,3,T,T,T,T,T,T,T,T,3,3]`

| 指标 | 数值 |
|---|---:|
| Final Loss | 1.9519 |
| Final Top-1 | 54.9480 |
| Final Top-5 | 78.5120 |
| Mean Top-1 over 20 epochs | 43.3467 |
| Mean Top-5 over 20 epochs | 67.5124 |
| Last-5 mean Top-1 | 53.7548 |

相对 W2A2：

| 指标 | Δ |
|---|---:|
| Final Top-1 | +1.9440 |
| Final Top-5 | +1.3580 |
| Final Loss | -0.0908 |
| Mean Top-1 | +2.1019 |
| Mean Top-5 | +2.1078 |
| Last-5 mean Top-1 | +1.8208 |

该 profile 在全部 20 个同 epoch validation 点上的 Top-1 都高于 W2A2。

Final Top-1 quality per effective bit：

```text
W2A2: 53.0040 / 2.0000 = 26.5020
3/T:  54.9480 / 2.0566 = 26.7173
relative improvement: +0.81%
```

Final Top-5 quality per effective bit 略低约 1.04%。因此 high-bit profile 的
绝对精度优势明确，但 Top-5 的 quality-per-bit 优势还不成立。

## 4. `[3,2,T,T,T,T,T,T,T,T,2,3]` 当前结果

截至 2026-11-12 本次记录时，该 experiment 已完成 epoch 0-6 的 full validation，
仍在继续运行。

| Epoch | W2A2 Loss | 3/2/T Loss | W2A2 Top-1 | 3/2/T Top-1 | Δ Top-1 | W2A2 Top-5 | 3/2/T Top-5 | Δ Top-5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.9317 | 4.8326 | 10.004 | 10.798 | +0.794 | 25.440 | 26.864 | +1.424 |
| 1 | 4.0062 | 3.8555 | 20.542 | 22.434 | +1.892 | 42.270 | 45.230 | +2.960 |
| 2 | 3.4565 | 3.4191 | 28.602 | 29.322 | +0.720 | 52.638 | 53.468 | +0.830 |
| 3 | 3.2404 | 3.1882 | 31.686 | 32.672 | +0.986 | 56.458 | 57.674 | +1.216 |
| 4 | 3.0653 | 3.0438 | 34.528 | 34.598 | +0.070 | 59.774 | 60.300 | +0.526 |
| 5 | 2.9064 | 2.8989 | 37.172 | 37.350 | +0.178 | 62.366 | 62.460 | +0.094 |
| 6 | 2.8169 | 2.7869 | 38.934 | 39.256 | +0.322 | 64.136 | 64.630 | +0.494 |

当前观察：

1. 绝对精度与 W2A2 很接近，epoch 4-6 的 Top-1 差距只有约 `+0.07` 到 `+0.32`。
2. 尚未落后 W2A2，但早期 `+1` 左右的优势已经明显收窄。
3. 该 profile 的有效 bit 比 W2A2 少 `5.50%`。

Epoch 6 quality per effective bit：

```text
Top-1:
W2A2: 38.934 / 2.0000 = 19.467
3/2/T: 39.256 / 1.8900 = 20.770
relative improvement: +6.70%

Top-5:
W2A2: 64.136 / 2.0000 = 32.068
3/2/T: 64.630 / 1.8900 = 34.196
relative improvement: +6.64%
```

因此当前更准确的表述是：

> 3/2/T 在约少 5.5% 有效 bit 的条件下，绝对精度与 W2A2 基本持平且略高，
> 但绝对收益不足以说明 depth-wise H-L-H 本身带来了强增益。

## 5. 与 high-budget 3/T profile 的关系

在 epoch 6：

```text
3/T H-L-H Top-1:   41.428
3/2/T H-L-H Top-1: 39.256
Δ:                 -2.172
```

high-budget `3/T` profile 的 effective mean 是 `2.0566`，比 3/2/T 多
`0.1666 bit/block`。目前形成清楚的 budget trade-off：

- `3/T`：绝对精度更好，适合证明高 bit 首尾结构的效果。
- `3/2/T`：绝对精度接近 W2A2，但有更好的 quality-per-effective-bit。

## 6. 当前判断

现阶段不能把 3/2/T 描述为显著优于 W2A2：

- epoch 4-6 的绝对 Top-1 差距只有 `+0.07` 到 `+0.32`。
- 最终结果尚未产生，后续学习率衰减可能改变差距。
- 若最终维持约 `+0.2` 到 `+0.5` Top-1，应把结论定位为“少 bit 下的
  Pareto improvement”，而不是“明显提高绝对精度”。

建议继续跑完 20 epoch，并在最终结果中同时报告：

- 同 epoch Top-1/Top-5 差值。
- final/best/last-5 average。
- realized effective bits 和 quality-per-bit。
- 与 high-budget 3/T profile 的 trade-off。

## 7. 路径

基线：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w2a2_official_cekl_t1_noaug_20ep_4gpu_20261112.log
/datadisk2/quyanyi/qat_runs/deit_tiny_w2a2_official_cekl_t1_noaug_20ep_4gpu_20261112/
```

3/T：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_qat_hlh_w3a3_mid_ternary_20ep_4gpu_20261112.log
/datadisk2/quyanyi/qat_runs/deit_tiny_qat_hlh_w3a3_mid_ternary_20ep_4gpu_20261112/
```

3/2/T：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_qat_hlh_w3_w2_mid_ternary_20ep_4gpu_20261112.log
/datadisk2/quyanyi/qat_runs/deit_tiny_qat_hlh_w3_w2_mid_ternary_20ep_4gpu_20261112/
```

代码入口：

```text
tmp_scripts/run_deit_tiny_qat_depth_profile_hlh_20ep_4gpu_20261112.sh
tmp_scripts/run_deit_tiny_qat_hlh_w3a3_mid_ternary_20ep_4gpu_20261112.sh
tmp_scripts/run_deit_tiny_qat_hlh_w3_w2_mid_ternary_20ep_4gpu_20261112.sh
```
