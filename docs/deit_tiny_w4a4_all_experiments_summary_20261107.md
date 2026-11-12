# DeiT-Tiny W4A4 注意力量化：全部实验总结

> 日期：2026-11-07  
> 模型：`deit_tiny_distilled_patch16_224`  
> 数据集：ImageNet-1K  
> 量化设置：主 block W4A4，QK reparam；patch/head W8A8  
> 硬件：实验使用 GPU `0,1`，最多两张卡；持续 baseline 使用 GPU `4,7`，未受干预  
> 精度口径：所有 Top-1 / Top-5 均为 50,000 张 ImageNet validation 的 full-val

---

## 1. 总结论

当前没有找到**稳定、明显值得推进**的 attention 量化增益。

已经确认的事情：

1. post-softmax attention 在 4-bit 下有大量精确零值，这是客观现象。
2. 但“降低零值率”本身不是有效目标。把有意义 attention 的零值率从 `27.82%` 降到 `7.71%` 后，首个 epoch Top-1 反而从 `15.554` 降到 `14.770`。
3. 全分布 JSD、总 row mass、`A·V` cosine 都属于过强监督，均失败。
4. 全局 softmax 温度和全局 LSQ step 缩放都没有收益，模型当前学到的缩放点基本就是最优点。
5. top-5 key mass 是唯一做出真实机制变化的辅助目标：
   - top-5 mass retention 从约 `0.948` 提到 `0.989`；
   - 但第一个对齐 epoch 只提升 `+0.21`，第二个 epoch 回撤 `-0.368`；
   - 因此不能作为稳定精度方案继续扩展。
6. 更值得调查的问题从“attention 尾部要不要保留”转成了：
   - post-softmax LSQ scale 在 epoch 间变化很大；
   - 部分 scale 进入负值区后被 `clip(1e-5)`，梯度为 0，形成不可恢复的量化死区。
7. 最终 upper-bound gate 表明：当前模型并没有因为 attention 概率量化“损失太多”而受限。
   - 直接把 post-softmax attention 改成全精度，Top-1 反而下降 `1.784`；
   - 恢复负 LSQ scale，基本不变；
   - 强制恢复 attention row mass，下降 `3.038`。
   - 因此模型已经把 4-bit attention distortion 纳入训练解，不能把这些 distortion 当成单纯误差去消除。
8. 逐层单独恢复 post-softmax attention FP 的 12 个 full-val 结果全部低于 baseline，最好的一层也下降 `0.132`。不存在被全局 FP bypass 掩盖的局部过量化层。
9. QK、V、attention proj、MLP fc1、MLP fc2 的 activation FP bypass 也全部为负；其中 MLP fc2 input 恢复 FP 后 Top-1 直接降到 `0.2540`。当前 A4 activation 已经成为有效前向的一部分，不能再当作只需要消除的量化噪声。

当前建议：

- 停止 attention-tail / zero-rate / fidelity / row-mass 修复方向；
- 不再通过“让 post-quant attention 更像 pre-quant attention”来设计 loss；
- 如果继续量化研究，目标应转为直接优化真实低比特前向的可训练自由度，而不是恢复全精度分布。

---

## 2. Baseline 曲线

日志：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_baseline_100ep_2gpu_20261107.log
```

| epoch | Top-1 | Top-5 |
|---:|---:|---:|
| 0 | 15.5540 | 37.3860 |
| 1 | 35.4040 | 63.4120 |
| 2 | 49.2300 | 76.1560 |
| 3 | 55.7920 | 81.1700 |
| 4 | 58.4980 | 83.0740 |
| 5 | 61.2720 | 84.7000 |
| 6 | 62.7760 | 85.8420 |
| 7 | 63.6360 | 86.2980 |
| 8 | 64.5100 | 86.8100 |
| 9 | 65.0000 | 86.9900 |
| 10 | 65.5200 | 87.4220 |
| 11 | 66.1620 | 87.6180 |
| 12 | 66.5160 | 87.7400 |
| 13 | 66.4700 | 87.9000 |
| 14 | 66.4260 | 87.8100 |
| 15 | 66.4420 | 87.6600 |
| 16 | 66.9840 | 88.1560 |

注意：OFQ checkpoint 文件名中的编号是 `epoch + 1`。例如：

- `checkpoint-9.pth.tar` 是 epoch 8 训练完后的 checkpoint，对应 baseline `64.5100`；
- `checkpoint-10.pth.tar` 是 epoch 9 训练完后的 checkpoint，对应 baseline `65.0000`。

---

## 3. 完整 full-val 实验总表

| 实验 | 起点 | 主要配置 | Top-1 | Top-5 | 结论 |
|---|---|---:|---:|---:|---|
| baseline | scratch | 无辅助 attention loss | 15.554 @ epoch0 | 37.386 | 对照 |
| attention JSD + mass | scratch | quant weight `0.08`，JSD `1`，mass `1` | 14.770 | 35.706 | 明显失败 |
| `A·V` cosine | scratch | output weight `6.25` | 7.514 | 21.000 | 严重失败 |
| row mass only | scratch | quant weight `2.2`，mass `1` | 15.182 | 36.350 | 低于 baseline |
| top-5 mass, epoch 1 | baseline checkpoint 8 | quant weight `1.5`，top-5 mass `1` | 64.720 | 86.818 | 仅 `+0.21` |
| top-5 mass, epoch 2 | top-k checkpoint 9 | 同上 | 64.632 | 86.808 | 低于 baseline `65.000` |

单点 full-val 波动约 `0.2`。因此：

- `64.720 vs 64.510` 的 `+0.21` 只能算弱信号；
- `64.632 vs 65.000` 的 `-0.368` 说明收益不可持续。

---

## 4. 从零训练的辅助 loss 实验

### 4.1 Attention JSD + row mass

目标：

```text
KD
+ JSD(normalize(pre_quant_attention), normalize(post_quant_attention))
+ relative_row_mass_error
```

最初诊断：

| 指标 | baseline 初始状态 |
|---|---:|
| meaningful attention zero rate | 0.278201 |
| all attention zero rate | 0.296596 |

`quant weight=0.08` 的完整 1 epoch 结果：

| 指标 | baseline | JSD + mass |
|---|---:|---:|
| Top-1 | 15.5540 | 14.7700 |
| Top-5 | 37.3860 | 35.7060 |
| meaningful zero rate | 0.278201 | 0.077073 |
| all attention zero rate | 0.296596 | 0.078036 |

解释：

- 零值率下降非常明显；
- 精度也明显下降；
- 这不是“量化问题没解决”，更像是模型原本已经把低比特的 zero-tail 纳入有效解，强行恢复全分布反而破坏了最优表示。

日志：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_attnquant_w008_1ep_20261107.log
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_attnquant_w008_evaldiag2_20261107.log
```

`weight=0.03` 的尝试在 epoch 0 约 `24%` 时停止，没有 full-val，因此不能用于判断精度：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_attnquant_w003_1ep_20261107.log
```

### 4.2 从 baseline checkpoint 7 resume 的 JSD + mass

配置：

```text
resume=baseline/checkpoint-7.pth.tar
start_epoch=7
attention_quant_weight=0.08
jsd_weight=1
mass_weight=1
```

训练在 epoch 7 约 `14%` 时停止，没有 full-val。只能确认接线和训练能运行，不能判断精度。

日志：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_attnquant_w008_late_ep7_20261107.log
```

### 4.3 `A·V` quantization cosine

目标：比较 pre-quant attention context 和 post-quant attention context 的方向。

`weight=2.0` 的尝试在 epoch 0 约 `45%` 时停止，没有 full-val。

`weight=6.25` 的完整结果：

| 指标 | baseline | `A·V` cosine |
|---|---:|---:|
| Top-1 | 15.5540 | 7.5140 |
| Top-5 | 37.3860 | 21.0000 |

该约束过强，直接破坏 attention context，明确关闭。

日志：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_attnoutput_w2_1ep_20261107.log
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_attnoutput_w625_1ep_20261107.log
```

### 4.4 Row mass only

配置：

```text
attention_quant_weight=2.2
mass_weight=1
JSD=0
```

结果：

| 指标 | baseline | row mass only |
|---|---:|---:|
| Top-1 | 15.5540 | 15.1820 |
| Top-5 | 37.3860 | 36.3500 |

仅约束每行总 mass 也低于 baseline，关闭。

日志：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_attnmass_w22_1ep_20261107.log
```

---

## 5. Softmax 温度扫描

在 baseline checkpoint 7 上对 softmax logits 统一乘不同 temperature scale。

Baseline 同 checkpoint：`63.6360 / 86.2980`。

| softmax scale | Top-1 | Top-5 |
|---:|---:|---:|
| 0.50 | 60.7880 | 84.3860 |
| 0.75 | 62.3260 | 85.5200 |
| 0.90 | 62.5880 | 85.7340 |
| 1.10 | 62.6100 | 85.7300 |

没有一个点超过 baseline。结论：通过改变 attention sharpness 去减少 zero-tail，不是有效方向。

日志：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_ckpt7_softmax_sx050_eval_20261107.log
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_ckpt7_softmax_sx075_eval_20261107.log
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_ckpt7_softmax_sx090_eval_20261107.log
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_ckpt7_softmax_sx110_eval_20261107.log
```

---

## 6. Post-softmax LSQ step 全局缩放

这里不是 softmax temperature，而是直接把 checkpoint 中 12 个 attention softmax quantizer 的学习步长 `s` 乘以全局因子。

Baseline checkpoint：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_baseline_100ep_2gpu_20261107/checkpoint-9.pth.tar
```

对应 full-val：`64.5100 / 86.8100`。

| LSQ step factor | Top-1 | Top-5 | meaningful zero rate | top-5 mass retention |
|---:|---:|---:|---:|---:|
| 0.50 | 62.2640 | 85.1360 | 0.447480 | 0.879078 |
| 0.75 | 64.0560 | 86.3480 | 0.564435 | 0.926883 |
| 1.00 | 64.5100 | 86.8100 | 0.642401 | 0.947958 |
| 1.25 | 64.5340 | 86.7980 | 0.697212 | 0.954546 |
| 1.50 | 64.0260 | 86.4300 | 0.738821 | 0.959682 |

关键结论：

- `factor=1.0` 精确复现 baseline，说明 eval-only 链路正确。
- 变大 `s` 能持续降低 zero rate、提高 top-k retention，但不会提高精度。
- `1.25` 只比 baseline 高 `0.024`，没有实际收益。
- 再次证明 zero-tail / mass 指标优化与分类精度目标不一致。

日志目录：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_attn_softmax_step_x1p0_eval_20261107.log
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_attn_softmax_step_x0p75_eval_20261107.log
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_attn_softmax_step_x1p25_eval_20261107.log
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_attn_softmax_step_x0p5_eval_20261107.log
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_attn_softmax_step_x1p5_eval_20261107.log
```

---

## 7. Top-5 key mass：唯一有效机制信号

### 7.1 目标

允许低概率尾部被 4-bit 量化为 0，只要求每个 query row 的前 5 个 key 的概率质量在 quant 后保留：

```text
top5_mass_pre  = sum(top5(pre_quant_attention))
top5_mass_post = sum(top5(pre_quant_attention 的 same indices) post_quant)
loss = ((top5_mass_post - top5_mass_pre) / top5_mass_pre)^2
```

该目标不要求 full attention 分布对齐，也不要求总 row mass 对齐。

### 7.2 第一轮：epoch 8

严格 resume baseline `checkpoint-8`，训练 epoch 8。

配置：

```text
attention_quant_weight=1.5
jsd_weight=0
row_mass_weight=0
topk_mass_weight=1
topk=5
```

| 指标 | initial checkpoint 8 | top-5 mass |
|---|---:|---:|
| Top-1 | 64.5100 | 64.7200 |
| Top-5 | 86.8100 | 86.8180 |
| meaningful zero rate | 0.646950 | 0.615107 |
| all attention zero rate | 0.683790 | 0.645439 |
| top-5 retention | 0.948279 | 0.988854 |

结果解释：

- 机制确实改变了量化行为；
- retention 提高约 `4.06` 个百分点；
- Top-1 只增加 `0.21`，刚达到单点波动量级；
- 必须继续第二个 epoch 判断稳定性。

日志：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_attntopkmass_w15_late_ep8_20261107.log
```

### 7.3 第二轮：epoch 9

从 top-k 第一轮 `checkpoint-9` 严格继续，训练 epoch 9。

| 指标 | baseline | top-5 mass |
|---|---:|---:|
| Top-1 | 65.0000 | 64.6320 |
| Top-5 | 86.9900 | 86.8080 |
| meaningful zero rate | 未在同轮测 | 0.612828 |
| all attention zero rate | 未在同轮测 | 0.640508 |
| top-5 retention | 未在同轮测 | 0.989620 |

关键结果：

- top-5 retention 仍从 `0.988854` 小幅升到 `0.989620`；
- zero rate 继续保持在较低水平；
- 但 Top-1 从第一轮的 `+0.21` 变成 `-0.368`。

结论：

**机制有效，但精度收益不稳定，当前 top-k mass 方案关闭。**

日志：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_attntopkmass_w15_ep9_from_ep8_20261107.log
```

---

## 8. Attention 零值率诊断

### 8.1 训练初始

| 指标 | 值 |
|---|---:|
| meaningful attention zero rate | 0.278201 |
| all attention zero rate | 0.296596 |

### 8.2 Baseline 早期 checkpoint 2

对应 full-val：`35.4040 / 63.4120`。

| 指标 | 值 |
|---|---:|
| meaningful zero rate | 0.646385 |
| all attention zero rate | 0.669981 |

日志：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_baseline_ep2_evaldiag_20261107.log
```

### 8.3 Baseline checkpoint 8

对应 full-val：`63.6360 / 86.2980`。

| 指标 | 值 |
|---|---:|
| meaningful zero rate | 0.646950 |
| all attention zero rate | 0.683790 |
| top-5 mass error | 0.048318 |
| top-5 mass retention | 0.948279 |

日志：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_ckpt8_quantdiag_20261107.log
```

观察：

- 模型训练过程中 zero rate 反而升高；
- 精度却从 15.554 提升到 64 左右；
- 因此“零值多 = 模型坏”这个假设不成立。

---

## 9. LSQ scale 稳定性诊断

### 9.1 epoch 间 scale 变化

比较 baseline checkpoint 9 和 checkpoint 10：

- 多数 layer 的 row-wise scale 平均绝对相对变化约 `25%-36%`；
- 95th percentile 相对变化约 `57%-86%`。

这说明 post-softmax quantizer 的 scale 不是稳定的静态校准参数，而是在持续大幅漂移。

### 9.2 Layer 间差异

- layer 3 的平均 scale 比 layer 11 大约 10 倍；
- layer 10 和 layer 11 在 top-k 训练后出现不同的重分配：
  - layer 11 mean scale 约变成 baseline 的 `0.45x`；
  - layer 10 从接近 clamp 下界恢复到非零尺度。

### 9.3 负 scale 与梯度死区

Baseline checkpoint 9 的 layer 10：

- `16 / 198` 个 post-softmax scale 为负。

Baseline checkpoint 10：

- `14 / 198` 个仍为负；
- 最小负值约 `-0.17`。

对应代码：

```python
s_scale = grad_scale(clip(alpha, 1e-5), s_grad_scale)
```

问题：

- forward 中负 scale 被当成 `1e-5`；
- `clip` 在负区间梯度为 0；
- scale 很难自己恢复；
- 对应 query row 会长期处于极不合理的量化状态。

这不是蒸馏目标问题，而是 LSQ 参数化/投影问题。

---

## 10. 工程冒烟与中途停止实验

### 10.1 冒烟

| 实验 | 结果 |
|---|---|
| baseline smoke 1 | 启动失败，未进入稳定训练 |
| baseline smoke 2 | 20 optimizer updates 正常结束 |
| attention quant smoke | 20 updates 正常结束 |
| attention output smoke | 20 updates 正常结束 |
| row mass smoke | 20 updates 正常结束 |
| top-k mass smoke | 20 updates 正常结束 |

### 10.2 没有 full-val 的尝试

| 实验 | 停止位置 | 可否用于结论 |
|---|---|---|
| attention JSD+mass `weight=0.03` | epoch 0 约 24% | 否 |
| `A·V` cosine `weight=2.0` | epoch 0 约 45% | 否 |
| late JSD+mass `weight=0.08`, checkpoint7 | epoch 7 约 14% | 否 |

这些实验只能证明代码可运行，不能用于判断精度。

---

## 11. Upper-bound gate 结果

统一起点：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_baseline_100ep_2gpu_20261107/checkpoint-17.pth.tar
```

该 checkpoint 对应 baseline epoch 16：

```text
Top-1=66.9840
Top-5=88.1560
```

### 11.1 Post-softmax attention 全精度 bypass

| 实验 | Top-1 | Top-5 | 相对 baseline |
|---|---:|---:|---:|
| baseline W4 attention | 66.9840 | 88.1560 | 0 |
| post-softmax attention FP bypass | 65.2000 | 87.0080 | -1.7840 |

结论：

- 直接把 post-softmax attention 改成全精度不是收益方向；
- 当前模型已经把 4-bit attention 的 distortion 学进了有效解；
- 这也解释了为什么 JSD、row mass、zero-tail 修复虽然让量化指标更像全精度，却持续损害精度。

日志：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_ep16_attn_softmax_fp_fixed_eval_20261107.log
```

第一次 FP 诊断因为开关未真正挂到 model，输出与 baseline 完全相同，已废弃：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_ep16_attn_softmax_fp_eval_20261107.log
```

### 11.2 恢复负 LSQ scale

#### current best checkpoint

checkpoint 17 只有 `1/198` 个 layer 10 post-softmax scale 为负。

| 实验 | Top-1 | Top-5 | 相对 baseline |
|---|---:|---:|---:|
| baseline | 66.9840 | 88.1560 | 0 |
| 负 scale 替换为同层 median | 66.9480 | 88.1600 | -0.0360 |

日志：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_ep16_deadscale_revived_eval_20261107.log
```

#### earlier checkpoint with more dead scales

baseline checkpoint 10 有 `14/198` 个 layer 10 scale 为负。

| 实验 | Top-1 | Top-5 | 相对 baseline |
|---|---:|---:|---:|
| baseline | 65.5200 | 87.4220 | 0 |
| 14 个负 scale 替换为同层 median | 65.5340 | 87.4960 | +0.0140 |

结论：负 scale 作为“静态恢复”没有可用收益。若后面还要研究，只能研究正值参数化和训练轨迹，不能再把恢复几个负值当作方法。

日志：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_ep10_deadscale_revived_eval_20261107.log
```

### 11.3 Post-quant row mass renormalization

| 实验 | Top-1 | Top-5 | 相对 baseline |
|---|---:|---:|---:|
| baseline | 66.9840 | 88.1560 | 0 |
| post-quant row-mass renorm | 63.9460 | 86.0360 | -3.0380 |
| 恢复负 scale + renorm | 63.9740 | 86.0500 | -3.0100 |

结论：

- row mass 不是当前模型缺失的自由度；
- 模型已经把低比特 attention 的 row-mass bias 吸收到分类解里；
- 强行把每行重新归一到 1 会直接破坏已学到的有效前向。

日志：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_ep16_attn_postquant_renorm_eval_20261107.log
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_ep16_deadscale_revived_plus_renorm_eval_20261107.log
```

### 11.4 逐层 FP sensitivity

每次只把 `blocks.i.attn.quan_a_softmax_fn` 从 A4 改成 FP，其余 11 层保持 W4A4。

Baseline：`66.9840 / 88.1560`。

| layer | Top-1 | Top-5 | ΔTop-1 |
|---:|---:|---:|---:|
| 0 | 66.7540 | 88.0740 | -0.2300 |
| 1 | 66.7680 | 88.1100 | -0.2160 |
| 2 | 66.6100 | 88.0640 | -0.3740 |
| 3 | 66.7920 | 88.0540 | -0.1920 |
| 4 | 66.7220 | 88.0940 | -0.2620 |
| 5 | 66.5080 | 88.0180 | -0.4760 |
| 6 | 66.7480 | 88.1360 | -0.2360 |
| 7 | 66.4920 | 87.9820 | -0.4920 |
| 8 | 66.6940 | 88.1400 | -0.2900 |
| 9 | 66.8340 | 88.1500 | -0.1500 |
| 10 | 66.6560 | 88.1280 | -0.3280 |
| 11 | 66.8520 | 88.1380 | -0.1320 |

结论：

- 12 层全部为负收益；
- 最接近 baseline 的 layer 11 仍低 `0.132`，属于无收益；
- 不存在“只看全层 FP 时被平均掩盖、单独一层其实有收益”的情况；
- attention softmax 量化方向的单层局部修补也关闭。

日志：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_ep16_attn_fp_layer0_eval_20261107.log
...
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_ep16_attn_fp_layer11_eval_20261107.log
```

### 11.5 Activation quantizer 组件敏感性

统一起点仍为 `deit_baseline_ep16_top1_66p984.pth.tar`。每个实验只把一个 activation quantizer family 从 A4 改成 FP，其他位置保持 W4A4。

Baseline：`66.9840 / 88.1560`。

| activation 组件 | 对应 quantizer | Top-1 | Top-5 | ΔTop-1 |
|---|---|---:|---:|---:|
| QK | `quan_a_qkx_fn` | 65.5580 | 87.1800 | -1.4260 |
| V | `quan_a_v_fn` | 66.7620 | 88.2220 | -0.2220 |
| attention proj input | `attn.proj.input_quant_fn` | 65.7100 | 87.4140 | -1.2740 |
| MLP fc1 input | `mlp.fc1.input_quant_fn` | 64.8160 | 86.7680 | -2.1680 |
| MLP fc2 input | `mlp.fc2.input_quant_fn` | 0.2540 | 1.0660 | -66.7300 |

结论：

- 没有发现可以通过局部恢复 FP 获得收益的 activation family；
- QK、attention proj、MLP fc1 对全精度恢复非常敏感；
- MLP fc2 input 恢复 FP 后几乎完全失效，说明 A4 quantizer 在这里不只是压缩，还起到了关键的数值约束/归一化作用；
- 继续用“让 quantized activation 更接近 FP activation”的 fidelity loss 没有依据。

日志：

```text
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_ep16_activation_fp_qkx_eval_20261107.log
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_ep16_activation_fp_v_eval_20261107.log
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_ep16_activation_fp_attn_proj_eval_20261107.log
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_ep16_activation_fp_mlp_fc1_eval_20261107.log
/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_ep16_activation_fp_mlp_fc2_eval_20261107.log
```

### 11.6 仍已接入但未验证的训练开关

负 scale projection 训练开关：

```text
--lsq-scale-floor 1e-5
--lsq-scale-floor-target attention_softmax
```

AOQ-style threshold/level separation 开关：

```text
--aoq-explore-attention-softmax
--aoq-explore-scale-ratio
--aoq-explore-threshold-ratio
```

两者的静态 upper-bound 支持度都很弱，因此只有在提出明确的“训练轨迹”假设时才值得运行；当前不进入下一轮。

---

## 12. 当前实验结论

### 已关闭

1. 降低 attention zero-tail。
2. 全 attention JSD。
3. 总 row mass 对齐。
4. `A·V` context cosine。
5. softmax 温度扫描。
6. 全局 post-softmax LSQ step 缩放。
7. 当前 top-5 mass loss 作为主精度方案。
8. post-softmax attention 全精度 bypass 作为修复方向。
9. 静态恢复负 LSQ scale。
10. post-quant row-mass renormalization。
11. 逐层恢复 post-softmax attention FP。
12. QK/V/attention-proj/MLP activation 的 FP bypass。

### 当前判断

attention probability quantization 本身不再是值得直接“修正”的对象，且逐层 gate 也确认不存在局部例外。其他 activation 组件的 FP bypass 同样全部为负。后续若要继续，应重新定义为：

1. 真实低比特前向下，直接优化 quantizer 的可学习自由度；
2. 研究 4-bit attention distortion 为何充当有效正则/偏置，而不是把它视为纯误差；
3. 将 activation 稳定性问题转移到 Q/K/V/proj/MLP 的 activation quantizer scale 轨迹；
4. 若提不出不依赖 pre/post-quant 对齐的新假设，则暂停 attention-tail 路线。

---

## 13. 相关代码和脚本

核心代码：

```text
qat_launch.py
third_party/OFQ/src/quantization/modules/attention.py
```

实验脚本：

```text
tmp_scripts/run_deit_tiny_w4a4_attnquant_gate_2gpu_20261107.sh
tmp_scripts/scale_attention_softmax_checkpoint_20261107.py
tmp_scripts/run_deit_attn_softmax_step_sweep_20261107.sh
tmp_scripts/revive_nonpositive_attention_softmax_scales_20261107.py
tmp_scripts/run_deit_attention_quant_upper_bound_20261107.sh
```

补充文档：

```text
docs/attention_rank_zero_tail_degeneracy_20261107.md
docs/deit_tiny_w4a4_attention_quant_experiments_20261107.md
```

