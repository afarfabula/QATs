# DeiT-Tiny W4A4 注意力量化实验记录

> 最新完整汇总见 `docs/deit_tiny_w4a4_all_experiments_summary_20261107.md`。本文保留为阶段性记录。  
> 后续 upper-bound 已完成：attention FP bypass 为 `65.2000`（比 baseline `66.9840` 低 `1.7840`），row-mass renorm 为 `63.9460`，静态负 scale 恢复无收益。attention-tail 修复路线已关闭。

> 日期：2026-11-07  
> 模型：DeiT-Tiny distilled, ImageNet-1K  
> 量化：主 block W4A4，QK reparam；patch/head W8A8  
> 用途：判断 attention 的 zero-tail、量化保真度和激活尺度稳定性中，哪一部分值得进入后续方法设计

## 1. 实验纪律

- 所有精度均为 50,000 张 ImageNet validation 的 full-val 结果。
- 训练对照严格从同一个 baseline checkpoint 和同一 epoch 开始。
- 实验只使用 GPU `0,1`；另一条 baseline 在 GPU `4,7`，未干预。
- 训练 loss 只用于观察，不作为精度收益证据。

## 2. Baseline

日志：`/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_baseline_100ep_2gpu_20261107.log`

| epoch | Top-1 | Top-5 |
|---:|---:|---:|
| 6 | 62.7760 | 85.8420 |
| 7 | 63.6360 | 86.2980 |
| 8 | 64.5100 | 86.8100 |
| 9 | 65.0000 | 86.9900 |

单点 full-val 的统计波动约为 `0.2`，低于这个幅度的变化不能单独视为收益。

## 3. 已关闭方向

### 3.1 全分布与总 mass 对齐

从零训练时加入 JSD 和 row-mass 约束，可以把有意义 attention 的零值率从约 `27.8%` 降到约 `7.7%`，但首个 full-val Top-1 从 `15.554` 降到 `14.770`。降低零值率本身不是有效目标。

### 3.2 输出方向约束

加入 `A·V` cosine 约束后，epoch 0 Top-1 仅为 `7.514`。约束 attention context 的方向太强。

### 3.3 全局 softmax 量化步长缩放

在 baseline `checkpoint-9`（对应上表 epoch 8，Top-1 `64.5100`）上整体缩放全部 12 层 post-softmax LSQ 步长：

| LSQ step factor | Top-1 | Top-5 |
|---:|---:|---:|
| 0.50 | 62.2640 | 85.1360 |
| 0.75 | 64.0560 | 86.3480 |
| 1.00 | 64.5100 | 86.8100 |
| 1.25 | 64.5340 | 86.7980 |
| 1.50 | 64.0260 | 86.4300 |

`factor=1.0` 精确复现 baseline，说明 eval-only 变换链路正确。最优区域就是当前学习到的步长，没有可直接利用的全局缩放收益。

## 4. 当前阳性信号：Top-5 Key Mass

动机：允许低概率尾部被 4-bit 量化为 0，只保护每个 query row 中 top-5 key 的概率质量。

配置：

```text
attention_quant_weight=1.5
jsd_weight=0
row_mass_weight=0
topk_mass_weight=1
topk=5
```

第一轮从 baseline `checkpoint-8` 严格 resume，训练一个 epoch：

日志：`/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_attntopkmass_w15_late_ep8_20261107.log`

| 指标 | initial checkpoint 8 | top-5 mass |
|---|---:|---:|
| full-val Top-1 | 64.5100 | 64.7200 |
| Top-5 mass retention | 0.9483 | 0.9889 |
| meaningful attention zero rate | 0.6470 | 0.6151 |
| all attention zero rate | 0.6838 | 0.6454 |

结论：

- 机制确实生效，关键 key 概率质量保留率提高约 `4.1` 个百分点。
- 精度提高 `+0.21`，刚达到单点噪声量级，尚不能视为稳定收益。
- 必须再跑一个对齐 epoch，与 baseline epoch 9 的 `65.0000` 比较。
- 第二轮结果已完成：`64.6320`，低于 baseline `65.0000`，当前 top-5 mass 方案不再继续扩展。

第二轮状态：

```text
experiment=deit_tiny_w4a4_attntopkmass_w15_ep9_from_ep8_20261107
resume=/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_attntopkmass_w15_late_ep8_20261107/checkpoint-9.pth.tar
log=/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_attntopkmass_w15_ep9_from_ep8_20261107.log
```

## 5. 激活稳定性证据

baseline 相邻 epoch 的 post-softmax quantizer scale 不是稳定的静态校准结果：

- 多数层的 row-wise scale 相对变化平均绝对值约 `25%-36%`；
- 95th percentile 相对变化约 `57%-86%`；
- layer 10 出现负的 LSQ alpha，forward 时实际被 clamp 到 `1e-5`；
- layer 3 的平均 scale 约比 layer 11 大 10 倍。

top-5 mass 训练后，layer 10/11 的 scale 又发生了大幅重分配。Layer 11 的 mean scale 变为 baseline 的约 `0.45x`，而 layer 10 从实际 clamp 下界恢复到非零尺度。

这说明“activation variance 很大”至少部分可直接归因于低比特概率在每个 query row 上的量化步长不稳定，而不是无法观测的抽象现象。

## 6. 一个明确的量化实现缺陷：LSQ scale 负值死区

baseline epoch 9 的 layer 10 中有 `16/198` 个 post-softmax LSQ scale 为负。`LsqQuantizer.forward` 内部使用：

```python
s_scale = grad_scale(clip(alpha, 1e-5), s_grad_scale)
```

当 `alpha < 0` 时，forward 的确把它当成 `1e-5`，但 `clip` 在负半轴的梯度为 0。因此这些位置不仅失去了合理的缩放步长，也失去了把 scale 学回来的梯度。它们对应的 query row 会近似变成常量极小 attention，而不是正常的 4-bit 概率。

这不是蒸馏 loss 问题，而是 LSQ scale 参数化/投影问题。最小修复是在 optimizer step 后把被选中的 scale 投影到正值下界，例如 `s.clamp_min_(1e-5)`，使下一轮 forward 位于非零梯度区域。

本实验已加入两个可选控制：

```text
--lsq-scale-floor 1e-5
--lsq-scale-floor-target attention_softmax
```

尚未把它算作收益，需单独训练并 full-val 验证。

## 7. 下一步，只在当前信号失败时启动

按优先级：

1. `top-5 mass` 第二 epoch 仍领先：保留并演进到 late-stage stabilization。
2. LSQ scale floor 是最小、最明确的量化修复，必须做单变量 gate。
3. post-quant row-mass renormalization 用于确认 row mass 是否是主要误差源；它是上界诊断，不默认作为最终硬件方案。
4. 若上述都没有收益：再测试只作用于 `quan_a_softmax_fn` 的 AOQ threshold/level separation，训练末回到真实 4-bit 点验证。
5. 若这些都没有收益：停止 attention-tail 方向，不再用更强蒸馏 loss 调参。
