# DeiT-Tiny H-L-H QAT 首版实验与比较口径

日期：2026-11-12

状态：实验入口已准备，尚未正式启动。当前已有 W2A2 官方目标基线占用
`1,4,5,7`，应等待其结束后再跑本实验。

## 1. 首版实验

实现入口：

```text
tmp_scripts/run_deit_tiny_qat_depth_profile_hlh_20ep_4gpu_20261112.sh
```

新增 CLI：

```text
--depth-bit-profile
--depth-weight-bit-profile
--depth-act-bit-profile
```

profile 按 Transformer block 的自然深度顺序解释，每个 entry 同时应用于该
block 的 attention 和 MLP。patch embedding 与 classifier head 继续固定 W8A8。

## 2. 为什么第一版用 4/2 平均 3-bit

把 4-bit 和 2-bit 做成平均 3-bit，可以和 uniform W3A3 对齐名义 bit budget：

```text
H-L-H:
blocks 0-2    W4A4
blocks 3-8    W2A2
blocks 9-11   W4A4

mean block precision
= (6 * 4 + 6 * 2) / 12
= 3
```

相比直接做 H8-L4-H8，这个设计更适合作为第一篇论文的主对照，因为：

1. 12 个 Transformer blocks 的平均 W/A bit 都是 3。
2. 参数数量完全不变。
3. patch/head 在两个实验中都保持 W8A8。
4. 不引入“因为 mixed profile 总 bit 更高所以更好”的明显混杂因素。
5. 能直接回答“不均匀花 3-bit 是否优于均匀花 3-bit”。

注意：这只在名义 W/A bit 上近似匹配。实际 BOPs、kernel 支持、scale 存储和
峰值内存仍要实测，不能只凭平均 bit 报效率。

## 2.1 三值 + 3-bit 的近似 W2A2 profile

三值 `{-1, 0, 1}` 的熵信息量是：

```text
log2(3) = 1.5849625 bit/value
```

因此真正三值不能按整数 1-bit 与 3-bit 精确平均成 2-bit。

首版使用 4 个 3-bit block 和 8 个三值 block：

```text
blocks 0-1    W3A3
blocks 2-9    W-T/A-T
blocks 10-11  W3A3

effective mean
= (4 * 3 + 8 * log2(3)) / 12
= 2.0566 bit
```

它比 uniform W2A2 多约 `2.83%` 的 block 有效 bit，因此可以和当前 W2A2 基线
做近似等预算比较，不需要重跑基线；但结果必须同时报告 effective bits 和
quality-per-bit，不能把结果严格描述成“同等 2-bit 预算”。

入口：

```text
tmp_scripts/run_deit_tiny_qat_hlh_w3a3_mid_ternary_20ep_4gpu_20261112.sh
```

CLI profile 也支持直接写：

```text
--depth-bit-profile 3,3,T,T,T,T,T,T,T,T,3,3
```

`T`、`ternary` 和 `1.58` 都映射到真实三值 quantizer。当前实现输出严格三个
量化 level，而不是二值符号函数。

## 3. 主要比较对象

### 3.1 必跑的同协议控制

| 实验 | Block 精度 | 目的 |
|---|---|---|
| Local uniform W3A3 | `[3]*12` | 同代码、同数据、同 epoch、同 KD 的直接主控制 |
| Local H-L-H | `[4,4,4,2,2,2,2,2,2,4,4,4]` | 检验非均匀 4/2 是否优于均匀 3 |
| Local uniform W2A2 | `[2]*12` | 下界，判断 H-L-H 是否只是在接近 W2 |
| Local uniform W4A4 | `[4]*12` | 上界，判断 H-L-H 与全 4-bit 的差距 |
| Local 3/T H-L-H | `[3,3,T,T,T,T,T,T,T,T,3,3]` | 与当前 W2A2 基线近似等预算比较 |

前两个是论文主表，后两个是诊断 bracket。

3/T profile 与当前 W2A2 基线比较时应同时给出：

```text
baseline block precision = 2.0000
3/T effective mean        = 2.0566
relative effective-bit increase = +2.83%
```

如果 3/T 只比 W2A2 高很少，不能证明 profile 有效；更合理的目标是：

- 绝对精度接近 W3A3。
- quality-per-effective-bit 明显高于 W3A3。
- 在相同 20 epoch 协议下优于当前 W2A2 基线。

同协议 uniform W3A3 启动方式：

```bash
DEPTH_BIT_PROFILE=3,3,3,3,3,3,3,3,3,3,3,3 \
EXP=deit_tiny_qat_uniform_w3a3_20ep_4gpu_20261112 \
bash tmp_scripts/run_deit_tiny_qat_depth_profile_hlh_20ep_4gpu_20261112.sh
```

### 3.2 OFQ 论文 3-bit 只能先作为外部参考

OFQ README 中 DeiT-T W3A3 是 `72.72` Top-1，但官方脚本条件是：

- 300 epochs。
- batch size 140 x 4 ranks。
- LR `5.47e-4`。
- hard + soft KD。
- QKR。
- 之后再做 CGA，`freeze_for_n_epochs=30`。
- patch/head 与主 block 的最终精度口径需要按官方代码重新审计。

当前本地首版是：

- 20 epochs。
- batch 64 x 4 ranks x grad accumulation 2 = 512/update。
- LR `5e-4`。
- CE(cls, GT) + KL(dist, FP teacher)。
- QKR。
- 没有跑 CGA。

因此 H-L-H 的 20 epoch 分数不能直接宣称超过或低于论文 `72.72`。论文数字应
放在“最终目标参考”列，主结论必须来自同协议 local uniform W3A3。

## 4. 比较表设计

建议最终表格至少包含：

| 方法 | Epochs | CGA | Block budget | Realized bits | BOPs | Top-1 | Top-5 | 备注 |
|---|---:|---|---|---|---:|---:|---:|---|
| OFQ public W3A3 | 300 + CGA | yes | uniform 3 | 按官方实现 | 按官方实现 | 72.72 | 外部参考 |
| Local uniform W3A3 | 同本地协议 | 先 no | uniform 3 | 3 | measured | 待跑 | 主控制 |
| Local H-L-H 4/2 | 同本地协议 | 先 no | avg 3 | 3 | measured | 待跑 | 主方法 |
| Local uniform W2A2 | 同本地协议 | 先 no | uniform 2 | 2 | measured | 已有/待对齐 | 下界 |
| Local uniform W4A4 | 同本地协议 | 先 no | uniform 4 | 4 | measured | 已有/待对齐 | 上界 |

如果只在模拟环境统计，可以额外报：

```text
weight_storage_bits
activation_bitops
attention_score_bitops
peak_activation_memory
measured_step_time
```

实际 wall time 不能直接由平均 bit 推算，因为当前 fake-quant QAT 不会自动获得
真实 low-bit kernel 加速。

## 5. H-L-H 的消融顺序

第一轮只跑：

```text
uniform3
H-L-H
```

只有在 H-L-H 有正向 signal 后，再跑：

```text
L-H-L: [2,2,2,4,4,4,4,4,4,2,2,2]
early-high only: [4,4,4,4,4,4,2,2,2,2,2,2]
late-high only:  [2,2,2,2,2,2,4,4,4,4,4,4]
sensitivity allocation: 同 average 3-bit，由逐层 sensitivity 做 DP/knapsack
```

这样可以区分：

- 是“high precision 总量”带来收益。
- 还是 early/late 的位置本身重要。
- 还是简单的自动 sensitivity allocation 已经能解释全部收益。

## 6. 关键指标

除 Top-1/Top-5，还建议记录：

- epoch 0/1/5/10/15/final 的 teacher logit KL。
- 各 block feature KL/MSE。
- attention-output KL。
- pre/post-softmax attention distribution fidelity。
- 每层 quantization error 和 scale 漂移。
- epoch 间 bin crossing / weight oscillation。
- 每个 profile 的实际 checkpoint size。

尤其应检验误差传播假设：

1. early high 是否显著降低早期 feature error。
2. middle low 是否被 middle 层 self-heal。
3. late high 是否主要改善 attention/logit 输出，而不是恢复中间特征。

已有文献存在相反信号：VBQ 报告 deeper layers 会 self-heal early quantization
error；Q-PACE 在 LLM QAT 中报告 later blocks 通常更敏感。因此 H-L-H 不能被当成
先验最优，必须由 ViT QAT 实验判定。

## 7. 运行顺序

### 7.1 静态检查

```bash
DRY_RUN=1 DEPTH_BIT_PROFILE=4,4,4,2,2,2,2,2,2,4,4,4 \
bash tmp_scripts/run_deit_tiny_qat_depth_profile_hlh_20ep_4gpu_20261112.sh
```

### 7.2 20-update smoke

```bash
MAX_UPDATES=20 SKIP_VALIDATE=1 CHECKPOINT_HIST=0 \
EXP=deit_tiny_qat_hlh_avg3_smoke_20261112 \
bash tmp_scripts/run_deit_tiny_qat_depth_profile_hlh_20ep_4gpu_20261112.sh
```

smoke 日志必须出现：

```text
Depth-wise QAT bit profile: blocks=12,
weight=[4, 4, 4, 2, 2, 2, 2, 2, 2, 4, 4, 4],
activation=[4, 4, 4, 2, 2, 2, 2, 2, 2, 4, 4, 4]
```

### 7.3 正式 gate

baseline 释放 GPU 后，先跑 local uniform W3A3 和 H-L-H；两项使用相同 seed、
数据、teacher、batch、LR 和 epoch。

## 8. 当前实现边界

- profile 当前按 12 个 Transformer block 生效。
- patch embedding 和 head 仍由 OFQ wrapper 固定 W8A8。
- `T` 使用内部 bit-0 sentinel 的对称三值 quantizer；activation 和 weight 都是
  三个量化 level。
- 三值 effective bit 使用 `log2(3)` 记账；若部署格式实际按 2-bit byte packing，
  必须另报 packed storage。
- W 与 A 可以共用 profile，也可以分别通过
  `--depth-weight-bit-profile` / `--depth-act-bit-profile` 指定。
- depth profile 与 epoch-wise `--progressive-bit-schedule` 目前禁止同时使用，
  避免两套 bit scheduler 静默覆盖。
- 尚未实现自动 sensitivity search 和 DP/knapsack profile optimizer。
- 首版 CGA 尚未跑；最终论文比较若要求 OFQ 300epoch + CGA，需要另做同协议
  CGA 入口和 checkpoint resume 验证。
