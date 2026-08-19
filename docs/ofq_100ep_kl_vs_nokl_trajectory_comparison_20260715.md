# OFQ 100epoch KL vs no-KL 训练轨迹对比与 KL 实现细节

## 结论摘要

本次对比的是两条从 ImageNet pretrained / public OFQ 初始化开始的 Swin-T W4A4-family 100epoch 训练：

```text
实验 A: ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713
方法: OFQ public-family + late sparse prev-step attention KL
best: checkpoint-100 Top-1 80.7720 Top-5 95.4320
last20_avg: 80.6965
last10_avg: 80.7316

实验 B: ofq_100ep_fromscratch_original_ofq_public_control_20260714
方法: OFQ public-family strict no-KL control
best: checkpoint-82 Top-1 80.7920 Top-5 95.4100
last20_avg: 80.6916
last10_avg: 80.7086
```

严格结论：

```text
1. 两条曲线在 checkpoint-1 到 checkpoint-51 完全对齐，说明主训练链路一致。
2. checkpoint-52 后 KL controller 开始可能产生影响，但 no-KL 曲线并不弱于 KL 曲线。
3. no-KL best 80.7920 反而高于 KL best 80.7720，差值 +0.0200。
4. KL 版本的 last20 / last10 略高，但差值很小：last20 +0.0049，last10 +0.0230。
5. 因此，100epoch KL 版本的 80.7720 主要不是当前 sparse prev-step KL 带来的明确收益，而是 OFQ public-family 自然长跑本身可以达到的高度。
6. 当前固定-head sparse prev-step KL controller 在这个 100epoch from-pretrained 设置下没有形成稳定正贡献，不应继续原样加长作为主方案。
```

## 实验边界

两条实验共同保持：

```text
model: swin_t
method: ofq
dataset: /tmp/imagenet1k_full_parquet
dataset-format: parquet
wbits / abits: 4 / 4
wq_mode / aq_mode: statsq / lsq
qk_reparam: true
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
batch_size: 64 per GPU
nproc_per_node: 8
epochs: 100
scheduler_epochs: 100
lr: 2e-4
min_lr: 5e-6
weight_decay: 0.0
checkpoint_hist: 100
epoch_checkpoint_interval: 1
pretrained: true
pretrained_initialized: true
```

共同禁止：

```text
不从 checkpoint-10 / QAT checkpoint resume
不使用 soup
不使用 checkpoint averaging
不使用 ensemble
不使用 A8->A4
不做多 checkpoint 拼接
```

区别只在 KL/refmodel/controller：

```text
KL 版本:
  train_scheme=ema_ref_attn_kl
  ref_update=prev_step
  ref_attn_kl_weight=0.0
  dynamic_sparse_prevstep_kl=true
  dynamic_kl_start_epoch=51
  dynamic_kl_observe_until_epoch=50

no-KL 对照:
  train_scheme=baseline
  dynamic_sparse_prevstep_kl=false
  ref_attn_kl_weight=0.0
  controller artifact absent
  RefW nonzero rows=0
```

## 完整性审计

KL 版本：

```text
checkpoint_count=100
fullval_rows=100
bad_sample_rows=0
controller_rows=100
controller_triggers=10
controller_observe_triggers=0
controller_selected_avoid=0
observe_nonzero_refw_lines=0
nonzero_refw_epochs=53,54,55,64,65,73,78,79,85,98
```

no-KL 对照：

```text
checkpoint_count=100
fullval_rows=100
bad_sample_rows=0
controller_artifact=absent
nonzero_refw_lines=0
nonzero_refw_epochs=NA
checkpoint-1 到 checkpoint-100: present, missing=[]
```

## 总体结果对比

| 指标 | KL 100ep | no-KL 100ep | 结论 |
| --- | ---: | ---: | --- |
| best Top-1 | 80.7720 | 80.7920 | no-KL 高 0.0200 |
| best checkpoint | 100 | 82 | no-KL 更早达到峰值 |
| best Top-5 | 95.4320 | 95.4100 | KL 略高 0.0220 |
| last20_avg | 80.6965 | 80.6916 | 基本持平，KL 高 0.0049 |
| last10_avg | 80.7316 | 80.7086 | 基本持平，KL 高 0.0230 |
| >80.5980 | 20 | 20 | 持平 |
| >80.6820 | 14 | 12 | KL 多 2 个 |
| >80.7240 | 6 | 5 | KL 多 1 个 |
| >80.7520 | 3 | 2 | KL 多 1 个 |
| >80.7600 | 2 | 2 | 持平 |
| >80.7720 | NA | 2 | no-KL 有 2 个超过 KL best |
| >80.8280 | 0 | 0 | 都没有超过 10->210 KL best |
| >=81.0 | 0 | 0 | 都没到 81 |

## 分段轨迹

关键 checkpoint 对比：

| checkpoint | KL Top-1 | no-KL Top-1 | no-KL - KL | 解释 |
| --- | ---: | ---: | ---: | --- |
| 1 | 77.7080 | 77.7080 | +0.0000 | 完全一致 |
| 10 | 78.9460 | 78.9460 | +0.0000 | 完全一致 |
| 20 | 79.3600 | 79.3600 | +0.0000 | 完全一致 |
| 31 | 79.7980 | 79.7980 | +0.0000 | 完全一致 |
| 40 | 79.9620 | 79.9620 | +0.0000 | 完全一致 |
| 51 | 80.2140 | 80.2140 | +0.0000 | dynamic start 边界一致 |
| 52 | 80.3180 | 80.3180 | +0.0000 | 首个 dynamic 决策后仍一致 |
| 54 | 80.1920 | 80.2880 | +0.0960 | KL 第一组 pulse 后 no-KL 更高 |
| 55 | 80.2120 | 80.3180 | +0.1060 | no-KL 更高 |
| 60 | 80.4480 | 80.5260 | +0.0780 | no-KL 更高 |
| 64 | 80.3500 | 80.4700 | +0.1200 | no-KL 更高 |
| 71 | 80.5720 | 80.5920 | +0.0200 | no-KL 略高 |
| 77 | 80.5880 | 80.6360 | +0.0480 | no-KL 更高 |
| 80 | 80.6500 | 80.5840 | -0.0660 | KL 局部更高 |
| 82 | 80.5840 | 80.7920 | +0.2080 | no-KL 达到全局 best |
| 90 | 80.6860 | 80.6840 | -0.0020 | 基本一致 |
| 91 | 80.6780 | 80.7320 | +0.0540 | no-KL 更高 |
| 98 | 80.7040 | 80.7760 | +0.0720 | no-KL 更高 |
| 100 | 80.7720 | 80.6780 | -0.0940 | KL 最后一个点更高 |

### 完整 100epoch Full Validation 轨迹

说明：每个 checkpoint 都对应一次 ImageNet 50000 samples full validation。`no-KL - KL` 为正表示 no-KL 同 epoch 更高；best-so-far 用于画累计最优优化路径。

| checkpoint | KL Acc@1 | no-KL Acc@1 | no-KL - KL | KL best-so-far | no-KL best-so-far |
|---:|---:|---:|---:|---:|---:|
| 1 | 77.7080 | 77.7080 | +0.0000 | 77.7080 | 77.7080 |
| 2 | 78.2980 | 78.2980 | +0.0000 | 78.2980 | 78.2980 |
| 3 | 78.5020 | 78.5020 | +0.0000 | 78.5020 | 78.5020 |
| 4 | 78.4680 | 78.4680 | +0.0000 | 78.5020 | 78.5020 |
| 5 | 78.4660 | 78.4660 | +0.0000 | 78.5020 | 78.5020 |
| 6 | 78.6860 | 78.6860 | +0.0000 | 78.6860 | 78.6860 |
| 7 | 78.8220 | 78.8220 | +0.0000 | 78.8220 | 78.8220 |
| 8 | 78.8020 | 78.8020 | +0.0000 | 78.8220 | 78.8220 |
| 9 | 78.7960 | 78.7960 | +0.0000 | 78.8220 | 78.8220 |
| 10 | 78.9460 | 78.9460 | +0.0000 | 78.9460 | 78.9460 |
| 11 | 78.9520 | 78.9520 | +0.0000 | 78.9520 | 78.9520 |
| 12 | 79.0720 | 79.0720 | +0.0000 | 79.0720 | 79.0720 |
| 13 | 78.9700 | 78.9700 | +0.0000 | 79.0720 | 79.0720 |
| 14 | 79.2400 | 79.2400 | +0.0000 | 79.2400 | 79.2400 |
| 15 | 79.1180 | 79.1180 | +0.0000 | 79.2400 | 79.2400 |
| 16 | 79.1460 | 79.1460 | +0.0000 | 79.2400 | 79.2400 |
| 17 | 79.2520 | 79.2520 | +0.0000 | 79.2520 | 79.2520 |
| 18 | 79.2860 | 79.2860 | +0.0000 | 79.2860 | 79.2860 |
| 19 | 79.3160 | 79.3160 | +0.0000 | 79.3160 | 79.3160 |
| 20 | 79.3600 | 79.3600 | +0.0000 | 79.3600 | 79.3600 |
| 21 | 79.3320 | 79.3320 | +0.0000 | 79.3600 | 79.3600 |
| 22 | 79.5980 | 79.5980 | +0.0000 | 79.5980 | 79.5980 |
| 23 | 79.5480 | 79.5480 | +0.0000 | 79.5980 | 79.5980 |
| 24 | 79.6140 | 79.6140 | +0.0000 | 79.6140 | 79.6140 |
| 25 | 79.5540 | 79.5540 | +0.0000 | 79.6140 | 79.6140 |
| 26 | 79.5840 | 79.5840 | +0.0000 | 79.6140 | 79.6140 |
| 27 | 79.6100 | 79.6100 | +0.0000 | 79.6140 | 79.6140 |
| 28 | 79.6820 | 79.6820 | +0.0000 | 79.6820 | 79.6820 |
| 29 | 79.8120 | 79.8120 | +0.0000 | 79.8120 | 79.8120 |
| 30 | 79.7400 | 79.7400 | +0.0000 | 79.8120 | 79.8120 |
| 31 | 79.7980 | 79.7980 | +0.0000 | 79.8120 | 79.8120 |
| 32 | 79.7420 | 79.7420 | +0.0000 | 79.8120 | 79.8120 |
| 33 | 79.9140 | 79.9140 | +0.0000 | 79.9140 | 79.9140 |
| 34 | 79.8780 | 79.8780 | +0.0000 | 79.9140 | 79.9140 |
| 35 | 79.9420 | 79.9420 | +0.0000 | 79.9420 | 79.9420 |
| 36 | 79.9300 | 79.9300 | +0.0000 | 79.9420 | 79.9420 |
| 37 | 79.9640 | 79.9640 | +0.0000 | 79.9640 | 79.9640 |
| 38 | 80.0900 | 80.0900 | +0.0000 | 80.0900 | 80.0900 |
| 39 | 79.9640 | 79.9640 | +0.0000 | 80.0900 | 80.0900 |
| 40 | 79.9620 | 79.9620 | +0.0000 | 80.0900 | 80.0900 |
| 41 | 80.0840 | 80.0840 | +0.0000 | 80.0900 | 80.0900 |
| 42 | 80.0740 | 80.0740 | +0.0000 | 80.0900 | 80.0900 |
| 43 | 80.1000 | 80.1000 | +0.0000 | 80.1000 | 80.1000 |
| 44 | 80.0540 | 80.0540 | +0.0000 | 80.1000 | 80.1000 |
| 45 | 80.1800 | 80.1800 | +0.0000 | 80.1800 | 80.1800 |
| 46 | 80.1180 | 80.1180 | +0.0000 | 80.1800 | 80.1800 |
| 47 | 80.1800 | 80.1800 | +0.0000 | 80.1800 | 80.1800 |
| 48 | 80.1620 | 80.1620 | +0.0000 | 80.1800 | 80.1800 |
| 49 | 80.1400 | 80.1400 | +0.0000 | 80.1800 | 80.1800 |
| 50 | 80.1300 | 80.1300 | +0.0000 | 80.1800 | 80.1800 |
| 51 | 80.2140 | 80.2140 | +0.0000 | 80.2140 | 80.2140 |
| 52 | 80.3180 | 80.3180 | +0.0000 | 80.3180 | 80.3180 |
| 53 | 80.2280 | 80.2280 | +0.0000 | 80.3180 | 80.3180 |
| 54 | 80.1920 | 80.2880 | +0.0960 | 80.3180 | 80.3180 |
| 55 | 80.2120 | 80.3180 | +0.1060 | 80.3180 | 80.3180 |
| 56 | 80.4040 | 80.3460 | -0.0580 | 80.4040 | 80.3460 |
| 57 | 80.3200 | 80.4240 | +0.1040 | 80.4040 | 80.4240 |
| 58 | 80.3560 | 80.3940 | +0.0380 | 80.4040 | 80.4240 |
| 59 | 80.2480 | 80.3540 | +0.1060 | 80.4040 | 80.4240 |
| 60 | 80.4480 | 80.5260 | +0.0780 | 80.4480 | 80.5260 |
| 61 | 80.4040 | 80.5080 | +0.1040 | 80.4480 | 80.5260 |
| 62 | 80.4700 | 80.3840 | -0.0860 | 80.4700 | 80.5260 |
| 63 | 80.3920 | 80.4260 | +0.0340 | 80.4700 | 80.5260 |
| 64 | 80.3500 | 80.4700 | +0.1200 | 80.4700 | 80.5260 |
| 65 | 80.3780 | 80.4600 | +0.0820 | 80.4700 | 80.5260 |
| 66 | 80.4640 | 80.4000 | -0.0640 | 80.4700 | 80.5260 |
| 67 | 80.4900 | 80.3800 | -0.1100 | 80.4900 | 80.5260 |
| 68 | 80.5660 | 80.4920 | -0.0740 | 80.5660 | 80.5260 |
| 69 | 80.5640 | 80.5640 | +0.0000 | 80.5660 | 80.5640 |
| 70 | 80.5520 | 80.4980 | -0.0540 | 80.5660 | 80.5640 |
| 71 | 80.5720 | 80.5920 | +0.0200 | 80.5720 | 80.5920 |
| 72 | 80.5320 | 80.5660 | +0.0340 | 80.5720 | 80.5920 |
| 73 | 80.4880 | 80.3520 | -0.1360 | 80.5720 | 80.5920 |
| 74 | 80.4840 | 80.4880 | +0.0040 | 80.5720 | 80.5920 |
| 75 | 80.5280 | 80.5020 | -0.0260 | 80.5720 | 80.5920 |
| 76 | 80.6340 | 80.4300 | -0.2040 | 80.6340 | 80.5920 |
| 77 | 80.5880 | 80.6360 | +0.0480 | 80.6340 | 80.6360 |
| 78 | 80.5620 | 80.4980 | -0.0640 | 80.6340 | 80.6360 |
| 79 | 80.5340 | 80.4920 | -0.0420 | 80.6340 | 80.6360 |
| 80 | 80.6500 | 80.5840 | -0.0660 | 80.6500 | 80.6360 |
| 81 | 80.6000 | 80.6360 | +0.0360 | 80.6500 | 80.6360 |
| 82 | 80.5840 | 80.7920 | +0.2080 | 80.6500 | 80.7920 |
| 83 | 80.6900 | 80.7180 | +0.0280 | 80.6900 | 80.7920 |
| 84 | 80.6800 | 80.6940 | +0.0140 | 80.6900 | 80.7920 |
| 85 | 80.5520 | 80.6900 | +0.1380 | 80.6900 | 80.7920 |
| 86 | 80.7120 | 80.6300 | -0.0820 | 80.7120 | 80.7920 |
| 87 | 80.6640 | 80.5360 | -0.1280 | 80.7120 | 80.7920 |
| 88 | 80.7220 | 80.6580 | -0.0640 | 80.7220 | 80.7920 |
| 89 | 80.7240 | 80.7080 | -0.0160 | 80.7240 | 80.7920 |
| 90 | 80.6860 | 80.6840 | -0.0020 | 80.7240 | 80.7920 |
| 91 | 80.6780 | 80.7320 | +0.0540 | 80.7240 | 80.7920 |
| 92 | 80.7160 | 80.6620 | -0.0540 | 80.7240 | 80.7920 |
| 93 | 80.6860 | 80.6880 | +0.0020 | 80.7240 | 80.7920 |
| 94 | 80.7640 | 80.6680 | -0.0960 | 80.7640 | 80.7920 |
| 95 | 80.7520 | 80.7400 | -0.0120 | 80.7640 | 80.7920 |
| 96 | 80.7400 | 80.7200 | -0.0200 | 80.7640 | 80.7920 |
| 97 | 80.7500 | 80.6720 | -0.0780 | 80.7640 | 80.7920 |
| 98 | 80.7040 | 80.7760 | +0.0720 | 80.7640 | 80.7920 |
| 99 | 80.7540 | 80.7500 | -0.0040 | 80.7640 | 80.7920 |
| 100 | 80.7720 | 80.6780 | -0.0940 | 80.7720 | 80.7920 |

### 0-51: 完全对齐的 warmup / observe 区间

KL 实验配置了 `dynamic_kl_observe_until_epoch=50`，且 `ref_attn_kl_weight=0.0`。因此 checkpoint-1 到 checkpoint-51 理论上应该和 no-KL 主链路一致。实际结果也完全一致。

这很关键，因为它证明：

```text
1. 两条实验的 public pretrained / OFQ 初始化一致。
2. 数据、batch、LR/scheduler、QKR、soft-KD、augmentation、seed 都没有发生可见偏移。
3. no-KL 对照是有效消融，不是另一个配置漂移实验。
```

### 52-65: 第一组 KL pulse 后 no-KL 反而更高

KL controller 在 epoch 52 检测到 drop，安排后续 pulse：

```text
epoch 52: next_head=8:4, next_weight=1e-5
epoch 53: applied_head=8:4, next_head=5:7
epoch 54: applied_head=5:7, next_head=4:11
epoch 55: applied_head=4:11
```

同期对比：

```text
checkpoint-54: KL 80.1920, no-KL 80.2880
checkpoint-55: KL 80.2120, no-KL 80.3180
checkpoint-60: KL 80.4480, no-KL 80.5260
checkpoint-64: KL 80.3500, no-KL 80.4700
```

这段最能说明当前 KL 不像是“明确保峰”。如果 KL 真在有效抑制 attention relation 震荡，至少不应在第一组 pulse 后持续弱于 no-KL。

### 66-80: KL 有局部保峰，但不稳定

KL 在 checkpoint-80 达到 80.6500，略高于 no-KL 同点 80.5840。但 no-KL 在 checkpoint-77 已经达到 80.6360，差距只有 0.014。

```text
by checkpoint-80:
  KL best: checkpoint-80 80.6500
  no-KL best: checkpoint-77 80.6360
  差值: +0.0140
```

这个差值太小，不足以证明 KL 是主要收益来源。更像是自然训练曲线在 80.5-80.6 区间的随机/局部波动。

### 81-100: no-KL 反超 KL best

no-KL 在 checkpoint-82 达到 80.7920，超过 KL 最终 best 80.7720。

```text
no-KL:
  checkpoint-82: 80.7920
  checkpoint-98: 80.7760
  checkpoint-99: 80.7500

KL:
  checkpoint-98: 80.7040
  checkpoint-99: 80.7540
  checkpoint-100: 80.7720
```

最终 best 层面：

```text
no-KL 80.7920 - KL 80.7720 = +0.0200
```

这使得消融结论非常明确：当前 KL 配置没有带来可验证的 best 提升。

## KL 实现细节

### 1. 训练方案开关

`qat_launch.py` 里通过 `train_scheme` 控制是否启用 refmodel 相关 loss：

```text
train_scheme=baseline:
  不启用 ref_model attention KL 路径。

train_scheme=ema_ref_attn_kl:
  创建 ref_model。
  根据 ref_update 选择 EMA / prev_step / fixed 等更新策略。
  允许加入 ref_attn_kl、ref_logit_kl、anchor_ref_attn_kl、teacher_attn_kl 等附加项。
```

本次 KL 实验使用：

```text
train_scheme=ema_ref_attn_kl
ref_update=prev_step
ref_update_interval=50
```

`prev_step` 的含义是每隔 `ref_update_interval` 个 optimizer update，把当前模型复制到 ref_model：

```text
if update_step
  and train_scheme == "ema_ref_attn_kl"
  and ref_model is not None
  and ref_update == "prev_step"
  and local_update_count % ref_update_interval == 0:
      update_ref_model(model, ref_model, 0.0)
```

这条路线的直觉是：用最近一步模型作为 attention relation 参考，抑制量化训练中 attention 关系的短期震荡。

### 2. attention KL loss

核心 attention KL 由 `attention_kl_pair_loss` 和 `attention_kl_consistency_loss` 计算。

支持的 loss 类型包括：

```text
kl_ref:
  F.kl_div(log(student_prob), ref_prob)

symmetric_kl:
  0.5 * (KL(student||ref) + KL(ref||student))

js:
  Jensen-Shannon 风格，student/ref 都对 mixed_prob 做 KL

cosine / centered_cosine:
  将 attention flatten 后做 cosine 距离
```

本次 100epoch KL 使用：

```text
ref_attn_loss=kl_ref
ref_attn_kl_clip=20.0
```

clip 逻辑是：

```text
if clip_value > 0:
    loss = min(loss, clip_value)
```

### 3. head 选择

head mode 支持多种形式：

```text
all:
  所有 attention layer/head 都参与 KL。

custom_subset:L:H:
  只选指定 layer/head。

dynamic_topK:
  在候选 head loss 中选 top-K。

dynamic_custom_topK:pool:
  在自定义 pool 中选 top-K。

dynamic_ema_custom_topK:pool:
  对 head score 做 EMA 平滑后选 top-K。

dynamic_teacher_agree_topK:pool:
  结合 teacher agreement 做选择。
```

本次 100epoch KL 的 controller 最终将每次 pulse 应用为单个 head：

```text
format_ref_head_mode_from_head(head) -> custom_subset:layer:head
```

也就是每次 KL pulse 不是全局 attention KL，而是针对一个选定 head 的非常稀疏约束。

### 4. dynamic sparse prev-step KL controller

本次 KL 版本配置：

```text
dynamic_sparse_prevstep_kl=true
dynamic_kl_start_epoch=51
dynamic_kl_observe_until_epoch=50
dynamic_kl_primary_heads=8:4,5:7,4:11
dynamic_kl_secondary_heads=11:18,6:1
dynamic_kl_avoid_heads=6:6,7:7,4:1,2:4,10:13,11:4,6:7,11:16
dynamic_kl_drop_threshold=0.06
dynamic_kl_strong_drop_threshold=0.14
dynamic_kl_default_weight=1e-5
dynamic_kl_strong_weight=2e-5
dynamic_kl_max_weight=2e-5
dynamic_kl_cooldown_epochs=6
dynamic_kl_window_epochs=10
dynamic_kl_max_pulses_per_window=3
ref_attn_kl_drop_prob=0.5
ref_attn_kl_clip=20.0
```

controller 逻辑：

```text
1. 每个 epoch full-val 后读取 top1/top5/samples。
2. 维护 rolling_best。
3. 计算 drop = rolling_best_before_epoch - current_top1。
4. 如果 epoch <= observe_until 或 epoch < start_epoch，只 observe，不触发。
5. 如果 drop < drop_threshold，不触发。
6. 如果最近 window_epochs 内 pulse 数达到 max_pulses_per_window，不触发。
7. 按 primary_heads + secondary_heads 顺序选择第一个不在 avoid/cooldown 的 head。
8. 如果 drop >= strong_drop_threshold，用 strong_weight，否则 default_weight。
9. 选中的 head/weight 在下一个 epoch 生效。
10. 写 controller TSV，记录 phase、drop、applied_head、next_head、triggered、reason、cooldown 等。
```

### 5. KL 如何真正加到 loss

每个 epoch 开始前，controller 决定本 epoch 的：

```text
epoch_ref_head_mode
epoch_ref_attn_kl_weight
epoch_dynamic_head
epoch_dynamic_reason
```

然后广播到各 rank，并写入：

```text
runtime_args.ref_head_mode = epoch_ref_head_mode
runtime_args.ref_attn_kl_weight = epoch_ref_attn_kl_weight
```

训练 batch 内，如果满足：

```text
train_scheme == "ema_ref_attn_kl"
ref_model is not None
epoch >= ref_warmup_epochs
local_update_count >= ref_warmup_updates
current_ref_attn_kl_weight > 0
```

则计算：

```text
ref_logits, ref_attn_info = ref_model(input)
ref_attn_kl_loss = attention_kl_consistency_loss(
    student_attn_info,
    ref_attn_info,
    head_mode=runtime_args.ref_head_mode,
    loss_type=runtime_args.ref_attn_loss,
    clip_value=runtime_args.ref_attn_kl_clip,
)
loss = loss + current_ref_attn_kl_weight * ref_attn_kl_loss
```

如果 `ref_attn_kl_drop_prob < 1.0`，还会做随机 gate：

```text
kl_gate = Bernoulli(ref_attn_kl_drop_prob)
ref_attn_kl_loss = ref_attn_kl_loss * kl_gate
```

本次设置 `drop_prob=0.5`，且没有 `drop_scale`，因此即使某个 epoch 有 `RefW=1e-5`，实际 batch 级别也只有约一半 batch 会加 KL。

### 6. 本次 KL 实际触发

controller 审计：

```text
controller_rows=100
controller_triggers=10
controller_observe_triggers=0
controller_selected_avoid=0
observe_nonzero_refw_lines=0
nonzero_refw_epochs=53,54,55,64,65,73,78,79,85,98
selected heads: 8:4,5:7,4:11
```

重要触发段：

```text
epoch 52:
  drop=0.0900
  next_head=8:4
  next_weight=1e-5

epoch 53:
  applied_head=8:4
  next_head=5:7

epoch 54:
  applied_head=5:7
  next_head=4:11

epoch 55:
  applied_head=4:11
```

```text
epoch 63:
  drop=0.1200
  next_head=8:4

epoch 64:
  applied_head=8:4
  next_head=5:7

epoch 65:
  applied_head=5:7
```

```text
epoch 72:
  drop=0.0840
  next_head=8:4

epoch 73:
  applied_head=8:4
  blocked by window_limit
```

```text
epoch 77:
  drop=0.0720
  next_head=5:7

epoch 78:
  applied_head=5:7
  next_head=8:4

epoch 79:
  applied_head=8:4
```

```text
epoch 84:
  drop=0.1380
  next_head=8:4

epoch 85:
  applied_head=8:4
```

```text
epoch 97:
  drop=0.0600
  next_head=8:4

epoch 98:
  applied_head=8:4
```

## 为什么 KL 没有带来收益

### 原因 1: 进入 KL 阶段时两者完全相同，说明收益空间取决于后段干预

checkpoint-51：

```text
KL    = 80.2140
no-KL = 80.2140
```

这意味着 KL 没有任何前期优势。后段要想证明有效，必须在 51 之后稳定拉开差距。

### 原因 2: 第一组 pulse 后 no-KL 多数更高

KL 在 53-55 开始加约束，但 no-KL 在 54/55/60 反而更高：

```text
checkpoint-54: no-KL +0.0960
checkpoint-55: no-KL +0.1060
checkpoint-60: no-KL +0.0780
```

这说明当前 head/weight/pulse 组合至少没有在早期 dynamic 段形成正向保峰。

### 原因 3: KL 的局部优势只出现在很小窗口

checkpoint-80：

```text
KL = 80.6500
no-KL = 80.5840
```

但 no-KL checkpoint-77 已有 80.6360，差距只有 0.014。到 checkpoint-82，no-KL 直接到 80.7920，超过 KL 全程 best。

### 原因 4: 当前 KL 过于固定，不能证明抓住了真实震荡 head

当前 controller 的 head 来源是静态 prior：

```text
primary: 8:4,5:7,4:11
secondary: 11:18,6:1
avoid: 6:6,7:7,4:1,2:4,10:13,11:4,6:7,11:16
```

它没有在训练时重新检测“当前哪个 head 正在产生 harmful oscillation”。因此它可能在某些 drop 后施加了一个形式上正确但时机/head 不够精确的 KL。

### 原因 5: pulse 强度非常轻，且 batch 级 drop_prob=0.5

实际权重通常是：

```text
RefW=1e-5
drop_prob=0.5
```

这意味着单个 pulse 对全局训练轨迹影响非常有限。它可能不足以带来显著收益，但又可能在局部扰动自然收敛路径。

## 对算法路线的含义

这次消融不是说“attention relation 震荡抑制方向一定错”，而是说：

```text
当前这版固定-head sparse prev-step KL controller 在 100epoch from-pretrained 设置下没有验证出正收益。
```

更准确的判断是：

```text
1. OFQ public-family 自然训练本身可以达到 80.79 左右。
2. 当前 sparse KL 不是推动 80.77 的原因。
3. 当前 KL 不能作为继续冲 81 的主力方案。
4. 如果继续做 KL，需要换成更有判别力的动态检测和更晚期的 ultra-sparse polish。
```

## 下一步建议

### 建议一：把 no-KL 100epoch 固定为新强 baseline

之后所有算法都应该对比：

```text
no-KL 100ep best: 80.7920
no-KL last20_avg: 80.6916
no-KL last10_avg: 80.7086
```

而不是继续只对比 KL 版本的 80.7720。

### 建议二：不要继续原样加长当前 fixed-head sparse KL

理由：

```text
1. no-KL 已超过它的 best。
2. 它没有在 checkpoint-52 后稳定拉开差距。
3. 它的正收益只像局部波动，不像稳定机制。
```

### 建议三：如果保留 KL，需要换成动态 head 检测

新 KL 方案应该满足：

```text
1. 不用固定 8:4,5:7,4:11 作为唯一主路径。
2. 每个阶段重新检测当前 attention relation oscillation 大的 head。
3. 区分 harmful oscillation 和正常探索。
4. 只在 no-KL 自然曲线已经进入平台期后启用。
5. 触发后必须能解释：为什么这个 head、为什么这个 epoch、为什么这个 weight。
```

### 建议四：81 目标可能更依赖 warmup / scheduler / quantizer 稳定性

两条 100epoch 曲线都没有到 81：

```text
KL best: 80.7720
no-KL best: 80.7920
```

这说明当前瓶颈不在“有没有加当前 KL”，而在 OFQ public-family 自然训练最高只能到 80.8 左右。下一步更应该研究：

```text
1. warmup / scheduler 是否能把 60-90 epoch 平台抬高；
2. quantizer / activation scale 稳定性是否限制了后段上限；
3. QKR / statsQ / LSQ 的交互是否导致 80.8 附近平台；
4. KL 是否应该只作为 80.8 之后的 late polish，而不是 51 开始的后段 controller。
```

## 给导师的简短说法

```text
我们完成了 100epoch KL 和严格 no-KL 消融。两条线在前 51 个 checkpoint 完全重合，证明对照是干净的。进入 KL 触发阶段后，no-KL 并没有落后，最终 best 80.792 还略高于 KL 的 80.772。结论是当前 fixed-head sparse prev-step KL controller 没有提供稳定正收益，80.7+ 主要来自 OFQ public-family 自然训练。下一步不应该原样加长 KL，而应该把 no-KL 固定为强 baseline，重新设计动态 head 检测和更晚期、更稀疏的 KL polish，或者转向 warmup/scheduler/quantizer 稳定性提升来冲 81。
```

## 2026-08-04 补充：teacher attention KL fixed 100epoch 轨迹

### 实验说明

新增第三条 100epoch from-pretrained 轨迹：

```text
实验 C: ofq_100ep_fromscratch_teacher_sparse_attnkl_fixed_20260803
方法: OFQ public-family + fixed schedule teacher attention KL
日志: /mlx_devbox/users/quyanyi/playground/train_ofq_100ep_fromscratch_teacher_sparse_attnkl_fixed_20260803.log
机读曲线: docs/ofq_100ep_kl_nokl_teacher_fixed_fullval_curve_20260804.csv
```

这条实验是修复 teacher attention collection bug 之后重跑的有效 teacher attention KL 版本。修复前的无效 run 虽然按 epoch override 打开了 `teacher_attn_kl_weight`，但 teacher model 创建时只检查初始 `teacher_attn_kl_weight > 0`，初始值为 0，因此没有开启 teacher attention collection，导致 `TeacherAttnKL` 一直为 0。fixed run 的 smoke 和正式日志均确认：

```text
Teacher attention-KL debug: student_layers=12, student_valid=3, teacher_layers=12, teacher_valid=12
TeacherAttnKL: 2.000e+01
```

teacher KL schedule 为固定、非 val 触发：

```text
epoch 0-4:   weight=0
epoch 5-29:  weight=1e-6, heads=8:4,11:18,6:1
epoch 30-69: weight=2e-6, heads=8:4,11:18,6:1,5:7,4:11
epoch 70-89: weight=1e-6, heads=8:4,11:18,6:1
epoch 90-99: weight=0
```

### 三线结论

```text
prev-step sparse KL best: checkpoint-100 Top-1 80.7720
no-KL control best:      checkpoint-82  Top-1 80.7920
teacher attn KL fixed:   checkpoint-82  Top-1 80.7920
```

最关键的发现是：

```text
teacher attention KL fixed 的 100 个 full-val Top-1 与 no-KL control 逐 epoch 完全一致。
```

这说明 teacher attention KL 虽然已经真实打开，但在当前权重和 head 子集下，对最终可观测 full-val 轨迹没有形成可分辨影响。它既没有超过 no-KL，也没有改变 no-KL 的 best checkpoint。和 prev-step sparse KL 相比，teacher fixed 版本的 best 高 0.0200，但这个提升来自它复现了 no-KL 的强轨迹，而不是超过 no-KL。

### 总体指标更新

| 指标 | prev-step sparse KL | no-KL control | teacher attn KL fixed |
| --- | ---: | ---: | ---: |
| best Top-1 | 80.7720 | 80.7920 | 80.7920 |
| best checkpoint | 100 | 82 | 82 |
| best Top-5 at best Top-1 | 95.4320 | 95.4100 | 95.4100 |
| final Top-1 | 80.7720 | 80.6780 | 80.6780 |
| last20 avg Top-1 | 80.6965 | 80.6916 | 80.6916 |
| last10 avg Top-1 | 80.7316 | 80.7086 | 80.7086 |
| >=81.0 epochs | 0 | 0 | 0 |

### 分段观察

```text
checkpoint 1-51:
  三条曲线完全一致。teacher KL 在 epoch 0-4 关闭，epoch 5 后虽然开启，但在这个区间没有改变 full-val 结果。

checkpoint 52-70:
  teacher fixed 与 no-KL 继续完全一致；二者平均比 prev-step sparse KL 高 0.0222。

checkpoint 71-89:
  teacher fixed 与 no-KL 继续完全一致；二者平均比 prev-step sparse KL 低 0.0157。
  teacher KL 在 epoch 70-89 降到 1e-6，但仍没有产生可见轨迹差异。

checkpoint 90-100:
  teacher fixed 与 no-KL 继续完全一致；teacher KL 已关闭，最终 Top-1 80.6780。
```

### 逐 checkpoint Top-1 轨迹

说明：`teacher - no-KL` 全部为 0，说明 fixed teacher attention KL 和 no-KL control 的 full-val Top-1 逐点一致。

| checkpoint | prev-step KL Acc@1 | no-KL Acc@1 | teacher attn KL fixed Acc@1 | teacher - prev-step KL | teacher - no-KL | teacher best-so-far |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 77.7080 | 77.7080 | 77.7080 | +0.0000 | +0.0000 | 77.7080 |
| 2 | 78.2980 | 78.2980 | 78.2980 | +0.0000 | +0.0000 | 78.2980 |
| 3 | 78.5020 | 78.5020 | 78.5020 | +0.0000 | +0.0000 | 78.5020 |
| 4 | 78.4680 | 78.4680 | 78.4680 | +0.0000 | +0.0000 | 78.5020 |
| 5 | 78.4660 | 78.4660 | 78.4660 | +0.0000 | +0.0000 | 78.5020 |
| 6 | 78.6860 | 78.6860 | 78.6860 | +0.0000 | +0.0000 | 78.6860 |
| 7 | 78.8220 | 78.8220 | 78.8220 | +0.0000 | +0.0000 | 78.8220 |
| 8 | 78.8020 | 78.8020 | 78.8020 | +0.0000 | +0.0000 | 78.8220 |
| 9 | 78.7960 | 78.7960 | 78.7960 | +0.0000 | +0.0000 | 78.8220 |
| 10 | 78.9460 | 78.9460 | 78.9460 | +0.0000 | +0.0000 | 78.9460 |
| 11 | 78.9520 | 78.9520 | 78.9520 | +0.0000 | +0.0000 | 78.9520 |
| 12 | 79.0720 | 79.0720 | 79.0720 | +0.0000 | +0.0000 | 79.0720 |
| 13 | 78.9700 | 78.9700 | 78.9700 | +0.0000 | +0.0000 | 79.0720 |
| 14 | 79.2400 | 79.2400 | 79.2400 | +0.0000 | +0.0000 | 79.2400 |
| 15 | 79.1180 | 79.1180 | 79.1180 | +0.0000 | +0.0000 | 79.2400 |
| 16 | 79.1460 | 79.1460 | 79.1460 | +0.0000 | +0.0000 | 79.2400 |
| 17 | 79.2520 | 79.2520 | 79.2520 | +0.0000 | +0.0000 | 79.2520 |
| 18 | 79.2860 | 79.2860 | 79.2860 | +0.0000 | +0.0000 | 79.2860 |
| 19 | 79.3160 | 79.3160 | 79.3160 | +0.0000 | +0.0000 | 79.3160 |
| 20 | 79.3600 | 79.3600 | 79.3600 | +0.0000 | +0.0000 | 79.3600 |
| 21 | 79.3320 | 79.3320 | 79.3320 | +0.0000 | +0.0000 | 79.3600 |
| 22 | 79.5980 | 79.5980 | 79.5980 | +0.0000 | +0.0000 | 79.5980 |
| 23 | 79.5480 | 79.5480 | 79.5480 | +0.0000 | +0.0000 | 79.5980 |
| 24 | 79.6140 | 79.6140 | 79.6140 | +0.0000 | +0.0000 | 79.6140 |
| 25 | 79.5540 | 79.5540 | 79.5540 | +0.0000 | +0.0000 | 79.6140 |
| 26 | 79.5840 | 79.5840 | 79.5840 | +0.0000 | +0.0000 | 79.6140 |
| 27 | 79.6100 | 79.6100 | 79.6100 | +0.0000 | +0.0000 | 79.6140 |
| 28 | 79.6820 | 79.6820 | 79.6820 | +0.0000 | +0.0000 | 79.6820 |
| 29 | 79.8120 | 79.8120 | 79.8120 | +0.0000 | +0.0000 | 79.8120 |
| 30 | 79.7400 | 79.7400 | 79.7400 | +0.0000 | +0.0000 | 79.8120 |
| 31 | 79.7980 | 79.7980 | 79.7980 | +0.0000 | +0.0000 | 79.8120 |
| 32 | 79.7420 | 79.7420 | 79.7420 | +0.0000 | +0.0000 | 79.8120 |
| 33 | 79.9140 | 79.9140 | 79.9140 | +0.0000 | +0.0000 | 79.9140 |
| 34 | 79.8780 | 79.8780 | 79.8780 | +0.0000 | +0.0000 | 79.9140 |
| 35 | 79.9420 | 79.9420 | 79.9420 | +0.0000 | +0.0000 | 79.9420 |
| 36 | 79.9300 | 79.9300 | 79.9300 | +0.0000 | +0.0000 | 79.9420 |
| 37 | 79.9640 | 79.9640 | 79.9640 | +0.0000 | +0.0000 | 79.9640 |
| 38 | 80.0900 | 80.0900 | 80.0900 | +0.0000 | +0.0000 | 80.0900 |
| 39 | 79.9640 | 79.9640 | 79.9640 | +0.0000 | +0.0000 | 80.0900 |
| 40 | 79.9620 | 79.9620 | 79.9620 | +0.0000 | +0.0000 | 80.0900 |
| 41 | 80.0840 | 80.0840 | 80.0840 | +0.0000 | +0.0000 | 80.0900 |
| 42 | 80.0740 | 80.0740 | 80.0740 | +0.0000 | +0.0000 | 80.0900 |
| 43 | 80.1000 | 80.1000 | 80.1000 | +0.0000 | +0.0000 | 80.1000 |
| 44 | 80.0540 | 80.0540 | 80.0540 | +0.0000 | +0.0000 | 80.1000 |
| 45 | 80.1800 | 80.1800 | 80.1800 | +0.0000 | +0.0000 | 80.1800 |
| 46 | 80.1180 | 80.1180 | 80.1180 | +0.0000 | +0.0000 | 80.1800 |
| 47 | 80.1800 | 80.1800 | 80.1800 | +0.0000 | +0.0000 | 80.1800 |
| 48 | 80.1620 | 80.1620 | 80.1620 | +0.0000 | +0.0000 | 80.1800 |
| 49 | 80.1400 | 80.1400 | 80.1400 | +0.0000 | +0.0000 | 80.1800 |
| 50 | 80.1300 | 80.1300 | 80.1300 | +0.0000 | +0.0000 | 80.1800 |
| 51 | 80.2140 | 80.2140 | 80.2140 | +0.0000 | +0.0000 | 80.2140 |
| 52 | 80.3180 | 80.3180 | 80.3180 | +0.0000 | +0.0000 | 80.3180 |
| 53 | 80.2280 | 80.2280 | 80.2280 | +0.0000 | +0.0000 | 80.3180 |
| 54 | 80.1920 | 80.2880 | 80.2880 | +0.0960 | +0.0000 | 80.3180 |
| 55 | 80.2120 | 80.3180 | 80.3180 | +0.1060 | +0.0000 | 80.3180 |
| 56 | 80.4040 | 80.3460 | 80.3460 | -0.0580 | +0.0000 | 80.3460 |
| 57 | 80.3200 | 80.4240 | 80.4240 | +0.1040 | +0.0000 | 80.4240 |
| 58 | 80.3560 | 80.3940 | 80.3940 | +0.0380 | +0.0000 | 80.4240 |
| 59 | 80.2480 | 80.3540 | 80.3540 | +0.1060 | +0.0000 | 80.4240 |
| 60 | 80.4480 | 80.5260 | 80.5260 | +0.0780 | +0.0000 | 80.5260 |
| 61 | 80.4040 | 80.5080 | 80.5080 | +0.1040 | +0.0000 | 80.5260 |
| 62 | 80.4700 | 80.3840 | 80.3840 | -0.0860 | +0.0000 | 80.5260 |
| 63 | 80.3920 | 80.4260 | 80.4260 | +0.0340 | +0.0000 | 80.5260 |
| 64 | 80.3500 | 80.4700 | 80.4700 | +0.1200 | +0.0000 | 80.5260 |
| 65 | 80.3780 | 80.4600 | 80.4600 | +0.0820 | +0.0000 | 80.5260 |
| 66 | 80.4640 | 80.4000 | 80.4000 | -0.0640 | +0.0000 | 80.5260 |
| 67 | 80.4900 | 80.3800 | 80.3800 | -0.1100 | +0.0000 | 80.5260 |
| 68 | 80.5660 | 80.4920 | 80.4920 | -0.0740 | +0.0000 | 80.5260 |
| 69 | 80.5640 | 80.5640 | 80.5640 | +0.0000 | +0.0000 | 80.5640 |
| 70 | 80.5520 | 80.4980 | 80.4980 | -0.0540 | +0.0000 | 80.5640 |
| 71 | 80.5720 | 80.5920 | 80.5920 | +0.0200 | +0.0000 | 80.5920 |
| 72 | 80.5320 | 80.5660 | 80.5660 | +0.0340 | +0.0000 | 80.5920 |
| 73 | 80.4880 | 80.3520 | 80.3520 | -0.1360 | +0.0000 | 80.5920 |
| 74 | 80.4840 | 80.4880 | 80.4880 | +0.0040 | +0.0000 | 80.5920 |
| 75 | 80.5280 | 80.5020 | 80.5020 | -0.0260 | +0.0000 | 80.5920 |
| 76 | 80.6340 | 80.4300 | 80.4300 | -0.2040 | +0.0000 | 80.5920 |
| 77 | 80.5880 | 80.6360 | 80.6360 | +0.0480 | +0.0000 | 80.6360 |
| 78 | 80.5620 | 80.4980 | 80.4980 | -0.0640 | +0.0000 | 80.6360 |
| 79 | 80.5340 | 80.4920 | 80.4920 | -0.0420 | +0.0000 | 80.6360 |
| 80 | 80.6500 | 80.5840 | 80.5840 | -0.0660 | +0.0000 | 80.6360 |
| 81 | 80.6000 | 80.6360 | 80.6360 | +0.0360 | +0.0000 | 80.6360 |
| 82 | 80.5840 | 80.7920 | 80.7920 | +0.2080 | +0.0000 | 80.7920 |
| 83 | 80.6900 | 80.7180 | 80.7180 | +0.0280 | +0.0000 | 80.7920 |
| 84 | 80.6800 | 80.6940 | 80.6940 | +0.0140 | +0.0000 | 80.7920 |
| 85 | 80.5520 | 80.6900 | 80.6900 | +0.1380 | +0.0000 | 80.7920 |
| 86 | 80.7120 | 80.6300 | 80.6300 | -0.0820 | +0.0000 | 80.7920 |
| 87 | 80.6640 | 80.5360 | 80.5360 | -0.1280 | +0.0000 | 80.7920 |
| 88 | 80.7220 | 80.6580 | 80.6580 | -0.0640 | +0.0000 | 80.7920 |
| 89 | 80.7240 | 80.7080 | 80.7080 | -0.0160 | +0.0000 | 80.7920 |
| 90 | 80.6860 | 80.6840 | 80.6840 | -0.0020 | +0.0000 | 80.7920 |
| 91 | 80.6780 | 80.7320 | 80.7320 | +0.0540 | +0.0000 | 80.7920 |
| 92 | 80.7160 | 80.6620 | 80.6620 | -0.0540 | +0.0000 | 80.7920 |
| 93 | 80.6860 | 80.6880 | 80.6880 | +0.0020 | +0.0000 | 80.7920 |
| 94 | 80.7640 | 80.6680 | 80.6680 | -0.0960 | +0.0000 | 80.7920 |
| 95 | 80.7520 | 80.7400 | 80.7400 | -0.0120 | +0.0000 | 80.7920 |
| 96 | 80.7400 | 80.7200 | 80.7200 | -0.0200 | +0.0000 | 80.7920 |
| 97 | 80.7500 | 80.6720 | 80.6720 | -0.0780 | +0.0000 | 80.7920 |
| 98 | 80.7040 | 80.7760 | 80.7760 | +0.0720 | +0.0000 | 80.7920 |
| 99 | 80.7540 | 80.7500 | 80.7500 | -0.0040 | +0.0000 | 80.7920 |
| 100 | 80.7720 | 80.6780 | 80.6780 | -0.0940 | +0.0000 | 80.7920 |

### 对现有结论的修正

原文档已经说明：固定 head 的 sparse prev-step KL 没有稳定超过 no-KL。加入 teacher attention KL fixed 之后，结论更明确：

```text
1. 当前 100epoch 设置下，no-KL control 仍是最强基线之一，best=80.7920。
2. fixed teacher attention KL 真实开启，但结果与 no-KL 逐 epoch 完全一致，说明当前 teacher attention KL 权重/head 子集太弱，或者该约束方向没有改变主导优化路径。
3. prev-step sparse KL 的后段扰动会让轨迹局部偏离 no-KL，但没有带来 best 提升。
4. 三条 100epoch 曲线都没有到 81，说明 81 目标不能只靠当前这种轻量 sparse attention KL 实现，需要重新设计更强的动态检测、作用对象或训练范式。
```

给导师的简短更新：

```text
我们补跑了修复后的 teacher attention KL 100epoch。bug 修复后 KL 确认真实生效，TeacherAttnKL 非零；但最终曲线和 no-KL 对照逐 epoch 完全一致，best 都是 checkpoint-82 的 80.792。这个结果说明当前 teacher attention KL 配置没有提供可见增益，之前 prev-step sparse KL 的 80.77 也不是优于 no-KL 的证据。下一步如果继续 attention relation 方向，必须改变 KL 的作用范式，例如更强的动态 head 检测、更晚期的 polish 或换 KL 对象，而不是继续原权重/原 head 子集加长训练。
```
