# OFQ 10->110 dynamic sparse prev-step KL goal

## 目标

运行一版 `checkpoint-10 -> checkpoint-110` 的 100 个 resumed epoch 实验，验证 dynamic sparse prev-step KL 是否能在保留原版 OFQ 自然高点的基础上，抑制后段 Attention relation 坏震荡和精度回落。

本实验不再做固定早期 pulse，不再压制原版 OFQ 的 `50-53` 自然高点窗口。prev-step KL 只作为后段动态稳定器，在检测到精度回落和候选 head 坏震荡时极稀疏触发。

## 实验背景

### 原版 OFQ 10->60

```text
best checkpoint: checkpoint-52
best Top-1: 80.7240
baseline: 80.5980
delta_vs_baseline: +0.1260
方案 C best: 80.6820
delta_vs_scheme_c: +0.0420
81.0 target: 未达到
```

原版 OFQ 已经证明 `checkpoint-10` resume 长跑可以超过 baseline 和方案 C，但 `checkpoint-52` 是峰值，后续 `checkpoint-53` 到 `checkpoint-60` 多数回落到方案 C 以下，不是稳定平台。

### 方案 C sparse prev-step KL

```text
best checkpoint: checkpoint-54
best Top-1: 80.6820
delta_vs_baseline: +0.0840
81.0 target: 未达到
```

方案 C 说明 sparse prev-step KL 有正信号，但没有超过原版 OFQ best。它的有效收益主要来自最后一组 `52/53` pulse，早期 `28/29`、`36/37`、`44/45` pulse 贡献不稳定。

### 两个短门控失败结论

连续 late pulse：

```text
checkpoint-49: 80.5580
checkpoint-50: 80.5500
checkpoint-51: 80.5480
checkpoint-52: 80.5420
checkpoint-53: 80.6120
```

单 pulse `52:0.00005`：

```text
checkpoint-49: 80.5580
checkpoint-50: 80.5360
checkpoint-51: 80.5940
checkpoint-52: 80.5700
checkpoint-53: 80.5980
```

结论：

```text
固定 KL pulse 即使很轻，也容易压低原版 OFQ 的 50-53 高点窗口。
下一版不能继续固定压 49-52，也不应在自然峰值形成前强行约束 Attention relation。
```

### 离线 Attention relation 震荡检测

分析目录：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/attn_relation_oscillation_analysis_20260710
```

使用数据：

```text
original checkpoint-48 到 checkpoint-60
scheme_c checkpoint-48 到 checkpoint-60
calibration samples: 512
```

推荐 KL heads：

```text
custom_subset:5:7,4:11,8:4,1:2,3:1
```

避让 heads：

```text
custom_subset:6:6,7:7,4:1,2:4,10:13,11:4,6:7,11:16
```

解释：

```text
8:4 是最可信的持续震荡候选。
5:7 和 4:11 更像 spike 型异常候选，只适合动态短促触发。
6:6、7:7、4:1、2:4、10:13、11:4、6:7、11:16 更像有益漂移或不应压制的 head，应进入 avoid/cooldown。
```

## 实验设定

### 起点

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
```

### 训练范围

```text
checkpoint-10 -> checkpoint-110
继续训练 100 个 resumed epoch
预期生成 checkpoint-11 到 checkpoint-110
```

### 原版 OFQ public-family 主链路

保持：

```text
method=ofq
wq_mode=statsq
aq_mode=lsq
qk_reparam=true
qk_reparam_type=0
kd_hard_and_soft=0
teacher_soft_temperature=2.75
no_resume_opt=true
batch_size=64
epoch_checkpoint_interval=1
checkpoint_hist=110
```

禁止：

```text
不使用 A8 -> A4
不使用 soup
不使用 checkpoint averaging
不使用 multi-checkpoint averaging
不使用 ensemble
```

## prev-step KL 框架

基础配置：

```text
train_scheme=ema_ref_attn_kl
ref_update=prev_step
ref_update_interval=50
ref_attn_loss=kl_ref
ref_attn_kl_drop_prob=0.50
ref_attn_kl_weight=0.0 by default
```

注意：默认 KL 关闭，只有 dynamic controller 明确触发时才开启。

## 核心策略

### 阶段划分

```text
epoch 10-60:
  只观察，不主动开 KL。
  不压制原版 OFQ 的自然高点窗口。
  记录 checkpoint full-val 和 attention relation 震荡。

epoch 61-110:
  启用 dynamic sparse prev-step KL controller。
  只有当精度相对 rolling best 回落，且候选 head 出现坏震荡 spike 时，下一 epoch 开启极稀疏 KL。
```

### 候选 head 池

Primary harmful candidate：

```text
8:4
```

Secondary spike candidates：

```text
5:7
4:11
6:1
11:18
```

Avoid / blacklist heads：

```text
6:6
7:7
4:1
2:4
10:13
11:4
6:7
11:16
```

### 动态触发规则

每个 epoch full-val 后更新：

```text
rolling_best_acc
current_acc
drop = rolling_best_acc - current_acc
每个候选 head 的 attention relation spike score
每个候选 head 的 cooldown 状态
当前 10epoch 窗口内 pulse 次数
```

触发条件：

```text
epoch >= 61
drop >= 0.06
候选 head 的 spike score 明显高于历史均值
该 head 不在 avoid list
该 head 不在 cooldown
当前 10epoch 窗口内 pulse 次数 < 3
```

动作：

```text
下一 epoch 对 top 1 bad head 开 KL
default weight = 1e-5
如果 drop >= 0.12 且 spike 极强，可用 weight = 2e-5
weight 绝不超过 3e-5
pulse duration = 1 epoch
同一个 head 触发后 cooldown = 5 epoch
每 10 epoch 最多 3 次 pulse
```

## 实现要求

1. 检查现有 `qat_launch.py` / OFQ train loop 是否支持动态修改：

```text
ref_head_mode
ref_attn_kl_weight
```

2. 如果不支持，新增轻量 controller：

```text
每个 epoch full-val 后，根据 rolling best、current_acc、drop、head spike score 决定下一 epoch:
  ref_attn_kl_weight
  ref_head_mode

controller 可写 JSON schedule。
下一 epoch 读取 JSON schedule。
避免大改训练主逻辑。
```

3. 默认 KL 关闭：

```text
ref_attn_kl_weight=0.0
```

4. 每次触发必须在日志中打印：

```text
epoch
selected head
weight
trigger reason
drop
spike score
cooldown state
10epoch window pulse count
```

5. 训练时持续记录：

```text
checkpoint id
Loss
Top-1
Top-5
Samples
RefW
selected KL head
trigger reason
delta_vs_baseline_80.5980
delta_vs_scheme_c_80.6820
delta_vs_original_80.7240
```

6. 维护中文进度文档和机器可读 TSV。

## 推荐实验名与路径

Experiment：

```text
ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710
```

输出目录：

```text
/tmp/qat_public_repro/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710
```

进度文档：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_progress_20260710.md
```

机器表：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_status_20260710.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_controller_20260710.tsv
```

## 启动前检查

必须确认：

```text
checkpoint-10 存在
ImageNet parquet 数据存在
teacher checkpoint 存在
GPU worker 8 卡可用
输出目录空间充足
experiment 名称和输出目录独立
不会覆盖原版 OFQ、方案 C、短门控实验
```

## 成功标准

最低通过：

```text
best Top-1 > 原版 OFQ best 80.7240
```

有效通过：

```text
至少 2 个 checkpoint > 80.7240
或至少 5 个 checkpoint > 方案 C 80.6820
```

强通过：

```text
best Top-1 >= 80.85
且最后 20 个 checkpoint 的均值高于原版 OFQ 后段均值
```

失败：

```text
best <= 80.7240
或 dynamic KL 触发后明显压低高点窗口
或后段没有任何稳定性提升
```

## 完成审计

最终必须确认：

```text
checkpoint-11 到 checkpoint-110 是否完整生成
full-val 行是否完整且 Samples=50000
epoch 10-60 没有主动 KL pulse
epoch 61-110 的 KL 只由 controller 触发
所有触发记录都有 selected head / weight / reason / drop / spike score / cooldown
avoid heads 没有被选中
best checkpoint
超过 80.5980、80.6820、80.7240 的 checkpoint 数量
是否达到 81.0
对比原版 OFQ、方案 C 和 81.0 目标
给出是否进入下一轮完整方案的结论
```

## 设计原则总结

```text
原版 OFQ 负责产生自然峰值。
prev-step sparse KL 只负责在峰值后抑制坏震荡和回落。
不再试图用 KL 提前制造峰值。
不再固定压制 49-52。
只动态约束少量坏震荡 head，并显式避让有益漂移 head。
```
