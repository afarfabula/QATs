# OFQ 10->110 dynamic sparse prev-step KL goal submit

## 可提交 goal 短文

完成文档 `/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_goal_submit_20260710.md` 中定义的 10->110 dynamic sparse prev-step KL 实验：在 OFQ public-family 主链路上，从 checkpoint-10 resume 到 checkpoint-110，先实现默认关闭、epoch 61 后才动态触发的 sparse prev-step KL controller，再启动训练并持续轮询记录结果，最终用完整中文实验日志和 TSV 审计是否超过原版 OFQ 10->60 best 80.7240。

## 详细方案入口

主方案文档：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_goal_20260710.md
```

本 submit 文档是提交 goal 时的短入口和执行约束汇总；实现、启动、监控和最终审计均以主方案文档为准。

## 背景结论

原版 OFQ 10->60 已验证 checkpoint-10 resume 长跑不是无效问题：

```text
output: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710
best checkpoint: checkpoint-52
best Top-1: 80.7240
```

方案 C sparse prev-step KL 有正信号但没有超过原版 OFQ best：

```text
output: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709
best checkpoint: checkpoint-54
best Top-1: 80.6820
```

两个 48->60 短门控说明固定 KL pulse 会压低原版 OFQ 的 50-53 自然高点，因此下一版不再做固定早期 pulse，而是把 prev-step KL 改成后段动态稳定器。

## 核心目标

运行一版 `checkpoint-10 -> checkpoint-110` 的 100 个 resumed epoch 实验，验证 dynamic sparse prev-step KL 是否能在保留原版 OFQ 自然高点的基础上，抑制后段 Attention relation 坏震荡和精度回落。

本实验必须聚焦 Attention relation 震荡抑制，使用 prev-step refmodel KL；不做 soup、不做 checkpoint averaging、不做 ensemble、不做 A8->A4。

## 起点和训练范围

起点 checkpoint：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
```

训练范围：

```text
checkpoint-10 -> checkpoint-110
继续训练 100 个 resumed epoch
预期生成 checkpoint-11 到 checkpoint-110
```

实验名：

```text
ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710
```

输出目录：

```text
/tmp/qat_public_repro/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710
```

## 必须保留的 OFQ public-family 主链路

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

启动后必须检查 `args.yaml`，确认这些关键参数真实生效，不能只相信命令行。

## prev-step KL controller 设计

基础 KL 框架：

```text
train_scheme=ema_ref_attn_kl
ref_update=prev_step
ref_update_interval=50
ref_attn_loss=kl_ref
ref_attn_kl_drop_prob=0.50
ref_attn_kl_weight=0.0 by default
```

阶段划分：

```text
epoch 10-60:
  只观察，不主动开 KL。
  不压制原版 OFQ 的自然高点窗口。

epoch 61-110:
  启用 dynamic sparse prev-step KL controller。
  只有当精度相对 rolling best 回落，并且候选 head 命中坏震荡规则时，下一 epoch 开启极稀疏 KL。
```

候选 head：

```text
primary: 8:4
secondary: 5:7,4:11,6:1,11:18
```

避让 head，禁止选择：

```text
6:6,7:7,4:1,2:4,10:13,11:4,6:7,11:16
```

触发规则：

```text
epoch >= 61
rolling_best_acc - current_acc >= 0.06
候选 head 不在 avoid list
候选 head 不在 cooldown
当前 10epoch 窗口 pulse 次数 < 3
```

动作：

```text
下一 epoch 对 top 1 bad head 开 KL
default weight = 1e-5
drop >= 0.12 时可用 2e-5
weight 绝不超过 3e-5
pulse duration = 1 epoch
同一个 head 触发后 cooldown = 5 epoch
每 10 epoch 最多 3 次 pulse
```

## 实现要求

必须在统一入口 `qat_launch.py` 的有效 OFQ runtime 路径中实现 controller，不要只改 `third_party/OFQ/train.py`。已知当前长跑实际走 `qat_launch.py -> invoke_ofq() -> ofq_spawn_entry_unified()`。

实现必须默认关闭，不影响原有 OFQ、方案 C 和其他实验。新增参数需要完整走通 parser、default、runtime config、训练日志和 `args.yaml`。

每个 epoch 开始前应用 controller 上一轮决策：

```text
ref_head_mode
ref_attn_kl_weight
```

每个 epoch full-val 后更新 controller 状态：

```text
rolling_best_acc
current_acc
drop
selected head
weight
cooldown
10epoch window pulse count
trigger reason
```

由于每个 epoch 做实时 512-sample attention probe 成本较高，第一版 controller 可使用离线 head harmful prior + validation drop 作为触发依据，但日志中必须明确记录这一点。

## 日志和产物

中文进度文档：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_progress_20260710.md
```

机器可读表：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_status_20260710.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_controller_20260710.tsv
```

每次轮询至少记录：

```text
checkpoint id
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

GPU 训练必须在真实 worker 环境中跑，不要用无 GPU 的 sandbox 结果判断模型代码。

## 验收标准

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

失败判定：

```text
best <= 80.7240
或 dynamic KL 触发后明显压低高点窗口
或后段没有任何稳定性提升
```

最终审计必须回答：

```text
checkpoint-11 到 checkpoint-110 是否完整生成
full-val 行是否完整且 Samples=50000
epoch 10-60 是否没有主动 KL pulse
epoch 61-110 的 KL 是否只由 controller 触发
avoid heads 是否从未被选中
best checkpoint 是哪个
超过 80.5980、80.6820、80.7240 的 checkpoint 数量
是否达到 81.0
是否值得进入下一轮完整方案
```
