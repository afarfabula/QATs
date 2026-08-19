# OFQ resume48->60 late-only sparse prev-step KL gate 进度记录

## 目标

验证 late-only sparse prev-step KL 是否能在原版 OFQ 的高点窗口附近进一步抬升或稳定精度，为后续完整 10->60 的 50epoch 长跑提供依据。

## 实验设定

```text
experiment: ofq_resume48_to60_lateonly_prevstep_refkl_gate_20260710
start checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-48.pth.tar
target range: checkpoint-49 -> checkpoint-60
expected resumed checkpoints: 12
output: /tmp/qat_public_repro/ofq_resume48_to60_lateonly_prevstep_refkl_gate_20260710
log: /mlx_devbox/users/quyanyi/playground/train_ofq_resume48_to60_lateonly_prevstep_refkl_gate_20260710.log
```

主链路：

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
checkpoint_hist=60
```

late-only sparse prev-step KL：

```text
train_scheme=ema_ref_attn_kl
ref_update=prev_step
ref_update_interval=50
ref_head_mode=custom_subset:6:1,8:4,8:9,11:18,11:4
ref_attn_loss=kl_ref
ref_attn_kl_drop_prob=0.50
ref_attn_kl_weight=0.0
ref_attn_kl_weight_epoch_overrides=49:0.00015,50:0.00020,51:0.00020,52:0.00015,56:0.00010
ref_warmup_epochs=49
```

禁止项：

```text
不使用 soup
不使用 checkpoint averaging
不使用 multi-checkpoint averaging
不使用 ensemble
不使用 A8 -> A4
不启用 28/29、36/37、44/45 早期 pulse
```

## 对照门槛

```text
baseline: 80.5980
方案 C best: 80.6820
原版 OFQ best: 80.7240
target: 81.0
```

门控标准：

```text
强通过: 至少 1 个 checkpoint > 80.7240，或至少 2 个 checkpoint > 80.6820
弱通过: 至少 1 个 checkpoint > 80.6820，且无明显训练异常
失败: 全部 checkpoint < 80.6820，或明显压低原版 OFQ 的 50-53 高点窗口
```

## 2026-07-10 启动准备

脚本：

```text
run script: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume48_to60_lateonly_prevstep_refkl_gate_20260710.sh
monitor script: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/monitor_ofq_resume48_to60_lateonly_prevstep_refkl_gate_20260710.sh
status tsv: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume48_to60_lateonly_prevstep_refkl_gate_status_20260710.tsv
refw tsv: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume48_to60_lateonly_prevstep_refkl_gate_refw_20260710.tsv
summary: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume48_to60_lateonly_prevstep_refkl_gate_monitor_summary_20260710.txt
```

预检查：

```text
checkpoint-48: exists, 329M
teacher checkpoint: exists, 109M
train parquet shards: 294
validation parquet shards: 14
GPU: 8 x NVIDIA H100 80GB HBM3 visible, idle before launch
/tmp free space: 436G
rootfs free space: 8.1G
输出策略: 输出目录放在 /tmp/qat_public_repro，避免占用 rootfs
```

## 2026-07-10 checkpoint-49

epoch 48 已完成，checkpoint-49 已生成：

```text
checkpoint_count: 1
latest_checkpoint: checkpoint-49.pth.tar
```

full ImageNet validation：

```text
checkpoint-49:
Test: [distributed-summary]  Loss: 0.8367  Acc@1: 80.5580  Acc@5: 95.4120  Samples: 50000
```

对比：

```text
checkpoint-49 Top-1: 80.5580
相对 baseline 80.5980: -0.0400
相对方案 C 80.6820: -0.1240
相对原版 OFQ best 80.7240: -0.1660
```

RefW 状态：

```text
epoch 48: RefW=0.000e+00，符合预期，未开 KL
epoch 49 start: RefW=1.500e-04，符合 49:0.00015 pulse 设定
```

阶段性状态：

```text
训练继续运行，进入 epoch 49。
checkpoint-49 未达到门控线；继续观察 checkpoint-50/51/52 高点窗口。
```

## 2026-07-10 checkpoint-50

epoch 49 已完成，checkpoint-50 已生成：

```text
checkpoint_count: 2
latest_checkpoint: checkpoint-50.pth.tar
```

full ImageNet validation：

```text
checkpoint-50:
TrainSummary: epoch=49 updates=2496 avg_step_time=0.322737s samples_per_step=512 samples_per_sec=1586.43
Test: [distributed-summary]  Time: 10.687s  Loss: 0.8323  Acc@1: 80.5500  Acc@5: 95.3960  Samples: 50000
```

对比：

```text
checkpoint-50 Top-1: 80.5500
相对 baseline 80.5980: -0.0480
相对方案 C 80.6820: -0.1320
相对原版 OFQ best 80.7240: -0.1740
相对原版 OFQ checkpoint-50 80.6300: -0.0800
```

RefW 状态：

```text
epoch 49: RefW=1.500e-04，符合 49:0.00015 pulse 设定
epoch 50 start: RefW=2.000e-04，符合 50:0.00020 pulse 设定
nonzero_refw_epochs: 49,50
```

阶段性状态：

```text
训练继续运行，进入 epoch 50。
checkpoint-49/50 均低于 baseline，也低于原版 OFQ 对应高点窗口。
如果 checkpoint-51/52 继续低于原版 OFQ 的 80.6160/80.7240，说明当前 late-only KL 偏强或时序不合适。
```

## 2026-07-10 checkpoint-51

epoch 50 已完成，checkpoint-51 已生成：

```text
checkpoint_count: 3
latest_checkpoint: checkpoint-51.pth.tar
```

full ImageNet validation：

```text
checkpoint-51:
TrainSummary: epoch=50 updates=2496 avg_step_time=0.324480s samples_per_step=512 samples_per_sec=1577.91
Test: [distributed-summary]  Time: 10.460s  Loss: 0.8329  Acc@1: 80.5480  Acc@5: 95.3680  Samples: 50000
```

对比：

```text
checkpoint-51 Top-1: 80.5480
相对 baseline 80.5980: -0.0500
相对方案 C 80.6820: -0.1340
相对原版 OFQ best 80.7240: -0.1760
相对原版 OFQ checkpoint-51 80.6160: -0.0680
```

RefW 状态：

```text
epoch 50: RefW=2.000e-04，符合 50:0.00020 pulse 设定
epoch 51 start: RefW=2.000e-04，符合 51:0.00020 pulse 设定
```

阶段性状态：

```text
训练继续运行，进入 epoch 51。
checkpoint-49/50/51 连续低于 baseline，也连续低于原版 OFQ 对应 checkpoint。
checkpoint-52 是本门控最关键观察点；若仍显著低于原版 OFQ checkpoint-52 80.7240，则可判定当前 late-only KL 偏强或时序不合适。
```

## 2026-07-10 checkpoint-52 / checkpoint-53 与提前停止

epoch 51 已完成，checkpoint-52 已生成：

```text
checkpoint_count: 4
latest_checkpoint: checkpoint-52.pth.tar
```

full ImageNet validation：

```text
checkpoint-52:
TrainSummary: epoch=51 updates=2496 avg_step_time=0.322604s samples_per_step=512 samples_per_sec=1587.09
Test: [distributed-summary]  Time: 10.482s  Loss: 0.8366  Acc@1: 80.5420  Acc@5: 95.3680  Samples: 50000
```

对比：

```text
checkpoint-52 Top-1: 80.5420
相对 baseline 80.5980: -0.0560
相对方案 C 80.6820: -0.1400
相对原版 OFQ best 80.7240: -0.1820
相对原版 OFQ checkpoint-52 80.7240: -0.1820
```

因为门控观察项要求看 checkpoint-52/53，checkpoint-52 后停止主进程，并从本实验 checkpoint-52 继续补跑 1 个 epoch 到 checkpoint-53。补跑使用同一实验目录、同一配置、独立日志：

```text
continue log: /mlx_devbox/users/quyanyi/playground/train_ofq_resume48_to60_lateonly_prevstep_refkl_gate_continue52_to53_20260710.log
resume: /tmp/qat_public_repro/ofq_resume48_to60_lateonly_prevstep_refkl_gate_20260710/checkpoint-52.pth.tar
```

checkpoint-53 full ImageNet validation：

```text
checkpoint-53:
TrainSummary: epoch=52 updates=2496 avg_step_time=0.322370s samples_per_step=512 samples_per_sec=1588.24
Test: [distributed-summary]  Time: 35.751s  Loss: 0.8323  Acc@1: 80.6120  Acc@5: 95.3600  Samples: 50000
```

对比：

```text
checkpoint-53 Top-1: 80.6120
相对 baseline 80.5980: +0.0140
相对方案 C 80.6820: -0.0700
相对原版 OFQ best 80.7240: -0.1120
相对原版 OFQ checkpoint-53 80.6680: -0.0560
```

RefW 状态：

```text
epoch 49: RefW=1.500e-04
epoch 50: RefW=2.000e-04
epoch 51: RefW=2.000e-04
epoch 52: RefW=1.500e-04
epoch 53 start: RefW=0.000e+00
nonzero_refw_epochs: 49,50,51,52
```

## 最终门控审计

目标交付物与证据：

```text
1. 独立实验名和输出目录:
   experiment=ofq_resume48_to60_lateonly_prevstep_refkl_gate_20260710
   output=/tmp/qat_public_repro/ofq_resume48_to60_lateonly_prevstep_refkl_gate_20260710

2. 启动脚本:
   /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume48_to60_lateonly_prevstep_refkl_gate_20260710.sh

3. 监控脚本:
   /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/monitor_ofq_resume48_to60_lateonly_prevstep_refkl_gate_20260710.sh

4. 进度文档:
   /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume48_to60_lateonly_prevstep_refkl_gate_progress_20260710.md

5. 机器可读结果表:
   /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume48_to60_lateonly_prevstep_refkl_gate_status_20260710.tsv

6. RefW 表:
   /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume48_to60_lateonly_prevstep_refkl_gate_refw_20260710.tsv

7. 预检查:
   checkpoint-48 exists, teacher exists, train shards=294, validation shards=14, 8x H100 visible, /tmp free=436G

8. 已生成 checkpoint:
   checkpoint-49, checkpoint-50, checkpoint-51, checkpoint-52, checkpoint-53

9. full-val:
   checkpoint-49 到 checkpoint-53 均为 Test: [distributed-summary] 且 Samples=50000

10. 停止策略:
   checkpoint-52 已明显压低原版高点；补跑 checkpoint-53 后仍低于方案 C 和原版对应点，因此按门控失败提前停止，未继续消耗到 checkpoint-60。
```

最终结果：

```text
best checkpoint: checkpoint-53
best Top-1: 80.6120
best Top-5: 95.3600
delta vs baseline 80.5980: +0.0140
delta vs 方案 C 80.6820: -0.0700
delta vs 原版 OFQ best 80.7240: -0.1120
```

门控判定：

```text
强通过: 否。没有 checkpoint > 80.7240；也没有 2 个 checkpoint > 80.6820。
弱通过: 否。没有 checkpoint > 80.6820。
失败: 是。checkpoint-52/53 明显低于原版 OFQ 的 80.7240/80.6680，高点窗口被压低。
```

结论：

```text
当前 late-only sparse prev-step KL 配方不建议进入完整 10->60 长跑。
49/50/51/52 的连续 pulse 虽然 RefW 生效正确，但明显压低原版 OFQ 的 50-53 高点窗口。
下一版应降低或推迟 KL：优先考虑只保留 52:0.00005 或 52:0.00008 的单 pulse，或改成 53/56 的后置轻 pulse，而不是 49-52 连续施加。
```
