# OFQ resume48->60 single-pulse52 sparse prev-step KL gate 进度记录

## 目标

验证单 pulse `52:0.00005` 的 sparse prev-step KL 是否能避免压低原版 OFQ 的 50-53 高点窗口，并判断是否值得进入完整 10->60 长跑。

## 背景

上一版 late-only sparse prev-step KL 已失败：

```text
checkpoint-49: 80.5580
checkpoint-50: 80.5500
checkpoint-51: 80.5480
checkpoint-52: 80.5420
checkpoint-53: 80.6120
```

失败原因：

```text
49/50/51/52 连续 pulse 虽然 RefW 生效正确，但明显压低原版 OFQ 的自然高点窗口。
原版 OFQ 对应高点:
checkpoint-50: 80.6300
checkpoint-51: 80.6160
checkpoint-52: 80.7240
checkpoint-53: 80.6680
```

## 实验设定

```text
experiment: ofq_resume48_to60_singlepulse52_refkl_gate_20260710
start checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-48.pth.tar
target range: checkpoint-49 -> checkpoint-53 first
conditional range: continue to checkpoint-60 only if checkpoint-52/53 passes gate
output: /tmp/qat_public_repro/ofq_resume48_to60_singlepulse52_refkl_gate_20260710
log: /mlx_devbox/users/quyanyi/playground/train_ofq_resume48_to60_singlepulse52_refkl_gate_20260710.log
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

single-pulse sparse prev-step KL：

```text
train_scheme=ema_ref_attn_kl
ref_update=prev_step
ref_update_interval=50
ref_head_mode=custom_subset:6:1,8:4,8:9,11:18,11:4
ref_attn_loss=kl_ref
ref_attn_kl_drop_prob=0.50
ref_attn_kl_weight=0.0
ref_attn_kl_weight_epoch_overrides=52:0.00005
ref_warmup_epochs=52
```

禁止项：

```text
不使用 49/50/51 连续 pulse
不使用 28/29、36/37、44/45 早期 pulse
不使用 soup
不使用 checkpoint averaging
不使用 multi-checkpoint averaging
不使用 ensemble
不使用 A8 -> A4
```

## 对照门槛

```text
baseline: 80.5980
方案 C best: 80.6820
原版 OFQ best: 80.7240
原版 OFQ checkpoint-52: 80.7240
原版 OFQ checkpoint-53: 80.6680
target: 81.0
```

门控标准：

```text
强通过: checkpoint-52 > 80.7240，或 checkpoint-53 > 80.6680 且 checkpoint-52 不显著低于原版
弱通过: checkpoint-52/53 至少一个 > 80.6820，或 checkpoint-52/53 接近原版窗口且差距 <= 0.03
失败: checkpoint-52 < 80.65，或 checkpoint-53 < 80.62，或 RefW 生效后继续明显压低 50-53 高点窗口
```

## 启动准备

脚本：

```text
run script: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume48_to60_singlepulse52_refkl_gate_20260710.sh
monitor script: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/monitor_ofq_resume48_to60_singlepulse52_refkl_gate_20260710.sh
status tsv: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume48_to60_singlepulse52_refkl_gate_status_20260710.tsv
refw tsv: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume48_to60_singlepulse52_refkl_gate_refw_20260710.tsv
summary: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume48_to60_singlepulse52_refkl_gate_monitor_summary_20260710.txt
```

预检查：

```text
checkpoint-48: exists, 329M
teacher checkpoint: exists, 109M
train parquet shards: 294
validation parquet shards: 14
GPU: 8 x NVIDIA H100 80GB HBM3 visible, idle before launch
/tmp free space: 434G
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
相对原版 OFQ checkpoint-49 80.5140: +0.0440
```

RefW 状态：

```text
epoch 48: RefW=0.000e+00
epoch 49 start: RefW=0.000e+00
符合预期：单 pulse 只在 epoch 52 开启
```

阶段性状态：

```text
训练继续运行，进入 epoch 49。
checkpoint-49 未达到门控线，但未受 KL 影响；继续观察 checkpoint-50/51 的无 KL 高点窗口。
```

## 2026-07-10 checkpoint-50 / checkpoint-51

epoch 49 和 epoch 50 已完成：

```text
checkpoint-50: /tmp/qat_public_repro/ofq_resume48_to60_singlepulse52_refkl_gate_20260710/checkpoint-50.pth.tar
checkpoint-51: /tmp/qat_public_repro/ofq_resume48_to60_singlepulse52_refkl_gate_20260710/checkpoint-51.pth.tar
checkpoint_count: 3
```

full ImageNet validation：

```text
checkpoint-50:
Test: [distributed-summary]  Loss: 0.8337  Acc@1: 80.5360  Acc@5: 95.3680  Samples: 50000

checkpoint-51:
Test: [distributed-summary]  Loss: 0.8341  Acc@1: 80.5940  Acc@5: 95.3800  Samples: 50000
```

对比：

```text
checkpoint-50 Top-1: 80.5360
相对 baseline 80.5980: -0.0620
相对方案 C 80.6820: -0.1460
相对原版 OFQ best 80.7240: -0.1880
相对原版 OFQ checkpoint-50 80.6300: -0.0940

checkpoint-51 Top-1: 80.5940
相对 baseline 80.5980: -0.0040
相对方案 C 80.6820: -0.0880
相对原版 OFQ best 80.7240: -0.1300
相对原版 OFQ checkpoint-51 80.6160: -0.0220
```

RefW 状态：

```text
epoch 49: RefW=0.000e+00
epoch 50: RefW=0.000e+00
epoch 51 start: RefW=0.000e+00
符合预期：单 pulse 只在 epoch 52 开启
```

阶段性状态：

```text
训练继续运行，进入 epoch 51。
checkpoint-50 低于原版对应点较多；checkpoint-51 接近 baseline，但仍低于原版对应点 0.022。
接下来 checkpoint-52 是核心门控点；若 checkpoint-52 < 80.65，则按失败处理。
```

## 2026-07-10 checkpoint-52 / checkpoint-53

epoch 51 和 epoch 52 已完成：

```text
checkpoint-52: /tmp/qat_public_repro/ofq_resume48_to60_singlepulse52_refkl_gate_20260710/checkpoint-52.pth.tar
checkpoint-53: /tmp/qat_public_repro/ofq_resume48_to60_singlepulse52_refkl_gate_20260710/checkpoint-53.pth.tar
checkpoint_count: 5
```

full ImageNet validation：

```text
checkpoint-52:
TrainSummary: epoch=51 updates=2496 avg_step_time=0.231231s samples_per_step=512 samples_per_sec=2214.24
Test: [distributed-summary]  Time: 11.000s  Loss: 0.8381  Acc@1: 80.5700  Acc@5: 95.3720  Samples: 50000

checkpoint-53:
TrainSummary: epoch=52 updates=2496 avg_step_time=0.322633s samples_per_step=512 samples_per_sec=1586.94
Test: [distributed-summary]  Time: 10.438s  Loss: 0.8331  Acc@1: 80.5980  Acc@5: 95.3720  Samples: 50000
```

对比：

```text
checkpoint-52 Top-1: 80.5700
相对 baseline 80.5980: -0.0280
相对方案 C 80.6820: -0.1120
相对原版 OFQ best 80.7240: -0.1540
相对原版 OFQ checkpoint-52 80.7240: -0.1540

checkpoint-53 Top-1: 80.5980
相对 baseline 80.5980: +0.0000
相对方案 C 80.6820: -0.0840
相对原版 OFQ best 80.7240: -0.1260
相对原版 OFQ checkpoint-53 80.6680: -0.0700
```

RefW 状态：

```text
epoch 48: RefW=0.000e+00
epoch 49: RefW=0.000e+00
epoch 50: RefW=0.000e+00
epoch 51: RefW=0.000e+00
epoch 52: RefW=5.000e-05
epoch 53 start: RefW=0.000e+00
nonzero_refw_epochs: 52
```

## 最终门控审计

目标交付物与证据：

```text
1. 独立实验名和输出目录:
   experiment=ofq_resume48_to60_singlepulse52_refkl_gate_20260710
   output=/tmp/qat_public_repro/ofq_resume48_to60_singlepulse52_refkl_gate_20260710

2. 启动脚本:
   /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume48_to60_singlepulse52_refkl_gate_20260710.sh

3. 监控脚本:
   /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/monitor_ofq_resume48_to60_singlepulse52_refkl_gate_20260710.sh

4. 进度文档:
   /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume48_to60_singlepulse52_refkl_gate_progress_20260710.md

5. 机器可读结果表:
   /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume48_to60_singlepulse52_refkl_gate_status_20260710.tsv

6. RefW 表:
   /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume48_to60_singlepulse52_refkl_gate_refw_20260710.tsv

7. 预检查:
   checkpoint-48 exists, teacher exists, train shards=294, validation shards=14, 8x H100 visible, /tmp free=434G

8. 已生成 checkpoint:
   checkpoint-49, checkpoint-50, checkpoint-51, checkpoint-52, checkpoint-53

9. full-val:
   checkpoint-49 到 checkpoint-53 均为 Test: [distributed-summary] 且 Samples=50000

10. RefW:
   只有 epoch 52 出现非零 RefW，符合 single pulse 52:0.00005 设计。

11. 停止策略:
   checkpoint-52=80.5700 < 80.65，且 checkpoint-53=80.5980 < 80.62；
   两个关键门控点均未通过，因此按失败提前停止，未继续到 checkpoint-60。
```

最终结果：

```text
best checkpoint: checkpoint-53
best Top-1: 80.5980
best Top-5: 95.3720
delta vs baseline 80.5980: +0.0000
delta vs 方案 C 80.6820: -0.0840
delta vs 原版 OFQ best 80.7240: -0.1260
```

门控判定：

```text
强通过: 否。checkpoint-52 未超过 80.7240；checkpoint-53 未超过 80.6680。
弱通过: 否。checkpoint-52/53 均未超过 80.6820，也没有接近原版窗口到 0.03 内。
失败: 是。checkpoint-52 < 80.65，checkpoint-53 < 80.62，且 50-53 高点窗口仍明显低于原版 OFQ。
```

结论：

```text
single pulse 52:0.00005 也不建议进入完整 10->60 长跑。
它比上一版连续 pulse 更轻，但仍未恢复原版 OFQ 的 50-53 高点窗口。
更关键的是，49-51 在没有 KL 的情况下也低于原版对应点，说明从 checkpoint-48 重新训练这个短窗口本身存在 run-to-run 波动；而 52 的轻 KL 没有带来补偿增益。
下一步不要继续降低 52 单点 KL；更合理的是后移到高点之后做稳定化，例如只在 53 或 56 开极轻 pulse，或者先做原版 resume48 的重复短门控，确认 run-to-run 波动幅度。
```
