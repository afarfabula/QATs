# OFQ resume10->210 late sparse prev-step KL progress

## 目标

运行一版 `checkpoint-10 -> checkpoint-210` 的 200 resumed epoch 长跑，保留 OFQ public-family 主链路，并在后段使用更稀疏、更晚启动的 prev-step attention relation KL，目标冲击 Top-1 `81.0`。

本实验不是重复 `resume10->110 dynamic sparse prev-step KL` 的原配置。上一轮完整审计已经证明：

```text
original OFQ 10->110 best: checkpoint-102 Top-1 80.7520
dynamic sparse prev-step KL 10->110 best: checkpoint-100 Top-1 80.7600
best 差距: 0.0080

original last20_avg: 80.6519
dynamic last20_avg: 80.6382
original last10_avg: 80.6520
dynamic last10_avg: 80.6316
```

因此旧 controller 不能作为强证据继续直接拉长；新实验要检验的是：在 200 epoch 更长后段中，prev-step KL 是否能作为“自然高点稳定器”，把原版 OFQ 已经能自然达到的 `80.75` 附近峰值继续推到 `81.0`。

## 实验名和路径

```text
experiment: ofq_resume10_to210_late_sparse_prevstep_refkl_20260712
output: /tmp/qat_public_repro/ofq_resume10_to210_late_sparse_prevstep_refkl_20260712
log: /mlx_devbox/users/quyanyi/playground/train_ofq_resume10_to210_late_sparse_prevstep_refkl_20260712.log
run script: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume10_to210_late_sparse_prevstep_refkl_20260712.sh
monitor script: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/monitor_ofq_resume10_to210_late_sparse_prevstep_refkl_20260712.sh
status TSV: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to210_late_sparse_prevstep_refkl_status_20260712.tsv
refw TSV: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to210_late_sparse_prevstep_refkl_refw_20260712.tsv
controller TSV: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to210_late_sparse_prevstep_refkl_controller_20260712.tsv
summary: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to210_late_sparse_prevstep_refkl_monitor_summary_20260712.txt
```

起点 checkpoint：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
```

## 主链路

保持 OFQ public-family：

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
checkpoint_hist=210
epochs=210
scheduler_epochs=210
lr=1.5e-5
min_lr=5e-6
weight_decay=0.0
```

## KL 方案设计

使用 `train_scheme=ema_ref_attn_kl` + `ref_update=prev_step`，但 base KL 权重保持 0，只允许 dynamic controller 触发 sparse pulse。

相对 10->110 dynamic KL 的关键变化：

```text
dynamic_kl_start_epoch: 91
dynamic_kl_observe_until_epoch: 90
drop_threshold: 0.08
strong_drop_threshold: 0.16
default_weight: 1e-5
strong_weight: 2e-5
max_weight: 2e-5
cooldown_epochs: 7
window_epochs: 12
max_pulses_per_window: 2
ref_attn_kl_clip: 20.0
ref_attn_kl_drop_prob: 0.50
```

设计理由：

```text
1. 原版 OFQ 在 90->110 自然达到 80.7520，说明早中期不应过早干预。
2. 旧 dynamic KL 从 61 开始，best 只高 0.0080，且 last20 / last10 低于原版；新方案推迟到 91 后，只做后段稳定器。
3. 旧方案 10 epoch 窗口允许 3 次 pulse；新方案 12 epoch 最多 2 次，降低 KL 过度改变自然收敛路径的风险。
4. 新增 KL clip=20.0，避免单 batch attention KL 极端值把训练局部拉偏。
5. primary heads 改为 8:4,5:7,4:11，把上一版真实选中的三组 head 提升为优先候选；secondary 只保留 11:18,6:1 作为补充。
```

## 对比阈值

```text
baseline: 80.5980
scheme C best: 80.6820
original OFQ 10->60 best: 80.7240
original OFQ 10->110 best: 80.7520
dynamic KL 10->110 best: 80.7600
target: 81.0
```

最低通过：

```text
best Top-1 > 80.7600
```

有效通过：

```text
至少 3 个 checkpoint > 80.7600
或 last20_avg > 80.70
```

强通过：

```text
best Top-1 >= 81.0
```

失败判据：

```text
best <= 80.7600 且 last20_avg <= original 10->110 last20_avg 80.6519
```

## 启动前检查

```text
worker: fdbd:dccd:cdc2:1234:0:b8::, port 9801
GPU: 8 x H100 visible, memory about 7 MiB, util 0
dataset: /tmp/imagenet1k_full_parquet/data, train shards 294, validation shards 14
/tmp free: about 340G
known output dirs of previous 10->110 runs: about 33G each
```

## 2026-07-12 launch

启动命令：

```text
cd /mlx_devbox/users/quyanyi/playground
MASTER_PORT=31841 OUT=/tmp/qat_public_repro nohup bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume10_to210_late_sparse_prevstep_refkl_20260712.sh >/tmp/ofq_resume10_to210_late_sparse_prevstep_refkl_20260712.nohup 2>&1 &
```

进程和 GPU：

```text
launcher pid: 141785
script pid: 141786
qat_launch pid: 141802
GPU 0-7 memory about 28427 MiB, utilization 99-100%
```

启动质量证据：

```text
Strict resume: loaded model from checkpoint-10; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Enabled EMA refmodel attention-KL scheme: ref_update=prev_step, ref_update_interval=50, attn_kl_weight=0.0
Enabled dynamic sparse prev-step KL controller: start_epoch=91, observe_until=90
primary_heads=['8:4', '5:7', '4:11']
secondary_heads=['11:18', '6:1']
drop_threshold=0.08
strong_drop_threshold=0.16
default_weight=1e-05
strong_weight=2e-05
max_weight=2e-05
controller_tsv=/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to210_late_sparse_prevstep_refkl_controller_20260712.tsv
Model swin_t created, param count:28608256
Scheduled epochs: 210
global_effective_batch=512
Train: 10 [0/2502] ... RefW: 0.000e+00
```

args.yaml 关键项：

```text
epochs: 210
train_scheme: ema_ref_attn_kl
dynamic_sparse_prevstep_kl: true
dynamic_kl_start_epoch: 91
dynamic_kl_observe_until_epoch: 90
dynamic_kl_drop_threshold: 0.08
dynamic_kl_strong_drop_threshold: 0.16
dynamic_kl_max_pulses_per_window: 2
dynamic_kl_window_epochs: 12
dynamic_kl_cooldown_epochs: 7
dynamic_kl_default_weight: 1e-05
dynamic_kl_strong_weight: 2e-05
dynamic_kl_max_weight: 2e-05
ref_update: prev_step
ref_attn_kl_weight: 0.0
ref_attn_kl_drop_prob: 0.5
ref_attn_kl_clip: 20.0
ref_head_mode: custom_subset:8:4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
batch_size: 64
checkpoint_hist: 210
epoch_checkpoint_interval: 1
```

启动后 monitor：

```text
output_exists=remote:/tmp/qat_public_repro/ofq_resume10_to210_late_sparse_prevstep_refkl_20260712
args_yaml=present
checkpoint_count=0
fullval_rows=0
nonzero_refw_lines=0
pre91_nonzero_refw_lines=0
controller_rows=0
controller_triggers=0
controller_pre91_triggers=0
```

## 2026-07-12 checkpoint-11

monitor 摘要：

```text
checkpoint_count=1
latest_checkpoint=checkpoint-11.pth.tar
fullval_rows=1
bad_sample_rows=0
best_fullval_line=checkpoint-11 Loss 0.8487 Acc@1 80.2920 Acc@5 95.3400 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original10to60_lines=0
above_original10to110_lines=0
above_dynamic10to110_lines=0
target_81_lines=0
last20_avg=80.2920
last10_avg=80.2920
nonzero_refw_lines=0
pre91_nonzero_refw_lines=0
controller_rows=1
controller_triggers=0
controller_pre91_triggers=0
```

对比 10->110 历史：

```text
original 10->110 checkpoint-11: 80.3360
dynamic 10->110 checkpoint-11: 80.3360
late sparse 10->210 checkpoint-11: 80.2920
delta vs historical aligned point: -0.0440
```

结论：

```text
第一个 resumed epoch 已完成，checkpoint/full-val/Samples/RefW/controller 都正常。
Top-1 比两条 10->110 历史同点低 0.0440，暂不立即干预；可能来自 210 scheduler horizon 改变 LR 曲线或正常波动。
下一步观察 checkpoint-20，如果持续系统性偏低，需要把 scheduler horizon 作为风险记录。
```

## 2026-07-12 checkpoint-20

monitor 摘要：

```text
checkpoint_count=10
latest_checkpoint=checkpoint-20.pth.tar
fullval_rows=10
bad_sample_rows=0
best_fullval_line=checkpoint-17 Loss 0.8428 Acc@1 80.4440 Acc@5 95.2980 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original10to60_lines=0
above_original10to110_lines=0
above_dynamic10to110_lines=0
target_81_lines=0
last20_avg=80.3494
last10_avg=80.3494
nonzero_refw_lines=0
pre91_nonzero_refw_lines=0
controller_rows=10
controller_triggers=0
controller_pre91_triggers=0
```

full-val：

```text
checkpoint-11: Acc@1 80.2920 Acc@5 95.3400 Samples 50000
checkpoint-12: Acc@1 80.3580 Acc@5 95.2660 Samples 50000
checkpoint-13: Acc@1 80.2740 Acc@5 95.2420 Samples 50000
checkpoint-14: Acc@1 80.3440 Acc@5 95.2980 Samples 50000
checkpoint-15: Acc@1 80.2840 Acc@5 95.3420 Samples 50000
checkpoint-16: Acc@1 80.4040 Acc@5 95.2620 Samples 50000
checkpoint-17: Acc@1 80.4440 Acc@5 95.2980 Samples 50000
checkpoint-18: Acc@1 80.3760 Acc@5 95.2540 Samples 50000
checkpoint-19: Acc@1 80.3100 Acc@5 95.3480 Samples 50000
checkpoint-20: Acc@1 80.4080 Acc@5 95.3080 Samples 50000
```

对比 original 10->110 同点：

```text
checkpoint-11: -0.0440
checkpoint-12: -0.0180
checkpoint-13: -0.0720
checkpoint-14: -0.0260
checkpoint-15: -0.0880
checkpoint-16: +0.0240
checkpoint-17: +0.0800
checkpoint-18: +0.0740
checkpoint-19: -0.0260
checkpoint-20: -0.0500
```

结论：

```text
checkpoint-11 到 checkpoint-20 已完整生成，Samples 全 50000，RefW=0，controller 只 observe 不触发。
早期曲线相对 original 10->110 并非单调系统性变差：11-15 偏低，16-18 反而更高，20 低 0.0500。
暂不修改训练；继续观察 checkpoint-30 和 checkpoint-40。如果后续 best/均值持续低于 historical observe 段，再把 210 scheduler horizon 或 runtime path 作为风险项处理。
```

## 2026-07-12 checkpoint-30

monitor 摘要：

```text
checkpoint_count=20
latest_checkpoint=checkpoint-30.pth.tar
fullval_rows=20
bad_sample_rows=0
best_fullval_line=checkpoint-26 Loss 0.8414 Acc@1 80.5480 Acc@5 95.3420 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original10to60_lines=0
above_original10to110_lines=0
above_dynamic10to110_lines=0
target_81_lines=0
last20_avg=80.4002
last10_avg=80.4510
nonzero_refw_lines=0
pre91_nonzero_refw_lines=0
controller_rows=20
controller_triggers=0
controller_pre91_triggers=0
```

checkpoint-21 到 checkpoint-30：

```text
checkpoint-21: Acc@1 80.4980 Acc@5 95.3380 Samples 50000
checkpoint-22: Acc@1 80.5180 Acc@5 95.3000 Samples 50000
checkpoint-23: Acc@1 80.4320 Acc@5 95.3520 Samples 50000
checkpoint-24: Acc@1 80.3440 Acc@5 95.3220 Samples 50000
checkpoint-25: Acc@1 80.4780 Acc@5 95.3100 Samples 50000
checkpoint-26: Acc@1 80.5480 Acc@5 95.3420 Samples 50000
checkpoint-27: Acc@1 80.5280 Acc@5 95.3160 Samples 50000
checkpoint-28: Acc@1 80.4220 Acc@5 95.2900 Samples 50000
checkpoint-29: Acc@1 80.3280 Acc@5 95.3380 Samples 50000
checkpoint-30: Acc@1 80.4140 Acc@5 95.3100 Samples 50000
```

对比 original 10->110 同点：

```text
checkpoint-21: +0.0320
checkpoint-22: -0.0260
checkpoint-23: +0.0600
checkpoint-24: -0.0280
checkpoint-25: +0.0180
checkpoint-26: +0.1680
checkpoint-27: +0.0620
checkpoint-28: -0.0480
checkpoint-29: -0.0320
checkpoint-30: -0.0740
```

结论：

```text
checkpoint-21 到 checkpoint-30 仍是干净 observe 段：RefW=0，controller_pre91_triggers=0，Samples 全 50000。
这一段没有显示 210 scheduler horizon 导致系统性劣化；checkpoint-26/27 明显高于 original 10->110 同点，但 checkpoint-28 到 30 回落。
继续保持训练，不做中途修改；下一段观察 checkpoint-40 是否进入 80.55-80.60 的自然高点区间。
```

## 2026-07-12 checkpoint-41

monitor 摘要：

```text
checkpoint_count=31
latest_checkpoint=checkpoint-41.pth.tar
fullval_rows=31
bad_sample_rows=0
best_fullval_line=checkpoint-35 Loss 0.8375 Acc@1 80.6060 Acc@5 95.3600 Samples 50000
above_baseline_lines=1
above_scheme_c_lines=0
above_original10to60_lines=0
above_original10to110_lines=0
above_dynamic10to110_lines=0
target_81_lines=0
last20_avg=80.4730
last10_avg=80.5036
nonzero_refw_lines=0
pre91_nonzero_refw_lines=0
controller_rows=31
controller_triggers=0
controller_pre91_triggers=0
```

checkpoint-31 到 checkpoint-41：

```text
checkpoint-31: Acc@1 80.4120 Acc@5 95.2920 Samples 50000
checkpoint-32: Acc@1 80.4620 Acc@5 95.3320 Samples 50000
checkpoint-33: Acc@1 80.4760 Acc@5 95.3360 Samples 50000
checkpoint-34: Acc@1 80.4700 Acc@5 95.3680 Samples 50000
checkpoint-35: Acc@1 80.6060 Acc@5 95.3600 Samples 50000
checkpoint-36: Acc@1 80.5440 Acc@5 95.3220 Samples 50000
checkpoint-37: Acc@1 80.4540 Acc@5 95.3540 Samples 50000
checkpoint-38: Acc@1 80.5560 Acc@5 95.3400 Samples 50000
checkpoint-39: Acc@1 80.5500 Acc@5 95.3240 Samples 50000
checkpoint-40: Acc@1 80.3220 Acc@5 95.3000 Samples 50000
checkpoint-41: Acc@1 80.5960 Acc@5 95.3820 Samples 50000
```

对比 original 10->110 同点：

```text
checkpoint-31: +0.0320
checkpoint-32: -0.1360
checkpoint-33: -0.0440
checkpoint-34: +0.0660
checkpoint-35: +0.1560
checkpoint-36: -0.0580
checkpoint-37: -0.0920
checkpoint-38: +0.0520
checkpoint-39: +0.0460
checkpoint-40: -0.1520
checkpoint-41: +0.0680
```

结论：

```text
checkpoint-31 到 checkpoint-41 仍然是干净 observe 段，RefW=0，controller_pre91_triggers=0，Samples 全 50000。
曲线波动较大：checkpoint-40 明显低于历史同点，但 checkpoint-35 达到 80.6060，略高于 original 10->110 checkpoint-36 的 80.6020。
目前没有证据说明训练坏掉；继续保持训练，下一段观察 checkpoint-50 是否稳定站上 80.60，并为 91 后 KL controller 提供足够高的自然 rolling best。
```

## 2026-07-12 checkpoint-50

monitor 摘要：

```text
checkpoint_count=40
latest_checkpoint=checkpoint-50.pth.tar
fullval_rows=40
bad_sample_rows=0
best_fullval_line=checkpoint-35 Loss 0.8375 Acc@1 80.6060 Acc@5 95.3600 Samples 50000
above_baseline_lines=1
above_scheme_c_lines=0
above_original10to60_lines=0
above_original10to110_lines=0
above_dynamic10to110_lines=0
target_81_lines=0
last20_avg=80.5106
last10_avg=80.5360
nonzero_refw_lines=0
pre91_nonzero_refw_lines=0
controller_rows=40
controller_triggers=0
controller_pre91_triggers=0
```

checkpoint-42 到 checkpoint-50：

```text
checkpoint-42: Acc@1 80.4560 Acc@5 95.3860 Samples 50000
checkpoint-43: Acc@1 80.5860 Acc@5 95.3480 Samples 50000
checkpoint-44: Acc@1 80.5680 Acc@5 95.3100 Samples 50000
checkpoint-45: Acc@1 80.5120 Acc@5 95.3420 Samples 50000
checkpoint-46: Acc@1 80.5480 Acc@5 95.3300 Samples 50000
checkpoint-47: Acc@1 80.5060 Acc@5 95.3840 Samples 50000
checkpoint-48: Acc@1 80.5520 Acc@5 95.3520 Samples 50000
checkpoint-49: Acc@1 80.5280 Acc@5 95.3300 Samples 50000
checkpoint-50: Acc@1 80.5080 Acc@5 95.3280 Samples 50000
```

对比 original 10->110 同点：

```text
checkpoint-42: -0.0920
checkpoint-43: +0.0120
checkpoint-44: +0.0720
checkpoint-45: -0.0300
checkpoint-46: +0.0160
checkpoint-47: -0.0720
checkpoint-48: -0.0320
checkpoint-49: -0.0540
checkpoint-50: -0.0720
```

结论：

```text
checkpoint-42 到 checkpoint-50 未刷新 best，当前 best 仍是 checkpoint-35 的 80.6060。
不过 last20_avg 提升到 80.5106，last10_avg 提升到 80.5360，说明曲线在变稳但高点还没打开。
pre91 RefW 仍为 0，controller 无触发，Samples 全 50000。继续跑到 checkpoint-60，看自然高点是否进入 80.62+ 区间。
```

## 2026-07-12 checkpoint-61

monitor 摘要：

```text
checkpoint_count=51
latest_checkpoint=checkpoint-61.pth.tar
fullval_rows=51
bad_sample_rows=0
best_fullval_line=checkpoint-61 Loss 0.8322 Acc@1 80.6360 Acc@5 95.4420 Samples 50000
above_baseline_lines=3
above_scheme_c_lines=0
above_original10to60_lines=0
above_original10to110_lines=0
above_dynamic10to110_lines=0
target_81_lines=0
last20_avg=80.5247
last10_avg=80.5280
nonzero_refw_lines=0
pre91_nonzero_refw_lines=0
controller_rows=51
controller_triggers=0
controller_pre91_triggers=0
```

checkpoint-51 到 checkpoint-61：

```text
checkpoint-51: Acc@1 80.4500 Acc@5 95.3800 Samples 50000
checkpoint-52: Acc@1 80.3860 Acc@5 95.3440 Samples 50000
checkpoint-53: Acc@1 80.4860 Acc@5 95.3340 Samples 50000
checkpoint-54: Acc@1 80.4780 Acc@5 95.3960 Samples 50000
checkpoint-55: Acc@1 80.5020 Acc@5 95.3840 Samples 50000
checkpoint-56: Acc@1 80.5900 Acc@5 95.3100 Samples 50000
checkpoint-57: Acc@1 80.5220 Acc@5 95.3560 Samples 50000
checkpoint-58: Acc@1 80.5220 Acc@5 95.2880 Samples 50000
checkpoint-59: Acc@1 80.6220 Acc@5 95.3420 Samples 50000
checkpoint-60: Acc@1 80.5360 Acc@5 95.3480 Samples 50000
checkpoint-61: Acc@1 80.6360 Acc@5 95.4420 Samples 50000
```

对比 original 10->110 同点：

```text
checkpoint-51: -0.0700
checkpoint-52: -0.1260
checkpoint-53: -0.0440
checkpoint-54: -0.0080
checkpoint-55: +0.0720
checkpoint-56: +0.0780
checkpoint-57: -0.0980
checkpoint-58: -0.0340
checkpoint-59: -0.0240
checkpoint-60: -0.0540
checkpoint-61: +0.1080
```

结论：

```text
checkpoint-51 到 checkpoint-61 仍为干净 observe 段：RefW=0，controller_pre91_triggers=0，Samples 全 50000。
自然高点已经进入 80.62+ 区间，当前 best checkpoint-61 Top-1 80.6360；这比 original 10->110 checkpoint-61 高 0.1080，但仍低于 original 10->110 在 checkpoint-59 的 80.6460。
当前 rolling best 足够作为后续 controller 的自然高点参考，但还没达到 scheme C 80.6820。继续跑到 checkpoint-70，观察是否能自然站上 80.65+。
```

## 2026-07-12 checkpoint-71

monitor 摘要：

```text
checkpoint_count=61
latest_checkpoint=checkpoint-71.pth.tar
fullval_rows=61
bad_sample_rows=0
best_fullval_line=checkpoint-61 Loss 0.8322 Acc@1 80.6360 Acc@5 95.4420 Samples 50000
above_baseline_lines=5
above_scheme_c_lines=0
above_original10to60_lines=0
above_original10to110_lines=0
above_dynamic10to110_lines=0
target_81_lines=0
last20_avg=80.5482
last10_avg=80.5684
nonzero_refw_lines=0
pre91_nonzero_refw_lines=0
controller_rows=61
controller_triggers=0
controller_pre91_triggers=0
```

checkpoint-62 到 checkpoint-71：

```text
checkpoint-62: Acc@1 80.5480 Acc@5 95.3720 Samples 50000
checkpoint-63: Acc@1 80.5280 Acc@5 95.4700 Samples 50000
checkpoint-64: Acc@1 80.5500 Acc@5 95.4100 Samples 50000
checkpoint-65: Acc@1 80.5740 Acc@5 95.3280 Samples 50000
checkpoint-66: Acc@1 80.5980 Acc@5 95.3500 Samples 50000
checkpoint-67: Acc@1 80.6140 Acc@5 95.4040 Samples 50000
checkpoint-68: Acc@1 80.5600 Acc@5 95.3720 Samples 50000
checkpoint-69: Acc@1 80.5440 Acc@5 95.3740 Samples 50000
checkpoint-70: Acc@1 80.5440 Acc@5 95.3600 Samples 50000
checkpoint-71: Acc@1 80.6240 Acc@5 95.3800 Samples 50000
```

对比 original 10->110 同点：

```text
checkpoint-62: -0.0700
checkpoint-63: -0.0220
checkpoint-64: -0.0460
checkpoint-65: -0.0220
checkpoint-66: -0.0400
checkpoint-67: -0.0220
checkpoint-68: -0.1100
checkpoint-69: +0.0580
checkpoint-70: +0.0080
checkpoint-71: +0.0540
```

结论：

```text
checkpoint-62 到 checkpoint-71 仍为干净 observe 段，RefW=0，controller_pre91_triggers=0，Samples 全 50000。
本段未刷新 checkpoint-61 的 best 80.6360，但 last20_avg 提升到 80.5482，last10_avg 提升到 80.5684，说明曲线均值继续回升。
checkpoint-68 明显低于 original 10->110 同点，但 69-71 重新转为持平或更高。继续跑到 checkpoint-80，观察能否重新冲击 80.65+ 自然高点。
```

## 2026-07-12 checkpoint-80

monitor 摘要：

```text
checkpoint_count=70
latest_checkpoint=checkpoint-80.pth.tar
fullval_rows=70
bad_sample_rows=0
best_fullval_line=checkpoint-79 Loss 0.8340 Acc@1 80.7080 Acc@5 95.3680 Samples 50000
above_baseline_lines=9
above_scheme_c_lines=2
above_original10to60_lines=0
above_original10to110_lines=0
above_dynamic10to110_lines=0
target_81_lines=0
last20_avg=80.5901
last10_avg=80.6106
nonzero_refw_lines=0
pre91_nonzero_refw_lines=0
controller_rows=70
controller_triggers=0
controller_pre91_triggers=0
```

checkpoint-72 到 checkpoint-80：

```text
checkpoint-72: Acc@1 80.6940 Acc@5 95.3340 Samples 50000
checkpoint-73: Acc@1 80.5820 Acc@5 95.3760 Samples 50000
checkpoint-74: Acc@1 80.6360 Acc@5 95.3700 Samples 50000
checkpoint-75: Acc@1 80.5060 Acc@5 95.3600 Samples 50000
checkpoint-76: Acc@1 80.6180 Acc@5 95.3520 Samples 50000
checkpoint-77: Acc@1 80.5840 Acc@5 95.3980 Samples 50000
checkpoint-78: Acc@1 80.5700 Acc@5 95.3500 Samples 50000
checkpoint-79: Acc@1 80.7080 Acc@5 95.3680 Samples 50000
checkpoint-80: Acc@1 80.5840 Acc@5 95.3680 Samples 50000
```

对比 original 10->110 同点：

```text
checkpoint-72: +0.0940
checkpoint-73: +0.0060
checkpoint-74: +0.0220
checkpoint-75: +0.0240
checkpoint-76: +0.1460
checkpoint-77: -0.1060
checkpoint-78: +0.0060
checkpoint-79: +0.0120
checkpoint-80: -0.0720
```

结论：

```text
checkpoint-72 到 checkpoint-80 仍为干净 observe 段，RefW=0，controller_pre91_triggers=0，Samples 全 50000。
自然 best 已经提高到 checkpoint-79 Top-1 80.7080，超过 scheme C 80.6820，但仍低于 original 10->60 best 80.7240、original 10->110 best 80.7520、dynamic 10->110 best 80.7600 和 81.0。
这一段说明 210 scheduler horizon 并没有压低自然高点；相反 checkpoint-72/76/79 多数比 original 10->110 同点更高。下一段 checkpoint-81 到 checkpoint-90 是 KL 触发前最后观察窗口，重点看 rolling best 能否自然到 80.72+。
```

## 2026-07-12 checkpoint-91

monitor 摘要：

```text
checkpoint_count=81
latest_checkpoint=checkpoint-91.pth.tar
fullval_rows=81
bad_sample_rows=0
best_fullval_line=checkpoint-79 Loss 0.8340 Acc@1 80.7080 Acc@5 95.3680 Samples 50000
above_baseline_lines=12
above_scheme_c_lines=3
above_original10to60_lines=0
above_original10to110_lines=0
above_dynamic10to110_lines=0
target_81_lines=0
last20_avg=80.5883
last10_avg=80.5754
nonzero_refw_lines=0
pre91_nonzero_refw_lines=0
controller_rows=81
controller_triggers=0
controller_pre91_triggers=0
```

checkpoint-81 到 checkpoint-91：

```text
checkpoint-81: Acc@1 80.5300 Acc@5 95.3340 Samples 50000
checkpoint-82: Acc@1 80.5400 Acc@5 95.3900 Samples 50000
checkpoint-83: Acc@1 80.5420 Acc@5 95.3440 Samples 50000
checkpoint-84: Acc@1 80.5960 Acc@5 95.3840 Samples 50000
checkpoint-85: Acc@1 80.4440 Acc@5 95.4020 Samples 50000
checkpoint-86: Acc@1 80.6240 Acc@5 95.3500 Samples 50000
checkpoint-87: Acc@1 80.5420 Acc@5 95.4280 Samples 50000
checkpoint-88: Acc@1 80.6500 Acc@5 95.3760 Samples 50000
checkpoint-89: Acc@1 80.5860 Acc@5 95.3920 Samples 50000
checkpoint-90: Acc@1 80.7000 Acc@5 95.3960 Samples 50000
checkpoint-91: Acc@1 80.5300 Acc@5 95.3880 Samples 50000
```

对比 original 10->110 同点：

```text
checkpoint-81: -0.0640
checkpoint-82: -0.0420
checkpoint-83: -0.0700
checkpoint-84: +0.0180
checkpoint-85: -0.2000
checkpoint-86: +0.1040
checkpoint-87: -0.0340
checkpoint-88: +0.0880
checkpoint-89: -0.0100
checkpoint-90: +0.0800
checkpoint-91: -0.1100
```

controller 边界：

```text
epoch 88: phase=observe, top1=80.5860, rolling_best=80.7080, drop=0.1220, triggered=0
epoch 89: phase=observe, top1=80.7000, rolling_best=80.7080, drop=0.0080, triggered=0
epoch 90: phase=observe, top1=80.5300, rolling_best=80.7080, drop=0.1780, triggered=0
```

结论：

```text
checkpoint-81 到 checkpoint-91 完成 KL 触发前最后 observe 窗口，RefW=0，controller_pre91_triggers=0，Samples 全 50000。
自然 rolling best 停在 checkpoint-79 的 80.7080，checkpoint-90 也达到 80.7000，但还没有超过 original 10->60 best 80.7240。
注意 controller TSV 中 epoch 90 对应 checkpoint-91 验证后，仍按 observe_only_before_start 处理；真正 dynamic 决策会从后续 epoch 开始。由于 checkpoint-91 回落到 80.5300，下一轮 dynamic controller 很可能检测到相对 rolling best 0.178 的 drop，并触发 strong pulse。继续观察 checkpoint-92 到 checkpoint-100。
```

## 2026-07-12 checkpoint-100

monitor 摘要：

```text
checkpoint_count=90
latest_checkpoint=checkpoint-100.pth.tar
fullval_rows=90
bad_sample_rows=0
best_fullval_line=checkpoint-99 Loss 0.8294 Acc@1 80.8280 Acc@5 95.4240 Samples 50000
above_baseline_lines=18
above_scheme_c_lines=5
above_original10to60_lines=1
above_original10to110_lines=1
above_dynamic10to110_lines=1
target_81_lines=0
last20_avg=80.5966
last10_avg=80.6178
nonzero_refw_lines=100
nonzero_refw_epochs=92,94
pre91_nonzero_refw_lines=0
controller_rows=90
controller_triggers=2
controller_pre91_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7
```

checkpoint-92 到 checkpoint-100：

```text
checkpoint-92: Acc@1 80.5440 Acc@5 95.3620 Samples 50000
checkpoint-93: Acc@1 80.6780 Acc@5 95.4180 Samples 50000
checkpoint-94: Acc@1 80.4660 Acc@5 95.3860 Samples 50000
checkpoint-95: Acc@1 80.6220 Acc@5 95.3520 Samples 50000
checkpoint-96: Acc@1 80.5480 Acc@5 95.4420 Samples 50000
checkpoint-97: Acc@1 80.6500 Acc@5 95.3300 Samples 50000
checkpoint-98: Acc@1 80.6920 Acc@5 95.4140 Samples 50000
checkpoint-99: Acc@1 80.8280 Acc@5 95.4240 Samples 50000
checkpoint-100: Acc@1 80.6200 Acc@5 95.3900 Samples 50000
```

controller / RefW：

```text
epoch 91: dynamic, drop 0.1640, next_head=8:4, next_weight=2e-05, triggered=1
epoch 92: applied_head=8:4, RefW max=2e-05, top1 80.6780, drop 0.0300, no trigger
epoch 93: drop 0.2420, next_head=5:7, next_weight=2e-05, triggered=1
epoch 94: applied_head=5:7, RefW max=2e-05, top1 80.6220, window_limit reached
epoch 95-100: no further pulse because window_limit 2/12 or drop below threshold
avoid heads selected: 0
pre91 RefW lines: 0
```

结论：

```text
checkpoint-99 是本实验当前关键突破：Top-1 80.8280，已经超过 original 10->60 best 80.7240、original 10->110 best 80.7520、dynamic 10->110 best 80.7600。
这说明新的 late sparse prev-step KL 设计相对旧 10->110 dynamic KL 有实质增益，至少 best acc 已经提高到 80.8280。
但目标 81.0 仍未达到，checkpoint-100 又回落到 80.6200，说明高点还不稳定。
controller 行为符合预期：只在 epoch 91 后触发，实际 RefW 只在 epoch 92 / 94 非零，且未选择 avoid heads。下一段观察 checkpoint-101 到 110，看 window limit 解除后是否继续触发并把 80.8280 高点稳定住或推向 81。
```

## 2026-07-12 checkpoint-110

monitor 摘要：

```text
checkpoint_count=100
latest_checkpoint=checkpoint-110.pth.tar
fullval_rows=100
bad_sample_rows=0
best_fullval_line=checkpoint-99 Loss 0.8294 Acc@1 80.8280 Acc@5 95.4240 Samples 50000
above_baseline_lines=22
above_scheme_c_lines=5
above_original10to60_lines=1
above_original10to110_lines=1
above_dynamic10to110_lines=1
target_81_lines=0
last20_avg=80.6014
last10_avg=80.5850
nonzero_refw_lines=200
nonzero_refw_epochs=92,94,105,107
pre91_nonzero_refw_lines=0
controller_rows=100
controller_triggers=4
controller_pre91_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7
```

checkpoint-101 到 checkpoint-110：

```text
checkpoint-101: Acc@1 80.6000 Acc@5 95.4080 Samples 50000
checkpoint-102: Acc@1 80.5780 Acc@5 95.3780 Samples 50000
checkpoint-103: Acc@1 80.5800 Acc@5 95.3880 Samples 50000
checkpoint-104: Acc@1 80.6320 Acc@5 95.3980 Samples 50000
checkpoint-105: Acc@1 80.6700 Acc@5 95.3740 Samples 50000
checkpoint-106: Acc@1 80.5940 Acc@5 95.3680 Samples 50000
checkpoint-107: Acc@1 80.6000 Acc@5 95.4220 Samples 50000
checkpoint-108: Acc@1 80.5660 Acc@5 95.4040 Samples 50000
checkpoint-109: Acc@1 80.4720 Acc@5 95.4200 Samples 50000
checkpoint-110: Acc@1 80.5580 Acc@5 95.3900 Samples 50000
```

controller / RefW：

```text
epoch 104: drop 0.1580, next_head=8:4, next_weight=1e-05, triggered=1
epoch 105: applied_head=8:4, RefW max=1e-05, top1 80.5940
epoch 106: drop 0.2280, next_head=5:7, next_weight=2e-05, triggered=1
epoch 107: applied_head=5:7, RefW max=2e-05, top1 80.5660
epoch 108-110: no further pulse due to window_limit 2/12
avoid heads selected: 0
pre91 RefW lines: 0
```

结论：

```text
checkpoint-101 到 checkpoint-110 没有守住 checkpoint-99 的 80.8280 高点，当前 best 仍为 checkpoint-99。
controller 在 epoch 104/106 再次触发，但对应 epoch 105/107 的 pulse 后 full-val 没有立刻回到 80.8；这一段反而说明 window_limit=2/12 在连续大 drop 时限制了补救频率。
当前结论仍是：新 late sparse prev-step KL 已经显著超过旧 10->110 best，但离 81.0 还有差距，且高点稳定性不足。
继续跑到 checkpoint-120，观察下一轮 cooldown/window 后是否能再次冲高。
```

## 2026-07-13 checkpoint-120

monitor 摘要：

```text
checkpoint_count=110
latest_checkpoint=checkpoint-120.pth.tar
fullval_rows=110
bad_sample_rows=0
best_fullval_line=checkpoint-99 Loss 0.8294 Acc@1 80.8280 Acc@5 95.4240 Samples 50000
above_baseline_lines=27
above_scheme_c_lines=6
above_original10to60_lines=1
above_original10to110_lines=1
above_dynamic10to110_lines=1
target_81_lines=0
last20_avg=80.5936
last10_avg=80.6022
nonzero_refw_lines=285
nonzero_refw_epochs=92,94,105,107,118,120
pre91_nonzero_refw_lines=0
controller_rows=110
controller_triggers=6
controller_pre91_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7
```

checkpoint-111 到 checkpoint-120：

```text
checkpoint-111: Acc@1 80.6780 Acc@5 95.3800 Samples 50000
checkpoint-112: Acc@1 80.5320 Acc@5 95.4620 Samples 50000
checkpoint-113: Acc@1 80.6300 Acc@5 95.3860 Samples 50000
checkpoint-114: Acc@1 80.6420 Acc@5 95.4280 Samples 50000
checkpoint-115: Acc@1 80.6880 Acc@5 95.3700 Samples 50000
checkpoint-116: Acc@1 80.5960 Acc@5 95.3340 Samples 50000
checkpoint-117: Acc@1 80.6020 Acc@5 95.3760 Samples 50000
checkpoint-118: Acc@1 80.5460 Acc@5 95.3820 Samples 50000
checkpoint-119: Acc@1 80.5600 Acc@5 95.3600 Samples 50000
checkpoint-120: Acc@1 80.5480 Acc@5 95.3980 Samples 50000
```

controller / RefW：

```text
epoch 111-116: no pulse because window_limit 2/12
epoch 117: drop 0.2820, next_head=8:4, next_weight=2e-05, triggered=1
epoch 118: applied_head=8:4, RefW max=2e-05, top1 80.5600
epoch 119: drop 0.2800, next_head=5:7, next_weight=2e-05, triggered=1
epoch 120: applied_head=5:7, RefW max=2e-05, checkpoint-120 top1 80.5480
avoid heads selected: 0
pre91 RefW lines: 0
```

结论：

```text
checkpoint-111 到 checkpoint-120 仍未刷新 checkpoint-99 的 80.8280。
window_limit 在 111-116 继续限制补救；117/119 再次触发 8:4/5:7 strong pulse，但到 checkpoint-120 尚未出现正向高点。
当前 KL 方案已经证明能产生 80.8280 的突破，但后续稳定性不足；继续跑到 checkpoint-130，观察 epoch 120 之后的 5:7 pulse 是否有滞后收益，以及下一轮窗口是否能再次冲高。
```

## 2026-07-13 checkpoint-131

monitor 摘要：

```text
checkpoint_count=121
latest_checkpoint=checkpoint-131.pth.tar
fullval_rows=121
bad_sample_rows=0
best_fullval_line=checkpoint-99 Loss 0.8294 Acc@1 80.8280 Acc@5 95.4240 Samples 50000
above_baseline_lines=34
above_scheme_c_lines=9
above_original10to60_lines=2
above_original10to110_lines=2
above_dynamic10to110_lines=2
target_81_lines=0
last20_avg=80.6169
last10_avg=80.6368
nonzero_refw_lines=307
nonzero_refw_epochs=92,94,105,107,118,120,131
pre91_nonzero_refw_lines=0
controller_rows=121
controller_triggers=7
controller_pre91_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7
```

checkpoint-121 到 checkpoint-131：

```text
checkpoint-121: Acc@1 80.6260 Acc@5 95.4120 Samples 50000
checkpoint-122: Acc@1 80.5720 Acc@5 95.4200 Samples 50000
checkpoint-123: Acc@1 80.6160 Acc@5 95.4240 Samples 50000
checkpoint-124: Acc@1 80.6120 Acc@5 95.3900 Samples 50000
checkpoint-125: Acc@1 80.6940 Acc@5 95.4020 Samples 50000
checkpoint-126: Acc@1 80.5520 Acc@5 95.4200 Samples 50000
checkpoint-127: Acc@1 80.8040 Acc@5 95.3740 Samples 50000
checkpoint-128: Acc@1 80.5880 Acc@5 95.4160 Samples 50000
checkpoint-129: Acc@1 80.7180 Acc@5 95.4140 Samples 50000
checkpoint-130: Acc@1 80.5640 Acc@5 95.4220 Samples 50000
checkpoint-131: Acc@1 80.6480 Acc@5 95.4400 Samples 50000
```

controller / RefW：

```text
epoch 121-129: no pulse, mostly window_limit 2/12
epoch 126: top1 80.8040, drop 0.0240, below threshold
epoch 130: drop 0.1800, next_head=8:4, next_weight=2e-05, triggered=1
epoch 131: applied_head=8:4, RefW max=2e-05
avoid heads selected: 0
pre91 RefW lines: 0
```

结论：

```text
checkpoint-121 到 checkpoint-131 出现第二个 80.8+ 点：checkpoint-127 Top-1 80.8040。
这说明 checkpoint-99 的 80.8280 不是完全孤立单点，但仍未达到 81.0，也未刷新 best。
controller 在 epoch 130 再次触发 8:4 strong pulse，checkpoint-131 已开始应用 RefW；下一段 checkpoint-132 到 140 需要观察这次 pulse 是否能带来新的 80.8+ 或更高峰值。
```

## 2026-07-13 checkpoint-140

monitor 摘要：

```text
checkpoint_count=131
latest_checkpoint=checkpoint-141.pth.tar
fullval_rows=130
bad_sample_rows=0
best_fullval_line=checkpoint-99 Loss 0.8294 Acc@1 80.8280 Acc@5 95.4240 Samples 50000
above_baseline_lines=42
above_scheme_c_lines=12
above_original10to60_lines=3
above_original10to110_lines=2
above_dynamic10to110_lines=2
target_81_lines=0
last20_avg=80.6420
last10_avg=80.6494
nonzero_refw_lines=400
nonzero_refw_epochs=92,94,105,107,118,120,131,133
pre91_nonzero_refw_lines=0
controller_rows=131
controller_triggers=8
controller_pre91_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7
```

checkpoint-132 到 checkpoint-140：

```text
checkpoint-132: Acc@1 80.5600 Acc@5 95.4340 Samples 50000
checkpoint-133: Acc@1 80.6960 Acc@5 95.3380 Samples 50000
checkpoint-134: Acc@1 80.6580 Acc@5 95.4360 Samples 50000
checkpoint-135: Acc@1 80.6200 Acc@5 95.4940 Samples 50000
checkpoint-136: Acc@1 80.7160 Acc@5 95.4280 Samples 50000
checkpoint-137: Acc@1 80.7280 Acc@5 95.4260 Samples 50000
checkpoint-138: Acc@1 80.6300 Acc@5 95.4620 Samples 50000
checkpoint-139: Acc@1 80.6260 Acc@5 95.4220 Samples 50000
checkpoint-140: Acc@1 80.6120 Acc@5 95.4140 Samples 50000
```

controller / RefW：

```text
epoch 132: drop 0.1320, next_head=5:7, next_weight=1e-05, triggered=1
epoch 133: applied_head=5:7, RefW max=1e-05, top1 80.6580
epoch 134-140: no further pulse due to window_limit 2/12
avoid heads selected: 0
pre91 RefW lines: 0
```

结论：

```text
checkpoint-132 到 checkpoint-140 没有刷新 checkpoint-99 的 80.8280，但 checkpoint-137 达到 80.7280，超过 original 10->60 best 80.7240。
这一段 last20_avg 提升到 80.6420，说明后段均值在改善；但 high-water mark 仍停在 80.8280，目标 81.0 未达到。
window_limit 仍然是主要控制因素：epoch 132 触发一次 5:7，后续 134-140 持续被限制，不能连续补救。
继续跑到 checkpoint-150，看下一个窗口是否出现第三个 80.8+ 点或刷新 best。
```

## 2026-07-13 checkpoint-150

monitor 摘要：

```text
checkpoint_count=140
latest_checkpoint=checkpoint-150.pth.tar
fullval_rows=140
bad_sample_rows=0
best_fullval_line=checkpoint-99 Loss 0.8294 Acc@1 80.8280 Acc@5 95.4240 Samples 50000
above_baseline_lines=48
above_scheme_c_lines=12
above_original10to60_lines=3
above_original10to110_lines=2
above_dynamic10to110_lines=2
target_81_lines=0
last20_avg=80.6201
last10_avg=80.5908
nonzero_refw_lines=500
nonzero_refw_epochs=92,94,105,107,118,120,131,133,144,146
pre91_nonzero_refw_lines=0
controller_rows=140
controller_triggers=10
controller_pre91_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7
```

checkpoint-141 到 checkpoint-150：

```text
checkpoint-141: Acc@1 80.6700 Acc@5 95.4600 Samples 50000
checkpoint-142: Acc@1 80.6720 Acc@5 95.3880 Samples 50000
checkpoint-143: Acc@1 80.6040 Acc@5 95.4340 Samples 50000
checkpoint-144: Acc@1 80.5620 Acc@5 95.4400 Samples 50000
checkpoint-145: Acc@1 80.4940 Acc@5 95.4340 Samples 50000
checkpoint-146: Acc@1 80.5000 Acc@5 95.4420 Samples 50000
checkpoint-147: Acc@1 80.6620 Acc@5 95.4040 Samples 50000
checkpoint-148: Acc@1 80.6220 Acc@5 95.3580 Samples 50000
checkpoint-149: Acc@1 80.6060 Acc@5 95.4000 Samples 50000
checkpoint-150: Acc@1 80.5160 Acc@5 95.3900 Samples 50000
```

controller / RefW：

```text
epoch 143: drop 0.2660, next_head=8:4, next_weight=2e-05, triggered=1
epoch 144: applied_head=8:4, RefW max=2e-05, top1 80.4940
epoch 145: drop 0.3280, next_head=5:7, next_weight=2e-05, triggered=1
epoch 146: applied_head=5:7, RefW max=2e-05, top1 80.6620
epoch 147-150: no further pulse due to window_limit 2/12
avoid heads selected: 0
pre91 RefW lines: 0
```

结论：

```text
checkpoint-141 到 checkpoint-150 是弱段，没有出现第三个 80.8+，也没有刷新 checkpoint-99 的 80.8280。
本段触发了 8:4 / 5:7 两次 strong pulse，但只把局部从 80.49 拉回到 80.66，未形成新高。
last20_avg 从 checkpoint-140 的 80.6420 回落到 80.6201，last10_avg 回落到 80.5908。
继续跑到 checkpoint-160；如果仍无新高，后续最终审计要明确指出当前 controller 主要制造少数高点但稳定性不足。
```

## 2026-07-13 checkpoint-160

monitor 摘要：

```text
checkpoint_count=150
latest_checkpoint=checkpoint-160.pth.tar
fullval_rows=150
bad_sample_rows=0
best_fullval_line=checkpoint-99 Loss 0.8294 Acc@1 80.8280 Acc@5 95.4240 Samples 50000
above_baseline_lines=57
above_scheme_c_lines=16
above_original10to60_lines=5
above_original10to110_lines=3
above_dynamic10to110_lines=3
target_81_lines=0
last20_avg=80.6235
last10_avg=80.6562
nonzero_refw_lines=600
nonzero_refw_epochs=92,94,105,107,118,120,131,133,144,146,157,159
pre91_nonzero_refw_lines=0
controller_rows=150
controller_triggers=12
controller_pre91_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7
```

checkpoint-151 到 checkpoint-160：

```text
checkpoint-151: Acc@1 80.6040 Acc@5 95.4000 Samples 50000
checkpoint-152: Acc@1 80.6380 Acc@5 95.3980 Samples 50000
checkpoint-153: Acc@1 80.6880 Acc@5 95.3740 Samples 50000
checkpoint-154: Acc@1 80.5660 Acc@5 95.4600 Samples 50000
checkpoint-155: Acc@1 80.6280 Acc@5 95.3960 Samples 50000
checkpoint-156: Acc@1 80.6980 Acc@5 95.3760 Samples 50000
checkpoint-157: Acc@1 80.7280 Acc@5 95.4260 Samples 50000
checkpoint-158: Acc@1 80.6020 Acc@5 95.4140 Samples 50000
checkpoint-159: Acc@1 80.6420 Acc@5 95.3820 Samples 50000
checkpoint-160: Acc@1 80.7680 Acc@5 95.4420 Samples 50000
```

controller / RefW：

```text
epoch 156: drop 0.1000, next_head=8:4, next_weight=1e-05, triggered=1
epoch 157: applied_head=8:4, RefW max=1e-05, top1 80.6020
epoch 158: drop 0.1860, next_head=5:7, next_weight=2e-05, triggered=1
epoch 159: applied_head=5:7, RefW max=2e-05, top1 80.7680
epoch 160: no trigger due to window_limit 2/12
avoid heads selected: 0
pre91 RefW lines: 0
```

结论：

```text
checkpoint-151 到 checkpoint-160 出现第三个超过旧 dynamic 10->110 best 的点：checkpoint-160 Top-1 80.7680。
这说明 late sparse prev-step KL 的高点不是单次偶然，但仍未刷新 checkpoint-99 的 80.8280，也没有达到 81.0。
本段 5:7 pulse 后的 checkpoint-160 有明显正向响应；继续跑到 checkpoint-170，观察下一轮窗口能否继续产生 80.8+ 并提高 high-water mark。
```

## 2026-07-13 checkpoint-171

monitor 摘要：

```text
checkpoint_count=161
latest_checkpoint=checkpoint-171.pth.tar
fullval_rows=161
bad_sample_rows=0
best_fullval_line=checkpoint-99 Loss 0.8294 Acc@1 80.8280 Acc@5 95.4240 Samples 50000
above_baseline_lines=64
above_scheme_c_lines=18
above_original10to60_lines=5
above_original10to110_lines=3
above_dynamic10to110_lines=3
target_81_lines=0
last20_avg=80.6397
last10_avg=80.6318
nonzero_refw_lines=650
nonzero_refw_epochs=92,94,105,107,118,120,131,133,144,146,157,159,170
pre91_nonzero_refw_lines=0
controller_rows=161
controller_triggers=13
controller_pre91_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7
```

checkpoint-161 到 checkpoint-171：

```text
checkpoint-161: Acc@1 80.5180 Acc@5 95.4300 Samples 50000
checkpoint-162: Acc@1 80.7060 Acc@5 95.3920 Samples 50000
checkpoint-163: Acc@1 80.5340 Acc@5 95.4420 Samples 50000
checkpoint-164: Acc@1 80.5780 Acc@5 95.4500 Samples 50000
checkpoint-165: Acc@1 80.6480 Acc@5 95.4260 Samples 50000
checkpoint-166: Acc@1 80.5840 Acc@5 95.3860 Samples 50000
checkpoint-167: Acc@1 80.6600 Acc@5 95.4840 Samples 50000
checkpoint-168: Acc@1 80.6920 Acc@5 95.4020 Samples 50000
checkpoint-169: Acc@1 80.6680 Acc@5 95.4180 Samples 50000
checkpoint-170: Acc@1 80.6340 Acc@5 95.4300 Samples 50000
checkpoint-171: Acc@1 80.6140 Acc@5 95.3860 Samples 50000
```

controller / RefW：

```text
epoch 161-168: no pulse due to window_limit 2/12
epoch 169: drop 0.1940, next_head=8:4, next_weight=2e-05, triggered=1
epoch 170: applied_head=8:4, RefW max=2e-05, top1 80.6140
avoid heads selected: 0
pre91 RefW lines: 0
```

结论：

```text
checkpoint-161 到 checkpoint-171 没有刷新 checkpoint-99 的 80.8280，也没有新的 80.8+。
这一段整体维持在 80.6+，但 high-water mark 没有继续抬升。epoch 169 触发 8:4 strong pulse，checkpoint-170 已应用，后续需要看是否有滞后收益。
继续跑到 checkpoint-180；如果仍无新高，说明 160 之后 controller 更偏向维持均值而非继续推峰值。
```

## 2026-07-13 checkpoint-181

monitor 摘要：

```text
checkpoint_count=171
latest_checkpoint=checkpoint-181.pth.tar
fullval_rows=171
bad_sample_rows=0
best_fullval_line=checkpoint-99 Loss 0.8294 Acc@1 80.8280 Acc@5 95.4240 Samples 50000
above_baseline_lines=72
above_scheme_c_lines=21
above_original10to60_lines=6
above_original10to110_lines=3
above_dynamic10to110_lines=3
target_81_lines=0
last20_avg=80.6403
last10_avg=80.6488
nonzero_refw_lines=700
nonzero_refw_epochs=92,94,105,107,118,120,131,133,144,146,157,159,170,172
pre91_nonzero_refw_lines=0
controller_rows=171
controller_triggers=14
controller_pre91_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7
```

checkpoint-172 到 checkpoint-181：

```text
checkpoint-172: Acc@1 80.7180 Acc@5 95.4400 Samples 50000
checkpoint-173: Acc@1 80.6440 Acc@5 95.4180 Samples 50000
checkpoint-174: Acc@1 80.6820 Acc@5 95.4700 Samples 50000
checkpoint-175: Acc@1 80.6180 Acc@5 95.4000 Samples 50000
checkpoint-176: Acc@1 80.7460 Acc@5 95.3800 Samples 50000
checkpoint-177: Acc@1 80.5480 Acc@5 95.3840 Samples 50000
checkpoint-178: Acc@1 80.6600 Acc@5 95.4240 Samples 50000
checkpoint-179: Acc@1 80.6620 Acc@5 95.3780 Samples 50000
checkpoint-180: Acc@1 80.4940 Acc@5 95.4720 Samples 50000
checkpoint-181: Acc@1 80.7160 Acc@5 95.3820 Samples 50000
```

controller / RefW：

```text
epoch 172: applied_head=5:7, RefW max=1e-05, top1 80.6440
epoch 173-181: no further pulse due to window_limit 2/12
avoid heads selected: 0
pre91 RefW lines: 0
```

结论：

```text
checkpoint-172 到 checkpoint-181 没有刷新 checkpoint-99 的 80.8280，也没有新的 80.8+。
本段多次达到 80.7 左右，但仍没有接近 81.0；window_limit 继续压制连续补救。
last20_avg 为 80.6403，说明后段均值保持在 80.64 左右。继续跑到 checkpoint-190，观察最后 30 个 checkpoint 前是否还有一次高点窗口。
```

## 2026-07-13 checkpoint-191

monitor 摘要：

```text
checkpoint_count=181
latest_checkpoint=checkpoint-191.pth.tar
fullval_rows=181
bad_sample_rows=0
best_fullval_line=checkpoint-99 Loss 0.8294 Acc@1 80.8280 Acc@5 95.4240 Samples 50000
above_baseline_lines=81
above_scheme_c_lines=26
above_original10to60_lines=7
above_original10to110_lines=4
above_dynamic10to110_lines=3
target_81_lines=0
last20_avg=80.6602
last10_avg=80.6716
nonzero_refw_lines=800
nonzero_refw_epochs=92,94,105,107,118,120,131,133,144,146,157,159,170,172,183,185
pre91_nonzero_refw_lines=0
controller_rows=181
controller_triggers=16
controller_pre91_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7
```

checkpoint-182 到 checkpoint-191：

```text
checkpoint-182: Acc@1 80.7020 Acc@5 95.4320 Samples 50000
checkpoint-183: Acc@1 80.6500 Acc@5 95.3580 Samples 50000
checkpoint-184: Acc@1 80.6860 Acc@5 95.4220 Samples 50000
checkpoint-185: Acc@1 80.7060 Acc@5 95.4080 Samples 50000
checkpoint-186: Acc@1 80.6380 Acc@5 95.4620 Samples 50000
checkpoint-187: Acc@1 80.7200 Acc@5 95.4000 Samples 50000
checkpoint-188: Acc@1 80.5920 Acc@5 95.4180 Samples 50000
checkpoint-189: Acc@1 80.6180 Acc@5 95.3920 Samples 50000
checkpoint-190: Acc@1 80.6460 Acc@5 95.4360 Samples 50000
checkpoint-191: Acc@1 80.7580 Acc@5 95.4380 Samples 50000
```

controller / RefW：

```text
epoch 182: drop 0.1780, next_head=8:4, next_weight=2e-05, triggered=1
epoch 183: applied_head=8:4, RefW max=2e-05, top1 80.6860
epoch 184: drop 0.1220, next_head=5:7, next_weight=1e-05, triggered=1
epoch 185: applied_head=5:7, RefW max=1e-05, top1 80.6380
epoch 186-191: no further pulse due to window_limit 2/12 or drop below threshold
avoid heads selected: 0
pre91 RefW lines: 0
```

结论：

```text
checkpoint-182 到 checkpoint-191 没有刷新 checkpoint-99 的 80.8280，也没有达到 81.0。
但本段高点稳定性明显变好：checkpoint-182/185/187/191 都在 80.70+，last20_avg=80.6602，last10_avg=80.6716，均为当前后段较高水平。
checkpoint-191 达到 80.7580，几乎追平旧 dynamic 10->110 best 80.7600，但仍低于本实验 best。
进入最后 20 个 checkpoint，重点看能否在 192-210 中再出现 80.8+ 或接近 81.0；如果没有，最终结论应是“best 提升到 80.8280，但未达 81，后段均值有改善而高点不足”。
```

## 2026-07-13 checkpoint-202

monitor 摘要：

```text
checkpoint_count=192
latest_checkpoint=checkpoint-202.pth.tar
fullval_rows=192
bad_sample_rows=0
best_fullval_line=checkpoint-99 Loss 0.8294 Acc@1 80.8280 Acc@5 95.4240 Samples 50000
above_baseline_lines=89
above_scheme_c_lines=30
above_original10to60_lines=9
above_original10to110_lines=6
above_dynamic10to110_lines=4
target_81_lines=0
last20_avg=80.6597
last10_avg=80.6578
nonzero_refw_lines=900
nonzero_refw_epochs=92,94,105,107,118,120,131,133,144,146,157,159,170,172,183,185,196,198
pre91_nonzero_refw_lines=0
controller_rows=192
controller_triggers=18
controller_pre91_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7
```

checkpoint-192 到 checkpoint-202：

```text
checkpoint-192: Acc@1 80.6020 Acc@5 95.3840 Samples 50000
checkpoint-193: Acc@1 80.6820 Acc@5 95.3940 Samples 50000
checkpoint-194: Acc@1 80.6860 Acc@5 95.3940 Samples 50000
checkpoint-195: Acc@1 80.7820 Acc@5 95.3920 Samples 50000
checkpoint-196: Acc@1 80.5840 Acc@5 95.4620 Samples 50000
checkpoint-197: Acc@1 80.6620 Acc@5 95.4580 Samples 50000
checkpoint-198: Acc@1 80.6640 Acc@5 95.4440 Samples 50000
checkpoint-199: Acc@1 80.7100 Acc@5 95.4300 Samples 50000
checkpoint-200: Acc@1 80.5120 Acc@5 95.4360 Samples 50000
checkpoint-201: Acc@1 80.7600 Acc@5 95.4140 Samples 50000
checkpoint-202: Acc@1 80.5360 Acc@5 95.4940 Samples 50000
```

controller / RefW：

```text
epoch 195: drop 0.2440, next_head=8:4, next_weight=2e-05, triggered=1
epoch 196: applied_head=8:4, RefW max=2e-05, top1 80.6620
epoch 197: drop 0.1640, next_head=5:7, next_weight=2e-05, triggered=1
epoch 198: applied_head=5:7, RefW max=2e-05, top1 80.7100
epoch 199-202: no further pulse due to window_limit 2/12 or drop below threshold
avoid heads selected: 0
pre91 RefW lines: 0
```

结论：

```text
checkpoint-192 到 checkpoint-202 没有刷新 checkpoint-99 的 80.8280，也没有达到 81.0。
本段有 checkpoint-195 Top-1 80.7820 和 checkpoint-201 Top-1 80.7600，但都低于本实验 best。
最终 8 个 checkpoint 将决定是否出现最后一次 80.8+ 或 81.0；如果没有，最终结论基本确定为 best 80.8280、未达 81，但显著超过所有 10->110 历史 best。
```

## 2026-07-13 final audit

最终 monitor 摘要：

```text
checkpoint_count=200
latest_checkpoint=checkpoint-210.pth.tar
fullval_rows=200
bad_sample_rows=0
best_fullval_line=checkpoint-99 Loss 0.8294 Acc@1 80.8280 Acc@5 95.4240 Samples 50000
above_baseline_lines=96
above_scheme_c_lines=33
above_original10to60_lines=10
above_original10to110_lines=7
above_dynamic10to110_lines=4
target_81_lines=0
last20_avg=80.6592
last10_avg=80.6542
nonzero_refw_lines=950
nonzero_refw_epochs=92,94,105,107,118,120,131,133,144,146,157,159,170,172,183,185,196,198,209
pre91_nonzero_refw_lines=0
controller_rows=200
controller_triggers=19
controller_pre91_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7
```

checkpoint-203 到 checkpoint-210：

```text
checkpoint-203: Acc@1 80.6100 Acc@5 95.4500 Samples 50000
checkpoint-204: Acc@1 80.5820 Acc@5 95.4000 Samples 50000
checkpoint-205: Acc@1 80.6900 Acc@5 95.4180 Samples 50000
checkpoint-206: Acc@1 80.7000 Acc@5 95.4820 Samples 50000
checkpoint-207: Acc@1 80.7540 Acc@5 95.4160 Samples 50000
checkpoint-208: Acc@1 80.6640 Acc@5 95.4320 Samples 50000
checkpoint-209: Acc@1 80.6320 Acc@5 95.4660 Samples 50000
checkpoint-210: Acc@1 80.6140 Acc@5 95.3940 Samples 50000
```

最终统计：

```text
rows=200
bad_sample_rows=0
best=checkpoint-99
best_loss=0.8294
best_top1=80.8280
best_top5=95.4240
above_baseline_80.5980=96
above_scheme_c_80.6820=33
above_original10to60_80.7240=10
above_original10to110_80.7520=7
above_dynamic10to110_80.7600=4
target_81=0
last20_avg=80.6592
last10_avg=80.6542
```

完成性审计：

```text
checkpoint-11 到 checkpoint-210: remote count=200, no missing checkpoint printed
full-val rows: 200
Samples=50000: all rows, bad_sample_rows=0
args.yaml: present
epochs: 210
train_scheme: ema_ref_attn_kl
dynamic_sparse_prevstep_kl: true
dynamic_kl_start_epoch: 91
dynamic_kl_observe_until_epoch: 90
dynamic_kl_drop_threshold: 0.08
dynamic_kl_strong_drop_threshold: 0.16
dynamic_kl_max_pulses_per_window: 2
dynamic_kl_window_epochs: 12
dynamic_kl_cooldown_epochs: 7
dynamic_kl_default_weight: 1e-05
dynamic_kl_strong_weight: 2e-05
dynamic_kl_max_weight: 2e-05
ref_update: prev_step
ref_attn_kl_weight: 0.0
ref_attn_kl_drop_prob: 0.5
ref_attn_kl_clip: 20.0
ref_head_mode: custom_subset:8:4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
batch_size: 64
checkpoint_hist: 210
epoch_checkpoint_interval: 1
```

controller 审计：

```text
controller_rows=200
controller_triggers=19
controller_pre91_triggers=0
controller_selected_avoid=0
pre91_nonzero_refw_lines=0
nonzero_refw_lines=950
nonzero_refw_epochs=92,94,105,107,118,120,131,133,144,146,157,159,170,172,183,185,196,198,209
selected heads: 8:4,5:7
avoid heads selected: 0
```

最终结论：

```text
本实验完成 checkpoint-10 -> checkpoint-210 的 200 resumed epoch late sparse prev-step KL 长跑。

目标 81.0 没有达到：
best Top-1 = 80.8280
target_81_lines = 0

但相比所有 10->110 历史 best，本实验有明确提升：
original OFQ 10->110 best = 80.7520
dynamic sparse prev-step KL 10->110 best = 80.7600
本实验 best = 80.8280

超过历史 dynamic best 80.7600 的 checkpoint 数量为 4，分别包括 checkpoint-99 / checkpoint-127 / checkpoint-160 / checkpoint-195 等窗口中的高点。
超过 original 10->110 best 80.7520 的 checkpoint 数量为 7。
超过 original 10->60 best 80.7240 的 checkpoint 数量为 10。

后段均值：
last20_avg = 80.6592
last10_avg = 80.6542

结论上，这个 late sparse prev-step KL 方案是有效的：它把 best 从旧动态 KL 的 80.7600 推到 80.8280，并多次复现 80.76+ 高点。
但它还不是 81.0 方案：主要问题是高点不稳定，window_limit=2/12 下连续大 drop 时不能持续补救，导致 best 停在 80.8280。

如果继续做下一轮，方向应是改 controller，而不是重新证明 prev-step KL 是否有效：
1. 在 90 后保留 late-start，不要回到 61 过早触发。
2. 放宽或动态调整 window_limit，例如接近 80.8 后允许更密集的小权重 pulse。
3. 对 80.75+ 后的回落设计保峰策略，而不只是基于 rolling-best drop 被动触发。
4. 继续避免 soup / checkpoint averaging / A8->A4；本实验是单 checkpoint 训练链路。
```

## 轮询关注

```text
checkpoint_count / latest_checkpoint
fullval_rows / bad_sample_rows
best_fullval_line
above_baseline_lines
above_scheme_c_lines
above_original10to60_lines
above_original10to110_lines
above_dynamic10to110_lines
target_81_lines
last20_avg / last10_avg
nonzero_refw_lines / nonzero_refw_epochs
pre91_nonzero_refw_lines
controller_rows / controller_triggers
controller_pre91_triggers
controller_selected_avoid
controller_next_heads
```

## 完成审计清单

最终必须确认：

```text
checkpoint-11 到 checkpoint-210 是否完整生成
full-val rows 是否为 200 且 Samples=50000
epoch <= 90 是否没有主动 KL pulse
epoch >= 91 的 KL 是否只由 controller 触发
avoid heads 是否从未被选中
args.yaml 是否保持预期配置
best checkpoint / Top-1 / Top-5
超过 80.5980 / 80.6820 / 80.7240 / 80.7520 / 80.7600 / 81.0 的 checkpoint 数量
last20_avg / last10_avg
与 original OFQ 10->110 和 dynamic KL 10->110 的最终对比
```
