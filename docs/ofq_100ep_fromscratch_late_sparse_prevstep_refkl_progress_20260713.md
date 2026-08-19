# OFQ 100ep from-scratch late sparse prev-step KL progress

## 目标

执行一版 Swin-T W4A4-family / OFQ public-family 的 100 epoch 重头训练实验，使用 late sparse prev-step attention KL，目标冲击 Top-1 81.0。

这里的“重头训练”指：

```text
从 ImageNet pretrained / public OFQ 初始化开始
不从 checkpoint-10 或其他 QAT checkpoint resume
不是随机初始化
```

## 历史依据

已有关键结果：

```text
original OFQ 10->110 best: 80.7520
dynamic sparse prev-step KL 10->110 best: 80.7600
late sparse prev-step KL 10->210 best: 80.8280
late sparse prev-step KL 10->210 target_81: 0
```

结论：

```text
prev-step KL 有效，但旧 controller 未达到 81.0。
新实验用 100 epoch from-scratch/pretrained-init 范式验证：更早进入 KL 主阶段、更积极保峰，是否能超过 80.8280 并冲击 81.0。
```

## 实验名和路径

```text
experiment: ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713
output: /tmp/qat_public_repro/ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713
log: /mlx_devbox/users/quyanyi/playground/train_ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713.log
run script: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713.sh
monitor script: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/monitor_ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713.sh
status TSV: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_100ep_fromscratch_late_sparse_prevstep_refkl_status_20260713.tsv
refw TSV: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_100ep_fromscratch_late_sparse_prevstep_refkl_refw_20260713.tsv
controller TSV: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_100ep_fromscratch_late_sparse_prevstep_refkl_controller_20260713.tsv
summary: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_100ep_fromscratch_late_sparse_prevstep_refkl_monitor_summary_20260713.txt
```

## 主链路

```text
method=ofq
model=swin_t
data=/tmp/imagenet1k_full_parquet
dataset-format=parquet
wbits=4
abits=4
wq_mode=statsq
aq_mode=lsq
qk_reparam=true
qk_reparam_type=0
kd_hard_and_soft=0
teacher_soft_temperature=2.75
batch_size=64 per GPU
nproc_per_node=8
epochs=100
scheduler_epochs=100
epoch_checkpoint_interval=1
checkpoint_hist=100
```

禁止：

```text
resume from QAT checkpoint
soup
checkpoint averaging
ensemble
A8->A4
multi-checkpoint 拼接
```

## KL 范式

```text
train_scheme=ema_ref_attn_kl
ref_update=prev_step
ref_update_interval=50
ref_attn_kl_weight=0.0
dynamic_sparse_prevstep_kl=true
dynamic_kl_start_epoch=51
dynamic_kl_observe_until_epoch=50
```

controller：

```text
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
ref_attn_kl_clip=20.0
ref_attn_kl_drop_prob=0.5
ref_attn_loss=kl_ref
```

说明：

```text
当前第一版优先不改代码，使用静态 controller 配置跑完 100 epoch。
Stage D 的后段降权策略暂不实现，避免为降权引入额外代码风险。
```

## 成功标准

```text
最低通过: best Top-1 > 80.8280
有效通过: best >= 80.90，或至少 3 个 checkpoint > 80.8280，或 last20_avg >= 80.70
强通过: best Top-1 >= 81.0
失败判据: best <= 80.8280 且 last20_avg <= 80.6592
```

## 启动前检查

```text
worker: fdbd:dccd:cdc2:1234:0:b8::, port 9801
GPU: 8 x H100 idle, memory about 7 MiB, util 0
dataset: /tmp/imagenet1k_full_parquet/data
/tmp free: about 285G
```

## 2026-07-13 launch

启动命令：

```text
cd /mlx_devbox/users/quyanyi/playground
MASTER_PORT=31851 OUT=/tmp/qat_public_repro nohup bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713.sh >/tmp/ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713.nohup 2>&1 &
```

启动核验：

```text
launcher pid: 160494
script pid: 160495
qat_launch pid: 160510
GPU 0-7 memory about 28429 MiB, util 98-100%
args_yaml=present
```

dry-run / 实际命令确认：

```text
没有 --resume
epochs=100
scheduler-epochs=100
train_scheme=ema_ref_attn_kl
ref_update=prev_step
ref_attn_kl_weight=0.0
dynamic_sparse_prevstep_kl=true
dynamic_kl_start_epoch=51
dynamic_kl_observe_until_epoch=50
batch_size=64
checkpoint_hist=100
epoch_checkpoint_interval=1
```

args.yaml 关键项：

```text
epochs: 100
resume:
train_scheme: ema_ref_attn_kl
dynamic_sparse_prevstep_kl: true
dynamic_kl_start_epoch: 51
dynamic_kl_observe_until_epoch: 50
dynamic_kl_drop_threshold: 0.06
dynamic_kl_strong_drop_threshold: 0.14
dynamic_kl_max_pulses_per_window: 3
dynamic_kl_window_epochs: 10
dynamic_kl_cooldown_epochs: 6
ref_update: prev_step
ref_attn_kl_weight: 0.0
ref_attn_kl_clip: 20.0
batch_size: 64
checkpoint_hist: 100
epoch_checkpoint_interval: 1
pretrained: true
pretrained_initialized: true
```

日志证据：

```text
Loaded state_dict from checkpoint '/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth'
Enabled EMA refmodel attention-KL scheme: ref_update=prev_step, attn_kl_weight=0.0
Scheduled epochs: 100
Enabled dynamic sparse prev-step KL controller: start_epoch=51, observe_until=50
Train: 0 [0/2502] ... LR: 2.000e-04 RefW: 0.000e+00
```

启动后 monitor：

```text
checkpoint_count=0
fullval_rows=0
bad_sample_rows=0
nonzero_refw_lines=0
observe_nonzero_refw_lines=0
controller_rows=0
controller_triggers=0
controller_observe_triggers=0
```

## 2026-07-13 checkpoint-1

monitor 摘要：

```text
checkpoint_count=1
latest_checkpoint=checkpoint-1.pth.tar
fullval_rows=1
bad_sample_rows=0
best_fullval_line=checkpoint-1 Loss 0.9907 Acc@1 77.7080 Acc@5 94.0940 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original10to60_lines=0
above_original10to110_lines=0
above_dynamic10to110_lines=0
above_late10to210_lines=0
target_81_lines=0
last20_avg=77.7080
last10_avg=77.7080
nonzero_refw_lines=0
observe_nonzero_refw_lines=0
controller_rows=1
controller_triggers=0
controller_observe_triggers=0
controller_selected_avoid=0
```

结论：

```text
第一个 from-pretrained QAT epoch 已完成。
checkpoint-1 Top-1 77.7080，Samples=50000，RefW=0，controller observe 无触发。
这符合从 pretrained/public OFQ 初始化开始的 warmup 预期；继续观察 checkpoint-10，判断 20 epoch warmup 是否恢复到合理轨道。
```

## 2026-07-13 checkpoint-10

monitor 摘要：

```text
checkpoint_count=10
latest_checkpoint=checkpoint-10.pth.tar
fullval_rows=10
bad_sample_rows=0
best_fullval_line=checkpoint-10 Loss 0.9089 Acc@1 78.9460 Acc@5 94.7940 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original10to60_lines=0
above_original10to110_lines=0
above_dynamic10to110_lines=0
above_late10to210_lines=0
target_81_lines=0
last20_avg=78.5494
last10_avg=78.5494
nonzero_refw_lines=0
observe_nonzero_refw_lines=0
controller_rows=10
controller_triggers=0
controller_observe_triggers=0
controller_selected_avoid=0
```

checkpoint-1 到 checkpoint-10：

```text
checkpoint-1: Acc@1 77.7080 Acc@5 94.0940 Samples 50000
checkpoint-2: Acc@1 78.2980 Acc@5 94.5060 Samples 50000
checkpoint-3: Acc@1 78.5020 Acc@5 94.4740 Samples 50000
checkpoint-4: Acc@1 78.4680 Acc@5 94.6440 Samples 50000
checkpoint-5: Acc@1 78.4660 Acc@5 94.6660 Samples 50000
checkpoint-6: Acc@1 78.6860 Acc@5 94.6080 Samples 50000
checkpoint-7: Acc@1 78.8220 Acc@5 94.6720 Samples 50000
checkpoint-8: Acc@1 78.8020 Acc@5 94.7680 Samples 50000
checkpoint-9: Acc@1 78.7960 Acc@5 94.6840 Samples 50000
checkpoint-10: Acc@1 78.9460 Acc@5 94.7940 Samples 50000
```

结论：

```text
checkpoint-1 到 checkpoint-10 完整生成，Samples 全 50000。
observe 阶段仍然干净：RefW=0，controller_triggers=0，controller_observe_triggers=0。
Top-1 从 77.7080 稳定恢复到 78.9460，符合 from-pretrained QAT warmup 预期，但还明显低于 resume10 系列的 80+ 起点。
继续观察 checkpoint-20；如果 20 epoch 仍未接近 80，需要把 from-scratch 100ep 的 warmup 速度作为风险记录。
```

## 2026-07-13 checkpoint-20

monitor 摘要：

```text
checkpoint_count=20
latest_checkpoint=checkpoint-20.pth.tar
fullval_rows=20
bad_sample_rows=0
best_fullval_line=checkpoint-20 Loss 0.8964 Acc@1 79.3600 Acc@5 94.9120 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original10to60_lines=0
above_original10to110_lines=0
above_dynamic10to110_lines=0
above_late10to210_lines=0
target_81_lines=0
last20_avg=78.8603
last10_avg=79.1712
nonzero_refw_lines=0
observe_nonzero_refw_lines=0
controller_rows=20
controller_triggers=0
controller_observe_triggers=0
controller_selected_avoid=0
```

checkpoint-11 到 checkpoint-20：

```text
checkpoint-11: Acc@1 78.9520 Acc@5 94.7920 Samples 50000
checkpoint-12: Acc@1 79.0720 Acc@5 94.8280 Samples 50000
checkpoint-13: Acc@1 78.9700 Acc@5 94.7980 Samples 50000
checkpoint-14: Acc@1 79.2400 Acc@5 94.8700 Samples 50000
checkpoint-15: Acc@1 79.1180 Acc@5 94.8340 Samples 50000
checkpoint-16: Acc@1 79.1460 Acc@5 94.9020 Samples 50000
checkpoint-17: Acc@1 79.2520 Acc@5 94.8520 Samples 50000
checkpoint-18: Acc@1 79.2860 Acc@5 94.7720 Samples 50000
checkpoint-19: Acc@1 79.3160 Acc@5 94.9980 Samples 50000
checkpoint-20: Acc@1 79.3600 Acc@5 94.9120 Samples 50000
```

结论：

```text
checkpoint-20 Top-1 79.3600，warmup 稳定上升，但仍明显低于 80。
observe 阶段仍然干净：RefW=0，controller_triggers=0，controller_observe_triggers=0，Samples 全 50000。
这说明 100epoch from-scratch/pretrained-init 的前 20 epoch warmup 速度偏慢；继续跑到 checkpoint-30，如果仍不能接近 80，需要将 warmup 不足列为本方案主要风险。
```

## 2026-07-13 checkpoint-31

monitor 摘要：

```text
checkpoint_count=31
latest_checkpoint=checkpoint-31.pth.tar
fullval_rows=31
bad_sample_rows=0
best_fullval_line=checkpoint-29 Loss 0.8771 Acc@1 79.8120 Acc@5 95.1060 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original10to60_lines=0
above_original10to110_lines=0
above_dynamic10to110_lines=0
above_late10to210_lines=0
target_81_lines=0
last20_avg=79.4316
last10_avg=79.6540
nonzero_refw_lines=0
observe_nonzero_refw_lines=0
controller_rows=31
controller_triggers=0
controller_observe_triggers=0
controller_selected_avoid=0
```

checkpoint-21 到 checkpoint-31：

```text
checkpoint-21: Acc@1 79.3320 Acc@5 94.9580 Samples 50000
checkpoint-22: Acc@1 79.5980 Acc@5 95.0800 Samples 50000
checkpoint-23: Acc@1 79.5480 Acc@5 95.0140 Samples 50000
checkpoint-24: Acc@1 79.6140 Acc@5 94.9840 Samples 50000
checkpoint-25: Acc@1 79.5540 Acc@5 95.0520 Samples 50000
checkpoint-26: Acc@1 79.5840 Acc@5 95.0760 Samples 50000
checkpoint-27: Acc@1 79.6100 Acc@5 95.0740 Samples 50000
checkpoint-28: Acc@1 79.6820 Acc@5 95.1060 Samples 50000
checkpoint-29: Acc@1 79.8120 Acc@5 95.1060 Samples 50000
checkpoint-30: Acc@1 79.7400 Acc@5 95.0860 Samples 50000
checkpoint-31: Acc@1 79.7980 Acc@5 95.0560 Samples 50000
```

结论：

```text
checkpoint-31 仍未过 80，当前 best checkpoint-29 Top-1 79.8120。
observe 阶段仍然干净：RefW=0，controller_triggers=0，Samples 全 50000。
warmup 不足风险已经明确：100 epoch from-pretrained-init 到第 31 epoch 仍比 81 目标低 1.1880。
继续跑到 checkpoint-40；如果 checkpoint-40 仍低于 80.2，则这版 100 epoch 方案冲击 81 的空间会非常有限。
```

## 2026-07-13 checkpoint-40

monitor 摘要：

```text
checkpoint_count=40
latest_checkpoint=checkpoint-40.pth.tar
fullval_rows=40
bad_sample_rows=0
best_fullval_line=checkpoint-38 Loss 0.8580 Acc@1 80.0900 Acc@5 95.1500 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original10to60_lines=0
above_original10to110_lines=0
above_dynamic10to110_lines=0
above_late10to210_lines=0
target_81_lines=0
last20_avg=79.7629
last10_avg=79.9184
nonzero_refw_lines=0
observe_nonzero_refw_lines=0
controller_rows=40
controller_triggers=0
controller_observe_triggers=0
controller_selected_avoid=0
```

checkpoint-32 到 checkpoint-40：

```text
checkpoint-32: Acc@1 79.7420 Acc@5 95.1080 Samples 50000
checkpoint-33: Acc@1 79.9140 Acc@5 95.0900 Samples 50000
checkpoint-34: Acc@1 79.8780 Acc@5 95.0960 Samples 50000
checkpoint-35: Acc@1 79.9420 Acc@5 95.1520 Samples 50000
checkpoint-36: Acc@1 79.9300 Acc@5 95.1580 Samples 50000
checkpoint-37: Acc@1 79.9640 Acc@5 95.2060 Samples 50000
checkpoint-38: Acc@1 80.0900 Acc@5 95.1500 Samples 50000
checkpoint-39: Acc@1 79.9640 Acc@5 95.2480 Samples 50000
checkpoint-40: Acc@1 79.9620 Acc@5 95.2240 Samples 50000
```

结论：

```text
checkpoint-40 best 只有 80.0900，低于预设风险线 80.2。
observe 阶段仍然干净：RefW=0，controller_triggers=0，Samples 全 50000。
这版 100 epoch from-pretrained-init 的 warmup 明显不足；即使 51 后 KL 正常触发，冲击 81.0 的空间已经偏小。
继续跑到 checkpoint-50，观察进入 dynamic 前 rolling best 能否再抬升；但需要在最终审计中把 early warmup 不足作为主要失败原因候选。
```

## 2026-07-14 checkpoint-51

monitor 摘要：

```text
checkpoint_count=51
latest_checkpoint=checkpoint-51.pth.tar
fullval_rows=51
bad_sample_rows=0
best_fullval_line=checkpoint-51 Loss 0.8486 Acc@1 80.2140 Acc@5 95.3500 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original10to60_lines=0
above_original10to110_lines=0
above_dynamic10to110_lines=0
above_late10to210_lines=0
target_81_lines=0
last20_avg=80.0411
last10_avg=80.1352
nonzero_refw_lines=0
observe_nonzero_refw_lines=0
controller_rows=51
controller_triggers=0
controller_observe_triggers=0
controller_selected_avoid=0
```

checkpoint-41 到 checkpoint-51：

```text
checkpoint-41: Acc@1 80.0840 Acc@5 95.2360 Samples 50000
checkpoint-42: Acc@1 80.0740 Acc@5 95.2280 Samples 50000
checkpoint-43: Acc@1 80.1000 Acc@5 95.1340 Samples 50000
checkpoint-44: Acc@1 80.0540 Acc@5 95.2660 Samples 50000
checkpoint-45: Acc@1 80.1800 Acc@5 95.2760 Samples 50000
checkpoint-46: Acc@1 80.1180 Acc@5 95.2260 Samples 50000
checkpoint-47: Acc@1 80.1800 Acc@5 95.3040 Samples 50000
checkpoint-48: Acc@1 80.1620 Acc@5 95.2260 Samples 50000
checkpoint-49: Acc@1 80.1400 Acc@5 95.2080 Samples 50000
checkpoint-50: Acc@1 80.1300 Acc@5 95.3160 Samples 50000
checkpoint-51: Acc@1 80.2140 Acc@5 95.3500 Samples 50000
```

controller 边界：

```text
epoch 48: observe, top1 80.1400, rolling_best 80.1800, drop 0.0400, triggered=0
epoch 49: observe, top1 80.1300, rolling_best 80.1800, drop 0.0500, triggered=0
epoch 50: observe, top1 80.2140, rolling_best 80.2140, drop 0.0000, triggered=0
```

结论：

```text
checkpoint-51 已越过 dynamic start 边界，但 controller 尚未触发，因为 checkpoint-51 刷新 rolling best 到 80.2140，drop=0。
observe 阶段完整干净：RefW=0，controller_observe_triggers=0，Samples 全 50000。
进入 dynamic 前 rolling best 只有 80.2140，明显低于 200ep late sparse KL 的 80.7080 observe 高点；这进一步说明本 100ep from-scratch 版本 warmup 偏慢，冲击 81 空间不足。
继续跑到 checkpoint-60，观察 dynamic controller 是否开始触发，以及是否能快速把曲线推到 80.5+。
```

## 2026-07-14 checkpoint-61

monitor 摘要：

```text
checkpoint_count=61
latest_checkpoint=checkpoint-61.pth.tar
fullval_rows=61
bad_sample_rows=0
best_fullval_line=checkpoint-60 Loss 0.8424 Acc@1 80.4480 Acc@5 95.4260 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original10to60_lines=0
above_original10to110_lines=0
above_dynamic10to110_lines=0
above_late10to210_lines=0
target_81_lines=0
last20_avg=80.2241
last10_avg=80.3130
nonzero_refw_lines=150
nonzero_refw_epochs=53,54,55
observe_nonzero_refw_lines=0
controller_rows=61
controller_triggers=3
controller_observe_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7,4:11
```

checkpoint-51 到 checkpoint-61：

```text
checkpoint-51: Acc@1 80.2140 Acc@5 95.3500 Samples 50000
checkpoint-52: Acc@1 80.3180 Acc@5 95.2800 Samples 50000
checkpoint-53: Acc@1 80.2280 Acc@5 95.3780 Samples 50000
checkpoint-54: Acc@1 80.1920 Acc@5 95.2840 Samples 50000
checkpoint-55: Acc@1 80.2120 Acc@5 95.2620 Samples 50000
checkpoint-56: Acc@1 80.4040 Acc@5 95.3180 Samples 50000
checkpoint-57: Acc@1 80.3200 Acc@5 95.2600 Samples 50000
checkpoint-58: Acc@1 80.3560 Acc@5 95.2720 Samples 50000
checkpoint-59: Acc@1 80.2480 Acc@5 95.2560 Samples 50000
checkpoint-60: Acc@1 80.4480 Acc@5 95.4260 Samples 50000
checkpoint-61: Acc@1 80.4040 Acc@5 95.3580 Samples 50000
```

controller / RefW：

```text
epoch 52: drop 0.0900, next_head=8:4, next_weight=1e-05, triggered=1
epoch 53: applied_head=8:4, RefW max=1e-05, next_head=5:7, triggered=1
epoch 54: applied_head=5:7, RefW max=1e-05, next_head=4:11, triggered=1
epoch 55: applied_head=4:11, RefW max=1e-05
epoch 56-60: no further pulse due to window_limit or drop below threshold
avoid heads selected: 0
observe RefW lines: 0
```

结论：

```text
dynamic 阶段已经正常触发，且没有选中 avoid heads。
KL 初期把 best 从 80.2140 推到 checkpoint-60 的 80.4480，但仍低于 baseline 80.5980。
当前实验的主要问题已经不是 controller 是否能触发，而是 from-pretrained 100 epoch 的前期 rolling best 太低；即使 KL 生效，冲 81 的空间明显不足。
继续跑到 checkpoint-70，观察是否能至少过 baseline 80.5980。
```

## 2026-07-14 checkpoint-71

monitor 摘要：

```text
checkpoint_count=71
latest_checkpoint=checkpoint-71.pth.tar
fullval_rows=71
bad_sample_rows=0
best_fullval_line=checkpoint-71 Loss 0.8339 Acc@1 80.5720 Acc@5 95.3720 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original10to60_lines=0
above_original10to110_lines=0
above_dynamic10to110_lines=0
above_late10to210_lines=0
target_81_lines=0
last20_avg=80.3964
last10_avg=80.4798
nonzero_refw_lines=250
nonzero_refw_epochs=53,54,55,64,65
observe_nonzero_refw_lines=0
controller_rows=71
controller_triggers=5
controller_observe_triggers=0
controller_selected_avoid=0
```

checkpoint-62 到 checkpoint-71：

```text
checkpoint-62: Acc@1 80.4700 Acc@5 95.3760 Samples 50000
checkpoint-63: Acc@1 80.3920 Acc@5 95.3760 Samples 50000
checkpoint-64: Acc@1 80.3500 Acc@5 95.3480 Samples 50000
checkpoint-65: Acc@1 80.3780 Acc@5 95.3300 Samples 50000
checkpoint-66: Acc@1 80.4640 Acc@5 95.3880 Samples 50000
checkpoint-67: Acc@1 80.4900 Acc@5 95.3480 Samples 50000
checkpoint-68: Acc@1 80.5660 Acc@5 95.3920 Samples 50000
checkpoint-69: Acc@1 80.5640 Acc@5 95.4040 Samples 50000
checkpoint-70: Acc@1 80.5520 Acc@5 95.4120 Samples 50000
checkpoint-71: Acc@1 80.5720 Acc@5 95.3720 Samples 50000
```

controller / RefW：

```text
epoch 63: drop 0.1200, next_head=8:4, next_weight=1e-05, triggered=1
epoch 64: applied_head=8:4, next_head=5:7, triggered=1
epoch 65: applied_head=5:7, no further trigger because drop below threshold
epoch 66-71: no further pulse because drop below threshold
avoid heads selected: 0
observe RefW lines: 0
```

结论：

```text
checkpoint-71 best 80.5720，仍未超过 baseline 80.5980。
controller 运行正常，但因为 rolling best 低、drop 很快低于阈值，后续没有继续 pulse；这说明 KL 并没有弥补前期 warmup 不足。
当前 100ep from-pretrained-init 方案冲击 81.0 的希望已经很低。继续跑到 checkpoint-80，观察是否能越过 baseline 并接近 80.7，否则最终大概率判定该范式失败。
```

## 2026-07-14 checkpoint-81

monitor 摘要：

```text
checkpoint_count=81
latest_checkpoint=checkpoint-81.pth.tar
fullval_rows=81
bad_sample_rows=0
best_fullval_line=checkpoint-80 Loss 0.8298 Acc@1 80.6500 Acc@5 95.4240 Samples 50000
above_baseline_lines=3
above_scheme_c_lines=0
above_original10to60_lines=0
above_original10to110_lines=0
above_dynamic10to110_lines=0
above_late10to210_lines=0
target_81_lines=0
last20_avg=80.5199
last10_avg=80.5600
nonzero_refw_lines=400
nonzero_refw_epochs=53,54,55,64,65,73,78,79
observe_nonzero_refw_lines=0
controller_rows=81
controller_triggers=8
controller_observe_triggers=0
controller_selected_avoid=0
```

checkpoint-72 到 checkpoint-81：

```text
checkpoint-72: Acc@1 80.5320 Acc@5 95.3760 Samples 50000
checkpoint-73: Acc@1 80.4880 Acc@5 95.4120 Samples 50000
checkpoint-74: Acc@1 80.4840 Acc@5 95.3820 Samples 50000
checkpoint-75: Acc@1 80.5280 Acc@5 95.4420 Samples 50000
checkpoint-76: Acc@1 80.6340 Acc@5 95.3640 Samples 50000
checkpoint-77: Acc@1 80.5880 Acc@5 95.4240 Samples 50000
checkpoint-78: Acc@1 80.5620 Acc@5 95.4000 Samples 50000
checkpoint-79: Acc@1 80.5340 Acc@5 95.3940 Samples 50000
checkpoint-80: Acc@1 80.6500 Acc@5 95.4240 Samples 50000
checkpoint-81: Acc@1 80.6000 Acc@5 95.4160 Samples 50000
```

controller / RefW：

```text
epoch 72: drop 0.0840, next_head=8:4, next_weight=1e-05, triggered=1
epoch 73: applied_head=8:4, no trigger due to window_limit
epoch 77: drop 0.0720, next_head=5:7, triggered=1
epoch 78: applied_head=5:7, next_head=8:4, triggered=1
epoch 79: applied_head=8:4, top1 80.6500
epoch 80: no trigger, drop below threshold
avoid heads selected: 0
observe RefW lines: 0
```

结论：

```text
checkpoint-80 达到 80.6500，终于超过 baseline 80.5980，但仍低于 scheme C 80.6820，更远低于 81.0。
controller 在本段继续正常触发，没有选择 avoid heads；但由于前期 rolling best 太低，KL 只能把曲线推到 80.65 左右。
当前已基本可以判断：这版 100epoch from-pretrained-init 范式无法冲击 81.0。继续跑到 checkpoint-90 和 100，完成最终审计。
```

## 2026-07-14 checkpoint-91

monitor 摘要：

```text
checkpoint_count=91
latest_checkpoint=checkpoint-91.pth.tar
fullval_rows=91
bad_sample_rows=0
best_fullval_line=checkpoint-89 Loss 0.8259 Acc@1 80.7240 Acc@5 95.4340 Samples 50000
above_baseline_lines=11
above_scheme_c_lines=5
above_original10to60_lines=0
above_original10to110_lines=0
above_dynamic10to110_lines=0
above_late10to210_lines=0
target_81_lines=0
last20_avg=80.6146
last10_avg=80.6692
nonzero_refw_lines=450
nonzero_refw_epochs=53,54,55,64,65,73,78,79,85
observe_nonzero_refw_lines=0
controller_rows=91
controller_triggers=9
controller_observe_triggers=0
controller_selected_avoid=0
```

checkpoint-82 到 checkpoint-91：

```text
checkpoint-82: Acc@1 80.5840 Acc@5 95.4180 Samples 50000
checkpoint-83: Acc@1 80.6900 Acc@5 95.4920 Samples 50000
checkpoint-84: Acc@1 80.6800 Acc@5 95.4620 Samples 50000
checkpoint-85: Acc@1 80.5520 Acc@5 95.5160 Samples 50000
checkpoint-86: Acc@1 80.7120 Acc@5 95.4860 Samples 50000
checkpoint-87: Acc@1 80.6640 Acc@5 95.4520 Samples 50000
checkpoint-88: Acc@1 80.7220 Acc@5 95.4020 Samples 50000
checkpoint-89: Acc@1 80.7240 Acc@5 95.4340 Samples 50000
checkpoint-90: Acc@1 80.6860 Acc@5 95.4500 Samples 50000
checkpoint-91: Acc@1 80.6780 Acc@5 95.4340 Samples 50000
```

controller / RefW：

```text
epoch 84: drop 0.1380, next_head=8:4, next_weight=1e-05, triggered=1
epoch 85: applied_head=8:4, top1 80.7120
epoch 86-91: no further trigger because drop below threshold
avoid heads selected: 0
observe RefW lines: 0
```

结论：

```text
checkpoint-89 达到 80.7240，追平 original 10->60 best，但没有超过 original 10->110 best 80.7520，更没有接近 81.0。
last10_avg=80.6692，说明后段均值进入 80.6+，但前期 warmup 太慢导致最终高度不足。
controller 运行正常，KL 在 checkpoint-85 后帮助曲线上到 80.7 左右，但本方案已经不可能在剩余 9 个 checkpoint 内补足到 81.0。继续跑完 checkpoint-100，完成最终审计。
```

## 2026-07-14 final audit

最终 monitor 摘要：

```text
checkpoint_count=100
latest_checkpoint=checkpoint-100.pth.tar
fullval_rows=100
bad_sample_rows=0
best_fullval_line=checkpoint-100 Loss 0.8213 Acc@1 80.7720 Acc@5 95.4320 Samples 50000
above_baseline_lines=20
above_scheme_c_lines=14
above_original10to60_lines=6
above_original10to110_lines=3
above_dynamic10to110_lines=2
above_late10to210_lines=0
target_81_lines=0
last20_avg=80.6965
last10_avg=80.7316
nonzero_refw_lines=500
nonzero_refw_epochs=53,54,55,64,65,73,78,79,85,98
observe_nonzero_refw_lines=0
controller_rows=100
controller_triggers=10
controller_observe_triggers=0
controller_selected_avoid=0
```

checkpoint-92 到 checkpoint-100：

```text
checkpoint-92: Acc@1 80.7160 Acc@5 95.4880 Samples 50000
checkpoint-93: Acc@1 80.6860 Acc@5 95.4660 Samples 50000
checkpoint-94: Acc@1 80.7640 Acc@5 95.4760 Samples 50000
checkpoint-95: Acc@1 80.7520 Acc@5 95.4800 Samples 50000
checkpoint-96: Acc@1 80.7400 Acc@5 95.5600 Samples 50000
checkpoint-97: Acc@1 80.7500 Acc@5 95.4420 Samples 50000
checkpoint-98: Acc@1 80.7040 Acc@5 95.4960 Samples 50000
checkpoint-99: Acc@1 80.7540 Acc@5 95.4780 Samples 50000
checkpoint-100: Acc@1 80.7720 Acc@5 95.4320 Samples 50000
```

最终统计：

```text
rows=100
bad_sample_rows=0
best=checkpoint-100
best_loss=0.8213
best_top1=80.7720
best_top5=95.4320
above_baseline_80.5980=20
above_scheme_c_80.6820=14
above_original10to60_80.7240=6
above_original10to110_80.7520=3
above_dynamic10to110_80.7600=2
above_late10to210_80.8280=0
target_81=0
last20_avg=80.6965
last10_avg=80.7316
```

完成性审计：

```text
checkpoint-1 到 checkpoint-100: remote count=100, no missing checkpoint printed
full-val rows: 100
Samples=50000: all rows, bad_sample_rows=0
args.yaml: present
epochs: 100
resume: empty
train_scheme: ema_ref_attn_kl
dynamic_sparse_prevstep_kl: true
dynamic_kl_start_epoch: 51
dynamic_kl_observe_until_epoch: 50
dynamic_kl_drop_threshold: 0.06
dynamic_kl_strong_drop_threshold: 0.14
dynamic_kl_max_pulses_per_window: 3
dynamic_kl_window_epochs: 10
dynamic_kl_cooldown_epochs: 6
ref_update: prev_step
ref_attn_kl_weight: 0.0
ref_attn_kl_clip: 20.0
batch_size: 64
checkpoint_hist: 100
epoch_checkpoint_interval: 1
pretrained: true
pretrained_initialized: true
```

controller 审计：

```text
controller_rows=100
controller_triggers=10
controller_observe_triggers=0
controller_selected_avoid=0
observe_nonzero_refw_lines=0
nonzero_refw_lines=500
nonzero_refw_epochs=53,54,55,64,65,73,78,79,85,98
selected heads: 8:4,5:7,4:11
avoid heads selected: 0
```

最终结论：

```text
本实验完成 100 epoch from-pretrained-init / public OFQ 初始化训练，没有从 QAT checkpoint resume。

目标 81.0 没有达到：
best Top-1 = 80.7720
target_81_lines = 0

相对历史：
original OFQ 10->110 best = 80.7520
dynamic sparse prev-step KL 10->110 best = 80.7600
late sparse prev-step KL 10->210 best = 80.8280
本实验 best = 80.7720

本实验略高于 10->110 dynamic best，但低于 10->210 late sparse best 80.8280。

有效性：
above_dynamic10to110_lines=2
above_original10to110_lines=3
above_original10to60_lines=6
last20_avg=80.6965，接近但未达到有效通过线 80.70
last10_avg=80.7316

失败原因判断：
主要问题是 from-pretrained 100 epoch 的前期 warmup 太慢。checkpoint-40 best 只有 80.0900，checkpoint-51 rolling best 只有 80.2140，导致 dynamic KL 从 epoch 51 开始时基线过低。
controller 本身运行正常，observe 阶段没有 KL，dynamic 阶段没有选中 avoid heads，且后段把曲线推到 80.7+。
但是前期高度不足，最终只能到 80.7720，无法达到 81.0，也无法超过 10->210 late sparse KL 的 80.8280。

结论：该 100 epoch from-pretrained-init late sparse prev-step KL 范式不达标；它可以超过旧 10->110 dynamic best，但不是 81.0 方案。若继续做 100 epoch，需要改 warmup/init 范式，而不是只改后段 KL controller。
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
above_late10to210_lines
target_81_lines
last20_avg / last10_avg
nonzero_refw_lines / nonzero_refw_epochs
observe_nonzero_refw_lines
controller_rows / controller_triggers
controller_observe_triggers
controller_selected_avoid
controller_next_heads
```

## 完成审计清单

```text
checkpoint-1 到 checkpoint-100 是否完整生成
full-val rows 是否为 100 且 Samples=50000
observe 阶段是否没有有效 KL pulse
dynamic 阶段 KL 是否只由 controller 触发
avoid heads 是否从未被选中
args.yaml 是否符合预期
best checkpoint / Top-1 / Top-5
超过 80.5980 / 80.6820 / 80.7240 / 80.7520 / 80.7600 / 80.8280 / 81.0 的 checkpoint 数量
last20_avg / last10_avg
controller 触发时机、head、weight、drop、window limit、cooldown
```
