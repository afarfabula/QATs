# OFQ 100ep from-scratch original no-KL control progress

## 目标

执行一版 Swin-T W4A4-family / OFQ public-family 的 100 epoch 严格 no-KL 对照实验，和上一版 `ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713` 做消融对比。

本实验的核心问题：

```text
在相同 public pretrained 初始化、相同 OFQ public-family 主链路、相同训练长度下，
checkpoint-100 Top-1 80.7720 是 OFQ 自然长跑带来的，
还是 sparse prev-step KL 后段保峰带来的收益？
```

## 实验名和路径

```text
experiment: ofq_100ep_fromscratch_original_ofq_public_control_20260714
output: /tmp/qat_public_repro/ofq_100ep_fromscratch_original_ofq_public_control_20260714
log: /mlx_devbox/users/quyanyi/playground/train_ofq_100ep_fromscratch_original_ofq_public_control_20260714.log
run script: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_100ep_fromscratch_original_ofq_public_control_20260714.sh
monitor script: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/monitor_ofq_100ep_fromscratch_original_ofq_public_control_20260714.sh
status TSV: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_100ep_fromscratch_original_ofq_public_control_status_20260714.tsv
refw TSV: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_100ep_fromscratch_original_ofq_public_control_refw_20260714.tsv
summary: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_100ep_fromscratch_original_ofq_public_control_monitor_summary_20260714.txt
```

## 严格消融边界

和上一版 KL 实验保持一致：

```text
从 ImageNet pretrained / public OFQ 初始化开始
不从 checkpoint-10 或其他 QAT checkpoint resume
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
lr=2e-4
min_lr=5e-6
weight_decay=0.0
epoch_checkpoint_interval=1
checkpoint_hist=100
same augmentation flags
same seed=42
每个 epoch full validation
```

唯一变量：

```text
关闭 train_scheme=ema_ref_attn_kl
关闭 refmodel / prev-step ref update
关闭 ref_attn_kl_weight
关闭 dynamic sparse prev-step KL controller
训练全程 RefW=0
无 controller pulse / controller artifact
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

## 对比阈值

```text
baseline: 80.5980
scheme C best: 80.6820
original OFQ 10->60 best: 80.7240
original OFQ 10->110 best: 80.7520
dynamic KL 10->110 best: 80.7600
100ep sparse prev-step KL best: 80.7720
late sparse prev-step KL 10->210 best: 80.8280
target: 81.0
```

上一版 KL 100epoch 结果：

```text
best: checkpoint-100 Top-1 80.7720 Top-5 95.4320
last20_avg: 80.6965
last10_avg: 80.7316
controller_triggers: 10
nonzero_refw_epochs: 53,54,55,64,65,73,78,79,85,98
```

## 解释矩阵

```text
no-KL best 明显低于 80.7720:
  sparse prev-step KL 对后段保峰有正贡献；下一步应保留 KL，优化前 50 epoch warmup。

no-KL best 接近 80.7720:
  80.7720 主要可能来自 OFQ 自然长跑；当前 KL 增益很小，需要重新设计触发机制或 head 检测。

no-KL best 高于 80.7720:
  当前 KL 可能干扰自然收敛；下一步应降低 KL 频率/权重，或只做更晚期 ultra-sparse polish。
```

## 最终审计清单

```text
checkpoint-1 到 checkpoint-100 是否完整生成
full-val rows 是否 100
Samples 是否全为 50000
args.yaml 是否 present
args.yaml 是否 resume 为空
args.yaml 是否 train_scheme=baseline
args.yaml 是否 dynamic_sparse_prevstep_kl=false
日志中是否没有 Enabled EMA refmodel attention-KL scheme
日志中是否没有 Enabled dynamic sparse prev-step KL controller
RefW 是否始终为 0
controller artifact 是否 absent
best checkpoint / Top-1 / Top-5
checkpoint-40 / checkpoint-51 / checkpoint-80 / checkpoint-100 Top-1
last20_avg / last10_avg
超过 80.5980 / 80.6820 / 80.7240 / 80.7520 / 80.7600 / 80.7720 / 80.8280 / 81.0 的 checkpoint 数量
和上一版 100ep sparse prev-step KL 的 best 80.7720 / last20_avg 80.6965 / last10_avg 80.7316 对比
```

## 2026-07-14 preflight

本实验使用新脚本，不修改训练代码：

```text
run script created
monitor script created
progress doc created
```

本地数据检查：

```text
train_shards=294
validation_shards=14
```

磁盘检查：

```text
/tmp free: about 463G
/mlx_devbox/users/quyanyi/playground free: about 7.9G
```

下一步：

```text
执行 dry-run，确认命令没有 --resume、--train-scheme、--ref-*、--dynamic-*，并且无 soup / checkpoint averaging / ensemble / A8->A4。
```

## 2026-07-14 dry-run

脚本语法检查：

```text
run script bash -n: passed
monitor script bash -n: passed
```

dry-run 核心命令确认：

```text
--epochs 100
--scheduler-epochs 100
--batch-size 64
--lr 0.0002
--min-lr 5e-06
--weight-decay 0.0
--checkpoint-hist 100
--epoch-checkpoint-interval 1
--wq-bitw 4
--aq-bitw 4
--wq-mode statsq
--aq-mode lsq
--pretrained
--pretrained_initialized
--use-kd
--kd_hard_and_soft 0
--teacher-soft-temperature 2.75
--quantized
--qk_reparam
--qk_reparam_type 0
--smoothing 0.1
--mixup 0.0
--cutmix 0.0
--aa rand-m9-mstd0.5-inc1
--color-jitter 0.4
--reprob 0.25
--seed 42
```

dry-run 禁止项审计：

```text
run script 中没有 --resume
run script 中没有 --train-scheme
run script 中没有 --ref-*
run script 中没有 --dynamic-*
实际 dry-run OFQ command 中没有 --resume
实际 dry-run OFQ command 中没有 --train-scheme
实际 dry-run OFQ command 中没有 --ref-*
实际 dry-run OFQ command 中没有 --dynamic-*
无 soup / checkpoint averaging / ensemble / A8->A4 参数
```

worker 启动前检查：

```text
worker: fdbd:dccd:cdc2:1234:0:b8::, port 9801
GPU: 8 x H100 idle, memory about 7 MiB, util 0
output dir: absent
```

下一步：

```text
在真实 worker GPU 环境后台启动训练，并在启动后核验 args.yaml、首条 Train 日志、GPU、RefW=0。
```

## 2026-07-14 invalid local launch

曾尝试在当前 shell 后台启动：

```text
launcher_pid=1084633
```

审计结果：

```text
进程已退出
日志只到脚本 preflight
log shows: no-gpu-device
没有进入 qat_launch.py / OFQ train.py
remote worker GPU 仍然 idle
output dir 仍未建立有效训练产物
```

结论：

```text
这次不是有效训练启动；当前 shell 不是 GPU worker shell。
下一步改为通过 worker SSH 在真实 worker 上后台启动同一个脚本。
```

## 2026-07-14 worker launch

启动命令：

```text
cd /mlx_devbox/users/quyanyi/playground
MASTER_PORT=31861 OUT=/tmp/qat_public_repro nohup bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_100ep_fromscratch_original_ofq_public_control_20260714.sh >/tmp/ofq_100ep_fromscratch_original_ofq_public_control_20260714.nohup 2>&1 &
```

进程：

```text
worker_launcher_pid=175191
script pid=175192
qat_launch pid=175207
tee pid=175208
```

启动质量证据：

```text
worker log shows gpu-device-present
Model swin_t created, param count:28608256
Scheduled epochs: 100
Train: 0 [0/2502] ... LR: 2.000e-04 RefW: 0.000e+00 AnchorRefW: 0.000e+00
Train: 0 [500/2502] ... LR: 2.000e-04 RefW: 0.000e+00 AnchorRefW: 0.000e+00
```

GPU：

```text
8 x H100 stable training
memory per GPU about 28351 MiB
utilization per GPU 100%
```

args.yaml 关键项：

```text
resume: ''
train_scheme: baseline
dynamic_sparse_prevstep_kl: false
ref_attn_kl_weight: 0.0
ref_update: ema
epochs: 100
scheduler_epochs: 100
batch_size: 64
lr: 0.0002
min_lr: 5.0e-06
weight_decay: 0.0
checkpoint_hist: 100
epoch_checkpoint_interval: 1
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
qk_reparam: true
qk_reparam_type: 0
pretrained: true
pretrained_initialized: true
```

启动后 monitor：

```text
checkpoint_count=0
fullval_rows=0
bad_sample_rows=0
args_yaml=present
controller_artifact=absent
nonzero_refw_lines=0
nonzero_refw_epochs=NA
target_81_lines=0
```

结论：

```text
no-KL 100epoch 严格对照实验已在真实 worker GPU 环境有效启动。
当前证据确认：无 resume、train_scheme=baseline、dynamic controller disabled、RefW=0、controller artifact absent。
继续轮询 checkpoint-1 / full-val 结果。
```

## 2026-07-14 checkpoint-1

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
above_kl100ep_lines=0
above_late10to210_lines=0
target_81_lines=0
nonzero_refw_lines=0
controller_artifact=absent
```

checkpoint-1 结果：

```text
Test: [distributed-summary] Loss 0.9907 Acc@1 77.7080 Acc@5 94.0940 Samples 50000
```

结论：

```text
checkpoint-1 已完整生成，full validation samples=50000。
RefW 仍为 0，controller artifact absent，符合严格 no-KL 对照要求。
checkpoint-1 Top-1 77.7080，和上一版 100ep sparse prev-step KL 实验的 checkpoint-1 完全一致；这说明前 1 个 epoch 的 public pretrained / OFQ 主链路对齐，没有因为去掉 KL 参数而改变 early warmup 起点。
继续低频轮询 checkpoint-10；若 checkpoint-10 仍与上一版 KL 的 78.9460 接近，则可以进一步确认 no-KL 对照在 observe 阶段与 KL 实验主链路严格一致。
```

## 2026-07-14 checkpoint-10

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
above_kl100ep_lines=0
above_late10to210_lines=0
target_81_lines=0
last20_avg=78.5494
last10_avg=78.5494
controller_artifact=absent
nonzero_refw_lines=0
nonzero_refw_epochs=NA
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

GPU / 训练状态：

```text
8 x H100, memory about 28303 MiB, util 100%
training continues into epoch 10
```

结论：

```text
checkpoint-1 到 checkpoint-10 已完整生成，Samples 全 50000。
RefW 始终为 0，controller artifact absent，符合严格 no-KL 对照要求。
checkpoint-1 到 checkpoint-10 的 Top-1 序列与上一版 100ep sparse prev-step KL 实验完全一致：
77.7080, 78.2980, 78.5020, 78.4680, 78.4660, 78.6860, 78.8220, 78.8020, 78.7960, 78.9460。
这说明在前 10 epoch observe / no effective KL 阶段，两条链路主训练配置严格对齐。
继续轮询 checkpoint-20；checkpoint-20 后再和上一版 KL 的 checkpoint-20 Top-1 79.3600 对比。
```

## 2026-07-14 checkpoint-20

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
above_kl100ep_lines=0
above_late10to210_lines=0
target_81_lines=0
last20_avg=78.8603
last10_avg=79.1712
controller_artifact=absent
nonzero_refw_lines=0
nonzero_refw_epochs=NA
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

GPU / 训练状态：

```text
8 x H100, memory about 28303 MiB, util 100%
training continues into epoch 20
```

结论：

```text
checkpoint-1 到 checkpoint-20 已完整生成，Samples 全 50000。
RefW 始终为 0，controller artifact absent，no-KL 对照边界仍然干净。
checkpoint-20 Top-1 79.3600，与上一版 100ep sparse prev-step KL 实验 checkpoint-20 Top-1 79.3600 完全一致。
这进一步确认：在 KL 实验 dynamic_kl_start_epoch=51 之前，两条链路的 early warmup 是严格对齐的；当前 no-KL 对照可以作为后续 epoch 51+ 是否由 KL 带来差异的有效消融基线。
继续轮询 checkpoint-40；checkpoint-40 后和上一版 KL 的 checkpoint-40 best 80.0900 对齐检查。
```

## 2026-07-14 checkpoint-40

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
above_kl100ep_lines=0
above_late10to210_lines=0
target_81_lines=0
last20_avg=79.7629
last10_avg=79.9184
controller_artifact=absent
nonzero_refw_lines=0
nonzero_refw_epochs=NA
```

checkpoint-21 到 checkpoint-40：

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

GPU / 训练状态：

```text
8 x H100, memory about 28303 MiB, util 100%
training continues into epoch 40
```

结论：

```text
checkpoint-1 到 checkpoint-40 已完整生成，Samples 全 50000。
RefW 始终为 0，controller artifact absent，no-KL 对照边界仍然干净。
checkpoint-40 阶段 best 为 checkpoint-38 Top-1 80.0900，与上一版 100ep sparse prev-step KL 实验 checkpoint-40 阶段 best 80.0900 完全一致。
这说明到 epoch 40 为止，两条链路几乎是同一条 OFQ public-family warmup 曲线。由于上一版 KL 在 epoch 51 后才可能触发，本 no-KL 对照已经为后续判断 KL 后段贡献提供了有效基线。
继续轮询 checkpoint-51；checkpoint-51 是上一版 KL dynamic start 边界，用来确认两条链路进入后段前的 rolling best 是否仍然一致。
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
above_kl100ep_lines=0
above_late10to210_lines=0
target_81_lines=0
last20_avg=80.0411
last10_avg=80.1352
controller_artifact=absent
nonzero_refw_lines=0
nonzero_refw_epochs=NA
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

GPU / 训练状态：

```text
8 x H100, memory about 28303 MiB, util 100%
training continues into epoch 51
```

结论：

```text
checkpoint-1 到 checkpoint-51 已完整生成，Samples 全 50000。
RefW 始终为 0，controller artifact absent，no-KL 对照边界仍然干净。
checkpoint-51 Top-1 80.2140，与上一版 100ep sparse prev-step KL 实验 checkpoint-51 Top-1 80.2140 完全一致。
这说明在上一版 KL dynamic_kl_start_epoch=51 的边界处，两条链路仍然完全对齐；因此后续 checkpoint-52 之后的差异，才是判断 sparse prev-step KL 后段保峰贡献的关键证据。
继续轮询 checkpoint-80；重点比较 checkpoint-60 / 71 / 80 与上一版 KL 的后段曲线。
```

## 2026-07-15 checkpoint-80

monitor 摘要：

```text
checkpoint_count=80
latest_checkpoint=checkpoint-80.pth.tar
fullval_rows=80
bad_sample_rows=0
best_fullval_line=checkpoint-77 Loss 0.8307 Acc@1 80.6360 Acc@5 95.4080 Samples 50000
above_baseline_lines=1
above_scheme_c_lines=0
above_original10to60_lines=0
above_original10to110_lines=0
above_dynamic10to110_lines=0
above_kl100ep_lines=0
above_late10to210_lines=0
target_81_lines=0
last20_avg=80.4861
last10_avg=80.5140
controller_artifact=absent
nonzero_refw_lines=0
nonzero_refw_epochs=NA
```

checkpoint-52 到 checkpoint-80：

```text
checkpoint-52: Acc@1 80.3180 Acc@5 95.2800 Samples 50000
checkpoint-53: Acc@1 80.2280 Acc@5 95.3780 Samples 50000
checkpoint-54: Acc@1 80.2880 Acc@5 95.2980 Samples 50000
checkpoint-55: Acc@1 80.3180 Acc@5 95.3580 Samples 50000
checkpoint-56: Acc@1 80.3460 Acc@5 95.3520 Samples 50000
checkpoint-57: Acc@1 80.4240 Acc@5 95.3300 Samples 50000
checkpoint-58: Acc@1 80.3940 Acc@5 95.4120 Samples 50000
checkpoint-59: Acc@1 80.3540 Acc@5 95.2980 Samples 50000
checkpoint-60: Acc@1 80.5260 Acc@5 95.3500 Samples 50000
checkpoint-61: Acc@1 80.5080 Acc@5 95.3500 Samples 50000
checkpoint-62: Acc@1 80.3840 Acc@5 95.3500 Samples 50000
checkpoint-63: Acc@1 80.4260 Acc@5 95.3700 Samples 50000
checkpoint-64: Acc@1 80.4700 Acc@5 95.4200 Samples 50000
checkpoint-65: Acc@1 80.4600 Acc@5 95.3820 Samples 50000
checkpoint-66: Acc@1 80.4000 Acc@5 95.3640 Samples 50000
checkpoint-67: Acc@1 80.3800 Acc@5 95.4040 Samples 50000
checkpoint-68: Acc@1 80.4920 Acc@5 95.3920 Samples 50000
checkpoint-69: Acc@1 80.5640 Acc@5 95.3780 Samples 50000
checkpoint-70: Acc@1 80.4980 Acc@5 95.4500 Samples 50000
checkpoint-71: Acc@1 80.5920 Acc@5 95.3900 Samples 50000
checkpoint-72: Acc@1 80.5660 Acc@5 95.3840 Samples 50000
checkpoint-73: Acc@1 80.3520 Acc@5 95.3820 Samples 50000
checkpoint-74: Acc@1 80.4880 Acc@5 95.4520 Samples 50000
checkpoint-75: Acc@1 80.5020 Acc@5 95.4240 Samples 50000
checkpoint-76: Acc@1 80.4300 Acc@5 95.3160 Samples 50000
checkpoint-77: Acc@1 80.6360 Acc@5 95.4080 Samples 50000
checkpoint-78: Acc@1 80.4980 Acc@5 95.4560 Samples 50000
checkpoint-79: Acc@1 80.4920 Acc@5 95.3860 Samples 50000
checkpoint-80: Acc@1 80.5840 Acc@5 95.4720 Samples 50000
```

和上一版 100ep sparse prev-step KL 的同期对比：

```text
checkpoint-51:
  no-KL = 80.2140
  KL    = 80.2140

checkpoint-60:
  no-KL = 80.5260
  KL    = 80.4480

checkpoint-71:
  no-KL = 80.5920
  KL    = 80.5720

checkpoint-80:
  no-KL = 80.5840
  KL    = 80.6500

best by checkpoint-80:
  no-KL best = checkpoint-77 Top-1 80.6360
  KL best    = checkpoint-80 Top-1 80.6500
  KL - no-KL = +0.0140
```

GPU / 训练状态：

```text
8 x H100, memory about 28303 MiB, util 100%
training continues into epoch 80
```

阶段结论：

```text
checkpoint-1 到 checkpoint-80 已完整生成，Samples 全 50000。
RefW 始终为 0，controller artifact absent，no-KL 对照边界仍然干净。
到 checkpoint-80 为止，no-KL 自然 OFQ 曲线已经达到 best 80.6360，并且在 checkpoint-60 / 71 上不低于上一版 KL，checkpoint-80 best 只比 KL 低 0.0140。
这说明上一版 sparse prev-step KL 在 51->80 阶段没有表现出明确、显著的正增益；目前 80.6 左右主要可以由 OFQ public-family 自然长跑达到。
继续跑到 checkpoint-100 做最终消融审计，最终结论仍以 best / last20_avg / last10_avg / above-threshold counts 为准。
```

## 2026-07-15 final audit

最终 monitor 摘要：

```text
checkpoint_count=100
latest_checkpoint=checkpoint-100.pth.tar
fullval_rows=100
bad_sample_rows=0
best_fullval_line=checkpoint-82 Loss 0.8276 Acc@1 80.7920 Acc@5 95.4100 Samples 50000
above_baseline_lines=20
above_scheme_c_lines=12
above_original10to60_lines=5
above_original10to110_lines=2
above_dynamic10to110_lines=2
above_kl100ep_lines=2
above_late10to210_lines=0
target_81_lines=0
last20_avg=80.6916
last10_avg=80.7086
controller_artifact=absent
nonzero_refw_lines=0
nonzero_refw_epochs=NA
```

checkpoint-81 到 checkpoint-100：

```text
checkpoint-81: Acc@1 80.6360 Acc@5 95.5100 Samples 50000
checkpoint-82: Acc@1 80.7920 Acc@5 95.4100 Samples 50000
checkpoint-83: Acc@1 80.7180 Acc@5 95.4440 Samples 50000
checkpoint-84: Acc@1 80.6940 Acc@5 95.4620 Samples 50000
checkpoint-85: Acc@1 80.6900 Acc@5 95.4640 Samples 50000
checkpoint-86: Acc@1 80.6300 Acc@5 95.4440 Samples 50000
checkpoint-87: Acc@1 80.5360 Acc@5 95.4600 Samples 50000
checkpoint-88: Acc@1 80.6580 Acc@5 95.5180 Samples 50000
checkpoint-89: Acc@1 80.7080 Acc@5 95.4600 Samples 50000
checkpoint-90: Acc@1 80.6840 Acc@5 95.4500 Samples 50000
checkpoint-91: Acc@1 80.7320 Acc@5 95.4300 Samples 50000
checkpoint-92: Acc@1 80.6620 Acc@5 95.4300 Samples 50000
checkpoint-93: Acc@1 80.6880 Acc@5 95.4140 Samples 50000
checkpoint-94: Acc@1 80.6680 Acc@5 95.4600 Samples 50000
checkpoint-95: Acc@1 80.7400 Acc@5 95.4660 Samples 50000
checkpoint-96: Acc@1 80.7200 Acc@5 95.4600 Samples 50000
checkpoint-97: Acc@1 80.6720 Acc@5 95.4980 Samples 50000
checkpoint-98: Acc@1 80.7760 Acc@5 95.4620 Samples 50000
checkpoint-99: Acc@1 80.7500 Acc@5 95.5380 Samples 50000
checkpoint-100: Acc@1 80.6780 Acc@5 95.4840 Samples 50000
```

关键 checkpoint：

```text
checkpoint-40: Acc@1 79.9620
checkpoint-51: Acc@1 80.2140
checkpoint-60: Acc@1 80.5260
checkpoint-71: Acc@1 80.5920
checkpoint-80: Acc@1 80.5840
checkpoint-82: Acc@1 80.7920
checkpoint-90: Acc@1 80.6840
checkpoint-100: Acc@1 80.6780
```

完成性审计：

```text
checkpoint-1 到 checkpoint-100: present, missing=[]
full-val rows: 100
Samples=50000: all rows, bad_sample_rows=0
args.yaml: present
resume: ''
train_scheme: baseline
dynamic_sparse_prevstep_kl: false
ref_attn_kl_weight: 0.0
epochs: 100
scheduler_epochs: 100
batch_size: 64
lr: 0.0002
min_lr: 5e-06
weight_decay: 0.0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
qk_reparam: true
qk_reparam_type: 0
checkpoint_hist: 100
epoch_checkpoint_interval: 1
pretrained: true
pretrained_initialized: true
controller artifact: absent
RefW nonzero rows: 0
training process: ended
GPU after completion: 8 x H100 idle, about 7 MiB each
```

退出说明：

```text
日志末尾有 TCPStore / ProcessGroupNCCL 退出 warning。
这是 rank 退出时 TCPStore 已关闭导致的清理阶段 warning；checkpoint-100、full-val row=100、wall_seconds、输出目录和所有 checkpoint 均已完整存在。
因此该 warning 不影响本实验完成性和结果有效性。
```

与上一版 100ep sparse prev-step KL 的严格消融对比：

```text
100ep sparse prev-step KL:
  best: checkpoint-100 Top-1 80.7720 Top-5 95.4320
  last20_avg: 80.6965
  last10_avg: 80.7316
  above_baseline_lines: 20
  above_scheme_c_lines: 14
  above_original10to60_lines: 6
  above_original10to110_lines: 3
  above_dynamic10to110_lines: 2
  above_late10to210_lines: 0
  target_81_lines: 0

100ep no-KL strict control:
  best: checkpoint-82 Top-1 80.7920 Top-5 95.4100
  last20_avg: 80.6916
  last10_avg: 80.7086
  above_baseline_lines: 20
  above_scheme_c_lines: 12
  above_original10to60_lines: 5
  above_original10to110_lines: 2
  above_dynamic10to110_lines: 2
  above_kl100ep_lines: 2
  above_late10to210_lines: 0
  target_81_lines: 0
```

差值：

```text
best: no-KL 80.7920 - KL 80.7720 = +0.0200
last20_avg: no-KL 80.6916 - KL 80.6965 = -0.0049
last10_avg: no-KL 80.7086 - KL 80.7316 = -0.0230
```

阶段对比：

```text
checkpoint-51:
  no-KL = 80.2140
  KL    = 80.2140

checkpoint-60:
  no-KL = 80.5260
  KL    = 80.4480

checkpoint-71:
  no-KL = 80.5920
  KL    = 80.5720

checkpoint-80:
  no-KL = 80.5840
  KL    = 80.6500

best:
  no-KL = 80.7920
  KL    = 80.7720
```

最终结论：

```text
本实验完成了 100epoch 严格 no-KL 对照。
它和上一版 KL 实验在 checkpoint-1 到 checkpoint-51 完全对齐，说明 public pretrained / OFQ 主链路一致。
从 checkpoint-52 后，no-KL 自然曲线并不弱于 KL 曲线；最终 best 80.7920 反而高于 KL best 80.7720。

因此，上一版 100epoch sparse prev-step KL 的 80.7720 主要不是 KL 带来的明确收益，而是 OFQ public-family 自然长跑可以达到的高度。
当前 sparse prev-step KL controller 在 100epoch from-pretrained 设置下没有显示出稳定正贡献；它可能只是对个别局部回落有微弱保峰效果，但不是推动 80.77 的主要原因，也不是冲 81 的关键。

下一步如果继续做算法改进，不建议继续沿用当前固定-head sparse prev-step KL 配置直接加长。
更合理的方向是：
1. 先把 no-KL public-family baseline 固定为新的强对照；
2. 若继续做 KL，需要重新设计更晚、更少、更有判别力的触发逻辑，或动态 head 检测；
3. 81.0 目标更可能需要 warmup / scheduler / quantizer 稳定性改造，而不是当前版本的 sparse prev-step KL controller。
```
