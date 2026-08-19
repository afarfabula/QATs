# OFQ resume10->110 original OFQ public-family progress

## 目标

运行一版原版 OFQ public-family `checkpoint-10 -> checkpoint-110` 的 100 epoch 对照训练，用来和 dynamic sparse prev-step KL 10->110 长跑做直接对比。

本实验必须不启用：

```text
train_scheme=ema_ref_attn_kl
ref_update / ref_model
ref_attn_kl_weight
ref_attn_kl_weight_epoch_overrides
dynamic sparse prev-step KL controller
anchor-ref KL
soup
checkpoint averaging
ensemble
A8->A4
```

## 对照对象

dynamic sparse prev-step KL 10->110：

```text
progress: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_progress_20260710.md
output: /tmp/qat_public_repro/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710
best checkpoint: checkpoint-100
best Top-1: 80.7600
best Top-5: 95.4020
above_baseline_lines: 30
above_scheme_c_lines: 7
above_original_lines: 1
target_81_lines: 0
```

## 实验名和路径

```text
experiment: ofq_resume10_to110_original_ofq_public_20260711
output: /tmp/qat_public_repro/ofq_resume10_to110_original_ofq_public_20260711
log: /mlx_devbox/users/quyanyi/playground/train_ofq_resume10_to110_original_ofq_public_20260711.log
run script: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume10_to110_original_ofq_public_20260711.sh
monitor script: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/monitor_ofq_resume10_to110_original_ofq_public_20260711.sh
status TSV: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_original_ofq_public_status_20260711.tsv
refw TSV: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_original_ofq_public_refw_20260711.tsv
summary: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_original_ofq_public_monitor_summary_20260711.txt
```

## 配置

起点：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
```

保持 OFQ public-family 主链路：

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
epochs=110
scheduler_epochs=110
```

## 成功/对比指标

对比阈值：

```text
baseline: 80.5980
scheme C best: 80.6820
original OFQ 10->60 best: 80.7240
dynamic KL 10->110 best: 80.7600
81.0 target
```

最终审计：

```text
checkpoint-11 到 checkpoint-110 是否完整生成
full-val rows 是否完整且 Samples=50000
RefW 是否始终为 0
args.yaml 是否没有 train_scheme/ref/dynamic KL 参数
best checkpoint
超过 80.5980 / 80.6820 / 80.7240 / 80.7600 的 checkpoint 数量
是否达到 81.0
与 dynamic sparse prev-step KL 的 best、稳定性和后段均值对比
```

## 2026-07-11 preflight

已完成：

```text
bash -n run script: passed
bash -n monitor script: passed
dry-run: passed
forbidden KL/ref/controller args in dry-run: none
worker GPU: 8 x H100 visible, idle
checkpoint-10: exists
teacher checkpoint: exists
train shards: 294
validation shards: 14
```

dry-run 核心命令确认：

```text
--epochs 110 --scheduler-epochs 110
--resume checkpoint-10.pth.tar
--no-resume-opt
--checkpoint-hist 110
--epoch-checkpoint-interval 1
--wq-mode statsq --aq-mode lsq
--qk_reparam --qk_reparam_type 0
--kd_hard_and_soft 0
--teacher-soft-temperature 2.75
```

确认不包含：

```text
--train-scheme
--ref-*
--anchor-ref-*
--dynamic-*
ema_ref
attn-kl
```

下一步：

```text
在 worker 上后台启动训练。
启动后检查 args.yaml 和第一条 Train 日志，确认原版链路 RefW=0。
```

## 2026-07-11 launch

启动命令：

```text
cd /mlx_devbox/users/quyanyi/playground
MASTER_PORT=31821 OUT=/tmp/qat_public_repro nohup bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume10_to110_original_ofq_public_20260711.sh >/tmp/ofq_resume10_to110_original_ofq_public_20260711.nohup 2>&1 &
```

进程：

```text
launcher pid: 126825
script pid: 126826
qat_launch pid: 126842
8 rank processes spawned
```

启动质量证据：

```text
Strict resume: loaded model from checkpoint-10; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Model swin_t created, param count:28608256
Train: 10 [0/2502] ... RefW: 0.000e+00
```

args.yaml 关键项：

```text
train_scheme: baseline
dynamic_sparse_prevstep_kl: false
ref_attn_kl_weight: 0.0
ref_attn_kl_weight_epoch_overrides: {}
ref_head_mode: all
ref_update: ema
ref_attn_kl_drop_prob: 1.0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
batch_size: 64
checkpoint_hist: 110
epoch_checkpoint_interval: 1
qk_reparam: true
qk_reparam_type: 0
```

说明：

```text
args.yaml 会保留 dynamic/ref 相关默认字段，但 dynamic_sparse_prevstep_kl=false、train_scheme=baseline、RefW=0，且启动命令没有 --train-scheme / --ref-* / --dynamic-*。
因此该 run 按原版 OFQ public-family 对照处理。
```

启动后 monitor：

```text
output_exists=remote:/tmp/qat_public_repro/ofq_resume10_to110_original_ofq_public_20260711
args_yaml=present
checkpoint_count=0
fullval_rows=0
nonzero_refw_lines=0
```

## 2026-07-11 checkpoint-11

monitor 摘要：

```text
checkpoint_count=1
latest_checkpoint=checkpoint-11.pth.tar
fullval_rows=1
bad_sample_rows=0
best_fullval_line=checkpoint-11 Loss 0.8448 Acc@1 80.3360 Acc@5 95.2440 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original_lines=0
above_dynamic_lines=0
target_81_lines=0
nonzero_refw_lines=0
```

对比 dynamic KL 10->110：

```text
dynamic checkpoint-11: Acc@1 80.3360
original checkpoint-11: Acc@1 80.3360
delta original - dynamic: 0.0000
```

结论：

```text
原版 OFQ 10->110 对照已完成第一个 resumed epoch。
checkpoint-11 与 dynamic KL run 的 checkpoint-11 完全一致，说明起点和早期主链路对齐。
RefW=0，符合无 KL 对照要求。
```

## 2026-07-11 checkpoint-15

monitor 摘要：

```text
checkpoint_count=5
latest_checkpoint=checkpoint-15.pth.tar
fullval_rows=5
bad_sample_rows=0
best_fullval_line=checkpoint-12 Loss 0.8443 Acc@1 80.3760 Acc@5 95.2920 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original_lines=0
above_dynamic_lines=0
target_81_lines=0
last20_avg=80.3600
nonzero_refw_lines=0
```

full-val 明细：

```text
checkpoint-11: Acc@1 80.3360 Acc@5 95.2440 Samples 50000
checkpoint-12: Acc@1 80.3760 Acc@5 95.2920 Samples 50000
checkpoint-13: Acc@1 80.3460 Acc@5 95.2740 Samples 50000
checkpoint-14: Acc@1 80.3700 Acc@5 95.3000 Samples 50000
checkpoint-15: Acc@1 80.3720 Acc@5 95.3000 Samples 50000
```

对比 dynamic KL 10->110：

```text
dynamic checkpoint-11..15:
80.3360, 80.3760, 80.3460, 80.3700, 80.3720

original checkpoint-11..15:
80.3360, 80.3760, 80.3460, 80.3700, 80.3720

delta: all 0.0000
```

结论：

```text
原版 OFQ 10->110 对照在前 5 个 checkpoint 与 dynamic KL run 完全一致。
说明 dynamic run 在 10-15 期间虽然走 baseline-equivalent training behavior，但额外 runtime 路径尚未产生可观测差异。
RefW 始终为 0，符合无 KL 对照要求。
```

## 2026-07-11 checkpoint-18

monitor 摘要：

```text
checkpoint_count=8
latest_checkpoint=checkpoint-18.pth.tar
fullval_rows=8
bad_sample_rows=0
best_fullval_line=checkpoint-16 Loss 0.8435 Acc@1 80.3800 Acc@5 95.3140 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original_lines=0
above_dynamic_lines=0
target_81_lines=0
last20_avg=80.3558
nonzero_refw_lines=0
```

full-val 明细：

```text
checkpoint-11: Acc@1 80.3360 Acc@5 95.2440 Samples 50000
checkpoint-12: Acc@1 80.3760 Acc@5 95.2920 Samples 50000
checkpoint-13: Acc@1 80.3460 Acc@5 95.2740 Samples 50000
checkpoint-14: Acc@1 80.3700 Acc@5 95.3000 Samples 50000
checkpoint-15: Acc@1 80.3720 Acc@5 95.3000 Samples 50000
checkpoint-16: Acc@1 80.3800 Acc@5 95.3140 Samples 50000
checkpoint-17: Acc@1 80.3640 Acc@5 95.3100 Samples 50000
checkpoint-18: Acc@1 80.3020 Acc@5 95.2420 Samples 50000
```

对比 dynamic KL 10->110：

```text
dynamic checkpoint-11..18:
80.3360,80.3760,80.3460,80.3700,80.3720,80.3800,80.3640,80.3020

original checkpoint-11..18:
80.3360,80.3760,80.3460,80.3700,80.3720,80.3800,80.3640,80.3020

delta: all 0.0000
```

结论：

```text
原版 OFQ 对照到 checkpoint-18 仍与 dynamic KL run 早期完全一致。
RefW=0，证明无 KL 对照路径正常。
继续监控到 checkpoint-30/40/60，确认原版对照是否复现 dynamic run 前 60 observe 段和后段无 KL 差异。
```

## 2026-07-11 checkpoint-23

monitor 摘要：

```text
checkpoint_count=13
latest_checkpoint=checkpoint-23.pth.tar
fullval_rows=13
bad_sample_rows=0
best_fullval_line=checkpoint-22 Loss 0.8378 Acc@1 80.5440 Acc@5 95.3640 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original_lines=0
above_dynamic_lines=0
target_81_lines=0
last20_avg=80.3863
nonzero_refw_lines=0
```

full-val 明细：

```text
checkpoint-20: Acc@1 80.4580 Acc@5 95.3380 Samples 50000
checkpoint-21: Acc@1 80.4660 Acc@5 95.3400 Samples 50000
checkpoint-22: Acc@1 80.5440 Acc@5 95.3640 Samples 50000
checkpoint-23: Acc@1 80.3720 Acc@5 95.3100 Samples 50000
```

对比 dynamic KL 10->110：

```text
dynamic checkpoint-20: 80.4580
dynamic checkpoint-21: 80.4660
dynamic checkpoint-22: 80.5440
dynamic checkpoint-23: 80.3720

original checkpoint-20: 80.4580
original checkpoint-21: 80.4660
original checkpoint-22: 80.5440
original checkpoint-23: 80.3720

delta: all 0.0000
```

结论：

```text
原版 OFQ 对照到 checkpoint-23 仍与 dynamic run 的 observe 段完全一致。
这说明 dynamic run 在 KL 触发前的数值路径和无 KL 原版对照至少到 checkpoint-23 完全一致。
RefW=0，符合无 KL 对照要求。
```

## 2026-07-11 checkpoint-34

monitor 摘要：

```text
checkpoint_count=24
latest_checkpoint=checkpoint-34.pth.tar
fullval_rows=24
bad_sample_rows=0
best_fullval_line=checkpoint-32 Loss 0.8373 Acc@1 80.5980 Acc@5 95.3180 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original_lines=0
above_dynamic_lines=0
target_81_lines=0
last20_avg=80.4246
nonzero_refw_lines=0
```

最近 full-val：

```text
checkpoint-30: Acc@1 80.4880 Acc@5 95.3300 Samples 50000
checkpoint-31: Acc@1 80.3800 Acc@5 95.2800 Samples 50000
checkpoint-32: Acc@1 80.5980 Acc@5 95.3180 Samples 50000
checkpoint-33: Acc@1 80.5200 Acc@5 95.3880 Samples 50000
checkpoint-34: Acc@1 80.4040 Acc@5 95.3000 Samples 50000
```

对比 dynamic KL 10->110：

```text
dynamic checkpoint-30: 80.4880
dynamic checkpoint-31: 80.3800
dynamic checkpoint-32: 80.5980
dynamic checkpoint-33: 80.5200
dynamic checkpoint-34: 80.4040

original checkpoint-30: 80.4880
original checkpoint-31: 80.3800
original checkpoint-32: 80.5980
original checkpoint-33: 80.5200
original checkpoint-34: 80.4040

delta: all 0.0000
```

结论：

```text
原版 OFQ 对照到 checkpoint-34 仍与 dynamic run 的 observe 段完全一致。
dynamic run 前 60 epoch 的“observe-only”结果并非由 refmodel/KL runtime 造成差异，至少到 checkpoint-34 与原版完全对齐。
RefW=0，符合无 KL 对照要求。
```

## 2026-07-11 checkpoint-36

monitor 摘要：

```text
checkpoint_count=26
latest_checkpoint=checkpoint-36.pth.tar
fullval_rows=26
bad_sample_rows=0
best_fullval_line=checkpoint-36 Loss 0.8394 Acc@1 80.6020 Acc@5 95.3420 Samples 50000
above_baseline_lines=1
above_scheme_c_lines=0
above_original_lines=0
above_dynamic_lines=0
target_81_lines=0
last20_avg=80.4396
nonzero_refw_lines=0
```

最近 full-val：

```text
checkpoint-32: Acc@1 80.5980 Acc@5 95.3180 Samples 50000
checkpoint-33: Acc@1 80.5200 Acc@5 95.3880 Samples 50000
checkpoint-34: Acc@1 80.4040 Acc@5 95.3000 Samples 50000
checkpoint-35: Acc@1 80.4500 Acc@5 95.3840 Samples 50000
checkpoint-36: Acc@1 80.6020 Acc@5 95.3420 Samples 50000
```

对比 dynamic KL 10->110：

```text
dynamic checkpoint-35: 80.4500
dynamic checkpoint-36: 80.6020

original checkpoint-35: 80.4500
original checkpoint-36: 80.6020

delta: all 0.0000
```

结论：

```text
原版 OFQ 对照到 checkpoint-36 仍与 dynamic run 的 observe 段完全一致。
checkpoint-36 首次超过 baseline 80.5980，与 dynamic run 同点完全一致。
RefW=0，符合无 KL 对照要求。
```

## 2026-07-11 checkpoint-40

monitor 摘要：

```text
checkpoint_count=30
latest_checkpoint=checkpoint-40.pth.tar
fullval_rows=30
bad_sample_rows=0
best_fullval_line=checkpoint-36 Loss 0.8394 Acc@1 80.6020 Acc@5 95.3420 Samples 50000
above_baseline_lines=1
above_scheme_c_lines=0
above_original_lines=0
above_dynamic_lines=0
target_81_lines=0
last20_avg=80.4680
nonzero_refw_lines=0
```

最近 full-val：

```text
checkpoint-36: Acc@1 80.6020 Acc@5 95.3420 Samples 50000
checkpoint-37: Acc@1 80.5460 Acc@5 95.4000 Samples 50000
checkpoint-38: Acc@1 80.5040 Acc@5 95.3220 Samples 50000
checkpoint-39: Acc@1 80.5040 Acc@5 95.3260 Samples 50000
checkpoint-40: Acc@1 80.4740 Acc@5 95.3960 Samples 50000
```

对比 dynamic KL 10->110：

```text
dynamic checkpoint-38: 80.5040
dynamic checkpoint-39: 80.5040
dynamic checkpoint-40: 80.4740

original checkpoint-38: 80.5040
original checkpoint-39: 80.5040
original checkpoint-40: 80.4740

delta: all 0.0000
```

args.yaml 复核：

```text
train_scheme: baseline
dynamic_sparse_prevstep_kl: false
ref_attn_kl_weight: 0.0
ref_attn_kl_weight_epoch_overrides: {}
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
batch_size: 64
checkpoint_hist: 110
epoch_checkpoint_interval: 1
qk_reparam: true
qk_reparam_type: 0
wq_mode: statsq
aq_mode: lsq
epochs: 110
no_resume_opt: true
```

结论：

```text
原版 OFQ 对照到 checkpoint-40 仍与 dynamic run 的 observe 段完全一致。
因此 dynamic run 在 KL/controller 尚未触发前没有引入额外训练路径差异；当前差异比较仍可从 KL 真正触发后的区间开始分析。
checkpoint-40 仍低于 baseline / scheme C / original 10->60 best / dynamic best / 81.0。
RefW=0，符合无 KL 对照要求。
```

## 2026-07-11 checkpoint-50

monitor 摘要：

```text
checkpoint_count=40
latest_checkpoint=checkpoint-50.pth.tar
fullval_rows=40
bad_sample_rows=0
best_fullval_line=checkpoint-36 Loss 0.8394 Acc@1 80.6020 Acc@5 95.3420 Samples 50000
above_baseline_lines=1
above_scheme_c_lines=0
above_original_lines=0
above_dynamic_lines=0
target_81_lines=0
last20_avg=80.5263
nonzero_refw_lines=0
```

最近 full-val：

```text
checkpoint-41: Acc@1 80.5280 Acc@5 95.3220 Samples 50000
checkpoint-42: Acc@1 80.5480 Acc@5 95.3680 Samples 50000
checkpoint-43: Acc@1 80.5740 Acc@5 95.3700 Samples 50000
checkpoint-44: Acc@1 80.4960 Acc@5 95.2740 Samples 50000
checkpoint-45: Acc@1 80.5420 Acc@5 95.3280 Samples 50000
checkpoint-46: Acc@1 80.5320 Acc@5 95.3120 Samples 50000
checkpoint-47: Acc@1 80.5780 Acc@5 95.3820 Samples 50000
checkpoint-48: Acc@1 80.5840 Acc@5 95.3080 Samples 50000
checkpoint-49: Acc@1 80.5820 Acc@5 95.3620 Samples 50000
checkpoint-50: Acc@1 80.5800 Acc@5 95.3700 Samples 50000
```

对比 dynamic KL 10->110：

```text
dynamic checkpoint-41: 80.5280
dynamic checkpoint-42: 80.5480
dynamic checkpoint-43: 80.5740
dynamic checkpoint-44: 80.4960
dynamic checkpoint-45: 80.5420
dynamic checkpoint-46: 80.5320
dynamic checkpoint-47: 80.5780
dynamic checkpoint-48: 80.5840
dynamic checkpoint-49: 80.5820
dynamic checkpoint-50: 80.5800

original checkpoint-41: 80.5280
original checkpoint-42: 80.5480
original checkpoint-43: 80.5740
original checkpoint-44: 80.4960
original checkpoint-45: 80.5420
original checkpoint-46: 80.5320
original checkpoint-47: 80.5780
original checkpoint-48: 80.5840
original checkpoint-49: 80.5820
original checkpoint-50: 80.5800

delta: all 0.0000
```

结论：

```text
原版 OFQ 对照到 checkpoint-50 仍与 dynamic run 的 observe 段完全一致。
checkpoint-41 到 checkpoint-50 都低于 baseline 80.5980；当前唯一超过 baseline 的点仍是 checkpoint-36。
这进一步确认 dynamic KL run 在 controller/KL 触发前与原版 OFQ 主链路完全可比，后续差异应重点看 checkpoint-61 之后。
RefW=0，符合无 KL 对照要求。
```

## 2026-07-11 checkpoint-61

monitor 摘要：

```text
checkpoint_count=51
latest_checkpoint=checkpoint-61.pth.tar
fullval_rows=51
bad_sample_rows=0
best_fullval_line=checkpoint-59 Loss 0.8309 Acc@1 80.6460 Acc@5 95.3240 Samples 50000
above_baseline_lines=3
above_scheme_c_lines=0
above_original_lines=0
above_dynamic_lines=0
target_81_lines=0
last20_avg=80.5473
nonzero_refw_lines=0
```

最近 full-val：

```text
checkpoint-51: Acc@1 80.5200 Acc@5 95.3340 Samples 50000
checkpoint-52: Acc@1 80.5120 Acc@5 95.3500 Samples 50000
checkpoint-53: Acc@1 80.5300 Acc@5 95.3280 Samples 50000
checkpoint-54: Acc@1 80.4860 Acc@5 95.3340 Samples 50000
checkpoint-55: Acc@1 80.4300 Acc@5 95.3240 Samples 50000
checkpoint-56: Acc@1 80.5120 Acc@5 95.3700 Samples 50000
checkpoint-57: Acc@1 80.6200 Acc@5 95.4140 Samples 50000
checkpoint-58: Acc@1 80.5560 Acc@5 95.3520 Samples 50000
checkpoint-59: Acc@1 80.6460 Acc@5 95.3240 Samples 50000
checkpoint-60: Acc@1 80.5900 Acc@5 95.4140 Samples 50000
checkpoint-61: Acc@1 80.5280 Acc@5 95.4380 Samples 50000
```

对比 dynamic KL 10->110：

```text
dynamic checkpoint-51: 80.5200
dynamic checkpoint-52: 80.5120
dynamic checkpoint-53: 80.5300
dynamic checkpoint-54: 80.4860
dynamic checkpoint-55: 80.4300
dynamic checkpoint-56: 80.5120
dynamic checkpoint-57: 80.6200
dynamic checkpoint-58: 80.5560
dynamic checkpoint-59: 80.6460
dynamic checkpoint-60: 80.5900
dynamic checkpoint-61: 80.5280

original checkpoint-51: 80.5200
original checkpoint-52: 80.5120
original checkpoint-53: 80.5300
original checkpoint-54: 80.4860
original checkpoint-55: 80.4300
original checkpoint-56: 80.5120
original checkpoint-57: 80.6200
original checkpoint-58: 80.5560
original checkpoint-59: 80.6460
original checkpoint-60: 80.5900
original checkpoint-61: 80.5280

delta: all 0.0000
```

结论：

```text
原版 OFQ 对照到 checkpoint-61 仍与 dynamic run 完全一致。
checkpoint-59 刷新原版当前最好 Top-1 到 80.6460，但仍低于 scheme C 80.6820、original 10->60 best 80.7240、dynamic best 80.7600 和 81.0。
checkpoint-57 / checkpoint-59 / checkpoint-60 附近是原版 OFQ 的自然高点窗口；后续 dynamic KL 的有效性应看 KL 触发后是否能在这个自然高点上方继续抬升。
RefW=0，符合无 KL 对照要求。
```

## 2026-07-11 checkpoint-70

monitor 摘要：

```text
checkpoint_count=60
latest_checkpoint=checkpoint-70.pth.tar
fullval_rows=60
bad_sample_rows=0
best_fullval_line=checkpoint-68 Loss 0.8372 Acc@1 80.6700 Acc@5 95.3960 Samples 50000
above_baseline_lines=7
above_scheme_c_lines=0
above_original_lines=0
above_dynamic_lines=0
target_81_lines=0
last20_avg=80.5628
nonzero_refw_lines=0
```

原版 full-val：

```text
checkpoint-62: Acc@1 80.6180 Acc@5 95.3320 Samples 50000
checkpoint-63: Acc@1 80.5500 Acc@5 95.4160 Samples 50000
checkpoint-64: Acc@1 80.5960 Acc@5 95.3500 Samples 50000
checkpoint-65: Acc@1 80.5960 Acc@5 95.3940 Samples 50000
checkpoint-66: Acc@1 80.6380 Acc@5 95.3740 Samples 50000
checkpoint-67: Acc@1 80.6360 Acc@5 95.3300 Samples 50000
checkpoint-68: Acc@1 80.6700 Acc@5 95.3960 Samples 50000
checkpoint-69: Acc@1 80.4860 Acc@5 95.3820 Samples 50000
checkpoint-70: Acc@1 80.5360 Acc@5 95.3800 Samples 50000
```

dynamic KL 对比：

```text
checkpoint-62: original 80.6180, dynamic 80.6180, delta +0.0000
checkpoint-63: original 80.5500, dynamic 80.5500, delta +0.0000
checkpoint-64: original 80.5960, dynamic 80.5300, delta -0.0660
checkpoint-65: original 80.5960, dynamic 80.5620, delta -0.0340
checkpoint-66: original 80.6380, dynamic 80.6580, delta +0.0200
checkpoint-67: original 80.6360, dynamic 80.6700, delta +0.0340
checkpoint-68: original 80.6700, dynamic 80.6520, delta -0.0180
checkpoint-69: original 80.4860, dynamic 80.6160, delta +0.1300
checkpoint-70: original 80.5360, dynamic 80.6500, delta +0.1140
```

dynamic controller / RefW 证据：

```text
epoch 62: triggered next head 8:4 weight 1e-05, applied RefW still 0
epoch 63: applied head 8:4 RefW max 1e-05, next head 5:7
epoch 64: applied head 5:7 RefW max 1e-05, next head 4:11
epoch 65: applied head 4:11 RefW max 1e-05
epoch 66-70: applied RefW 0

original epoch 62-70: RefW max 0 for every checked epoch
```

结论：

```text
checkpoint-62/63 仍完全一致；dynamic KL 实际应用后的 checkpoint-64/65 短期低于原版，checkpoint-66/67 略高，checkpoint-68 与原版自然高点接近但略低。
checkpoint-69/70 处 dynamic 明显高于原版，主要因为原版从 checkpoint-68 的 80.6700 回落到 80.4860/80.5360，而 dynamic 维持在 80.6160/80.6500。
这说明当前 KL 的早期收益更像“抑制自然高点后的回落”，不是立即把峰值抬到 scheme C 以上；到 checkpoint-70 为止，两者都还没有超过 scheme C 80.6820、original 10->60 best 80.7240、dynamic best 80.7600 或 81.0。
RefW=0，原版对照仍满足无 KL 要求。
```

## 2026-07-11 checkpoint-81

monitor 摘要：

```text
checkpoint_count=71
latest_checkpoint=checkpoint-81.pth.tar
fullval_rows=71
bad_sample_rows=0
best_fullval_line=checkpoint-79 Loss 0.8329 Acc@1 80.6960 Acc@5 95.3980 Samples 50000
above_baseline_lines=12
above_scheme_c_lines=2
above_original_lines=0
above_dynamic_lines=0
target_81_lines=0
last20_avg=80.5920
nonzero_refw_lines=0
```

原版 full-val：

```text
checkpoint-71: Acc@1 80.5700 Acc@5 95.3620 Samples 50000
checkpoint-72: Acc@1 80.6000 Acc@5 95.2940 Samples 50000
checkpoint-73: Acc@1 80.5760 Acc@5 95.3900 Samples 50000
checkpoint-74: Acc@1 80.6140 Acc@5 95.4340 Samples 50000
checkpoint-75: Acc@1 80.4820 Acc@5 95.3860 Samples 50000
checkpoint-76: Acc@1 80.4720 Acc@5 95.3660 Samples 50000
checkpoint-77: Acc@1 80.6900 Acc@5 95.4580 Samples 50000
checkpoint-78: Acc@1 80.5640 Acc@5 95.3700 Samples 50000
checkpoint-79: Acc@1 80.6960 Acc@5 95.3980 Samples 50000
checkpoint-80: Acc@1 80.6560 Acc@5 95.4120 Samples 50000
checkpoint-81: Acc@1 80.5940 Acc@5 95.4240 Samples 50000
```

dynamic KL 对比：

```text
checkpoint-71: original 80.5700, dynamic 80.5720, delta +0.0020
checkpoint-72: original 80.6000, dynamic 80.5840, delta -0.0160
checkpoint-73: original 80.5760, dynamic 80.5000, delta -0.0760
checkpoint-74: original 80.6140, dynamic 80.5420, delta -0.0720
checkpoint-75: original 80.4820, dynamic 80.7120, delta +0.2300
checkpoint-76: original 80.4720, dynamic 80.4920, delta +0.0200
checkpoint-77: original 80.6900, dynamic 80.5920, delta -0.0980
checkpoint-78: original 80.5640, dynamic 80.5320, delta -0.0320
checkpoint-79: original 80.6960, dynamic 80.6320, delta -0.0640
checkpoint-80: original 80.6560, dynamic 80.4960, delta -0.1600
checkpoint-81: original 80.5940, dynamic 80.6840, delta +0.0900
```

结论：

```text
checkpoint-71 到 checkpoint-81 不再支持“dynamic KL 持续领先”的简单结论。
dynamic 在 checkpoint-75 和 checkpoint-81 有明显高点，但原版在 checkpoint-77 / checkpoint-79 / checkpoint-80 形成更稳定的自然高点窗口，并在 checkpoint-79 达到 80.6960。
到 checkpoint-81 为止，原版 best 80.6960 已超过 scheme C 80.6820，但仍低于 original 10->60 best 80.7240、dynamic KL final best 80.7600 和 81.0。
这一段说明原版 OFQ 自身后段仍有自然抬升，最终比较必须看完整 10->110 的 best、窗口均值和后段稳定性，不能只看 checkpoint-69/70 的局部领先。
RefW=0，原版对照仍满足无 KL 要求。
```

## 2026-07-12 checkpoint-90

monitor 摘要：

```text
checkpoint_count=80
latest_checkpoint=checkpoint-90.pth.tar
fullval_rows=80
bad_sample_rows=0
best_fullval_line=checkpoint-79 Loss 0.8329 Acc@1 80.6960 Acc@5 95.3980 Samples 50000
above_baseline_lines=15
above_scheme_c_lines=2
above_original_lines=0
above_dynamic_lines=0
target_81_lines=0
last20_avg=80.5902
nonzero_refw_lines=0
```

原版 full-val：

```text
checkpoint-82: Acc@1 80.5820 Acc@5 95.3660 Samples 50000
checkpoint-83: Acc@1 80.6120 Acc@5 95.4320 Samples 50000
checkpoint-84: Acc@1 80.5780 Acc@5 95.3580 Samples 50000
checkpoint-85: Acc@1 80.6440 Acc@5 95.4120 Samples 50000
checkpoint-86: Acc@1 80.5200 Acc@5 95.3660 Samples 50000
checkpoint-87: Acc@1 80.5760 Acc@5 95.3540 Samples 50000
checkpoint-88: Acc@1 80.5620 Acc@5 95.3620 Samples 50000
checkpoint-89: Acc@1 80.5960 Acc@5 95.4120 Samples 50000
checkpoint-90: Acc@1 80.6200 Acc@5 95.3720 Samples 50000
```

dynamic KL 对比：

```text
checkpoint-82: original 80.5820, dynamic 80.6700, delta +0.0880
checkpoint-83: original 80.6120, dynamic 80.5840, delta -0.0280
checkpoint-84: original 80.5780, dynamic 80.5960, delta +0.0180
checkpoint-85: original 80.6440, dynamic 80.5520, delta -0.0920
checkpoint-86: original 80.5200, dynamic 80.6660, delta +0.1460
checkpoint-87: original 80.5760, dynamic 80.5620, delta -0.0140
checkpoint-88: original 80.5620, dynamic 80.5760, delta +0.0140
checkpoint-89: original 80.5960, dynamic 80.6100, delta +0.0140
checkpoint-90: original 80.6200, dynamic 80.6420, delta +0.0220
```

结论：

```text
checkpoint-82 到 checkpoint-90 原版没有刷新 checkpoint-79 的 best 80.6960，也没有超过 original 10->60 best 80.7240、dynamic best 80.7600 或 81.0。
dynamic 在这一段有轻微均值优势，尤其 checkpoint-82 / 86 / 90 高于原版；但峰值仍未超过原版 checkpoint-79 的 80.6960。
到 checkpoint-90 为止，KL 的优势更像分段稳定性/局部回落抑制，而不是稳定抬高峰值；最终还要看 90-110 后段是否出现 dynamic 的 checkpoint-100 高点，以及原版是否也自然出现同等高点。
RefW=0，原版对照仍满足无 KL 要求。
```

## 2026-07-12 checkpoint-100

monitor 摘要：

```text
checkpoint_count=90
latest_checkpoint=checkpoint-100.pth.tar
fullval_rows=90
bad_sample_rows=0
best_fullval_line=checkpoint-96 Loss 0.8280 Acc@1 80.7500 Acc@5 95.4240 Samples 50000
above_baseline_lines=23
above_scheme_c_lines=5
above_original_lines=1
above_dynamic_lines=0
target_81_lines=0
last20_avg=80.6201
nonzero_refw_lines=0
```

原版 full-val：

```text
checkpoint-91: Acc@1 80.6400 Acc@5 95.3340 Samples 50000
checkpoint-92: Acc@1 80.5540 Acc@5 95.3600 Samples 50000
checkpoint-93: Acc@1 80.6780 Acc@5 95.4200 Samples 50000
checkpoint-94: Acc@1 80.6880 Acc@5 95.3460 Samples 50000
checkpoint-95: Acc@1 80.6120 Acc@5 95.4080 Samples 50000
checkpoint-96: Acc@1 80.7500 Acc@5 95.4240 Samples 50000
checkpoint-97: Acc@1 80.5420 Acc@5 95.3940 Samples 50000
checkpoint-98: Acc@1 80.7120 Acc@5 95.3840 Samples 50000
checkpoint-99: Acc@1 80.6620 Acc@5 95.4000 Samples 50000
checkpoint-100: Acc@1 80.6800 Acc@5 95.4400 Samples 50000
```

dynamic KL 对比：

```text
checkpoint-91: original 80.6400, dynamic 80.6460, delta +0.0060
checkpoint-92: original 80.5540, dynamic 80.5700, delta +0.0160
checkpoint-93: original 80.6780, dynamic 80.6860, delta +0.0080
checkpoint-94: original 80.6880, dynamic 80.6920, delta +0.0040
checkpoint-95: original 80.6120, dynamic 80.5660, delta -0.0460
checkpoint-96: original 80.7500, dynamic 80.6260, delta -0.1240
checkpoint-97: original 80.5420, dynamic 80.7060, delta +0.1640
checkpoint-98: original 80.7120, dynamic 80.5560, delta -0.1560
checkpoint-99: original 80.6620, dynamic 80.6400, delta -0.0220
checkpoint-100: original 80.6800, dynamic 80.7600, delta +0.0800
```

结论：

```text
checkpoint-91 到 checkpoint-100 是目前最关键的对照段：dynamic 在 checkpoint-100 达到 80.7600，但原版无 KL 在 checkpoint-96 已达到 80.7500，只低 0.0100。
原版 checkpoint-96 首次超过 original 10->60 best 80.7240；这说明“10->110 原版 OFQ public-family 自然长跑”本身就几乎复现 dynamic KL 的最好峰值。
dynamic KL 在 checkpoint-100 的峰值仍是当前全局最高，但优势非常薄；如果最终 last20 / 90-110 均值没有明显更高，则 KL 的贡献更可能是改变峰值出现位置和局部稳定性，而不是显著抬高上限。
到 checkpoint-100 为止，两者都没有达到 81.0；原版 RefW=0，仍满足无 KL 对照要求。
```

## 2026-07-12 checkpoint-110 final audit

最终 monitor 摘要：

```text
checkpoint_count=100
latest_checkpoint=checkpoint-110.pth.tar
fullval_rows=100
bad_sample_rows=0
best_fullval_line=checkpoint-102 Loss 0.8284 Acc@1 80.7520 Acc@5 95.4300 Samples 50000
above_baseline_lines=33
above_scheme_c_lines=7
above_original_lines=2
above_dynamic_lines=0
target_81_lines=0
last20_avg=80.6519
nonzero_refw_lines=0
```

checkpoint-101 到 checkpoint-110：

```text
checkpoint-101: Acc@1 80.6800 Acc@5 95.4320 Samples 50000
checkpoint-102: Acc@1 80.7520 Acc@5 95.4300 Samples 50000
checkpoint-103: Acc@1 80.6080 Acc@5 95.3520 Samples 50000
checkpoint-104: Acc@1 80.6160 Acc@5 95.3600 Samples 50000
checkpoint-105: Acc@1 80.6640 Acc@5 95.4300 Samples 50000
checkpoint-106: Acc@1 80.6120 Acc@5 95.3880 Samples 50000
checkpoint-107: Acc@1 80.6380 Acc@5 95.4740 Samples 50000
checkpoint-108: Acc@1 80.6280 Acc@5 95.3780 Samples 50000
checkpoint-109: Acc@1 80.6360 Acc@5 95.4140 Samples 50000
checkpoint-110: Acc@1 80.6860 Acc@5 95.3360 Samples 50000
```

dynamic KL 对比：

```text
checkpoint-101: original 80.6800, dynamic 80.5780, delta -0.1020
checkpoint-102: original 80.7520, dynamic 80.6820, delta -0.0700
checkpoint-103: original 80.6080, dynamic 80.5960, delta -0.0120
checkpoint-104: original 80.6160, dynamic 80.6440, delta +0.0280
checkpoint-105: original 80.6640, dynamic 80.5740, delta -0.0900
checkpoint-106: original 80.6120, dynamic 80.6240, delta +0.0120
checkpoint-107: original 80.6380, dynamic 80.6340, delta -0.0040
checkpoint-108: original 80.6280, dynamic 80.7240, delta +0.0960
checkpoint-109: original 80.6360, dynamic 80.6000, delta -0.0360
checkpoint-110: original 80.6860, dynamic 80.6600, delta -0.0260
```

最终统计：

```text
original OFQ:
rows=100
bad_sample_rows=0
best=checkpoint-102 80.7520
above_baseline_80.5980=33
above_scheme_c_80.6820=7
above_original_10to60_best_80.7240=2
above_dynamic_best_80.7600=0
target_81=0
last20_avg=80.6519
last10_avg=80.6520
nonzero_refw_lines=0

dynamic sparse prev-step KL:
rows=100
bad_sample_rows=0
best=checkpoint-100 80.7600
above_baseline_80.5980=30
above_scheme_c_80.6820=7
above_original_10to60_best_80.7240=1
above_dynamic_best_80.7600=0
target_81=0
last20_avg=80.6382
last10_avg=80.6316
```

完成性审计：

```text
checkpoint-11 到 checkpoint-110: remote count=100, no missing checkpoint printed
full-val rows: 100
Samples=50000: all rows, bad_sample_rows=0
RefW: nonzero_lines=0, epochs=NA
args.yaml: present
train_scheme: baseline
dynamic_sparse_prevstep_kl: false
ref_attn_kl_weight: 0.0
ref_attn_kl_weight_epoch_overrides: {}
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
batch_size: 64
checkpoint_hist: 110
epoch_checkpoint_interval: 1
qk_reparam: true
qk_reparam_type: 0
wq_mode: statsq
aq_mode: lsq
epochs: 110
no_resume_opt: true
```

最终结论：

```text
原版 OFQ public-family 10->110 对照实验完整跑完，且是干净的无 KL 对照：checkpoint 完整、full-val 完整、Samples 全 50000、RefW 始终为 0、args.yaml 保持 baseline / no dynamic / no ref_attn_kl。

核心结论不是“dynamic KL 明显赢”，而是“原版 OFQ 长跑几乎追平 dynamic KL best”：
dynamic best = checkpoint-100 80.7600
original best = checkpoint-102 80.7520
差距只有 0.0080

原版在 last20_avg 和 last10_avg 上反而更高：
original last20_avg 80.6519 > dynamic last20_avg 80.6382
original last10_avg 80.6520 > dynamic last10_avg 80.6316

因此，这个 10->110 设定下，dynamic sparse prev-step KL 没有证明显著抬高最终上限；它主要改变了局部峰值出现位置，并在部分窗口抑制回落，但原版 OFQ 自然长跑本身已经达到同一水平。

两条 run 都没有达到 81.0。若后续还要坚持 Attention relation 震荡抑制方向，不能再只用本次 sparse KL 的 best acc 作为强证据；需要换更强的干预定义或更严格的统计评价，例如多 seed、窗口均值、震荡指标-收益相关性，而不是单点 best。
```
