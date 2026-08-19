# OFQ resume10->110 dynamic sparse prev-step KL progress

## 目标

按照 `/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_goal_20260710.md` 执行 10->110 的 100epoch dynamic sparse prev-step KL 实验。

核心要求：

```text
保持 OFQ public-family 主链路
从 checkpoint-10 resume 到 checkpoint-110
epoch 10-60 只观测，不主动开 KL
epoch 61-110 只在 rolling best 精度回落后极稀疏触发 prev-step KL
不使用 soup / checkpoint averaging / ensemble / A8->A4
持续维护中文进度文档和机器 TSV
```

## 实验名和路径

```text
experiment: ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710
output: /tmp/qat_public_repro/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710
log: /mlx_devbox/users/quyanyi/playground/train_ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710.log
run script: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710.sh
monitor script: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/monitor_ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710.sh
status TSV: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_status_20260710.tsv
refw TSV: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_refw_20260710.tsv
controller TSV: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_controller_20260710.tsv
summary: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_monitor_summary_20260710.txt
```

## 关键 baseline

```text
baseline: 80.5980
scheme C best: 80.6820
original OFQ 10->60 best: checkpoint-52 Top-1 80.7240
81.0 target: 未达到，需要本实验验证
```

最低通过：

```text
best Top-1 > 80.7240
```

有效通过：

```text
至少 2 个 checkpoint > 80.7240
或至少 5 个 checkpoint > 80.6820
```

强通过：

```text
best Top-1 >= 80.85
且最后 20 个 checkpoint 均值高于原版 OFQ 后段均值
```

## 2026-07-10 实现记录

已在统一 OFQ runtime `qat_launch.py` 中实现 default-off dynamic sparse prev-step KL controller。

新增能力：

```text
--dynamic-sparse-prevstep-kl
--dynamic-kl-start-epoch
--dynamic-kl-observe-until-epoch
--dynamic-kl-primary-heads
--dynamic-kl-secondary-heads
--dynamic-kl-avoid-heads
--dynamic-kl-drop-threshold
--dynamic-kl-strong-drop-threshold
--dynamic-kl-default-weight
--dynamic-kl-strong-weight
--dynamic-kl-max-weight
--dynamic-kl-cooldown-epochs
--dynamic-kl-window-epochs
--dynamic-kl-max-pulses-per-window
--dynamic-kl-controller-tsv
--dynamic-kl-prior-source
```

实现位置：

```text
qat_launch.py
build_ofq()
build_ofq_runtime_overrides()
build_ofq_runtime_config()
run_unified_ofq()
DynamicSparsePrevStepKLController
```

controller 逻辑：

```text
epoch <= 60: observe_only, RefW=0
epoch >= 61: 如果 rolling_best_acc - current_acc >= 0.06，则从候选 head 中选择一个非 cooldown、非 avoid head，在下一 epoch 开 1 个 epoch 的 prev-step KL
drop >= 0.12 时权重 2e-5，否则 1e-5
最大权重 3e-5
同一 head cooldown 5 epoch
10epoch 窗口最多 3 次 pulse
```

第一版不做每个 epoch 的实时 512-sample attention probe；controller 使用离线 head harmful prior + full-val drop，日志字段 `prior_source=offline_attn_relation_oscillation_20260710_no_live_probe` 显式记录。

候选 head：

```text
primary: 8:4
secondary: 5:7,4:11,6:1,11:18
```

禁止选择：

```text
6:6,7:7,4:1,2:4,10:13,11:4,6:7,11:16
```

## 2026-07-10 本地预检

已确认：

```text
checkpoint-10 exists:
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar

teacher exists:
/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth

ImageNet parquet:
train shards: 294
validation shards: 14
```

已通过：

```text
python -m py_compile qat_launch.py
python qat_launch.py --help | rg dynamic-kl
bash -n tmp_scripts/run_ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710.sh
bash -n tmp_scripts/monitor_ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710.sh
DRY_RUN=1 bash tmp_scripts/run_ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710.sh
```

dry-run 命令检查：

```text
包含 --dynamic-sparse-prevstep-kl
包含 --dynamic-kl-start-epoch 61
包含 --dynamic-kl-observe-until-epoch 60
包含 --dynamic-kl-primary-heads 8:4
包含 --dynamic-kl-secondary-heads 5:7,4:11,6:1,11:18
包含 --dynamic-kl-avoid-heads 6:6,7:7,4:1,2:4,10:13,11:4,6:7,11:16
包含 --ref-attn-kl-weight 0.0
不包含 --ref-attn-kl-weight-epoch-overrides
```

一次本地普通 shell 误触发已在进入有效训练前 Ctrl-C 中断；本地无 `/dev/nvidia*`，该日志不作为训练证据。已检查无残留 `qat_launch.py/train.py` 训练进程，无输出 checkpoint。

待确认：

```text
真实 GPU worker 8 卡可用
/tmp 空间足够
输出目录不会覆盖已有实验
```

## 启动命令

计划在真实 GPU worker 中启动：

```text
cd /mlx_devbox/users/quyanyi/playground
MASTER_PORT=31811 OUT=/tmp/qat_public_repro nohup bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710.sh >/tmp/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710.nohup 2>&1 &
```

## 2026-07-10 16:39 UTC 启动记录

真实 worker 预检：

```text
worker: fdbd:dccd:cdc2:1234:0:b8::, ssh port 9801
GPU: 8 x NVIDIA H100 80GB HBM3 visible
torch.cuda.is_available: True
torch.cuda.device_count: 8
/tmp free: about 430G
root fs free: about 8.0G, output stays under /tmp
checkpoint-10: exists
teacher checkpoint: exists
train shards: 294
validation shards: 14
same experiment process before launch: none
same experiment output files before launch: none
```

启动命令：

```text
cd /mlx_devbox/users/quyanyi/playground
MASTER_PORT=31811 OUT=/tmp/qat_public_repro nohup bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710.sh >/tmp/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710.nohup 2>&1 &
```

启动进程：

```text
launcher pid: 100154
script pid: 100156
qat_launch pid: 100173
spawned 8 rank processes
```

启动质量证据：

```text
Strict resume: loaded model from checkpoint-10; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Enabled EMA refmodel attention-KL scheme: ref_update=prev_step, ref_update_interval=50, attn_kl_weight=0.0, head_mode=custom_subset:8:4
Enabled dynamic sparse prev-step KL controller: start_epoch=61, observe_until=60, primary_heads=['8:4'], secondary_heads=['5:7', '4:11', '6:1', '11:18'], avoid_heads=['2:4', '4:1', '6:6', '6:7', '7:7', '10:13', '11:4', '11:16']
DynamicSparsePrevStepKLControllerApply: epoch=10, head_mode=custom_subset:8:4, head=, weight=0.000e+00, reason=observe_only
Train: 10 [0/2502] ... RefW: 0.000e+00
```

GPU 启动后状态：

```text
8 卡显存约 28427 MiB/card
8 卡 GPU util 99-100%
```

monitor 摘要：

```text
output_exists=remote:/tmp/qat_public_repro/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710
args_yaml=present
controller_exists=local_doc_tsv
checkpoint_count=0
fullval_rows=0
pre61_nonzero_refw_lines=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

结论：

```text
训练已正确进入 epoch 10，当前仍在第一个 resumed epoch 内。
epoch 10 处于 observe_only，RefW=0，符合 epoch 10-60 不主动开 KL 的约束。
下一步继续轮询 checkpoint-11 和第一次 full-val。
```

## 2026-07-10 16:53 UTC checkpoint-11

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
target_81_lines=0
nonzero_refw_lines=0
pre61_nonzero_refw_lines=0
controller_rows=1
controller_triggers=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

日志证据：

```text
TrainSummary: epoch=10 updates=2496 avg_step_time=0.228127s samples_per_sec=2244.36
Test: [distributed-summary] Loss 0.8448 Acc@1 80.3360 Acc@5 95.2440 Samples 50000
DynamicSparsePrevStepKLController: epoch=10, phase=observe, top1=80.3360, rolling_best=80.3360, drop=0.0000, applied_weight=0.000e+00, triggered=False, reason=observe_only_before_start
DynamicSparsePrevStepKLControllerApply: epoch=11, head_mode=custom_subset:8:4, head=, weight=0.000e+00, reason=observe_only
Train: 11 [0/2502] ... RefW: 0.000e+00
```

checkpoint 文件：

```text
/tmp/qat_public_repro/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710/checkpoint-11.pth.tar
/tmp/qat_public_repro/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710/last.pth.tar
```

结论：

```text
checkpoint-11 已保存并完成 full-val。
epoch 10/11 仍为观测期，RefW=0，controller 没有触发，Samples=50000。
Top-1 80.3360 低于 baseline 80.5980，属于刚从 checkpoint-10 resume 后的早期波动；继续观察长跑趋势。
```

## 2026-07-10 17:04 UTC checkpoint-12

monitor 摘要：

```text
checkpoint_count=2
latest_checkpoint=checkpoint-12.pth.tar
fullval_rows=2
bad_sample_rows=0
best_fullval_line=checkpoint-12 Loss 0.8443 Acc@1 80.3760 Acc@5 95.2920 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.3560
nonzero_refw_lines=0
pre61_nonzero_refw_lines=0
controller_rows=2
controller_triggers=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

日志证据：

```text
TrainSummary: epoch=11 updates=2496 avg_step_time=0.228240s samples_per_sec=2243.25
Test: [distributed-summary] Loss 0.8443 Acc@1 80.3760 Acc@5 95.2920 Samples 50000
DynamicSparsePrevStepKLController: epoch=11, phase=observe, top1=80.3760, rolling_best=80.3760, drop=0.0000, applied_weight=0.000e+00, triggered=False, reason=observe_only_before_start
DynamicSparsePrevStepKLControllerApply: epoch=12, head_mode=custom_subset:8:4, head=, weight=0.000e+00, reason=observe_only
```

结论：

```text
checkpoint-12 已保存并完成 full-val。
Top-1 从 checkpoint-11 的 80.3360 小幅回升到 80.3760。
epoch 10-12 仍完全符合观测期要求：RefW=0，无 controller 触发，无 avoid head 选择。
```

## 2026-07-10 17:07 UTC 重启记录

重启原因：

```text
目标文档要求 controller 触发记录包含 selected head / weight / reason / drop / spike score / cooldown。
第一版 controller TSV 已记录 head / weight / reason / drop / cooldown，但缺少单独 spike_score 字段。
当前只跑到 checkpoint-12，重启成本很低，因此停止预跑、补齐 spike_score 字段后重新从 checkpoint-10 干净启动。
```

代码/脚本修正：

```text
qat_launch.py:
  controller TSV header 新增 applied_spike_score 和 next_spike_score
  controller log 新增 applied_spike_score 和 next_spike_score
  trigger reason 中保留 spike_score
  broadcast decision 同步 spike_score

monitor script:
  适配 controller TSV 新列索引
```

停止预跑：

```text
stopped process group: 100152
GPU released: 8 cards back to about 7 MiB used, util 0
```

清理并重启：

```text
rm -rf /tmp/qat_public_repro/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710
rm -f train log / status TSV / refw TSV / controller TSV / monitor summary
cd /mlx_devbox/users/quyanyi/playground
MASTER_PORT=31811 OUT=/tmp/qat_public_repro nohup bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710.sh >/tmp/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710.nohup 2>&1 &
```

新 run 进程：

```text
launcher pid: 111828
script pid: 111829
qat_launch pid: 111845
```

新 run 启动命令仍保持：

```text
checkpoint-10 -> checkpoint-110
dynamic_sparse_prevstep_kl=true
dynamic_kl_start_epoch=61
dynamic_kl_observe_until_epoch=60
ref_attn_kl_weight=0.0
ref_attn_kl_weight_epoch_overrides={}
```

新版启动确认：

```text
Strict resume: loaded model from checkpoint-10; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Enabled EMA refmodel attention-KL scheme: ref_update=prev_step, ref_update_interval=50, attn_kl_weight=0.0, head_mode=custom_subset:8:4
Enabled dynamic sparse prev-step KL controller: start_epoch=61, observe_until=60
DynamicSparsePrevStepKLControllerApply: epoch=10, weight=0.000e+00, spike_score=0.000000, reason=observe_only
Train: 10 [0/2502] ... RefW: 0.000e+00
```

controller TSV header 已包含审计字段：

```text
epoch phase top1 top5 samples rolling_best drop applied_head applied_weight applied_spike_score next_head next_weight next_spike_score triggered reason prior_source cooldown_state window_pulses
```

monitor 摘要：

```text
output_exists=remote:/tmp/qat_public_repro/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710
args_yaml=present
checkpoint_count=0
fullval_rows=0
controller_rows=0
pre61_nonzero_refw_lines=0
```

结论：

```text
新版 run 已经正确进入 epoch 10。
当前 controller 记录字段满足 selected head / weight / reason / drop / spike score / cooldown 的最终审计要求。
下一步继续轮询 checkpoint-11 和第一次 full-val。
```

## 2026-07-10 17:20 UTC 新版 checkpoint-11

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
target_81_lines=0
last20_avg=80.3360
nonzero_refw_lines=0
pre61_nonzero_refw_lines=0
controller_rows=1
controller_triggers=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

日志证据：

```text
TrainSummary: epoch=10 updates=2496 avg_step_time=0.229075s samples_per_sec=2235.08
Test: [distributed-summary] Loss 0.8448 Acc@1 80.3360 Acc@5 95.2440 Samples 50000
DynamicSparsePrevStepKLController: epoch=10, phase=observe, top1=80.3360, rolling_best=80.3360, drop=0.0000, applied_weight=0.000e+00, applied_spike_score=0.000000, next_weight=0.000e+00, next_spike_score=0.000000, triggered=False, reason=observe_only_before_start
DynamicSparsePrevStepKLControllerApply: epoch=11, head=, weight=0.000e+00, spike_score=0.000000, reason=observe_only
```

controller TSV 行：

```text
10 observe 80.3360 95.2440 50000 rolling_best=80.3360 drop=0.0000 applied_weight=0 applied_spike_score=0 next_weight=0 next_spike_score=0 triggered=0 reason=observe_only_before_start cooldown={}
```

结论：

```text
新版 checkpoint-11 已保存并完成 full-val。
Samples=50000，RefW=0，controller 没有触发，spike_score 字段完整。
当前结果低于 baseline，但与重启前 checkpoint-11 一致，说明重启只改变日志审计字段，不改变训练行为。
```

## 2026-07-10 17:41 UTC checkpoint-13

monitor 摘要：

```text
checkpoint_count=3
latest_checkpoint=checkpoint-13.pth.tar
fullval_rows=3
bad_sample_rows=0
best_fullval_line=checkpoint-12 Loss 0.8443 Acc@1 80.3760 Acc@5 95.2920 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.3527
nonzero_refw_lines=0
pre61_nonzero_refw_lines=0
controller_rows=3
controller_triggers=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

full-val 明细：

```text
checkpoint-11: Loss 0.8448 Acc@1 80.3360 Acc@5 95.2440 Samples 50000
checkpoint-12: Loss 0.8443 Acc@1 80.3760 Acc@5 95.2920 Samples 50000
checkpoint-13: Loss 0.8443 Acc@1 80.3460 Acc@5 95.2740 Samples 50000
```

controller 明细：

```text
epoch 10: observe, top1 80.3360, drop 0.0000, applied_weight 0, applied_spike_score 0, next_weight 0, triggered 0
epoch 11: observe, top1 80.3760, drop 0.0000, applied_weight 0, applied_spike_score 0, next_weight 0, triggered 0
epoch 12: observe, top1 80.3460, drop 0.0300, applied_weight 0, applied_spike_score 0, next_weight 0, triggered 0
```

训练健康：

```text
GPU util: 98-100%
epoch throughput: about 2235-2260 samples/sec
checkpoint files: checkpoint-11, checkpoint-12, checkpoint-13
```

结论：

```text
新版 run 已稳定跑到 checkpoint-13。
epoch 10-12 仍处于 observe-only 阶段，RefW 全为 0，controller 没有 pre-61 trigger，也没有 avoid head 被选中。
当前 best 为 checkpoint-12 Top-1 80.3760，仍低于 baseline 80.5980；继续观察原版 OFQ 风格的中后段恢复趋势。
```

## 2026-07-10 18:02 UTC checkpoint-15

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
target_81_lines=0
last20_avg=80.3600
nonzero_refw_lines=0
pre61_nonzero_refw_lines=0
controller_rows=5
controller_triggers=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

full-val 明细：

```text
checkpoint-11: Loss 0.8448 Acc@1 80.3360 Acc@5 95.2440 Samples 50000
checkpoint-12: Loss 0.8443 Acc@1 80.3760 Acc@5 95.2920 Samples 50000
checkpoint-13: Loss 0.8443 Acc@1 80.3460 Acc@5 95.2740 Samples 50000
checkpoint-14: Loss 0.8467 Acc@1 80.3700 Acc@5 95.3000 Samples 50000
checkpoint-15: Loss 0.8443 Acc@1 80.3720 Acc@5 95.3000 Samples 50000
```

controller 明细：

```text
epoch 10: observe, drop 0.0000, weight 0, spike_score 0, triggered 0
epoch 11: observe, drop 0.0000, weight 0, spike_score 0, triggered 0
epoch 12: observe, drop 0.0300, weight 0, spike_score 0, triggered 0
epoch 13: observe, drop 0.0060, weight 0, spike_score 0, triggered 0
epoch 14: observe, drop 0.0040, weight 0, spike_score 0, triggered 0
```

训练健康：

```text
GPU util: 97-100%
epoch throughput: about 2235-2260 samples/sec
checkpoint files: checkpoint-11 through checkpoint-15
```

结论：

```text
前 5 个 resumed checkpoint 全部完成，Samples=50000。
epoch 10-14 严格保持 observe-only：RefW=0、controller_triggers=0、controller_selected_avoid=0。
精度目前在 80.34-80.38 平台，仍低于 baseline 80.5980；这属于 10->60 原版 OFQ 长跑早期段，继续观察后续自然恢复。
```

## 2026-07-10 18:49 UTC checkpoint-20

monitor 摘要：

```text
checkpoint_count=10
latest_checkpoint=checkpoint-20.pth.tar
fullval_rows=10
bad_sample_rows=0
best_fullval_line=checkpoint-20 Loss 0.8388 Acc@1 80.4580 Acc@5 95.3380 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.3640
nonzero_refw_lines=0
pre61_nonzero_refw_lines=0
controller_rows=10
controller_triggers=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

最近 full-val：

```text
checkpoint-16: Acc@1 80.3800 Acc@5 95.3140 Samples 50000
checkpoint-17: Acc@1 80.3640 Acc@5 95.3100 Samples 50000
checkpoint-18: Acc@1 80.3020 Acc@5 95.2420 Samples 50000
checkpoint-19: Acc@1 80.3360 Acc@5 95.2660 Samples 50000
checkpoint-20: Acc@1 80.4580 Acc@5 95.3380 Samples 50000
```

controller 明细：

```text
epoch 15: observe, top1 80.3800, drop 0.0000, weight 0, spike_score 0, triggered 0
epoch 16: observe, top1 80.3640, drop 0.0160, weight 0, spike_score 0, triggered 0
epoch 17: observe, top1 80.3020, drop 0.0780, weight 0, spike_score 0, triggered 0
epoch 18: observe, top1 80.3360, drop 0.0440, weight 0, spike_score 0, triggered 0
epoch 19: observe, top1 80.4580, drop 0.0000, weight 0, spike_score 0, triggered 0
```

训练健康：

```text
GPU util: 97-100%
epoch throughput: about 2256-2259 samples/sec in recent epochs
checkpoint files: checkpoint-11 through checkpoint-20
```

结论：

```text
前 10 个 full-val 均完整，Samples=50000。
observe-only 阶段规则仍严格成立：RefW=0，controller_triggers=0，controller_pre61_triggers=0，controller_selected_avoid=0。
Top-1 从 checkpoint-15 附近的 80.37 上升到 checkpoint-20 的 80.4580，仍低于 baseline 80.5980，但开始向上恢复。
继续轮询到 checkpoint-30/40/50/60，重点看是否复现原版 OFQ 中段恢复和 50-53 自然高点。
```

## 2026-07-10 19:28 UTC checkpoint-24

monitor 摘要：

```text
checkpoint_count=14
latest_checkpoint=checkpoint-24.pth.tar
fullval_rows=14
bad_sample_rows=0
best_fullval_line=checkpoint-22 Loss 0.8378 Acc@1 80.5440 Acc@5 95.3640 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.3853
nonzero_refw_lines=0
pre61_nonzero_refw_lines=0
controller_rows=14
controller_triggers=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

最近 full-val：

```text
checkpoint-20: Acc@1 80.4580 Acc@5 95.3380 Samples 50000
checkpoint-21: Acc@1 80.4660 Acc@5 95.3400 Samples 50000
checkpoint-22: Acc@1 80.5440 Acc@5 95.3640 Samples 50000
checkpoint-23: Acc@1 80.3720 Acc@5 95.3100 Samples 50000
checkpoint-24: Acc@1 80.3720 Acc@5 95.2980 Samples 50000
```

controller 明细：

```text
epoch 20: observe, top1 80.4660, drop 0.0000, weight 0, spike_score 0, triggered 0
epoch 21: observe, top1 80.5440, drop 0.0000, weight 0, spike_score 0, triggered 0
epoch 22: observe, top1 80.3720, drop 0.1720, weight 0, spike_score 0, triggered 0
epoch 23: observe, top1 80.3720, drop 0.1720, weight 0, spike_score 0, triggered 0
```

训练健康：

```text
GPU util: 97-100%
checkpoint files: checkpoint-11 through checkpoint-24
recent errors: none
```

结论：

```text
run 已推进到 checkpoint-24，仍处于 observe-only 阶段。
checkpoint-22 达到 80.5440，距离 baseline 80.5980 还差 0.0540，说明早期曲线正在恢复但尚未超过原版 baseline。
即使 epoch 22/23 的 drop 已超过 0.06，controller 仍没有触发，这是预期行为，因为 epoch <= 60 只观测。
后续继续观察 checkpoint-30/40/50/60 是否接近原版 OFQ 的自然高点窗口。
```

## 2026-07-10 19:50 UTC checkpoint-26

monitor 摘要：

```text
checkpoint_count=16
latest_checkpoint=checkpoint-26.pth.tar
fullval_rows=16
bad_sample_rows=0
best_fullval_line=checkpoint-22 Loss 0.8378 Acc@1 80.5440 Acc@5 95.3640 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.3896
nonzero_refw_lines=0
pre61_nonzero_refw_lines=0
controller_rows=16
controller_triggers=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

最近 full-val：

```text
checkpoint-22: Acc@1 80.5440 Acc@5 95.3640 Samples 50000
checkpoint-23: Acc@1 80.3720 Acc@5 95.3100 Samples 50000
checkpoint-24: Acc@1 80.3720 Acc@5 95.2980 Samples 50000
checkpoint-25: Acc@1 80.4600 Acc@5 95.3220 Samples 50000
checkpoint-26: Acc@1 80.3800 Acc@5 95.3140 Samples 50000
```

controller 明细：

```text
epoch 21: observe, top1 80.5440, drop 0.0000, weight 0, spike_score 0, triggered 0
epoch 22: observe, top1 80.3720, drop 0.1720, weight 0, spike_score 0, triggered 0
epoch 23: observe, top1 80.3720, drop 0.1720, weight 0, spike_score 0, triggered 0
epoch 24: observe, top1 80.4600, drop 0.0840, weight 0, spike_score 0, triggered 0
epoch 25: observe, top1 80.3800, drop 0.1640, weight 0, spike_score 0, triggered 0
```

结论：

```text
run 已推进到 checkpoint-26，仍保持 observe-only，无 pre-61 KL。
checkpoint-22 仍是当前 best 80.5440，尚未超过 baseline 80.5980。
即使 epoch 22/24/25 的 drop 超过 dynamic trigger 阈值 0.06，controller 仍未触发，符合 epoch 10-60 只观测的硬约束。
下一步继续轮询到 checkpoint-30/40，观察是否超过 baseline 并向原版 OFQ 中段高点靠近。
```

## 2026-07-10 20:06 UTC checkpoint-28

monitor 摘要：

```text
checkpoint_count=18
latest_checkpoint=checkpoint-28.pth.tar
fullval_rows=18
bad_sample_rows=0
best_fullval_line=checkpoint-22 Loss 0.8378 Acc@1 80.5440 Acc@5 95.3640 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.3983
nonzero_refw_lines=0
pre61_nonzero_refw_lines=0
controller_rows=18
controller_triggers=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

最近 full-val：

```text
checkpoint-22: Acc@1 80.5440 Acc@5 95.3640 Samples 50000
checkpoint-23: Acc@1 80.3720 Acc@5 95.3100 Samples 50000
checkpoint-24: Acc@1 80.3720 Acc@5 95.2980 Samples 50000
checkpoint-25: Acc@1 80.4600 Acc@5 95.3220 Samples 50000
checkpoint-26: Acc@1 80.3800 Acc@5 95.3140 Samples 50000
checkpoint-27: Acc@1 80.4660 Acc@5 95.3280 Samples 50000
checkpoint-28: Acc@1 80.4700 Acc@5 95.2960 Samples 50000
```

controller 明细：

```text
epoch 21: observe, top1 80.5440, drop 0.0000, weight 0, spike_score 0, triggered 0
epoch 22: observe, top1 80.3720, drop 0.1720, weight 0, spike_score 0, triggered 0
epoch 23: observe, top1 80.3720, drop 0.1720, weight 0, spike_score 0, triggered 0
epoch 24: observe, top1 80.4600, drop 0.0840, weight 0, spike_score 0, triggered 0
epoch 25: observe, top1 80.3800, drop 0.1640, weight 0, spike_score 0, triggered 0
epoch 26: observe, top1 80.4660, drop 0.0780, weight 0, spike_score 0, triggered 0
epoch 27: observe, top1 80.4700, drop 0.0740, weight 0, spike_score 0, triggered 0
```

结论：

```text
run 已推进到 checkpoint-28，仍保持 observe-only，无 pre-61 KL。
checkpoint-22 仍是当前 best 80.5440；checkpoint-27/28 有小幅恢复，但未超过 checkpoint-22，也未超过 baseline 80.5980。
多个 epoch drop 已超过 0.06 但 controller 没触发，说明 10-60 只观测硬约束正常工作。
下一步继续轮询到 checkpoint-30/40。
```

## 2026-07-10 20:24 UTC checkpoint-30

monitor 摘要：

```text
checkpoint_count=20
latest_checkpoint=checkpoint-30.pth.tar
fullval_rows=20
bad_sample_rows=0
best_fullval_line=checkpoint-22 Loss 0.8378 Acc@1 80.5440 Acc@5 95.3640 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.4009
nonzero_refw_lines=0
pre61_nonzero_refw_lines=0
controller_rows=20
controller_triggers=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

最近 full-val：

```text
checkpoint-27: Acc@1 80.4660 Acc@5 95.3280 Samples 50000
checkpoint-28: Acc@1 80.4700 Acc@5 95.2960 Samples 50000
checkpoint-29: Acc@1 80.3600 Acc@5 95.3560 Samples 50000
checkpoint-30: Acc@1 80.4880 Acc@5 95.3300 Samples 50000
```

controller 明细：

```text
epoch 26: observe, top1 80.4660, drop 0.0780, weight 0, spike_score 0, triggered 0
epoch 27: observe, top1 80.4700, drop 0.0740, weight 0, spike_score 0, triggered 0
epoch 28: observe, top1 80.3600, drop 0.1840, weight 0, spike_score 0, triggered 0
epoch 29: observe, top1 80.4880, drop 0.0560, weight 0, spike_score 0, triggered 0
```

结论：

```text
run 已推进到 checkpoint-30，仍保持 observe-only，无 pre-61 KL。
checkpoint-30 Top-1 80.4880，较 checkpoint-28 有恢复，但仍未超过 checkpoint-22 的 80.5440，也未超过 baseline 80.5980。
截至 checkpoint-30，20 个 full-val 均完整，Samples=50000，controller 未触发、未选 avoid head。
下一步继续观察到 checkpoint-40/50/60，重点看中段是否恢复到原版 OFQ baseline 以上。
```

## 2026-07-10 20:53 UTC checkpoint-33

monitor 摘要：

```text
checkpoint_count=23
latest_checkpoint=checkpoint-33.pth.tar
fullval_rows=23
bad_sample_rows=0
best_fullval_line=checkpoint-32 Loss 0.8373 Acc@1 80.5980 Acc@5 95.3180 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.4229
nonzero_refw_lines=0
pre61_nonzero_refw_lines=0
controller_rows=23
controller_triggers=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

最近 full-val：

```text
checkpoint-29: Acc@1 80.3600 Acc@5 95.3560 Samples 50000
checkpoint-30: Acc@1 80.4880 Acc@5 95.3300 Samples 50000
checkpoint-31: Acc@1 80.3800 Acc@5 95.2800 Samples 50000
checkpoint-32: Acc@1 80.5980 Acc@5 95.3180 Samples 50000
checkpoint-33: Acc@1 80.5200 Acc@5 95.3880 Samples 50000
```

controller 明细：

```text
epoch 29: observe, top1 80.4880, drop 0.0560, weight 0, spike_score 0, triggered 0
epoch 30: observe, top1 80.3800, drop 0.1640, weight 0, spike_score 0, triggered 0
epoch 31: observe, top1 80.5980, drop 0.0000, weight 0, spike_score 0, triggered 0
epoch 32: observe, top1 80.5200, drop 0.0780, weight 0, spike_score 0, triggered 0
```

结论：

```text
run 已推进到 checkpoint-33，仍处于 observe-only，无 pre-61 KL。
checkpoint-32 Top-1 80.5980，已经追平 baseline 80.5980，但还没有超过；above_baseline_lines 仍为 0。
仍未超过 scheme C best 80.6820，也未超过 original OFQ best 80.7240。
到 checkpoint-33 为止，所有 full-val 样本数都是 50000，controller 触发数仍为 0。
```

## 2026-07-10 21:16 UTC checkpoint-35

monitor 摘要：

```text
checkpoint_count=25
latest_checkpoint=checkpoint-35.pth.tar
fullval_rows=25
bad_sample_rows=0
best_fullval_line=checkpoint-32 Loss 0.8373 Acc@1 80.5980 Acc@5 95.3180 Samples 50000
above_baseline_lines=0
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.4285
nonzero_refw_lines=0
pre61_nonzero_refw_lines=0
controller_rows=25
controller_triggers=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

最近 full-val：

```text
checkpoint-31: Acc@1 80.3800 Acc@5 95.2800 Samples 50000
checkpoint-32: Acc@1 80.5980 Acc@5 95.3180 Samples 50000
checkpoint-33: Acc@1 80.5200 Acc@5 95.3880 Samples 50000
checkpoint-34: Acc@1 80.4040 Acc@5 95.3000 Samples 50000
checkpoint-35: Acc@1 80.4500 Acc@5 95.3840 Samples 50000
```

controller 明细：

```text
epoch 31: observe, top1 80.5980, drop 0.0000, weight 0, spike_score 0, triggered 0
epoch 32: observe, top1 80.5200, drop 0.0780, weight 0, spike_score 0, triggered 0
epoch 33: observe, top1 80.4040, drop 0.1940, weight 0, spike_score 0, triggered 0
epoch 34: observe, top1 80.4500, drop 0.1480, weight 0, spike_score 0, triggered 0
```

结论：

```text
run 已推进到 checkpoint-35，仍处于 observe-only，无 pre-61 KL。
当前 best 仍为 checkpoint-32 Top-1 80.5980，追平 baseline 但没有超过。
checkpoint-33 到 checkpoint-35 回落后仍保持 80.40+，未出现训练异常；controller 在 drop 超阈值时仍保持不触发，符合 10-60 只观测约束。
下一步继续观察 checkpoint-40/50/60 是否超过 baseline 并进入原版 OFQ 自然高点窗口。
```

## 2026-07-10 21:54 UTC checkpoint-39

monitor 摘要：

```text
checkpoint_count=29
latest_checkpoint=checkpoint-39.pth.tar
fullval_rows=29
bad_sample_rows=0
best_fullval_line=checkpoint-36 Loss 0.8394 Acc@1 80.6020 Acc@5 95.3420 Samples 50000
above_baseline_lines=1
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.4672
nonzero_refw_lines=0
pre61_nonzero_refw_lines=0
controller_rows=29
controller_triggers=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

最近 full-val：

```text
checkpoint-34: Acc@1 80.4040 Acc@5 95.3000 Samples 50000
checkpoint-35: Acc@1 80.4500 Acc@5 95.3840 Samples 50000
checkpoint-36: Acc@1 80.6020 Acc@5 95.3420 Samples 50000
checkpoint-37: Acc@1 80.5460 Acc@5 95.4000 Samples 50000
checkpoint-38: Acc@1 80.5040 Acc@5 95.3220 Samples 50000
checkpoint-39: Acc@1 80.5040 Acc@5 95.3260 Samples 50000
```

controller 明细：

```text
epoch 35: observe, top1 80.6020, drop 0.0000, weight 0, spike_score 0, triggered 0
epoch 36: observe, top1 80.5460, drop 0.0560, weight 0, spike_score 0, triggered 0
epoch 37: observe, top1 80.5040, drop 0.0980, weight 0, spike_score 0, triggered 0
epoch 38: observe, top1 80.5040, drop 0.0980, weight 0, spike_score 0, triggered 0
```

结论：

```text
run 已推进到 checkpoint-39，仍处于 observe-only，无 pre-61 KL。
checkpoint-36 Top-1 80.6020，首次超过 baseline 80.5980，但只高 0.004，仍未超过 scheme C best 80.6820 和 original OFQ best 80.7240。
截至 checkpoint-39，29 个 full-val 全部 Samples=50000，controller 仍没有触发、没有选 avoid head。
下一步继续观察 checkpoint-40/50/60；关键是能否进入原版 OFQ 在 50-53 附近的自然高点窗口。
```

## 2026-07-10 22:33 UTC checkpoint-43

monitor 摘要：

```text
checkpoint_count=33
latest_checkpoint=checkpoint-43.pth.tar
fullval_rows=33
bad_sample_rows=0
best_fullval_line=checkpoint-36 Loss 0.8394 Acc@1 80.6020 Acc@5 95.3420 Samples 50000
above_baseline_lines=1
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.4814
nonzero_refw_lines=0
pre61_nonzero_refw_lines=0
controller_rows=33
controller_triggers=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

最近 full-val：

```text
checkpoint-39: Acc@1 80.5040 Acc@5 95.3260 Samples 50000
checkpoint-40: Acc@1 80.4740 Acc@5 95.3960 Samples 50000
checkpoint-41: Acc@1 80.5280 Acc@5 95.3220 Samples 50000
checkpoint-42: Acc@1 80.5480 Acc@5 95.3680 Samples 50000
checkpoint-43: Acc@1 80.5740 Acc@5 95.3700 Samples 50000
```

controller 明细：

```text
epoch 39: observe, top1 80.4740, drop 0.1280, weight 0, spike_score 0, triggered 0
epoch 40: observe, top1 80.5280, drop 0.0740, weight 0, spike_score 0, triggered 0
epoch 41: observe, top1 80.5480, drop 0.0540, weight 0, spike_score 0, triggered 0
epoch 42: observe, top1 80.5740, drop 0.0280, weight 0, spike_score 0, triggered 0
```

结论：

```text
run 已推进到 checkpoint-43，仍处于 observe-only，无 pre-61 KL。
checkpoint-43 回升到 80.5740，接近但未超过 checkpoint-36 best 80.6020。
last20_avg 提升到 80.4814，说明中段曲线在恢复，但仍未超过 scheme C best 80.6820 和 original OFQ best 80.7240。
继续观察 checkpoint-50/53/60 的自然高点窗口。
```

## 2026-07-10 23:13 UTC checkpoint-47

monitor 摘要：

```text
checkpoint_count=37
latest_checkpoint=checkpoint-47.pth.tar
fullval_rows=37
bad_sample_rows=0
best_fullval_line=checkpoint-36 Loss 0.8394 Acc@1 80.6020 Acc@5 95.3420 Samples 50000
above_baseline_lines=1
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.5049
nonzero_refw_lines=0
pre61_nonzero_refw_lines=0
controller_rows=37
controller_triggers=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

最近 full-val：

```text
checkpoint-44: Acc@1 80.4960 Acc@5 95.2740 Samples 50000
checkpoint-45: Acc@1 80.5420 Acc@5 95.3280 Samples 50000
checkpoint-46: Acc@1 80.5320 Acc@5 95.3120 Samples 50000
checkpoint-47: Acc@1 80.5780 Acc@5 95.3820 Samples 50000
```

controller 明细：

```text
epoch 43: observe, top1 80.4960, drop 0.1060, weight 0, spike_score 0, triggered 0
epoch 44: observe, top1 80.5420, drop 0.0600, weight 0, spike_score 0, triggered 0
epoch 45: observe, top1 80.5320, drop 0.0700, weight 0, spike_score 0, triggered 0
epoch 46: observe, top1 80.5780, drop 0.0240, weight 0, spike_score 0, triggered 0
```

结论：

```text
run 已推进到 checkpoint-47，仍处于 observe-only，无 pre-61 KL。
checkpoint-47 Top-1 80.5780，接近 baseline 80.5980，但仍未超过 checkpoint-36 best 80.6020。
last20_avg 提升到 80.5049，曲线继续缓慢恢复。
下一步进入 checkpoint-50/53 自然高点窗口，这是对比原版 OFQ 10->60 的关键位置。
```

## 2026-07-10 23:36 UTC checkpoint-50

monitor 摘要：

```text
checkpoint_count=40
latest_checkpoint=checkpoint-50.pth.tar
fullval_rows=39
bad_sample_rows=0
best_fullval_line=checkpoint-36 Loss 0.8394 Acc@1 80.6020 Acc@5 95.3420 Samples 50000
above_baseline_lines=1
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.5217
nonzero_refw_lines=0
pre61_nonzero_refw_lines=0
controller_rows=40
controller_triggers=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

最近 full-val：

```text
checkpoint-45: Acc@1 80.5420 Acc@5 95.3280 Samples 50000
checkpoint-46: Acc@1 80.5320 Acc@5 95.3120 Samples 50000
checkpoint-47: Acc@1 80.5780 Acc@5 95.3820 Samples 50000
checkpoint-48: Acc@1 80.5840 Acc@5 95.3080 Samples 50000
checkpoint-49: Acc@1 80.5820 Acc@5 95.3620 Samples 50000
checkpoint-50: Acc@1 80.5800 Acc@5 95.3700 Samples 50000
```

controller 明细：

```text
epoch 47: observe, top1 80.5840, drop 0.0180, weight 0, spike_score 0, triggered 0
epoch 48: observe, top1 80.5820, drop 0.0200, weight 0, spike_score 0, triggered 0
epoch 49: observe, top1 80.5800, drop 0.0220, weight 0, spike_score 0, triggered 0
```

结论：

```text
run 已进入原版 OFQ 的自然高点窗口附近，但 checkpoint-48/49/50 都在 80.58 左右，尚未超过 checkpoint-36 best 80.6020，也未超过 scheme C/original best。
与原版 OFQ 10->60 的 checkpoint-50 80.6300、checkpoint-52 80.7240 相比，目前同窗口明显偏低。
仍严格满足 observe-only：RefW=0、controller_triggers=0、controller_selected_avoid=0。
下一步继续重点观察 checkpoint-52/53。
```

## 2026-07-11 00:00 UTC checkpoint-52

monitor 摘要：

```text
checkpoint_count=42
latest_checkpoint=checkpoint-52.pth.tar
fullval_rows=42
bad_sample_rows=0
best_fullval_line=checkpoint-36 Loss 0.8394 Acc@1 80.6020 Acc@5 95.3420 Samples 50000
above_baseline_lines=1
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.5290
nonzero_refw_lines=0
pre61_nonzero_refw_lines=0
controller_rows=42
controller_triggers=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

关键窗口 full-val：

```text
checkpoint-48: Acc@1 80.5840 Acc@5 95.3080 Samples 50000
checkpoint-49: Acc@1 80.5820 Acc@5 95.3620 Samples 50000
checkpoint-50: Acc@1 80.5800 Acc@5 95.3700 Samples 50000
checkpoint-51: Acc@1 80.5200 Acc@5 95.3340 Samples 50000
checkpoint-52: Acc@1 80.5120 Acc@5 95.3500 Samples 50000
```

对比原版 OFQ 10->60：

```text
original checkpoint-50: 80.6300
current  checkpoint-50: 80.5800
delta: -0.0500

original checkpoint-52: 80.7240
current  checkpoint-52: 80.5120
delta: -0.2120
```

controller 明细：

```text
epoch 48: observe, top1 80.5820, drop 0.0200, weight 0, spike_score 0, triggered 0
epoch 49: observe, top1 80.5800, drop 0.0220, weight 0, spike_score 0, triggered 0
epoch 50: observe, top1 80.5200, drop 0.0820, weight 0, spike_score 0, triggered 0
epoch 51: observe, top1 80.5120, drop 0.0900, weight 0, spike_score 0, triggered 0
```

结论：

```text
run 已推进到 checkpoint-52，仍处于 observe-only，无 pre-61 KL。
关键自然高点窗口没有复现原版 OFQ 高点：current checkpoint-52 只有 80.5120，低于 original checkpoint-52 80.7240。
当前 best 仍是 checkpoint-36 80.6020；只超过 baseline 1 个 checkpoint，未超过 scheme C best 80.6820，也未超过 original best 80.7240。
这说明即使 controller 在 10-60 完全不触发，仅 train_scheme=ema_ref_attn_kl + refmodel/attention collection 路径也可能已经使运行与原版 OFQ 对照不完全等价，需要后续审计原因。
下一步继续观察 checkpoint-53/60；同时记录这个偏差，后续需要对比 args.yaml、attention collection、refmodel clone 和 static graph 等运行路径是否引入额外差异。
```

## 2026-07-11 00:16 UTC checkpoint-54

monitor 摘要：

```text
checkpoint_count=44
latest_checkpoint=checkpoint-54.pth.tar
fullval_rows=44
bad_sample_rows=0
best_fullval_line=checkpoint-36 Loss 0.8394 Acc@1 80.6020 Acc@5 95.3420 Samples 50000
above_baseline_lines=1
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.5336
nonzero_refw_lines=0
pre61_nonzero_refw_lines=0
controller_rows=44
controller_triggers=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

关键窗口 full-val：

```text
checkpoint-50: Acc@1 80.5800 Acc@5 95.3700 Samples 50000
checkpoint-51: Acc@1 80.5200 Acc@5 95.3340 Samples 50000
checkpoint-52: Acc@1 80.5120 Acc@5 95.3500 Samples 50000
checkpoint-53: Acc@1 80.5300 Acc@5 95.3280 Samples 50000
checkpoint-54: Acc@1 80.4860 Acc@5 95.3340 Samples 50000
```

对比原版 OFQ 10->60：

```text
original checkpoint-52: 80.7240
current  checkpoint-52: 80.5120
delta: -0.2120

original checkpoint-53: 80.6680
current  checkpoint-53: 80.5300
delta: -0.1380

original checkpoint-54: 80.6460
current  checkpoint-54: 80.4860
delta: -0.1600
```

controller 明细：

```text
epoch 50: observe, top1 80.5200, drop 0.0820, weight 0, spike_score 0, triggered 0
epoch 51: observe, top1 80.5120, drop 0.0900, weight 0, spike_score 0, triggered 0
epoch 52: observe, top1 80.5300, drop 0.0720, weight 0, spike_score 0, triggered 0
epoch 53: observe, top1 80.4860, drop 0.1160, weight 0, spike_score 0, triggered 0
```

结论：

```text
run 已推进到 checkpoint-54，仍处于 observe-only，无 pre-61 KL。
原版 OFQ 的 50-54 自然高点没有复现；当前窗口最高仅 checkpoint-50 80.5800，低于 baseline 80.5980。
当前 best 仍是 checkpoint-36 80.6020，只略高于 baseline 0.0040。
这版在 10-60 虽然没有主动 KL，但由于启用了 ema_ref_attn_kl/runtime attention collection/refmodel 路径，已经不能等价视作原版 OFQ 对照；这个偏差需要在最终审计中作为重要风险记录。
下一步继续跑过 checkpoint-60，确认 observe 段完整性；epoch 61 以后 controller 才允许触发。
```

## 2026-07-11 00:49 UTC checkpoint-57

monitor 摘要：

```text
checkpoint_count=47
latest_checkpoint=checkpoint-57.pth.tar
fullval_rows=47
bad_sample_rows=0
best_fullval_line=checkpoint-57 Loss 0.8299 Acc@1 80.6200 Acc@5 95.4140 Samples 50000
above_baseline_lines=2
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.5318
nonzero_refw_lines=0
pre61_nonzero_refw_lines=0
controller_rows=47
controller_triggers=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

最近 full-val：

```text
checkpoint-53: Acc@1 80.5300 Acc@5 95.3280 Samples 50000
checkpoint-54: Acc@1 80.4860 Acc@5 95.3340 Samples 50000
checkpoint-55: Acc@1 80.4300 Acc@5 95.3240 Samples 50000
checkpoint-56: Acc@1 80.5120 Acc@5 95.3700 Samples 50000
checkpoint-57: Acc@1 80.6200 Acc@5 95.4140 Samples 50000
```

controller 明细：

```text
epoch 53: observe, top1 80.4860, drop 0.1160, weight 0, spike_score 0, triggered 0
epoch 54: observe, top1 80.4300, drop 0.1720, weight 0, spike_score 0, triggered 0
epoch 55: observe, top1 80.5120, drop 0.0900, weight 0, spike_score 0, triggered 0
epoch 56: observe, top1 80.6200, drop 0.0000, weight 0, spike_score 0, triggered 0
```

结论：

```text
run 已推进到 checkpoint-57，仍处于 observe-only，无 pre-61 KL。
checkpoint-57 成为新 best 80.6200，超过 baseline 80.5980 共 0.0220；above_baseline_lines 从 1 增至 2。
仍低于 scheme C best 80.6820 和 original OFQ best 80.7240。
下一步重点观察 checkpoint-60 以及 epoch 61 后 dynamic controller 是否按 drop>=0.06 规则触发。
```

## 2026-07-11 01:19 UTC checkpoint-60

monitor 摘要：

```text
checkpoint_count=50
latest_checkpoint=checkpoint-60.pth.tar
fullval_rows=50
bad_sample_rows=0
best_fullval_line=checkpoint-59 Loss 0.8309 Acc@1 80.6460 Acc@5 95.3240 Samples 50000
above_baseline_lines=3
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.5473
nonzero_refw_lines=0
pre61_nonzero_refw_lines=0
controller_rows=50
controller_triggers=0
controller_pre61_triggers=0
controller_selected_avoid=0
```

observe 段关键 full-val：

```text
checkpoint-57: Acc@1 80.6200 Acc@5 95.4140 Samples 50000
checkpoint-58: Acc@1 80.5560 Acc@5 95.3520 Samples 50000
checkpoint-59: Acc@1 80.6460 Acc@5 95.3240 Samples 50000
checkpoint-60: Acc@1 80.5900 Acc@5 95.4140 Samples 50000
```

对比目标：

```text
baseline 80.5980: checkpoint-57 和 checkpoint-59 超过，checkpoint-60 略低
scheme C best 80.6820: 未超过
original OFQ best 80.7240: 未超过
81.0 target: 未达到
```

controller 审计：

```text
epoch 10-59: phase=observe
RefW: 0
controller_triggers: 0
controller_pre61_triggers: 0
controller_selected_avoid: 0
Samples: 每个 full-val 都是 50000
```

结论：

```text
10-60 observe 段已经完整跑完，硬约束满足：没有主动 KL，没有 pre-61 trigger，没有 avoid head。
best 为 checkpoint-59 Top-1 80.6460，超过 baseline 80.5980，但低于 scheme C 80.6820 和 original OFQ 80.7240。
这个结果比原版 OFQ 10->60 的 best 80.7240 低 0.0780，说明当前 dynamic controller 版本即使前 60 epoch 不触发 KL，也不是完全等价的原版 OFQ 对照路径。
下一步进入 epoch 61+，观察 dynamic controller 是否仅在 drop>=0.06 且 cooldown/window 允许时触发，并确认触发 head/weight/spike_score/reason/cooldown 记录完整。
```

## 2026-07-11 01:42 UTC dynamic controller first trigger

monitor 摘要：

```text
checkpoint_count=53
latest_checkpoint=checkpoint-63.pth.tar
fullval_rows=52
bad_sample_rows=0
best_fullval_line=checkpoint-59 Loss 0.8309 Acc@1 80.6460 Acc@5 95.3240 Samples 50000
above_baseline_lines=4
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.5508
controller_rows=53
controller_triggers=1
controller_pre61_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4
```

边界 full-val：

```text
checkpoint-60: Acc@1 80.5900 Acc@5 95.4140 Samples 50000
checkpoint-61: Acc@1 80.5280 Acc@5 95.4380 Samples 50000
checkpoint-62: Acc@1 80.6180 Acc@5 95.3320 Samples 50000
```

controller 决策：

```text
epoch 60: phase=observe, top1 80.5280, rolling_best 80.6460, drop 0.1180, triggered 0
epoch 61: phase=dynamic, top1 80.6180, rolling_best 80.6460, drop 0.0280, triggered 0, reason=drop_below_threshold:0.0280<0.0600
epoch 62: phase=dynamic, top1 80.5500, rolling_best 80.6460, drop 0.0960, next_head=8:4, next_weight=1e-05, next_spike_score=1.000000, triggered 1, reason=offline_prior_validation_drop
```

KL 生效日志：

```text
Train: 63 [0/2502] ... RefAttnKL: 3.084e+01 ... RefW: 1.000e-05
```

审计：

```text
trigger epoch: 62
applied epoch: 63
selected head: 8:4
weight: 1e-5
spike_score: 1.000000
avoid head selected: no
pre61 trigger: no
cooldown: {\"8:4\": 68}
```

结论：

```text
dynamic controller 首次触发符合设计：epoch 61 未触发，因为 drop 0.028 < 0.06；epoch 62 drop 0.096 >= 0.06 后，选择 primary head 8:4，在下一 epoch 63 启用 1e-5 KL。
触发记录包含 selected head / weight / reason / drop / spike_score / cooldown，且没有选择 blacklist head。
下一步观察 checkpoint-63/64，确认一次 pulse 对 full-val 的影响；同时确认 cooldown/window 规则是否继续生效。
```

## 2026-07-11 02:05 UTC checkpoint-64

monitor 摘要：

```text
checkpoint_count=54
latest_checkpoint=checkpoint-64.pth.tar
fullval_rows=54
bad_sample_rows=0
best_fullval_line=checkpoint-59 Loss 0.8309 Acc@1 80.6460 Acc@5 95.3240 Samples 50000
above_baseline_lines=4
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.5513
nonzero_refw_lines=88
nonzero_refw_epochs=63,64
pre61_nonzero_refw_lines=0
controller_rows=54
controller_triggers=2
controller_pre61_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7
```

dynamic full-val：

```text
checkpoint-61: Acc@1 80.5280 Acc@5 95.4380 Samples 50000
checkpoint-62: Acc@1 80.6180 Acc@5 95.3320 Samples 50000
checkpoint-63: Acc@1 80.5500 Acc@5 95.4160 Samples 50000
checkpoint-64: Acc@1 80.5300 Acc@5 95.3660 Samples 50000
```

controller 明细：

```text
epoch 61: dynamic, top1 80.6180, drop 0.0280, triggered 0, reason=drop_below_threshold
epoch 62: dynamic, top1 80.5500, drop 0.0960, next_head=8:4, next_weight=1e-5, next_spike_score=1.0, triggered 1, cooldown {8:4:68}
epoch 63: dynamic, top1 80.5300, drop 0.1160, applied_head=8:4, applied_weight=1e-5, next_head=5:7, next_weight=1e-5, next_spike_score=0.8, triggered 1, cooldown {8:4:68,5:7:69}
```

KL 生效：

```text
epoch 63: RefW=1e-5, selected head 8:4
epoch 64: RefW=1e-5, selected head 5:7
pre61_nonzero_refw_lines=0
controller_selected_avoid=0
```

结论：

```text
dynamic controller 的前两次触发符合规则：只在 epoch>=61 后触发，drop 达标，权重 1e-5，选择 8:4 和 5:7，未选 blacklist。
但 full-val 没看到收益：checkpoint-63 80.5500，checkpoint-64 80.5300，均低于 observe 段 best checkpoint-59 80.6460。
需要继续观察 cooldown 和窗口限制；当前已经有 2 次 trigger，仍在每 10epoch 最多 3 次的限制内。
```

## 2026-07-11 02:45 UTC checkpoint-68

monitor 摘要：

```text
checkpoint_count=58
latest_checkpoint=checkpoint-68.pth.tar
fullval_rows=58
bad_sample_rows=0
best_fullval_line=checkpoint-67 Loss 0.8348 Acc@1 80.6700 Acc@5 95.4260 Samples 50000
above_baseline_lines=7
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.5666
nonzero_refw_lines=150
nonzero_refw_epochs=63,64,65
pre61_nonzero_refw_lines=0
controller_rows=58
controller_triggers=3
controller_pre61_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7,4:11
```

dynamic full-val：

```text
checkpoint-63: Acc@1 80.5500 Acc@5 95.4160 Samples 50000
checkpoint-64: Acc@1 80.5300 Acc@5 95.3660 Samples 50000
checkpoint-65: Acc@1 80.5620 Acc@5 95.3280 Samples 50000
checkpoint-66: Acc@1 80.6580 Acc@5 95.3560 Samples 50000
checkpoint-67: Acc@1 80.6700 Acc@5 95.4260 Samples 50000
checkpoint-68: Acc@1 80.6520 Acc@5 95.3920 Samples 50000
```

controller 明细：

```text
epoch 62: dynamic, drop 0.0960, next_head=8:4, next_weight=1e-5, triggered 1, cooldown {8:4:68}
epoch 63: dynamic, applied_head=8:4, next_head=5:7, next_weight=1e-5, triggered 1, cooldown {8:4:68,5:7:69}
epoch 64: dynamic, applied_head=5:7, next_head=4:11, next_weight=1e-5, triggered 1, cooldown {8:4:68,5:7:69,4:11:70}
epoch 65: dynamic, applied_head=4:11, triggered 0, reason=drop_below_threshold
epoch 66: dynamic, no KL, triggered 0
epoch 67: dynamic, no KL, triggered 0
```

KL 生效：

```text
nonzero_refw_epochs=63,64,65
epoch 63: head 8:4, RefW=1e-5
epoch 64: head 5:7, RefW=1e-5
epoch 65: head 4:11, RefW=1e-5
epoch 68: RefW=0 after pulse window/cooldown
```

结论：

```text
dynamic controller 的第一次窗口触发 3 次，达到 10epoch 窗口上限但未超过限制。
未选择 blacklist head，pre61_nonzero_refw=0。
效果上，checkpoint-67 达到 80.6700，成为全 run 当前 best，超过 baseline 0.0720，但仍低于 scheme C best 80.6820 0.0120，也低于 original OFQ best 80.7240 0.0540。
三连 pulse 后没有立即超过 scheme C，但相比 checkpoint-63/64 有恢复；后续要看 cooldown 后是否能稳定维持或继续提升。
```

## 2026-07-11 03:18 UTC checkpoint-71

monitor 摘要：

```text
checkpoint_count=61
latest_checkpoint=checkpoint-71.pth.tar
fullval_rows=61
bad_sample_rows=0
best_fullval_line=checkpoint-67 Loss 0.8348 Acc@1 80.6700 Acc@5 95.4260 Samples 50000
above_baseline_lines=9
above_scheme_c_lines=0
above_original_lines=0
target_81_lines=0
last20_avg=80.5744
nonzero_refw_lines=150
nonzero_refw_epochs=63,64,65
pre61_nonzero_refw_lines=0
controller_rows=61
controller_triggers=3
controller_pre61_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7,4:11
```

dynamic full-val：

```text
checkpoint-66: Acc@1 80.6580 Acc@5 95.3560 Samples 50000
checkpoint-67: Acc@1 80.6700 Acc@5 95.4260 Samples 50000
checkpoint-68: Acc@1 80.6520 Acc@5 95.3920 Samples 50000
checkpoint-69: Acc@1 80.6160 Acc@5 95.3500 Samples 50000
checkpoint-70: Acc@1 80.6500 Acc@5 95.4240 Samples 50000
checkpoint-71: Acc@1 80.5720 Acc@5 95.3960 Samples 50000
```

controller 明细：

```text
epoch 65: applied_head=4:11, weight 1e-5, triggered 0, reason=drop_below_threshold
epoch 66: no KL, triggered 0
epoch 67: no KL, triggered 0
epoch 68: no KL, triggered 0, drop 0.0540 < 0.06
epoch 69: no KL, triggered 0, drop 0.0200 < 0.06
epoch 70: no KL, triggered 0, reason=window_limit:3/10
```

结论：

```text
run 已推进到 checkpoint-71。
第一次 dynamic pulse 窗口后，best 仍是 checkpoint-67 Top-1 80.6700，距离 scheme C 80.6820 还差 0.0120，距离 original OFQ 80.7240 还差 0.0540。
controller 后续遵守了窗口限制：epoch 70 drop=0.0980 但由于 10epoch window 内已有 3 次 pulse，没有继续触发。
nonzero RefW 只出现在 epoch 63/64/65，符合 pulse duration=1 epoch 的设计；未选择 blacklist head。
下一步继续观察窗口释放后是否再次触发，以及是否能超过 scheme C。
```

## 2026-07-11 03:58 UTC checkpoint-75

monitor 摘要：

```text
checkpoint_count=65
latest_checkpoint=checkpoint-75.pth.tar
fullval_rows=65
bad_sample_rows=0
best_fullval_line=checkpoint-75 Loss 0.8324 Acc@1 80.7120 Acc@5 95.3680 Samples 50000
above_baseline_lines=10
above_scheme_c_lines=1
above_original_lines=0
target_81_lines=0
last20_avg=80.5934
nonzero_refw_lines=200
nonzero_refw_epochs=63,64,65,74
pre61_nonzero_refw_lines=0
controller_rows=65
controller_triggers=4
controller_pre61_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7,4:11
```

dynamic full-val：

```text
checkpoint-69: Acc@1 80.6160 Acc@5 95.3500 Samples 50000
checkpoint-70: Acc@1 80.6500 Acc@5 95.4240 Samples 50000
checkpoint-71: Acc@1 80.5720 Acc@5 95.3960 Samples 50000
checkpoint-72: Acc@1 80.5840 Acc@5 95.4000 Samples 50000
checkpoint-73: Acc@1 80.5000 Acc@5 95.3460 Samples 50000
checkpoint-74: Acc@1 80.5420 Acc@5 95.3240 Samples 50000
checkpoint-75: Acc@1 80.7120 Acc@5 95.3680 Samples 50000
```

controller 明细：

```text
epoch 70: drop 0.0980, no trigger, reason=window_limit:3/10
epoch 71: drop 0.0860, no trigger, reason=window_limit:3/10
epoch 72: drop 0.1700, no trigger, reason=window_limit:3/10
epoch 73: drop 0.1280, next_head=8:4, next_weight=2e-5, next_spike_score=1.0, triggered 1
epoch 74: applied_head=8:4, applied_weight=2e-5, triggered 0, reason=drop_below_threshold
```

KL 生效：

```text
nonzero_refw_epochs=63,64,65,74
epoch 74: head 8:4, RefW=2e-5
```

结论：

```text
checkpoint-75 Top-1 80.7120，首次超过 scheme C best 80.6820，距离 original OFQ best 80.7240 只差 0.0120。
第二轮触发符合规则：等 10epoch window 释放后，epoch 73 drop=0.1280 >= strong threshold 0.12，因此使用 strong weight 2e-5；epoch 74 生效后 checkpoint-75 明显提升。
仍未超过 original OFQ best，也未达到 81.0，但已经满足“至少 5 个 checkpoint > scheme C”之前的更弱单点突破信号；后续要看是否能持续出现 >80.682 和是否能突破 80.724。
```

## 2026-07-11 04:32 UTC checkpoint-78

monitor 摘要：

```text
checkpoint_count=68
latest_checkpoint=checkpoint-78.pth.tar
fullval_rows=68
bad_sample_rows=0
best_fullval_line=checkpoint-75 Loss 0.8324 Acc@1 80.7120 Acc@5 95.3680 Samples 50000
above_baseline_lines=10
above_scheme_c_lines=1
above_original_lines=0
target_81_lines=0
last20_avg=80.5898
nonzero_refw_lines=300
nonzero_refw_epochs=63,64,65,74,76,77
pre61_nonzero_refw_lines=0
controller_rows=68
controller_triggers=6
controller_pre61_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7,4:11
```

dynamic full-val：

```text
checkpoint-75: Acc@1 80.7120 Acc@5 95.3680 Samples 50000
checkpoint-76: Acc@1 80.4920 Acc@5 95.4100 Samples 50000
checkpoint-77: Acc@1 80.5920 Acc@5 95.3900 Samples 50000
checkpoint-78: Acc@1 80.5320 Acc@5 95.3880 Samples 50000
```

controller 明细：

```text
epoch 73: drop 0.1280, next_head=8:4, next_weight=2e-5, triggered 1
epoch 74: applied_head=8:4, weight=2e-5, checkpoint-75 reached 80.7120
epoch 75: drop 0.2200, next_head=5:7, next_weight=2e-5, triggered 1
epoch 76: applied_head=5:7, drop 0.1200, next_head=4:11, next_weight=2e-5, triggered 1
epoch 77: applied_head=4:11, no next trigger due window_limit:3/10
```

结论：

```text
checkpoint-75 是当前全 run best 80.7120，超过 scheme C best 80.6820，但仍低于 original OFQ best 80.7240 0.0120。
第二轮 strong pulses 后，checkpoint-76/77/78 没有继续提升，分别回落到 80.4920 / 80.5920 / 80.5320。
controller 规则仍正确：pre61 没有 RefW，未选 blacklist，window_limit 正常阻止继续触发。
目前有效通过标准尚未满足：只 1 个 checkpoint > 80.6820，尚未达到“至少 5 个 checkpoint > scheme C”或“至少 2 个 checkpoint > original best”。
下一步继续观察后段是否还能在 cooldown/window 释放后再次冲击 original best 80.7240。
```

## 2026-07-11 05:05 UTC checkpoint-81

monitor 摘要：

```text
checkpoint_count=71
latest_checkpoint=checkpoint-81.pth.tar
fullval_rows=71
bad_sample_rows=0
best_fullval_line=checkpoint-75 Loss 0.8324 Acc@1 80.7120 Acc@5 95.3680 Samples 50000
above_baseline_lines=12
above_scheme_c_lines=2
above_original_lines=0
target_81_lines=0
last20_avg=80.5922
nonzero_refw_lines=300
nonzero_refw_epochs=63,64,65,74,76,77
pre61_nonzero_refw_lines=0
controller_rows=71
controller_triggers=6
controller_pre61_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7,4:11
```

最近 full-val：

```text
checkpoint-75: Acc@1 80.7120 Acc@5 95.3680 Samples 50000
checkpoint-76: Acc@1 80.4920 Acc@5 95.4100 Samples 50000
checkpoint-77: Acc@1 80.5920 Acc@5 95.3900 Samples 50000
checkpoint-78: Acc@1 80.5320 Acc@5 95.3880 Samples 50000
checkpoint-79: Acc@1 80.6320 Acc@5 95.3840 Samples 50000
checkpoint-80: Acc@1 80.4960 Acc@5 95.4400 Samples 50000
checkpoint-81: Acc@1 80.6840 Acc@5 95.4080 Samples 50000
```

controller 明细：

```text
epoch 78: drop 0.0800, no trigger, reason=window_limit:3/10
epoch 79: drop 0.2160, no trigger, reason=window_limit:3/10
epoch 80: drop 0.0280, no trigger, reason=drop_below_threshold
```

结论：

```text
checkpoint-81 Top-1 80.6840，成为第二个超过 scheme C best 80.6820 的 checkpoint。
当前 best 仍 checkpoint-75 Top-1 80.7120，距离 original OFQ best 80.7240 仍差 0.0120。
controller 继续遵守窗口限制，没有额外触发，也没有选择 blacklist head。
有效通过标准仍未满足：above_scheme_c_lines=2，尚未达到至少 5 个；above_original_lines=0。
```

## 2026-07-11 06:00 UTC checkpoint-87

monitor 摘要：

```text
checkpoint_count=77
latest_checkpoint=checkpoint-87.pth.tar
fullval_rows=77
bad_sample_rows=0
best_fullval_line=checkpoint-75 Loss 0.8324 Acc@1 80.7120 Acc@5 95.3680 Samples 50000
above_baseline_lines=14
above_scheme_c_lines=2
above_original_lines=0
target_81_lines=0
last20_avg=80.5943
nonzero_refw_lines=355
nonzero_refw_epochs=63,64,65,74,76,77,85,87
pre61_nonzero_refw_lines=0
controller_rows=77
controller_triggers=8
controller_pre61_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7,4:11
```

最近 full-val：

```text
checkpoint-81: Acc@1 80.6840 Acc@5 95.4080 Samples 50000
checkpoint-82: Acc@1 80.6700 Acc@5 95.3860 Samples 50000
checkpoint-83: Acc@1 80.5840 Acc@5 95.3820 Samples 50000
checkpoint-84: Acc@1 80.5960 Acc@5 95.3800 Samples 50000
checkpoint-85: Acc@1 80.5520 Acc@5 95.3260 Samples 50000
checkpoint-86: Acc@1 80.6660 Acc@5 95.3420 Samples 50000
checkpoint-87: Acc@1 80.5620 Acc@5 95.4480 Samples 50000
```

controller 明细：

```text
epoch 82: drop 0.1280, no trigger, reason=window_limit:3/10
epoch 83: drop 0.1160, no trigger, reason=window_limit:3/10
epoch 84: drop 0.1600, next_head=8:4, next_weight=2e-5, triggered 1
epoch 85: applied_head=8:4, drop 0.0460, no next trigger
epoch 86: drop 0.1500, next_head=5:7, next_weight=2e-5, triggered 1
epoch 87: applied_head=5:7, RefW=2e-5
```

结论：

```text
run 已推进到 checkpoint-87。
当前 best 仍是 checkpoint-75 Top-1 80.7120；尚未超过 original OFQ best 80.7240。
above_scheme_c_lines 仍为 2，尚未达到有效通过标准的至少 5 个 checkpoint > 80.6820。
controller 仍遵守 pre61、blacklist 和 window/cooldown 规则，但 strong pulse 后没有持续推高曲线。
下一步继续观察到 checkpoint-100/110，最终审计是否达到有效通过或强通过标准。
```

## 2026-07-11 06:55 UTC checkpoint-92

monitor 摘要：

```text
checkpoint_count=82
latest_checkpoint=checkpoint-92.pth.tar
fullval_rows=82
bad_sample_rows=0
best_fullval_line=checkpoint-75 Loss 0.8324 Acc@1 80.7120 Acc@5 95.3680 Samples 50000
above_baseline_lines=17
above_scheme_c_lines=2
above_original_lines=0
target_81_lines=0
last20_avg=80.5928
nonzero_refw_lines=450
nonzero_refw_epochs=63,64,65,74,76,77,85,87,88
pre61_nonzero_refw_lines=0
controller_rows=82
controller_triggers=9
controller_pre61_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7,4:11
```

最近 full-val：

```text
checkpoint-81: Acc@1 80.6840 Acc@5 95.4080 Samples 50000
checkpoint-82: Acc@1 80.6700 Acc@5 95.3860 Samples 50000
checkpoint-83: Acc@1 80.5840 Acc@5 95.3820 Samples 50000
checkpoint-84: Acc@1 80.5960 Acc@5 95.3800 Samples 50000
checkpoint-85: Acc@1 80.5520 Acc@5 95.3260 Samples 50000
checkpoint-86: Acc@1 80.6660 Acc@5 95.3420 Samples 50000
checkpoint-87: Acc@1 80.5620 Acc@5 95.4480 Samples 50000
checkpoint-88: Acc@1 80.5760 Acc@5 95.4020 Samples 50000
checkpoint-89: Acc@1 80.6100 Acc@5 95.4080 Samples 50000
checkpoint-90: Acc@1 80.6420 Acc@5 95.4500 Samples 50000
checkpoint-91: Acc@1 80.6460 Acc@5 95.3900 Samples 50000
checkpoint-92: Acc@1 80.5700 Acc@5 95.3880 Samples 50000
```

controller 明细：

```text
epoch 84: drop 0.1600, next_head=8:4, next_weight=2e-5, triggered 1
epoch 85: applied_head=8:4, no next trigger, drop 0.0460
epoch 86: drop 0.1500, next_head=5:7, next_weight=2e-5, triggered 1
epoch 87: applied_head=5:7, drop 0.1360, next_head=4:11, next_weight=2e-5, triggered 1
epoch 88: applied_head=4:11, no next trigger due window_limit
epoch 89-91: no trigger due window_limit
```

结论：

```text
run 已推进到 checkpoint-92。
当前 best 仍是 checkpoint-75 Top-1 80.7120，尚未超过 original OFQ best 80.7240。
above_scheme_c_lines=2，仍未达到有效通过标准的至少 5 个 checkpoint > 80.6820。
第三轮 strong pulse 没有带来新的高点；checkpoint-89/90/91 保持 80.61-80.65，但没超过 scheme C。
controller 规则继续正常：pre61 无 RefW，无 blacklist，window_limit 生效。
```

## 2026-07-11 08:01 UTC checkpoint-98

monitor 摘要：

```text
checkpoint_count=88
latest_checkpoint=checkpoint-98.pth.tar
fullval_rows=88
bad_sample_rows=0
best_fullval_line=checkpoint-75 Loss 0.8324 Acc@1 80.7120 Acc@5 95.3680 Samples 50000
above_baseline_lines=21
above_scheme_c_lines=5
above_original_lines=0
target_81_lines=0
last20_avg=80.6159
nonzero_refw_lines=517
nonzero_refw_epochs=63,64,65,74,76,77,85,87,88,96,98
pre61_nonzero_refw_lines=0
controller_rows=88
controller_triggers=11
controller_pre61_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7,4:11
```

最近 full-val：

```text
checkpoint-93: Acc@1 80.6860 Acc@5 95.3840 Samples 50000
checkpoint-94: Acc@1 80.6920 Acc@5 95.4160 Samples 50000
checkpoint-95: Acc@1 80.5660 Acc@5 95.3780 Samples 50000
checkpoint-96: Acc@1 80.6260 Acc@5 95.4680 Samples 50000
checkpoint-97: Acc@1 80.7060 Acc@5 95.4600 Samples 50000
checkpoint-98: Acc@1 80.5560 Acc@5 95.4080 Samples 50000
```

controller 明细：

```text
epoch 92: top1 80.6860, drop 0.0260, no trigger
epoch 93: top1 80.6920, drop 0.0200, no trigger
epoch 94: top1 80.5660, drop 0.1460, no trigger due window_limit
epoch 95: top1 80.6260, drop 0.0860, next_head=8:4, next_weight=1e-5, triggered 1
epoch 96: applied_head=8:4, top1 80.7060, no next trigger
epoch 97: top1 80.5560, drop 0.1560, next_head=5:7, next_weight=2e-5, triggered 1
epoch 98: applied_head=5:7, RefW=2e-5
```

结论：

```text
run 已推进到 checkpoint-98。
有效通过标准中的“至少 5 个 checkpoint > 方案 C best 80.6820”已经满足：checkpoint-75, 81, 93, 94, 97。
当前 best 仍是 checkpoint-75 Top-1 80.7120，尚未超过 original OFQ best 80.7240，也未达到 81.0。
controller 仍符合规则：pre61 无 RefW，无 blacklist，window/cooldown 生效。
下一步继续跑到 checkpoint-110 完整结束，做最终审计。
```

## 2026-07-11 09:02 UTC checkpoint-104

monitor 摘要：

```text
checkpoint_count=94
latest_checkpoint=checkpoint-104.pth.tar
fullval_rows=93
bad_sample_rows=0
best_fullval_line=checkpoint-100 Loss 0.8324 Acc@1 80.7600 Acc@5 95.4020 Samples 50000
above_baseline_lines=24
above_scheme_c_lines=6
above_original_lines=1
target_81_lines=0
last20_avg=80.6254
nonzero_refw_lines=600
nonzero_refw_epochs=63,64,65,74,76,77,85,87,88,96,98,99
pre61_nonzero_refw_lines=0
controller_rows=94
controller_triggers=12
controller_pre61_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7,4:11
```

最近 full-val：

```text
checkpoint-93: Acc@1 80.6860 Acc@5 95.3840 Samples 50000
checkpoint-94: Acc@1 80.6920 Acc@5 95.4160 Samples 50000
checkpoint-95: Acc@1 80.5660 Acc@5 95.3780 Samples 50000
checkpoint-96: Acc@1 80.6260 Acc@5 95.4680 Samples 50000
checkpoint-97: Acc@1 80.7060 Acc@5 95.4600 Samples 50000
checkpoint-98: Acc@1 80.5560 Acc@5 95.4080 Samples 50000
checkpoint-99: Acc@1 80.6400 Acc@5 95.4060 Samples 50000
checkpoint-100: Acc@1 80.7600 Acc@5 95.4020 Samples 50000
checkpoint-101: Acc@1 80.5780 Acc@5 95.3980 Samples 50000
checkpoint-102: Acc@1 80.6820 Acc@5 95.4100 Samples 50000
checkpoint-103: Acc@1 80.5960 Acc@5 95.3200 Samples 50000
checkpoint-104: Acc@1 80.6440 Acc@5 95.3720 Samples 50000
```

controller 明细：

```text
epoch 95: drop 0.0860, next_head=8:4, next_weight=1e-5, triggered 1
epoch 96: applied_head=8:4, top1 80.7060, no next trigger
epoch 97: drop 0.1560, next_head=5:7, next_weight=2e-5, triggered 1
epoch 98: applied_head=5:7, drop 0.0720, next_head=4:11, next_weight=1e-5, triggered 1
epoch 99: applied_head=4:11, top1 80.7600, no next trigger
epoch 100-103: no trigger because window_limit:3/10
```

结论：

```text
checkpoint-100 Top-1 80.7600，首次超过 original OFQ best 80.7240，above_original_lines=1。
有效通过标准已经明确满足：above_scheme_c_lines=6，且 best > original OFQ best。
强通过仍未满足：best 80.7600 < 80.85，target_81_lines=0。
controller 规则仍正常：pre61 无 RefW，无 blacklist，window_limit 生效。
下一步继续跑到 checkpoint-110 完整结束，做最终审计和结论归档。
```

## 2026-07-11 10:20 UTC final audit

最终 monitor 摘要：

```text
checkpoint_count=100
latest_checkpoint=checkpoint-110.pth.tar
fullval_rows=100
bad_sample_rows=0
best_fullval_line=checkpoint-100 Loss 0.8324 Acc@1 80.7600 Acc@5 95.4020 Samples 50000
above_baseline_lines=30
above_scheme_c_lines=7
above_original_lines=1
target_81_lines=0
last20_avg=80.6382
nonzero_refw_lines=700
nonzero_refw_epochs=63,64,65,74,76,77,85,87,88,96,98,99,107,109
pre61_nonzero_refw_lines=0
controller_rows=100
controller_triggers=15
controller_pre61_triggers=0
controller_selected_avoid=0
controller_next_heads=8:4,5:7,4:11
```

最终 checkpoint：

```text
checkpoint-106: Acc@1 80.6240 Acc@5 95.4060 Samples 50000
checkpoint-107: Acc@1 80.6340 Acc@5 95.3800 Samples 50000
checkpoint-108: Acc@1 80.7240 Acc@5 95.4440 Samples 50000
checkpoint-109: Acc@1 80.6000 Acc@5 95.3860 Samples 50000
checkpoint-110: Acc@1 80.6600 Acc@5 95.3600 Samples 50000
```

最佳 checkpoint：

```text
checkpoint-100
Loss: 0.8324
Acc@1: 80.7600
Acc@5: 95.4020
Samples: 50000
delta_vs_baseline_80.5980: +0.1620
delta_vs_scheme_c_80.6820: +0.0780
delta_vs_original_80.7240: +0.0360
```

成功标准审计：

```text
最低通过: best Top-1 > original OFQ best 80.7240
结果: 80.7600 > 80.7240, 通过

有效通过:
  条件 A: 至少 2 个 checkpoint > 80.7240
  结果: above_original_lines=1, 不满足

  条件 B: 至少 5 个 checkpoint > 方案 C 80.6820
  结果: above_scheme_c_lines=7, 满足

强通过:
  条件 A: best Top-1 >= 80.85
  结果: 80.7600 < 80.85, 不满足

  条件 B: last20 average 高于原版 OFQ 后段均值
  结果: last20_avg=80.6382，原版后段均值未在本审计脚本内重新计算，不单独判定

81.0 target:
  target_81_lines=0, 未达到
```

controller 完成审计：

```text
epoch 10-60:
  phase=observe
  pre61_nonzero_refw_lines=0
  controller_pre61_triggers=0

epoch 61-110:
  controller_triggers=15
  nonzero_refw_epochs=63,64,65,74,76,77,85,87,88,96,98,99,107,109
  selected heads: 8:4,5:7,4:11
  avoid heads selected: 0
  spike score columns present: applied_spike_score,next_spike_score
  cooldown_state present in controller TSV
```

完整性审计：

```text
checkpoint files: checkpoint-11 through checkpoint-110, count=100
full-val rows: 100
bad sample rows: 0
all full-val Samples: 50000
training process: exited
GPU after finish: 8 cards back to about 7 MiB used, util 0
```

退出日志说明：

```text
训练结束后 rank 3/5 出现 TCPStore/NCCL heartbeat warning:
Failed to check the should dump flag on TCPStore / TCPStore server has shut down too early

该 warning 出现在 checkpoint-110、full-val、wall_seconds 和输出路径已经写出之后。
最终 checkpoint/full-val/controller TSV 均完整，因此本次按训练完成处理。
```

最终结论：

```text
本实验完成 10->110 的 100 epoch dynamic sparse prev-step KL 长跑。

算法目标层面:
  dynamic sparse prev-step KL 在后段产生有效正信号。
  best checkpoint-100 Top-1 80.7600，超过 original OFQ 10->60 best 80.7240。
  above_scheme_c_lines=7，满足有效通过标准。

限制:
  未达到 81.0。
  未达到 strong pass 的 best >= 80.85。
  above_original_lines=1，超过 original best 的 checkpoint 数还不多。

重要发现:
  10-60 observe-only 段虽然没有主动 KL，但因启用了 ema_ref_attn_kl/refmodel/attention collection runtime 路径，表现不完全等价于原版 OFQ 10->60；原版 checkpoint-52 80.7240 未复现。
  真正突破 original best 出现在后段 dynamic controller 触发之后，说明 sparse prev-step KL controller 方向有价值，但仍需改进稳定性和提高高点数量。
```

启动后立即检查：

```text
args.yaml 中 dynamic_sparse_prevstep_kl: true
args.yaml 中 train_scheme: ema_ref_attn_kl
args.yaml 中 ref_update: prev_step
args.yaml 中 ref_attn_kl_weight: 0.0
args.yaml 中 dynamic_kl_start_epoch: 61
args.yaml 中 dynamic_kl_observe_until_epoch: 60
args.yaml 中 kd_hard_and_soft: 0
args.yaml 中 teacher_soft_temperature: 2.75
args.yaml 中 batch_size: 64
args.yaml 中 checkpoint_hist: 110
```

## 轮询命令

```text
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/monitor_ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710.sh
cat /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_monitor_summary_20260710.txt
```

每次轮询关注：

```text
checkpoint_count
latest_checkpoint
fullval_rows
bad_sample_rows
best_fullval_line
above_baseline_lines
above_scheme_c_lines
above_original_lines
target_81_lines
pre61_nonzero_refw_lines
controller_pre61_triggers
controller_selected_avoid
controller_next_heads
```

## 最终审计清单

必须确认：

```text
checkpoint-11 到 checkpoint-110 是否完整生成
full-val rows 是否完整且 Samples=50000
epoch 10-60 是否没有主动 KL pulse
epoch 61-110 的 KL 是否只由 controller 触发
所有触发记录是否都有 selected head / weight / reason / drop / cooldown
avoid heads 是否从未被选中
best checkpoint 是哪个
超过 80.5980、80.6820、80.7240 的 checkpoint 数量
是否达到 81.0
是否值得进入下一轮完整方案
```
