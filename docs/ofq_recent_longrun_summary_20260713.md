# OFQ / prev-step KL recent long-run summary

## 背景

最近这一组实验围绕同一个问题展开：

```text
从 checkpoint-10 的 80% 左右起点继续 resume 长跑，
验证 OFQ public-family 原版链路和 sparse prev-step attention relation KL
是否能在单 checkpoint 训练链路下把 Top-1 推到 81.0。
```

约束：

```text
不使用 soup
不使用 checkpoint averaging
不使用 ensemble
不做 A8 -> A4
主链路保持 OFQ public-family: statsq / lsq / qk_reparam / soft-KD T=2.75 / no_resume_opt / batch_size=64
```

共同起点：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
```

核心阈值：

```text
baseline: 80.5980
scheme C best: 80.6820
original OFQ 10->60 best: 80.7240
original OFQ 10->110 best: 80.7520
dynamic KL 10->110 best: 80.7600
target: 81.0
```

## 总表

| 实验 | 区间 | KL / controller | Best Top-1 | Best checkpoint | >80.682 | >80.724 | >80.752 | >80.760 | >=81 | last20 | last10 | 结论 |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 原版 OFQ public-family | 10->60 | 无 KL | 80.7240 | checkpoint-52 | 1 | 0 | 0 | 0 | 0 | 80.5801 | 80.6120 | 原版 50 epoch 短长跑基线，第一次给出 80.7240 |
| dynamic sparse prev-step KL | 10->110 | 61 后触发，3/10 window | 80.7600 | checkpoint-100 | 7 | 1 | 1 | 0 | 0 | 80.6382 | 80.6316 | 比 10->60 best 高 0.036，但优势薄 |
| 原版 OFQ public-family | 10->110 | 无 KL | 80.7520 | checkpoint-102 | 7 | 2 | 0 | 0 | 0 | 80.6519 | 80.6520 | 原版自然长跑几乎追平 dynamic 10->110，且后段均值更高 |
| late sparse prev-step KL | 10->210 | 91 后触发，2/12 window | 80.8280 | checkpoint-99 | 33 | 10 | 7 | 4 | 0 | 80.6592 | 80.6542 | 当前最强方案；best 明显超过 10->110，但仍未达 81 |

## 实验 1: 原版 OFQ public-family 10->60

路径：

```text
progress: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to60_original_ofq_public_progress_20260710.md
status: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to60_original_ofq_public_status_20260710.tsv
summary: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to60_original_ofq_public_monitor_summary_20260710.txt
```

配置：

```text
epochs=60
scheduler_epochs=60
train_scheme=baseline
dynamic_sparse_prevstep_kl=false
RefW=0
```

结果：

```text
rows=50
bad_sample_rows=0
best=checkpoint-52
best_loss=0.8369
best_top1=80.7240
best_top5=95.3400
above_baseline_80.5980=6
above_scheme_c_80.6820=1
target_81=0
last20_avg=80.5801
last10_avg=80.6120
```

结论：

```text
原版 OFQ 10->60 能自然到 80.7240。
这是后续 dynamic KL 实验最初要超过的“原版长跑”基线。
```

## 实验 2: dynamic sparse prev-step KL 10->110

路径：

```text
progress: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_progress_20260710.md
status: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_status_20260710.tsv
controller: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_controller_20260710.tsv
summary: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_dynamic_sparse_prevstep_refkl_monitor_summary_20260710.txt
```

配置：

```text
epochs=110
scheduler_epochs=110
train_scheme=ema_ref_attn_kl
ref_update=prev_step
dynamic_sparse_prevstep_kl=true
dynamic_kl_start_epoch=61
dynamic_kl_observe_until_epoch=60
dynamic_kl_drop_threshold=0.06
dynamic_kl_strong_drop_threshold=0.12
dynamic_kl_default_weight=1e-05
dynamic_kl_strong_weight=2e-05
window_epochs=10
max_pulses_per_window=3
selected heads: 8:4,5:7,4:11
avoid heads selected: 0
```

结果：

```text
rows=100
bad_sample_rows=0
best=checkpoint-100
best_loss=0.8324
best_top1=80.7600
best_top5=95.4020
above_baseline_80.5980=30
above_scheme_c_80.6820=7
above_original10to60_80.7240=1
target_81=0
last20_avg=80.6382
last10_avg=80.6316
controller_triggers=15
pre61_nonzero_refw_lines=0
nonzero_refw_epochs=63,64,65,74,76,77,85,87,88,96,98,99,107,109
```

关键观察：

```text
checkpoint-100 Top-1 80.7600，第一次超过 original 10->60 best 80.7240。
但 above_original10to60 只有 1 个 checkpoint，且未达到 81。
```

结论：

```text
这个方案证明 sparse prev-step KL 有正信号，但优势很薄。
后续用原版 OFQ 10->110 对照后发现，80.7600 不能单独证明 KL 显著抬高上限。
```

## 实验 3: 原版 OFQ public-family 10->110

路径：

```text
progress: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_original_ofq_public_progress_20260711.md
status: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_original_ofq_public_status_20260711.tsv
summary: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to110_original_ofq_public_monitor_summary_20260711.txt
```

配置：

```text
epochs=110
scheduler_epochs=110
train_scheme=baseline
dynamic_sparse_prevstep_kl=false
RefW=0
```

结果：

```text
rows=100
bad_sample_rows=0
best=checkpoint-102
best_loss=0.8284
best_top1=80.7520
best_top5=95.4300
above_baseline_80.5980=33
above_scheme_c_80.6820=7
above_original10to60_80.7240=2
above_dynamic10to110_80.7600=0
target_81=0
last20_avg=80.6519
last10_avg=80.6520
nonzero_refw_lines=0
```

关键高点：

```text
checkpoint-96: 80.7500
checkpoint-102: 80.7520
```

与 dynamic 10->110 对比：

```text
dynamic best: 80.7600
original best: 80.7520
gap: 0.0080

dynamic last20_avg: 80.6382
original last20_avg: 80.6519

dynamic last10_avg: 80.6316
original last10_avg: 80.6520
```

结论：

```text
这个对照改变了判断：旧 dynamic KL 的 best 只比原版高 0.0080，
而后段均值还低于原版。
因此“61 后触发的旧 controller”不是 81 方案，也不能作为 KL 显著抬上限的强证据。
```

## 实验 4: late sparse prev-step KL 10->210

路径：

```text
progress: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to210_late_sparse_prevstep_refkl_progress_20260712.md
status: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to210_late_sparse_prevstep_refkl_status_20260712.tsv
controller: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to210_late_sparse_prevstep_refkl_controller_20260712.tsv
summary: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to210_late_sparse_prevstep_refkl_monitor_summary_20260712.txt
```

配置：

```text
epochs=210
scheduler_epochs=210
train_scheme=ema_ref_attn_kl
ref_update=prev_step
dynamic_sparse_prevstep_kl=true
dynamic_kl_start_epoch=91
dynamic_kl_observe_until_epoch=90
dynamic_kl_drop_threshold=0.08
dynamic_kl_strong_drop_threshold=0.16
dynamic_kl_default_weight=1e-05
dynamic_kl_strong_weight=2e-05
dynamic_kl_max_weight=2e-05
dynamic_kl_window_epochs=12
dynamic_kl_max_pulses_per_window=2
dynamic_kl_cooldown_epochs=7
ref_attn_kl_drop_prob=0.5
ref_attn_kl_clip=20.0
primary_heads=8:4,5:7,4:11
secondary_heads=11:18,6:1
actual selected heads=8:4,5:7
avoid heads selected=0
```

设计动机：

```text
旧 dynamic KL 从 61 开始，过早。
原版 OFQ 在 90->110 自然能达到 80.7520。
所以新方案推迟到 epoch 91 后，只做 late stabilizer，不干预 10->90 的自然高点形成。
同时把 window 从 3/10 改成 2/12，减少过度 pulse。
```

最终结果：

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
controller_triggers=19
pre91_nonzero_refw_lines=0
controller_pre91_triggers=0
controller_selected_avoid=0
nonzero_refw_epochs=92,94,105,107,118,120,131,133,144,146,157,159,170,172,183,185,196,198,209
```

关键高点：

```text
checkpoint-99: 80.8280
checkpoint-127: 80.8040
checkpoint-160: 80.7680
checkpoint-195: 80.7820
checkpoint-201: 80.7600
```

阶段观察：

```text
checkpoint-79: observe 段自然到 80.7080
checkpoint-99: dynamic 后达到全局 best 80.8280
checkpoint-127: 第二个 80.8+，说明 checkpoint-99 不是完全孤立单点
checkpoint-160: 5:7 pulse 后出现 80.7680
checkpoint-182->191: 多个 80.70+，后段稳定性改善
checkpoint-203->210: 未再刷新 best
```

结论：

```text
这是最近长跑里最强方案。
它明确超过所有 10->110 历史 best：
80.8280 > 80.7600 > 80.7520 > 80.7240

但仍未达到 81.0。
主要问题不是“prev-step KL 没有效”，而是 controller 高点稳定性不足：
高点能出现，但不能稳定维持，也不能继续推到 81。
```

## 横向结论

### 1. 问题定义的修正

早期我们怀疑“resume 到 81”任务可能本身定义有问题。最近长跑后的判断更具体：

```text
10->110 原版 OFQ 已经可以自然到 80.7520。
因此 80.75 左右不能算 KL 的强贡献。
必须超过 80.7600，最好多次超过，才算方法有新增价值。
```

200 epoch late sparse KL 达到：

```text
best=80.8280
above_dynamic10to110=4
```

这说明方法方向有效，但还没解决 81。

### 2. late-start 比 early-start 更合理

旧方案：

```text
start_epoch=61
best=80.7600
last20_avg=80.6382
```

新方案：

```text
start_epoch=91
best=80.8280
last20_avg=80.6592
```

结论：

```text
不要回到 61 过早触发。
90 之前保留 OFQ 自然形成高点，91 后再用 KL 做回落稳定器，是更合理的方向。
```

### 3. 当前 controller 的主要问题

当前 controller 是被动 drop-trigger：

```text
rolling_best - current_top1 >= threshold -> 触发 sparse pulse
window_limit=2/12
cooldown=7
```

观察到的问题：

```text
1. 高点后连续大 drop 时，window_limit 经常阻止继续补救。
2. pulse 后有时能产生高点，例如 checkpoint-99 / 160。
3. 但 pulse 不能稳定守住 80.8+。
4. 后段 last20 只到 80.6592，没有形成 80.7+ 均值平台。
```

### 4. 81 未达成的原因

当前证据支持：

```text
不是 OFQ 主链路完全上不去，已经到 80.8280。
也不是 prev-step KL 完全无效，已经多次超过旧 best。
真正瓶颈是高点稳定性和 controller 策略。
```

81 需要的不只是触发 KL，而是：

```text
在 80.75+ 后少掉下去，
或者掉下去后更快、更连续地补救，
形成多个 80.85 附近窗口。
```

## 下一步建议

如果继续这一方向，建议下一轮不要再证明 prev-step KL 是否有效，而是改 controller：

```text
1. 保留 late-start，仍从 epoch 90 后开始。
2. 对 80.75+ 后的回落引入 high-water protect mode。
3. window_limit 动态化：
   - 普通阶段保守 2/12
   - 若 rolling_best > 80.75 且 drop > 0.12，允许 3/12 或连续两个小 pulse
4. pulse weight 不一定加大，优先增加小权重 pulse 的连续性。
5. 保留 avoid heads，不引入 A8->A4 / soup / checkpoint averaging。
```

一个可验证的下一轮目标：

```text
best >= 80.90
above_dynamic10to110 >= 8
last20_avg >= 80.70
target_81_lines >= 1 作为强成功
```

## 对外汇报口径

可以这样说：

```text
我们先用原版 OFQ 做了 10->110 对照，发现原版本身能到 80.752，
所以旧 dynamic KL 的 80.760 其实优势很薄。

之后重新设计成 200 epoch late sparse prev-step KL：
90 epoch 前不干预，让 OFQ 自然形成高点；
90 epoch 后用 sparse prev-step KL 做回落稳定。

这个新方案 best 到 80.828，超过了所有 10->110 历史 best，
并且有 4 个 checkpoint 超过旧 dynamic best 80.760。

但还没有到 81。现在主要瓶颈不是 KL 无效，
而是 controller 不能稳定守住高点。下一步要做 high-water protect / 动态 window limit，
把 80.8+ 从单点提升成稳定窗口。
```
