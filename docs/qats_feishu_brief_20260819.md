# QAT / OFQ Swin-T W4A4 项目简报

## 1. 项目怎么启动训练

项目主目录：

```text
/mlx_devbox/users/quyanyi/playground/QATs
```

统一训练入口是 `qat_launch.py`。实际长跑不要手写长命令，优先用已经归档的脚本启动，脚本里固定了数据、模型、量化配置、日志路径、checkpoint 频率和分布式参数。

常用启动方式：

```bash
cd /mlx_devbox/users/quyanyi/playground/QATs

# 当前最好 200epoch fixed-cycle sparse prev-step KL
bash tmp_scripts/run_ofq_200ep_fromscratch_fixedcycle_group_sparse_prevstep_refkl_20260731.sh

# 最新完整 teacher KLD1 / FP teacher attention-KL 实验
bash tmp_scripts/run_ofq_100ep_fromscratch_teacher_sparse_attnkl_clipgrad_20260804.sh

# 最新一次 late-polish teacher KLD1 试验，当前只跑到 66 个 full-val 点
bash tmp_scripts/run_ofq_100ep_fromscratch_teacher_sparse_attnkl_latepolish_20260805.sh
```

核心公共配置：

```text
method=ofq
model=swin_t
wbits=4, abits=4
wq_mode=statsq, aq_mode=lsq
qk_reparam=true
data=/tmp/imagenet1k_full_parquet
teacher=/mlx_devbox/users/quyanyi/playground/QATs/checkpoints/pretrained/swin_t-704ceda3.pth
full validation samples=50000
```

这些实验建议表述为 `Swin-T W4A4-family`，不要直接写成 strict W4A4 SOTA，因为 first/last layer 等细节还没有单独重新审计。

## 2. 关键实验结果

| 实验 | epoch 范围 | 最好 Top-1 | 最好点 | 最后 Top-1 | 结论 |
|---|---:|---:|---:|---:|---|
| `ofq_100ep_fromscratch_original_ofq_public_control_20260714` | 0-99 | 80.7920 | epoch 81 | 80.6780 | 100ep no-KL 基线很强 |
| `ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713` | 0-99 | 80.7720 | epoch 99 | 80.7720 | 后段保峰更稳，但没有超过 no-KL best |
| `ofq_100ep_fromscratch_teacher_sparse_attnkl_fixed_20260803` | 0-99 | 80.7920 | epoch 81 | 80.6780 | teacher-KL 路径跑通，但轨迹等同 no-KL |
| `ofq_100ep_fromscratch_teacher_sparse_attnkl_clipgrad_20260804` | 0-99 | 80.8180 | epoch 99 | 80.8180 | 最新完整 teacher KLD1，有小幅提升但还不能单独证明收益 |
| `ofq_200ep_fromscratch_fixedcycle_group_sparse_prevstep_refkl_20260731` | 0-199 | 80.8680 | epoch 194 | 80.7080 | 当前单 checkpoint 最好结果 |
| `ofq_resume200_to300_fixedcycle_sparse_prevstep_refkl_20260802` | 200-299 | 80.8420 | epoch 209 | 80.6900 | 延长到 300ep 没有超过 200ep 最好点 |

当前最好的 checkpoint：

```text
/tmp/qat_public_repro/ofq_200ep_fromscratch_fixedcycle_group_sparse_prevstep_refkl_20260731/checkpoint-195.pth.tar
```

注意日志 epoch 是 0-based，所以 `epoch 194` 对应 `checkpoint-195`。

## 3. 最新 teacher KLD1 实验轨迹和新意

teacher KLD1 这组实验的新意不是简单加 KD，而是用 FP Swin-T teacher 的 attention relation 直接约束量化模型里最容易震荡的 attention heads。它和之前的 prev-step ref KL 不同：prev-step 是让当前量化模型对齐自己的上一状态，teacher KLD1 是把 FP teacher 当成外部稳定锚点，希望减少 late-stage attention relation 抖动。

最新完整实验是：

```text
ofq_100ep_fromscratch_teacher_sparse_attnkl_clipgrad_20260804
log: experiment_logs/fullval_ge10/playground__train_ofq_100ep_fromscratch_teacher_sparse_attnkl_clipgrad_20260804.log
script: tmp_scripts/run_ofq_100ep_fromscratch_teacher_sparse_attnkl_clipgrad_20260804.sh
```

它的 Top-1 轨迹：

| epoch | Top-1 |
|---:|---:|
| 0 | 77.7080 |
| 10 | 79.0580 |
| 20 | 79.4480 |
| 30 | 79.6880 |
| 40 | 80.1240 |
| 50 | 80.2640 |
| 60 | 80.4560 |
| 70 | 80.5660 |
| 80 | 80.5140 |
| 90 | 80.7400 |
| 99 | 80.8180 |

这个结果比 100ep no-KL best `80.7920` 高 `+0.0260`，但幅度很小；因此现在合理结论是“teacher attention-KL 路径可运行，并且没有破坏收敛，完整 100ep 达到 80.8180”，但还不能把它写成确定有效的主增益。

关键原因是 loss scale 仍偏保守：日志里 `TeacherAttnKL` 常被 clip 到 `20.0`，权重主要是 `1e-6` 到 `2e-6`，等效损失贡献约 `2e-5` 到 `4e-5`，对总 loss 影响很弱。它更像一次正确方向的低强度校准实验。

最新一次启动的改法是：

```text
ofq_100ep_fromscratch_teacher_sparse_attnkl_latepolish_20260805
```

它把 teacher KLD1 推迟到 epoch 60 以后再打开，意图是先保留 OFQ 自然 warmup，再用 FP teacher 做 late-stage attention polish。当前日志只到 66 个 full-val 点：

| epoch | Top-1 |
|---:|---:|
| 0 | 77.7080 |
| 10 | 78.9520 |
| 20 | 79.3320 |
| 30 | 79.7980 |
| 40 | 80.0840 |
| 50 | 80.2140 |
| 60 | 80.4120 |
| 65 | 80.4800 |

这条 late-polish 还没有跑到真正发力区间，当前最好是 epoch 59 的 `80.5260`。它的价值主要在设计上：验证“teacher-KLD1 只作为后段 polish，而不是全程扰动”的范式，但还不能作为最终结果汇报。

## 4. 当前判断

1. 目前最强可交付结果是 200epoch fixed-cycle sparse prev-step KL，Top-1 `80.8680`。
2. 100epoch teacher KLD1 完整版达到 `80.8180`，说明 FP teacher attention relation 可以接入长跑并保持稳定，但当前权重偏小，证据还不足以证明显著收益。
3. 下一步如果继续 teacher KLD1，建议先做 5-10epoch 小门槛，把等效 KL loss 从 `1e-5` 量级提高到 `1e-3` 到 `1e-2` 量级，再决定是否跑完整 100/200epoch。

相关日志和脚本已经归档并 push 到 `origin/main`，提交为：

```text
67251c2 Archive QAT experiment logs and scripts
```
