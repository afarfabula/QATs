# OFQ sparse pulse prev-step refKL 50epoch 方案 C goal

## 目标

从已验证的 `checkpoint-10` 起点继续训练 50 个 resumed epoch，保持原版 OFQ public-family 主链路，并加入方案 C：低频强脉冲式 sparse prev-step refmodel attention KL。

目标是验证该方案能否超过原版 OFQ direct resume baseline `80.5980`，并观察是否能达到 `81.0`。

## 已有结论

原版 OFQ direct resume `checkpoint-10 -> checkpoint-30` 已完整完成：

```text
best checkpoint: checkpoint-27
Top-1: 80.5980
Top-5: 95.3560
Loss: 0.8404
Samples: 50000
```

上一条 sparse prev-step refKL 实验异常停在 `checkpoint-26`：

```text
best checkpoint: checkpoint-26
Top-1: 80.4840
Top-5: 95.3280
Loss: 0.8391
Samples: 50000
```

该实验确认 `RefW` 和 `RefAttnKL` 能正常生效，但当前 sparse pulse 设置未超过 baseline。还确认 `ref_attn_kl_weight_epoch_overrides` 是精确 epoch 生效，不是持续生效。

## 固定起点

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
```

起点 full-val：

```text
Top-1: 80.3640
Top-5: 95.3140
Loss: 0.8453
Samples: 50000
```

## 主链路约束

保持 OFQ public-family 主链路：

```text
wq_mode: statsq
aq_mode: lsq
qk_reparam: true
qk_reparam_type: 0
teacher KD: enabled
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
resume: checkpoint-10
no_resume_opt: true
batch_size: 64
epoch_checkpoint_interval: 1
checkpoint_hist: 60
```

禁止：

```text
不使用 soup
不使用 checkpoint averaging
不使用 multi-checkpoint averaging
不使用 ensemble
不使用 A8->A4
不切换到非 Attention relation / refmodel 路线
```

## 训练长度

```text
epochs: 60
scheduler_epochs: 60
expected checkpoints: checkpoint-11 到 checkpoint-60
```

## 方案 C 配置

设计理念：低频强脉冲 sparse KL。先保留 OFQ 自然上升路径，再用少量两连 epoch 的 KL 脉冲做 attention relation reset / drift correction。

```text
train_scheme: ema_ref_attn_kl
ref_update: prev_step
ref_update_interval: 50
ref_head_mode: custom_subset:6:1,8:4,8:9,11:18,11:4
ref_warmup_epochs: 28
ref_attn_kl_weight: 0.0
ref_attn_kl_weight_epoch_overrides: 28:0.00030,29:0.00030,36:0.00035,37:0.00035,44:0.00035,45:0.00035,52:0.00030,53:0.00030
ref_attn_kl_drop_prob: 0.50
ref_attn_loss: kl_ref
```

关键观察点：

```text
epoch 28/29 -> checkpoint-30
epoch 36/37 -> checkpoint-38
epoch 44/45 -> checkpoint-46
epoch 52/53 -> checkpoint-54
```

## 建议产物路径

```text
script:
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709.sh

experiment:
ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709

log:
/mlx_devbox/users/quyanyi/playground/train_ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709.log

output:
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709

progress doc:
/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_sparse_pulse_prevstep_refkl_c_50epoch_progress_20260709.md
```

## 轮询和记录

1. 启动后持续轮询训练日志、输出目录和 checkpoint。
2. 每个 checkpoint 都记录 Top-1、Top-5、Loss、Samples、相对 baseline `80.5980` 的差值。
3. full-val 必须来自日志中的 `Test: [distributed-summary] ... Samples: 50000`。
4. 检查 epoch 28、29、36、37、44、45、52、53 是否出现预期非零 `RefW`。
5. 持续写入中文实验日志、阶段结论和异常。

## 成功标准

```text
超过 baseline: 任意单 checkpoint Top-1 > 80.5980
达到最终目标: 任意单 checkpoint Top-1 >= 81.0
```

## 异常处理

如果训练异常停止或 hang：

```text
不盲目重启
优先使用已有 checkpoint 做 full-val 和结论审计
记录停止位置、最后 checkpoint、日志尾部、是否有 traceback / OOM / NCCL / wall_seconds
```

## 完成判定

满足任一条件即可完成：

1. 跑完 `checkpoint-60`，并完成 `checkpoint-11..checkpoint-60` full-val 结果表和最终结论。
2. 中途出现单 checkpoint Top-1 >= `81.0`。
3. 训练异常结束，但已对所有可用 checkpoint 完成 full-val、baseline 对比、81 目标对比、RefW 生效检查和中文结论审计。
