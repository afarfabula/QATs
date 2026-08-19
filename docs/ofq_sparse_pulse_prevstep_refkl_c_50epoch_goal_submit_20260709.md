# 方案 C 50epoch sparse prev-step refKL goal 提交稿

## Goal 文字

继续推进 Swin-T W4A4-family QAT 的 `checkpoint-10 -> checkpoint-60` 50epoch 长跑验证，目标是在不使用 soup、checkpoint averaging、ensemble、A8->A4 或非 refmodel 路线的前提下，验证“方案 C：低频强脉冲 sparse prev-step refmodel Attention KL”能否超过原版 OFQ direct resume baseline。

固定从已验证的 10epoch checkpoint resume：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
```

保持原版 OFQ public-family 主链路：

```text
statsQ + LSQ activation + qk_reparam + teacher KD
kd_hard_and_soft=0
teacher_soft_temperature=2.75
no_resume_opt=true
batch_size=64
epoch_checkpoint_interval=1
checkpoint_hist=60
```

加入方案 C sparse pulse prev-step refKL：

```text
train_scheme=ema_ref_attn_kl
ref_update=prev_step
ref_update_interval=50
ref_head_mode=custom_subset:6:1,8:4,8:9,11:18,11:4
ref_warmup_epochs=28
ref_attn_kl_weight=0.0
ref_attn_kl_weight_epoch_overrides=28:0.00030,29:0.00030,36:0.00035,37:0.00035,44:0.00035,45:0.00035,52:0.00030,53:0.00030
ref_attn_kl_drop_prob=0.50
ref_attn_loss=kl_ref
```

执行时在真实 GPU worker TTY 中启动已准备好的脚本：

```bash
cd /mlx_devbox/users/quyanyi/playground
MASTER_PORT=31561 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709.sh
```

持续轮询训练进度、checkpoint、full-val 指标和 `RefW` 生效情况，并把中文实验日志写入：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_sparse_pulse_prevstep_refkl_c_50epoch_progress_20260709.md
```

判定标准：

```text
baseline: 原版 OFQ direct resume best checkpoint-27 Top-1 80.5980
成功超过 baseline: 任意单 checkpoint Top-1 > 80.5980
达到最终目标: 任意单 checkpoint Top-1 >= 81.0
full-val 要求: Test: [distributed-summary] 且 Samples=50000
```

异常处理：

```text
如果训练 hang 或异常停止，不盲目重启。
优先对已有 checkpoint 做 full-val，记录停止位置、最后 checkpoint、日志尾部、RefW 是否生效、是否出现 traceback/OOM/NCCL。
```

完成条件：

```text
1. 跑到 checkpoint-60，并完成 checkpoint-11..60 的 full-val 表、baseline 对比和结论。
2. 或中途任意 checkpoint Top-1 >= 81.0。
3. 或训练异常结束，但已完成所有可用 checkpoint 的 full-val、RefW 审计和中文结论记录。
```

## 相关文件

```text
goal submit doc:
/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_sparse_pulse_prevstep_refkl_c_50epoch_goal_submit_20260709.md

detailed goal doc:
/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_sparse_pulse_prevstep_refkl_c_50epoch_goal_20260709.md

progress doc:
/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_sparse_pulse_prevstep_refkl_c_50epoch_progress_20260709.md

launch script:
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709.sh

log:
/mlx_devbox/users/quyanyi/playground/train_ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709.log

output:
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709
```
