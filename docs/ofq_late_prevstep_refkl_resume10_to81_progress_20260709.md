# late selective prev-step refmodel attention KL 20epoch 实验记录

## 目标

验证自研 `late-only selective prev-step refmodel attention KL` 是否能在完整 20epoch 设置中超过原版 OFQ direct resume 基线。

原版 OFQ direct resume 基线：

```text
experiment: ofq_public_resume10_to30_20260709
best checkpoint: checkpoint-27
Top-1: 80.5980
Top-5: 95.3560
Loss: 0.8404
Samples: 50000
```

本实验成功标准：

1. 任意单 checkpoint Top-1 > `80.5980`：超过原版 OFQ direct resume 基线。
2. 任意单 checkpoint Top-1 >= `81.0`：达到最终目标。
3. 所有结果必须是单 checkpoint、full ImageNet raw validation、`Samples=50000`，不使用 soup / averaging / ensemble。

## 实验信息

```text
script: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume10_to30_late_prevstep_refkl_20260709.sh
experiment: ofq_resume10_to30_late_prevstep_refkl_20260709
log: /mlx_devbox/users/quyanyi/playground/train_ofq_resume10_to30_late_prevstep_refkl_20260709.log
output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_resume10_to30_late_prevstep_refkl_20260709
resume checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
```

## 关键配置

保持与原版 OFQ direct resume baseline 相同的主链路：

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
epochs: 30
scheduler_epochs: 30
epoch_checkpoint_interval: 1
checkpoint_hist: 30
```

新增机制：

```text
train_scheme: ema_ref_attn_kl
ref_update: prev_step
ref_update_interval: 50
ref_head_mode: custom_subset:6:1,8:4,8:9,11:18,11:4
ref_warmup_epochs: 23
ref_attn_kl_weight: 0.0
ref_attn_kl_weight_epoch_overrides: 23:0.0002,25:0.0005,28:0.0002
ref_attn_kl_drop_prob: 0.25
ref_attn_loss: kl_ref
```

启动确认：

```text
Strict resume: loaded model from .../recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Enabled EMA refmodel attention-KL scheme: ref_update=prev_step, ref_update_interval=50, ... head_mode=custom_subset:6:1,8:4,8:9,11:18,11:4, selected_head_map={6: (1,), 8: (4, 9), 11: (4, 18)}, warmup_epochs=23
```

说明：`RefW=0` 在 checkpoint-23 之前是预期行为，因为本实验设计为 late-only selective KL。

## 当前结果

| checkpoint | single checkpoint | raw val samples | RefW expected | Top-1 | Top-5 | Loss | 对比 OFQ baseline best 80.5980 |
|---|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | 50000 | 0 | 80.3640 | 95.3140 | 0.8453 | -0.2340 |
| checkpoint-11 | yes | 50000 | 0 | 80.3840 | 95.3220 | 0.8494 | -0.2140 |
| checkpoint-12 | yes | 50000 | 0 | 80.4540 | 95.3280 | 0.8444 | -0.1440 |
| checkpoint-13 | yes | 50000 | 0 | 80.2400 | 95.2960 | 0.8431 | -0.3580 |
| checkpoint-14 | yes | 50000 | 0 | 80.4500 | 95.2680 | 0.8431 | -0.1480 |
| checkpoint-15 | yes | 50000 | 0 | 80.2780 | 95.3400 | 0.8399 | -0.3200 |
| checkpoint-16 | yes | 50000 | 0 | 80.4480 | 95.2680 | 0.8435 | -0.1500 |
| checkpoint-17 | yes | 50000 | 0 | 80.4380 | 95.3080 | 0.8424 | -0.1600 |
| checkpoint-18 | yes | 50000 | 0 | 80.3180 | 95.2560 | 0.8391 | -0.2800 |
| checkpoint-19 | yes | 50000 | 0 | 80.3780 | 95.2760 | 0.8372 | -0.2200 |
| checkpoint-20 | yes | 50000 | 0 | 80.4100 | 95.3220 | 0.8372 | -0.1880 |
| checkpoint-21 | yes | 50000 | 0 | 80.3400 | 95.3480 | 0.8406 | -0.2580 |
| checkpoint-22 | yes | 50000 | 0 | 80.4220 | 95.2920 | 0.8409 | -0.1760 |
| checkpoint-23 | yes | 50000 | 0；随后 epoch 23 启用 2e-4 | 80.4660 | 95.3020 | 0.8385 | -0.1320 |
| checkpoint-24 | yes | 50000 | epoch 23 为 2e-4；随后 epoch 24 回到 0 | 80.4460 | 95.3100 | 0.8390 | -0.1520 |
| checkpoint-25 | yes | 50000 | 0；随后 epoch 25 启用 5e-4 | 80.4760 | 95.3160 | 0.8409 | -0.1220 |
| checkpoint-26 | yes | 50000 | epoch 25 为 5e-4；随后 epoch 26 回到 0 | 80.4840 | 95.3280 | 0.8391 | -0.1140 |

原始 full-val 摘要：

```text
checkpoint-11: Test: [distributed-summary]  Time: 35.088s  Loss: 0.8494  Acc@1: 80.3840  Acc@5: 95.3220  Samples: 50000
checkpoint-12: Test: [distributed-summary]  Time: 10.408s  Loss: 0.8444  Acc@1: 80.4540  Acc@5: 95.3280  Samples: 50000
checkpoint-13: Test: [distributed-summary]  Time: 10.399s  Loss: 0.8431  Acc@1: 80.2400  Acc@5: 95.2960  Samples: 50000
checkpoint-14: Test: [distributed-summary]  Time: 10.561s  Loss: 0.8431  Acc@1: 80.4500  Acc@5: 95.2680  Samples: 50000
checkpoint-15: Test: [distributed-summary]  Time: 10.561s  Loss: 0.8399  Acc@1: 80.2780  Acc@5: 95.3400  Samples: 50000
checkpoint-16: Test: [distributed-summary]  Time: 10.321s  Loss: 0.8435  Acc@1: 80.4480  Acc@5: 95.2680  Samples: 50000
checkpoint-17: Test: [distributed-summary]  Time: 10.363s  Loss: 0.8424  Acc@1: 80.4380  Acc@5: 95.3080  Samples: 50000
checkpoint-18: Test: [distributed-summary]  Time: 10.424s  Loss: 0.8391  Acc@1: 80.3180  Acc@5: 95.2560  Samples: 50000
checkpoint-19: Test: [distributed-summary]  Time: 10.418s  Loss: 0.8372  Acc@1: 80.3780  Acc@5: 95.2760  Samples: 50000
checkpoint-20: Test: [distributed-summary]  Time: 10.399s  Loss: 0.8372  Acc@1: 80.4100  Acc@5: 95.3220  Samples: 50000
checkpoint-21: Test: [distributed-summary]  Time: 10.430s  Loss: 0.8406  Acc@1: 80.3400  Acc@5: 95.3480  Samples: 50000
checkpoint-22: Test: [distributed-summary]  Time: 10.418s  Loss: 0.8409  Acc@1: 80.4220  Acc@5: 95.2920  Samples: 50000
checkpoint-23: Test: [distributed-summary]  Time: 10.376s  Loss: 0.8385  Acc@1: 80.4660  Acc@5: 95.3020  Samples: 50000
checkpoint-24: Test: [distributed-summary]  Time: 10.486s  Loss: 0.8390  Acc@1: 80.4460  Acc@5: 95.3100  Samples: 50000
checkpoint-25: Test: [distributed-summary]  Time: 10.419s  Loss: 0.8409  Acc@1: 80.4760  Acc@5: 95.3160  Samples: 50000
checkpoint-26: Test: [distributed-summary]  Time: 10.449s  Loss: 0.8391  Acc@1: 80.4840  Acc@5: 95.3280  Samples: 50000
```

## RefW 激活确认

`checkpoint-23` 的 full-val 来自 epoch 22 结束后的 checkpoint，因此它仍属于 `RefW=0` 的 pre-KL 训练结果。随后进入 epoch 23 后，日志确认 late selective prev-step refKL 已经生效：

```text
Trainable parameter policy: epoch=23, quant_only=False, policy=all, trainable=28608256, frozen=0
Train: 23 [   0/2502 (  0%)] ... RefAttnKL: 2.010e+01 ... RefW: 2.000e-04 ...
Train: 23 [  50/2502 (  2%)] ... RefAttnKL: 3.044e+01 (2.527e+01) ... RefW: 2.000e-04 ...
```

需要注意：代码中 `epoch_float_value(overrides, epoch, default)` 是精确 epoch 查询：

```text
return float(overrides.get(int(epoch), default))
```

因此当前 `23:0.0002,25:0.0005,28:0.0002` 的实际语义是脉冲式生效：只在 epoch 23、25、28 非零；epoch 24 会回到默认 `ref_attn_kl_weight=0.0`。日志已经验证 epoch 24 起始 `RefW=0.000e+00`。这不是训练异常，但说明当前实验不是连续 late KL，而是按指定 epoch 的 sparse pulse KL。

epoch 25 的第二次脉冲也已确认生效：

```text
Trainable parameter policy: epoch=25, quant_only=False, policy=all, trainable=28608256, frozen=0
Train: 25 [   0/2502 (  0%)] ... RefAttnKL: 2.961e+01 ... RefW: 5.000e-04 ...
Train: 25 [  50/2502 (  2%)] ... RefAttnKL: 4.015e+01 (3.488e+01) ... RefW: 5.000e-04 ...
```

epoch 26 已回到默认 `RefW=0.000e+00`：

```text
Trainable parameter policy: epoch=26, quant_only=False, policy=all, trainable=28608256, frozen=0
Train: 26 [   0/2502 (  0%)] ... RefAttnKL: 0.000e+00 ... RefW: 0.000e+00 ...
```

## 停止状态

截至 `2026-07-09 15:21 UTC`，训练没有正常跑到 `checkpoint-30`：

```text
latest checkpoint: checkpoint-26.pth.tar
latest log mtime: 2026-07-09 15:10:56 UTC
normal tail marker: no wall_seconds line
process check: no train.py / qat_launch / ofq_resume10_to30_late_prevstep process found
```

日志末尾停在 epoch 26 前段，未出现 Python traceback / CUDA OOM / NCCL error 等显式报错，也未出现脚本末尾 `wall_seconds=`，因此按“训练异常停止或外部中断”处理；不重启，直接使用已有 checkpoint 完成审计。

## 当前阶段结论

截至 `checkpoint-26`：

1. 训练已保存 `checkpoint-11` 到 `checkpoint-26`，所有可用 checkpoint 均有 full-val `Samples=50000` 证据。
2. epoch 23 已确认 `RefW=2.000e-04`，epoch 25 已确认 `RefW=5.000e-04`，且 `RefAttnKL` 均非零，late selective prev-step refKL 已实际启用。
3. 当前 best 为 `checkpoint-26` Top-1 `80.4840`，尚未超过原版 OFQ baseline best `80.5980`。
4. `checkpoint-24` 是第一份完整经过 epoch 23 refKL 训练后的 full-val checkpoint，Top-1 `80.4460`，没有超过 baseline。
5. epoch 25 的 `5e-4` 脉冲后，`checkpoint-26` 只小幅刷新到 `80.4840`，仍未超过 `80.5980`。
6. 因训练在 epoch 26 前段停止，未观察到 epoch 28 的第三次 `2e-4` 脉冲，也没有 `checkpoint-27..30`。

## 最终审计

按 goal 的异常结束完成判定，本次以所有可用 checkpoint 做结论审计：

1. 完整可用结果范围：`checkpoint-11..checkpoint-26`，共 16 个单 checkpoint full-val。
2. baseline 对比：当前 best `checkpoint-26` Top-1 `80.4840`，低于 OFQ direct resume baseline best `80.5980`，差 `-0.1140`。
3. 81.0 目标对比：当前 best `80.4840`，低于 `81.0`，差 `-0.5160`。
4. 机制有效性判断：late selective prev-step refKL 机制确实启用并产生了非零 KL，但在当前 sparse pulse 调度和已完成 checkpoint 范围内，没有带来超过原版 OFQ direct resume baseline 的收益。
