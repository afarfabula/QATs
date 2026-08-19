# 原版 OFQ 从 checkpoint-10 resume 20 epoch 到 81% 的任务定义验证记录

## 目标

本 goal 验证“从 10 epoch 左右、已经达到约 80% Top-1 的 checkpoint 继续训练 20 epoch，是否能够冲到 `81%+`”这个任务定义是否成立。

此前 clean AOQ-native / no-QKR / no-StatsQ 分支做了大量优化但收益很小。因此本实验不再优化多个新范式，也不从 `checkpoint-300` 或其他 late-stage checkpoint resume，而是回到已验证的 `checkpoint-10` 起点，使用更接近原版 OFQ / public-family OFQ 的有效链路直接跑 20 epoch。

## 误启动废弃记录

曾误启动过一条从 `checkpoint-300.pth.tar` resume 的实验：

```text
script: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_original_resume300_to320_20260709.sh
experiment: ofq_original_resume300_to320_20260709
```

该实验不符合本 goal，因为用户明确要求从 `10epoch、80 acc` 的 checkpoint resume 20 epoch。该进程已用 `Ctrl-C` 停止，`ps` 检查没有 `ofq_original_resume300_to320` / `checkpoint-300` / `qat_launch.py` / `third_party/OFQ/train.py` 残留训练进程。该误启动不作为本 goal 的结果。

## Resume 起点

唯一允许的 resume 起点：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
```

已验证 full ImageNet raw validation：

```text
Strict resume: loaded model from .../recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 33.148s  Loss: 0.8453  Acc@1: 80.3640  Acc@5: 95.3140  Samples: 50000
```

checkpoint metadata：

```text
epoch: 10
arch: swin_t
version: 2
optimizer_present: true
lr_scheduler_present: true
rng_state_present: true
state_dict_tensors: 545
```

checkpoint 内 args 关键口径：

```text
wq_mode: statsq
aq_mode: lsq
qk_reparam: true
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
mixup: 0.0
cutmix: 0.0
aa: rand-m9-mstd0.5-inc1
smoothing: 0.1
batch_size: 64
epoch_checkpoint_interval: 1
```

## 本次单实验设计

本次只跑一条实验，不做多版本搜索。

```text
experiment: ofq_public_resume10_to30_20260709
resume checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709
train log: /mlx_devbox/users/quyanyi/playground/train_ofq_public_resume10_to30_20260709.log
launcher: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_public_resume10_to30_20260709.sh
```

`checkpoint-10` 的 metadata 为 `epoch=10`。`qat_launch.py` 的 strict resume 会返回该 epoch，并从 `range(start_epoch, epochs)` 进入训练。因此本实验设置：

```text
--resume checkpoint-10.pth.tar
--epochs 30
--scheduler-epochs 30
--epoch-checkpoint-interval 1
--checkpoint-hist 30
```

预期训练 epoch 为 `10..29`，对应 20 个 resumed epoch，保存 `checkpoint-11.pth.tar` 到 `checkpoint-30.pth.tar`。

启动命令：

```bash
NO_COLOR=1 TERM=dumb mlx worker login 984521

MASTER_PORT=31531 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_public_resume10_to30_20260709.sh
```

## 当前状态

- 已完成：废弃并停止错误的 `checkpoint-300` 误启动。
- 已完成：确认 `checkpoint-10` 是 80.3640 full-val 起点。
- 已完成：确认 `checkpoint-10` metadata `epoch=10`。
- 已完成：创建正确的 `resume10_to30` 单实验脚本。
- 已完成：第一次正确起点启动尝试，模型权重 strict resume 成功但 optimizer 恢复失败。
- 已完成：使用 `--no-resume-opt` 重新启动 20 epoch 训练，并确认进入 epoch 10。
- 未完成：checkpoint 列表统计。
- 未完成：关键 checkpoint full-val。
- 未完成：最终结论。

## 2026-07-09 启动记录：第一次 checkpoint-10 尝试失败，改用 no-resume-opt

第一次启动命令已经正确使用 `checkpoint-10`：

```text
[QATs] command=... --epochs 30 --scheduler-epochs 30 --batch-size 64 --lr 1.5e-05 --min-lr 5e-06 --resume /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar ... --wq-mode statsq --aq-mode lsq ... --qk_reparam --qk_reparam_type 0 --teacher-soft-temperature 2.75 ...
Strict resume: loaded model from .../recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar; missing=0, unexpected=0
```

失败原因：

```text
ValueError: loaded state dict has a different number of parameter groups
```

解释：

模型权重恢复是正确的，`missing=0, unexpected=0`。失败发生在 optimizer state 恢复阶段，原因是当前训练代码构造出的 optimizer 参数组数量与 `checkpoint-10` 内保存的 optimizer state 不一致。这个问题不否定 `checkpoint-10` 作为权重起点；为避免把 optimizer 参数组差异混入验证，本 goal 改为使用 `--no-resume-opt`，只恢复模型权重，从 epoch 10 到 30 继续训练。

修正后的脚本：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_public_resume10_to30_20260709.sh
```

关键修正：

```text
--resume checkpoint-10.pth.tar --no-resume-opt
```

## 2026-07-09 启动记录：no-resume-opt 后正确进入 epoch 10

修正后重新启动，命令口径确认如下：

```text
[QATs] command=... --epochs 30 --scheduler-epochs 30 --batch-size 64 --lr 1.5e-05 --min-lr 5e-06 --resume /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar --no-resume-opt ... --wq-mode statsq --aq-mode lsq ... --qk_reparam --qk_reparam_type 0 --teacher-soft-temperature 2.75 ... --mixup 0.0 --cutmix 0.0 ...
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Scheduled epochs: 30
Effective batch alignment: per_gpu_effective_batch=64, loader_batch=64, accum=1, world_size=8, global_effective_batch=512
Trainable parameter policy: epoch=10, quant_only=False, policy=all, trainable=28608256, frozen=0
Train: 10 [   0/2502 (  0%)] ...
```

结论：本次正确实验已经从指定 `checkpoint-10` 权重启动，未使用 `checkpoint-300`，并进入 epoch 10 训练。后续等待 `checkpoint-11` 到 `checkpoint-30` 产出。

## 2026-07-09 训练进度：epoch 10 运行中

当前训练仍在第一个 resumed epoch，也就是 epoch 10。关键进度：

```text
Train: 10 [   0/2502 (  0%)] ...
Train: 10 [ 550/2502 ( 22%)] ...
Train: 10 [1100/2502 ( 44%)] ...
Train: 10 [1650/2502 ( 66%)] ...
```

输出目录当前已有：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/args.yaml
```

尚未产出 `checkpoint-11.pth.tar`，需要等待 epoch 10 完成后检查。

## 2026-07-09 训练进度：checkpoint-11 已产出，首个 resumed epoch 小幅上涨

epoch 10 已完成，训练自动做了 full ImageNet raw validation，并进入 epoch 11。

checkpoint 产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/checkpoint-11.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/last.pth.tar
```

checkpoint-11 文件大小：

```text
329M
```

训练摘要：

```text
TrainSummary: epoch=10 updates=2496 avg_step_time=0.223361s samples_per_step=512 samples_per_sec=2292.25
epoch:  10 g['lr']:  1.2034830302000045e-05
```

full-val 结果：

| checkpoint | strict W4A4 | OFQ public-family | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 checkpoint-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | yes | yes | 50000 | no | 80.3640 | 95.3140 | 0.8453 | baseline |
| checkpoint-11 | yes | yes | yes | 50000 | no | 80.3840 | 95.3220 | 0.8494 | +0.0200 |

原始摘要：

```text
Test: [distributed-summary]  Time: 35.047s  Loss: 0.8494  Acc@1: 80.3840  Acc@5: 95.3220  Samples: 50000
```

阶段结论：

1. 本次实验的第一条 checkpoint 产出链路正常，`checkpoint-11` 已保存。
2. 首个 resumed epoch 从 `80.3640` 小幅上涨到 `80.3840`，但距离 `81.0` 仍差 `0.6160`。
3. 训练已继续进入 epoch 11，继续等待后续 `checkpoint-12..checkpoint-30`。

## 2026-07-09 训练进度：checkpoint-12 已产出，第二个 resumed epoch 继续上涨

epoch 11 已完成，训练自动做了 full ImageNet raw validation，并进入 epoch 12。

checkpoint 产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/checkpoint-12.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/last.pth.tar
```

checkpoint metadata：

```text
checkpoint-11 epoch: 11, wq_mode: statsq, qk_reparam: True
checkpoint-12 epoch: 12, wq_mode: statsq, qk_reparam: True
```

训练摘要：

```text
TrainSummary: epoch=11 updates=2496 avg_step_time=0.223257s samples_per_step=512 samples_per_sec=2293.32
epoch:  11 g['lr']:  1.1546279191970826e-05
```

full-val 结果：

| checkpoint | strict W4A4 | OFQ public-family | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 checkpoint-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | yes | yes | 50000 | no | 80.3640 | 95.3140 | 0.8453 | baseline |
| checkpoint-11 | yes | yes | yes | 50000 | no | 80.3840 | 95.3220 | 0.8494 | +0.0200 |
| checkpoint-12 | yes | yes | yes | 50000 | no | 80.4540 | 95.3280 | 0.8444 | +0.0900 |

原始摘要：

```text
Test: [distributed-summary]  Time: 10.155s  Loss: 0.8444  Acc@1: 80.4540  Acc@5: 95.3280  Samples: 50000
```

阶段结论：

1. 第二个 resumed epoch 继续上涨，从 `80.3840` 到 `80.4540`，趋势是正的。
2. 目前仍未达到 `81.0`，距离目标还差 `0.5460`。

## 2026-07-09 训练进度：checkpoint-13 已产出，第三个 resumed epoch 回落

epoch 12 已完成，训练自动做了 full ImageNet raw validation，并进入 epoch 13。

checkpoint 产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/checkpoint-13.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/last.pth.tar
```

训练摘要：

```text
TrainSummary: epoch=12 updates=2496 avg_step_time=0.223248s samples_per_step=512 samples_per_sec=2293.41
epoch:  12 g['lr']:  1.1040786653757094e-05
```

full-val 结果：

| checkpoint | strict W4A4 | OFQ public-family | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 checkpoint-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | yes | yes | 50000 | no | 80.3640 | 95.3140 | 0.8453 | baseline |
| checkpoint-11 | yes | yes | yes | 50000 | no | 80.3840 | 95.3220 | 0.8494 | +0.0200 |
| checkpoint-12 | yes | yes | yes | 50000 | no | 80.4540 | 95.3280 | 0.8444 | +0.0900 |
| checkpoint-13 | yes | yes | yes | 50000 | no | 80.2400 | 95.2960 | 0.8431 | -0.1240 |

原始摘要：

```text
Test: [distributed-summary]  Time: 10.060s  Loss: 0.8431  Acc@1: 80.2400  Acc@5: 95.2960  Samples: 50000
```

阶段结论：

1. 第三个 resumed epoch 出现回落，`checkpoint-13` 从 `checkpoint-12` 的 `80.4540` 降到 `80.2400`。
2. 当前最佳仍是 `checkpoint-12`，Top-1 `80.4540`。
3. 训练进程仍在 worker `984521` 内继续运行，已确认不是另起实验，也没有从 `checkpoint-300` 启动；继续等待 `checkpoint-14..checkpoint-30`。

## 2026-07-09 训练进度：checkpoint-14 已产出，回到 80.45 附近

epoch 13 已完成，训练自动做了 full ImageNet raw validation，并进入 epoch 14。

checkpoint 产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/checkpoint-14.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/last.pth.tar
```

训练摘要：

```text
TrainSummary: epoch=13 updates=2496 avg_step_time=0.223337s samples_per_step=512 samples_per_sec=2292.50
epoch:  13 g['lr']:  1.0523891076445579e-05
```

full-val 结果：

| checkpoint | strict W4A4 | OFQ public-family | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 checkpoint-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | yes | yes | 50000 | no | 80.3640 | 95.3140 | 0.8453 | baseline |
| checkpoint-11 | yes | yes | yes | 50000 | no | 80.3840 | 95.3220 | 0.8494 | +0.0200 |
| checkpoint-12 | yes | yes | yes | 50000 | no | 80.4540 | 95.3280 | 0.8444 | +0.0900 |
| checkpoint-13 | yes | yes | yes | 50000 | no | 80.2400 | 95.2960 | 0.8431 | -0.1240 |
| checkpoint-14 | yes | yes | yes | 50000 | no | 80.4500 | 95.2680 | 0.8431 | +0.0860 |

原始摘要：

```text
Test: [distributed-summary]  Time: 10.091s  Loss: 0.8431  Acc@1: 80.4500  Acc@5: 95.2680  Samples: 50000
```

阶段结论：

1. `checkpoint-14` 回到 `80.45` 附近，但仍略低于当前最佳 `checkpoint-12` 的 `80.4540`。
2. 当前最佳仍是 `checkpoint-12`，Top-1 `80.4540`，距离 `81.0` 还差 `0.5460`。
3. 训练仍在同一条原版 OFQ public-family resume run 中继续执行，继续等待 `checkpoint-15..checkpoint-30`。

## 2026-07-09 训练进度：checkpoint-15 已产出，未超过当前最佳

epoch 14 已完成，训练自动做了 full ImageNet raw validation，并进入 epoch 15。

checkpoint 产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/checkpoint-15.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/last.pth.tar
```

训练摘要：

```text
TrainSummary: epoch=14 updates=2496 avg_step_time=0.222992s samples_per_step=512 samples_per_sec=2296.05
epoch:  14 g['lr']:  1.000125565129565e-05
```

full-val 结果：

| checkpoint | strict W4A4 | OFQ public-family | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 checkpoint-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | yes | yes | 50000 | no | 80.3640 | 95.3140 | 0.8453 | baseline |
| checkpoint-11 | yes | yes | yes | 50000 | no | 80.3840 | 95.3220 | 0.8494 | +0.0200 |
| checkpoint-12 | yes | yes | yes | 50000 | no | 80.4540 | 95.3280 | 0.8444 | +0.0900 |
| checkpoint-13 | yes | yes | yes | 50000 | no | 80.2400 | 95.2960 | 0.8431 | -0.1240 |
| checkpoint-14 | yes | yes | yes | 50000 | no | 80.4500 | 95.2680 | 0.8431 | +0.0860 |
| checkpoint-15 | yes | yes | yes | 50000 | no | 80.2780 | 95.3400 | 0.8399 | -0.0860 |

原始摘要：

```text
Test: [distributed-summary]  Time: 10.085s  Loss: 0.8399  Acc@1: 80.2780  Acc@5: 95.3400  Samples: 50000
```

阶段结论：

1. `checkpoint-15` 的 loss 降到 `0.8399`，但 Top-1 只有 `80.2780`，没有超过当前最佳。
2. 当前最佳仍是 `checkpoint-12`，Top-1 `80.4540`，距离 `81.0` 还差 `0.5460`。
3. 训练仍在同一条原版 OFQ public-family resume run 中继续执行，继续等待 `checkpoint-16..checkpoint-30`。

## 2026-07-09 训练进度：checkpoint-16 已产出，仍未超过当前最佳

epoch 15 已完成，训练自动做了 full ImageNet raw validation，并进入 epoch 16。

checkpoint 产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/checkpoint-16.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/last.pth.tar
```

训练摘要：

```text
TrainSummary: epoch=15 updates=2496 avg_step_time=0.223117s samples_per_step=512 samples_per_sec=2294.76
```

full-val 结果：

| checkpoint | strict W4A4 | OFQ public-family | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 checkpoint-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | yes | yes | 50000 | no | 80.3640 | 95.3140 | 0.8453 | baseline |
| checkpoint-11 | yes | yes | yes | 50000 | no | 80.3840 | 95.3220 | 0.8494 | +0.0200 |
| checkpoint-12 | yes | yes | yes | 50000 | no | 80.4540 | 95.3280 | 0.8444 | +0.0900 |
| checkpoint-13 | yes | yes | yes | 50000 | no | 80.2400 | 95.2960 | 0.8431 | -0.1240 |
| checkpoint-14 | yes | yes | yes | 50000 | no | 80.4500 | 95.2680 | 0.8431 | +0.0860 |
| checkpoint-15 | yes | yes | yes | 50000 | no | 80.2780 | 95.3400 | 0.8399 | -0.0860 |
| checkpoint-16 | yes | yes | yes | 50000 | no | 80.4480 | 95.2680 | 0.8435 | +0.0840 |

原始摘要：

```text
Test: [distributed-summary]  Time: 10.185s  Loss: 0.8435  Acc@1: 80.4480  Acc@5: 95.2680  Samples: 50000
```

阶段结论：

1. `checkpoint-16` 回到 `80.4480`，与 `checkpoint-14` 接近，但仍略低于当前最佳 `checkpoint-12` 的 `80.4540`。
2. 当前最佳仍是 `checkpoint-12`，Top-1 `80.4540`，距离 `81.0` 还差 `0.5460`。
3. 训练仍在同一条原版 OFQ public-family resume run 中继续执行，继续等待 `checkpoint-17..checkpoint-30`。

## 2026-07-09 训练进度：checkpoint-17 已产出，平台期继续

epoch 16 已完成，训练自动做了 full ImageNet raw validation，并进入 epoch 17。

checkpoint 产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/checkpoint-17.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/last.pth.tar
```

训练摘要：

```text
TrainSummary: epoch=16 updates=2496 avg_step_time=0.223039s samples_per_step=512 samples_per_sec=2295.56
```

full-val 结果：

| checkpoint | strict W4A4 | OFQ public-family | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 checkpoint-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | yes | yes | 50000 | no | 80.3640 | 95.3140 | 0.8453 | baseline |
| checkpoint-11 | yes | yes | yes | 50000 | no | 80.3840 | 95.3220 | 0.8494 | +0.0200 |
| checkpoint-12 | yes | yes | yes | 50000 | no | 80.4540 | 95.3280 | 0.8444 | +0.0900 |
| checkpoint-13 | yes | yes | yes | 50000 | no | 80.2400 | 95.2960 | 0.8431 | -0.1240 |
| checkpoint-14 | yes | yes | yes | 50000 | no | 80.4500 | 95.2680 | 0.8431 | +0.0860 |
| checkpoint-15 | yes | yes | yes | 50000 | no | 80.2780 | 95.3400 | 0.8399 | -0.0860 |
| checkpoint-16 | yes | yes | yes | 50000 | no | 80.4480 | 95.2680 | 0.8435 | +0.0840 |
| checkpoint-17 | yes | yes | yes | 50000 | no | 80.4380 | 95.3080 | 0.8424 | +0.0740 |

原始摘要：

```text
Test: [distributed-summary]  Time: 10.195s  Loss: 0.8424  Acc@1: 80.4380  Acc@5: 95.3080  Samples: 50000
```

阶段结论：

1. `checkpoint-17` 为 `80.4380`，仍处于 `80.4` 左右平台期。
2. 当前最佳仍是 `checkpoint-12`，Top-1 `80.4540`，距离 `81.0` 还差 `0.5460`。
3. 训练仍在同一条原版 OFQ public-family resume run 中继续执行，继续等待 `checkpoint-18..checkpoint-30`。

## 2026-07-09 训练进度：checkpoint-18 已产出，仍低于当前最佳

epoch 17 已完成，训练自动做了 full ImageNet raw validation，并进入 epoch 18。

checkpoint 产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/checkpoint-18.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/last.pth.tar
```

训练摘要：

```text
TrainSummary: epoch=17 updates=2496 avg_step_time=0.223107s samples_per_step=512 samples_per_sec=2294.87
```

full-val 结果：

| checkpoint | strict W4A4 | OFQ public-family | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 checkpoint-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | yes | yes | 50000 | no | 80.3640 | 95.3140 | 0.8453 | baseline |
| checkpoint-11 | yes | yes | yes | 50000 | no | 80.3840 | 95.3220 | 0.8494 | +0.0200 |
| checkpoint-12 | yes | yes | yes | 50000 | no | 80.4540 | 95.3280 | 0.8444 | +0.0900 |
| checkpoint-13 | yes | yes | yes | 50000 | no | 80.2400 | 95.2960 | 0.8431 | -0.1240 |
| checkpoint-14 | yes | yes | yes | 50000 | no | 80.4500 | 95.2680 | 0.8431 | +0.0860 |
| checkpoint-15 | yes | yes | yes | 50000 | no | 80.2780 | 95.3400 | 0.8399 | -0.0860 |
| checkpoint-16 | yes | yes | yes | 50000 | no | 80.4480 | 95.2680 | 0.8435 | +0.0840 |
| checkpoint-17 | yes | yes | yes | 50000 | no | 80.4380 | 95.3080 | 0.8424 | +0.0740 |
| checkpoint-18 | yes | yes | yes | 50000 | no | 80.3180 | 95.2560 | 0.8391 | -0.0460 |

原始摘要：

```text
Test: [distributed-summary]  Time: 10.271s  Loss: 0.8391  Acc@1: 80.3180  Acc@5: 95.2560  Samples: 50000
```

阶段结论：

1. `checkpoint-18` 为 `80.3180`，没有超过当前最佳。
2. 当前最佳仍是 `checkpoint-12`，Top-1 `80.4540`，距离 `81.0` 还差 `0.5460`。
3. 截至 `checkpoint-18`，从 `checkpoint-10` 继续 8 个 resumed epoch 后没有出现接近 `81.0` 的趋势；训练仍继续等待 `checkpoint-19..checkpoint-30`。

## 2026-07-09 训练进度：checkpoint-19 / checkpoint-20 已产出，仍未突破平台期

epoch 18 和 epoch 19 已完成，训练分别保存 `checkpoint-19` 和 `checkpoint-20`，并自动做了 full ImageNet raw validation。

checkpoint 产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/checkpoint-19.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/checkpoint-20.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/last.pth.tar
```

训练摘要：

```text
TrainSummary: epoch=18 updates=2496 avg_step_time=0.223001s samples_per_step=512 samples_per_sec=2295.95
TrainSummary: epoch=19 updates=2496 avg_step_time=0.223073s samples_per_step=512 samples_per_sec=2295.22
epoch:  19 g['lr']:  7.501087933778763e-06
```

full-val 结果：

| checkpoint | strict W4A4 | OFQ public-family | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 checkpoint-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | yes | yes | 50000 | no | 80.3640 | 95.3140 | 0.8453 | baseline |
| checkpoint-11 | yes | yes | yes | 50000 | no | 80.3840 | 95.3220 | 0.8494 | +0.0200 |
| checkpoint-12 | yes | yes | yes | 50000 | no | 80.4540 | 95.3280 | 0.8444 | +0.0900 |
| checkpoint-13 | yes | yes | yes | 50000 | no | 80.2400 | 95.2960 | 0.8431 | -0.1240 |
| checkpoint-14 | yes | yes | yes | 50000 | no | 80.4500 | 95.2680 | 0.8431 | +0.0860 |
| checkpoint-15 | yes | yes | yes | 50000 | no | 80.2780 | 95.3400 | 0.8399 | -0.0860 |
| checkpoint-16 | yes | yes | yes | 50000 | no | 80.4480 | 95.2680 | 0.8435 | +0.0840 |
| checkpoint-17 | yes | yes | yes | 50000 | no | 80.4380 | 95.3080 | 0.8424 | +0.0740 |
| checkpoint-18 | yes | yes | yes | 50000 | no | 80.3180 | 95.2560 | 0.8391 | -0.0460 |
| checkpoint-19 | yes | yes | yes | 50000 | no | 80.3780 | 95.2760 | 0.8372 | +0.0140 |
| checkpoint-20 | yes | yes | yes | 50000 | no | 80.4100 | 95.3220 | 0.8372 | +0.0460 |

原始摘要：

```text
Test: [distributed-summary]  Time: 10.150s  Loss: 0.8372  Acc@1: 80.3780  Acc@5: 95.2760  Samples: 50000
Test: [distributed-summary]  Time: 10.187s  Loss: 0.8372  Acc@1: 80.4100  Acc@5: 95.3220  Samples: 50000
```

阶段结论：

1. `checkpoint-19` 和 `checkpoint-20` 的 loss 降到 `0.8372`，但 Top-1 仍没有超过早期最佳。
2. 当前最佳仍是 `checkpoint-12`，Top-1 `80.4540`，距离 `81.0` 还差 `0.5460`。
3. 截至 `checkpoint-20`，从 `checkpoint-10` 继续 10 个 resumed epoch 后没有出现向 `81.0` 抬升的趋势；训练已进入 epoch 20，继续等待 `checkpoint-21..checkpoint-30`。

## 2026-07-09 训练进度：checkpoint-21 已产出，继续低于当前最佳

epoch 20 已完成，训练保存 `checkpoint-21`，并自动做了 full ImageNet raw validation。

checkpoint 产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/checkpoint-21.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/last.pth.tar
```

训练摘要：

```text
TrainSummary: epoch=20 updates=2496 avg_step_time=0.222928s samples_per_step=512 samples_per_sec=2296.70
```

full-val 结果：

| checkpoint | strict W4A4 | OFQ public-family | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 checkpoint-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | yes | yes | 50000 | no | 80.3640 | 95.3140 | 0.8453 | baseline |
| checkpoint-11 | yes | yes | yes | 50000 | no | 80.3840 | 95.3220 | 0.8494 | +0.0200 |
| checkpoint-12 | yes | yes | yes | 50000 | no | 80.4540 | 95.3280 | 0.8444 | +0.0900 |
| checkpoint-13 | yes | yes | yes | 50000 | no | 80.2400 | 95.2960 | 0.8431 | -0.1240 |
| checkpoint-14 | yes | yes | yes | 50000 | no | 80.4500 | 95.2680 | 0.8431 | +0.0860 |
| checkpoint-15 | yes | yes | yes | 50000 | no | 80.2780 | 95.3400 | 0.8399 | -0.0860 |
| checkpoint-16 | yes | yes | yes | 50000 | no | 80.4480 | 95.2680 | 0.8435 | +0.0840 |
| checkpoint-17 | yes | yes | yes | 50000 | no | 80.4380 | 95.3080 | 0.8424 | +0.0740 |
| checkpoint-18 | yes | yes | yes | 50000 | no | 80.3180 | 95.2560 | 0.8391 | -0.0460 |
| checkpoint-19 | yes | yes | yes | 50000 | no | 80.3780 | 95.2760 | 0.8372 | +0.0140 |
| checkpoint-20 | yes | yes | yes | 50000 | no | 80.4100 | 95.3220 | 0.8372 | +0.0460 |
| checkpoint-21 | yes | yes | yes | 50000 | no | 80.3400 | 95.3480 | 0.8406 | -0.0240 |

原始摘要：

```text
Test: [distributed-summary]  Time: 10.163s  Loss: 0.8406  Acc@1: 80.3400  Acc@5: 95.3480  Samples: 50000
```

阶段结论：

1. `checkpoint-21` 为 `80.3400`，仍低于当前最佳 `checkpoint-12`。
2. 当前最佳仍是 `checkpoint-12`，Top-1 `80.4540`，距离 `81.0` 还差 `0.5460`。
3. 截至 `checkpoint-21`，从 `checkpoint-10` 继续 11 个 resumed epoch 后仍没有向 `81.0` 抬升的趋势；训练已进入 epoch 21，继续等待 `checkpoint-22..checkpoint-30`。

## 2026-07-09 训练进度：checkpoint-22 已产出，仍未超过当前最佳

epoch 21 已完成，训练保存 `checkpoint-22`，并自动做了 full ImageNet raw validation。

checkpoint 产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/checkpoint-22.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/last.pth.tar
```

训练摘要：

```text
TrainSummary: epoch=21 updates=2496 avg_step_time=0.223149s samples_per_step=512 samples_per_sec=2294.43
```

full-val 结果：

| checkpoint | strict W4A4 | OFQ public-family | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 checkpoint-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | yes | yes | 50000 | no | 80.3640 | 95.3140 | 0.8453 | baseline |
| checkpoint-11 | yes | yes | yes | 50000 | no | 80.3840 | 95.3220 | 0.8494 | +0.0200 |
| checkpoint-12 | yes | yes | yes | 50000 | no | 80.4540 | 95.3280 | 0.8444 | +0.0900 |
| checkpoint-13 | yes | yes | yes | 50000 | no | 80.2400 | 95.2960 | 0.8431 | -0.1240 |
| checkpoint-14 | yes | yes | yes | 50000 | no | 80.4500 | 95.2680 | 0.8431 | +0.0860 |
| checkpoint-15 | yes | yes | yes | 50000 | no | 80.2780 | 95.3400 | 0.8399 | -0.0860 |
| checkpoint-16 | yes | yes | yes | 50000 | no | 80.4480 | 95.2680 | 0.8435 | +0.0840 |
| checkpoint-17 | yes | yes | yes | 50000 | no | 80.4380 | 95.3080 | 0.8424 | +0.0740 |
| checkpoint-18 | yes | yes | yes | 50000 | no | 80.3180 | 95.2560 | 0.8391 | -0.0460 |
| checkpoint-19 | yes | yes | yes | 50000 | no | 80.3780 | 95.2760 | 0.8372 | +0.0140 |
| checkpoint-20 | yes | yes | yes | 50000 | no | 80.4100 | 95.3220 | 0.8372 | +0.0460 |
| checkpoint-21 | yes | yes | yes | 50000 | no | 80.3400 | 95.3480 | 0.8406 | -0.0240 |
| checkpoint-22 | yes | yes | yes | 50000 | no | 80.4220 | 95.2920 | 0.8409 | +0.0580 |

原始摘要：

```text
Test: [distributed-summary]  Time: 10.255s  Loss: 0.8409  Acc@1: 80.4220  Acc@5: 95.2920  Samples: 50000
```

阶段结论：

1. `checkpoint-22` 为 `80.4220`，仍低于当前最佳 `checkpoint-12`。
2. 当前最佳仍是 `checkpoint-12`，Top-1 `80.4540`，距离 `81.0` 还差 `0.5460`。
3. 截至 `checkpoint-22`，从 `checkpoint-10` 继续 12 个 resumed epoch 后仍没有向 `81.0` 抬升的趋势；训练已进入 epoch 22，继续等待 `checkpoint-23..checkpoint-30`。

## 2026-07-09 训练进度：checkpoint-23 已产出，刷新本 run 当前最佳但仍未接近 81

epoch 22 已完成，训练保存 `checkpoint-23`，并自动做了 full ImageNet raw validation。

checkpoint 产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/checkpoint-23.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/last.pth.tar
```

训练摘要：

```text
TrainSummary: epoch=22 updates=2496 avg_step_time=0.223050s samples_per_step=512 samples_per_sec=2295.45
```

full-val 结果：

| checkpoint | strict W4A4 | OFQ public-family | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 checkpoint-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | yes | yes | 50000 | no | 80.3640 | 95.3140 | 0.8453 | baseline |
| checkpoint-11 | yes | yes | yes | 50000 | no | 80.3840 | 95.3220 | 0.8494 | +0.0200 |
| checkpoint-12 | yes | yes | yes | 50000 | no | 80.4540 | 95.3280 | 0.8444 | +0.0900 |
| checkpoint-13 | yes | yes | yes | 50000 | no | 80.2400 | 95.2960 | 0.8431 | -0.1240 |
| checkpoint-14 | yes | yes | yes | 50000 | no | 80.4500 | 95.2680 | 0.8431 | +0.0860 |
| checkpoint-15 | yes | yes | yes | 50000 | no | 80.2780 | 95.3400 | 0.8399 | -0.0860 |
| checkpoint-16 | yes | yes | yes | 50000 | no | 80.4480 | 95.2680 | 0.8435 | +0.0840 |
| checkpoint-17 | yes | yes | yes | 50000 | no | 80.4380 | 95.3080 | 0.8424 | +0.0740 |
| checkpoint-18 | yes | yes | yes | 50000 | no | 80.3180 | 95.2560 | 0.8391 | -0.0460 |
| checkpoint-19 | yes | yes | yes | 50000 | no | 80.3780 | 95.2760 | 0.8372 | +0.0140 |
| checkpoint-20 | yes | yes | yes | 50000 | no | 80.4100 | 95.3220 | 0.8372 | +0.0460 |
| checkpoint-21 | yes | yes | yes | 50000 | no | 80.3400 | 95.3480 | 0.8406 | -0.0240 |
| checkpoint-22 | yes | yes | yes | 50000 | no | 80.4220 | 95.2920 | 0.8409 | +0.0580 |
| checkpoint-23 | yes | yes | yes | 50000 | no | 80.4660 | 95.3020 | 0.8385 | +0.1020 |

原始摘要：

```text
Test: [distributed-summary]  Time: 10.162s  Loss: 0.8385  Acc@1: 80.4660  Acc@5: 95.3020  Samples: 50000
```

阶段结论：

1. `checkpoint-23` 刷新了本次 `ofq_public_resume10_to30_20260709` 直接 resume run 的当前最佳，从 `checkpoint-12` 的 `80.4540` 提高到 `80.4660`。
2. 这个提升很小，距离 `81.0` 仍差 `0.5340`，还不能说明 20epoch resume 任务定义成立。
3. 截至 `checkpoint-23`，从 `checkpoint-10` 继续 13 个 resumed epoch 后仍处于 `80.4x` 平台；训练已进入 epoch 23，继续等待 `checkpoint-24..checkpoint-30`。

## 2026-07-09 训练进度：checkpoint-24 已产出，继续刷新本 run 当前最佳

epoch 23 已完成，训练保存 `checkpoint-24`，并自动做了 full ImageNet raw validation。

checkpoint 产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/checkpoint-24.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/last.pth.tar
```

训练摘要：

```text
TrainSummary: epoch=23 updates=2496 avg_step_time=0.223103s samples_per_step=512 samples_per_sec=2294.90
```

full-val 结果：

| checkpoint | strict W4A4 | OFQ public-family | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 checkpoint-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | yes | yes | 50000 | no | 80.3640 | 95.3140 | 0.8453 | baseline |
| checkpoint-11 | yes | yes | yes | 50000 | no | 80.3840 | 95.3220 | 0.8494 | +0.0200 |
| checkpoint-12 | yes | yes | yes | 50000 | no | 80.4540 | 95.3280 | 0.8444 | +0.0900 |
| checkpoint-13 | yes | yes | yes | 50000 | no | 80.2400 | 95.2960 | 0.8431 | -0.1240 |
| checkpoint-14 | yes | yes | yes | 50000 | no | 80.4500 | 95.2680 | 0.8431 | +0.0860 |
| checkpoint-15 | yes | yes | yes | 50000 | no | 80.2780 | 95.3400 | 0.8399 | -0.0860 |
| checkpoint-16 | yes | yes | yes | 50000 | no | 80.4480 | 95.2680 | 0.8435 | +0.0840 |
| checkpoint-17 | yes | yes | yes | 50000 | no | 80.4380 | 95.3080 | 0.8424 | +0.0740 |
| checkpoint-18 | yes | yes | yes | 50000 | no | 80.3180 | 95.2560 | 0.8391 | -0.0460 |
| checkpoint-19 | yes | yes | yes | 50000 | no | 80.3780 | 95.2760 | 0.8372 | +0.0140 |
| checkpoint-20 | yes | yes | yes | 50000 | no | 80.4100 | 95.3220 | 0.8372 | +0.0460 |
| checkpoint-21 | yes | yes | yes | 50000 | no | 80.3400 | 95.3480 | 0.8406 | -0.0240 |
| checkpoint-22 | yes | yes | yes | 50000 | no | 80.4220 | 95.2920 | 0.8409 | +0.0580 |
| checkpoint-23 | yes | yes | yes | 50000 | no | 80.4660 | 95.3020 | 0.8385 | +0.1020 |
| checkpoint-24 | yes | yes | yes | 50000 | no | 80.5080 | 95.3220 | 0.8371 | +0.1440 |

原始摘要：

```text
Test: [distributed-summary]  Time: 10.172s  Loss: 0.8371  Acc@1: 80.5080  Acc@5: 95.3220  Samples: 50000
```

阶段结论：

1. `checkpoint-24` 继续刷新本次直接 resume run 的当前最佳，从 `checkpoint-23` 的 `80.4660` 提高到 `80.5080`。
2. 这说明后半段有小幅回升，但距离 `81.0` 仍差 `0.4920`，还不能说明任务定义成立。
3. 截至 `checkpoint-24`，从 `checkpoint-10` 继续 14 个 resumed epoch 后仍未接近 `81.0`；训练已进入 epoch 24，继续等待 `checkpoint-25..checkpoint-30`。

## 2026-07-09 训练进度：checkpoint-25 已产出，继续小幅刷新但仍不到 81

epoch 24 已完成，训练保存 `checkpoint-25`，并自动做了 full ImageNet raw validation。

checkpoint 产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/checkpoint-25.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/last.pth.tar
```

训练摘要：

```text
TrainSummary: epoch=24 updates=2496 avg_step_time=0.223078s samples_per_step=512 samples_per_sec=2295.16
```

full-val 结果：

| checkpoint | strict W4A4 | OFQ public-family | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 checkpoint-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | yes | yes | 50000 | no | 80.3640 | 95.3140 | 0.8453 | baseline |
| checkpoint-11 | yes | yes | yes | 50000 | no | 80.3840 | 95.3220 | 0.8494 | +0.0200 |
| checkpoint-12 | yes | yes | yes | 50000 | no | 80.4540 | 95.3280 | 0.8444 | +0.0900 |
| checkpoint-13 | yes | yes | yes | 50000 | no | 80.2400 | 95.2960 | 0.8431 | -0.1240 |
| checkpoint-14 | yes | yes | yes | 50000 | no | 80.4500 | 95.2680 | 0.8431 | +0.0860 |
| checkpoint-15 | yes | yes | yes | 50000 | no | 80.2780 | 95.3400 | 0.8399 | -0.0860 |
| checkpoint-16 | yes | yes | yes | 50000 | no | 80.4480 | 95.2680 | 0.8435 | +0.0840 |
| checkpoint-17 | yes | yes | yes | 50000 | no | 80.4380 | 95.3080 | 0.8424 | +0.0740 |
| checkpoint-18 | yes | yes | yes | 50000 | no | 80.3180 | 95.2560 | 0.8391 | -0.0460 |
| checkpoint-19 | yes | yes | yes | 50000 | no | 80.3780 | 95.2760 | 0.8372 | +0.0140 |
| checkpoint-20 | yes | yes | yes | 50000 | no | 80.4100 | 95.3220 | 0.8372 | +0.0460 |
| checkpoint-21 | yes | yes | yes | 50000 | no | 80.3400 | 95.3480 | 0.8406 | -0.0240 |
| checkpoint-22 | yes | yes | yes | 50000 | no | 80.4220 | 95.2920 | 0.8409 | +0.0580 |
| checkpoint-23 | yes | yes | yes | 50000 | no | 80.4660 | 95.3020 | 0.8385 | +0.1020 |
| checkpoint-24 | yes | yes | yes | 50000 | no | 80.5080 | 95.3220 | 0.8371 | +0.1440 |
| checkpoint-25 | yes | yes | yes | 50000 | no | 80.5140 | 95.3380 | 0.8399 | +0.1500 |

原始摘要：

```text
Test: [distributed-summary]  Time: 10.135s  Loss: 0.8399  Acc@1: 80.5140  Acc@5: 95.3380  Samples: 50000
```

阶段结论：

1. `checkpoint-25` 继续刷新本次直接 resume run 的当前最佳，从 `checkpoint-24` 的 `80.5080` 提高到 `80.5140`。
2. 后半段确实有很小的回升趋势，但距离 `81.0` 仍差 `0.4860`，还不能说明任务定义成立。
3. 截至 `checkpoint-25`，从 `checkpoint-10` 继续 15 个 resumed epoch 后仍未接近 `81.0`；训练已进入 epoch 25，继续等待 `checkpoint-26..checkpoint-30`。

## 2026-07-09 训练进度：checkpoint-26 已产出，较当前最佳回落

epoch 25 已完成，训练保存 `checkpoint-26`，并自动做了 full ImageNet raw validation。

checkpoint 产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/checkpoint-26.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/last.pth.tar
```

训练摘要：

```text
TrainSummary: epoch=25 updates=2496 avg_step_time=0.222962s samples_per_step=512 samples_per_sec=2296.35
```

full-val 结果：

| checkpoint | strict W4A4 | OFQ public-family | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 checkpoint-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | yes | yes | 50000 | no | 80.3640 | 95.3140 | 0.8453 | baseline |
| checkpoint-11 | yes | yes | yes | 50000 | no | 80.3840 | 95.3220 | 0.8494 | +0.0200 |
| checkpoint-12 | yes | yes | yes | 50000 | no | 80.4540 | 95.3280 | 0.8444 | +0.0900 |
| checkpoint-13 | yes | yes | yes | 50000 | no | 80.2400 | 95.2960 | 0.8431 | -0.1240 |
| checkpoint-14 | yes | yes | yes | 50000 | no | 80.4500 | 95.2680 | 0.8431 | +0.0860 |
| checkpoint-15 | yes | yes | yes | 50000 | no | 80.2780 | 95.3400 | 0.8399 | -0.0860 |
| checkpoint-16 | yes | yes | yes | 50000 | no | 80.4480 | 95.2680 | 0.8435 | +0.0840 |
| checkpoint-17 | yes | yes | yes | 50000 | no | 80.4380 | 95.3080 | 0.8424 | +0.0740 |
| checkpoint-18 | yes | yes | yes | 50000 | no | 80.3180 | 95.2560 | 0.8391 | -0.0460 |
| checkpoint-19 | yes | yes | yes | 50000 | no | 80.3780 | 95.2760 | 0.8372 | +0.0140 |
| checkpoint-20 | yes | yes | yes | 50000 | no | 80.4100 | 95.3220 | 0.8372 | +0.0460 |
| checkpoint-21 | yes | yes | yes | 50000 | no | 80.3400 | 95.3480 | 0.8406 | -0.0240 |
| checkpoint-22 | yes | yes | yes | 50000 | no | 80.4220 | 95.2920 | 0.8409 | +0.0580 |
| checkpoint-23 | yes | yes | yes | 50000 | no | 80.4660 | 95.3020 | 0.8385 | +0.1020 |
| checkpoint-24 | yes | yes | yes | 50000 | no | 80.5080 | 95.3220 | 0.8371 | +0.1440 |
| checkpoint-25 | yes | yes | yes | 50000 | no | 80.5140 | 95.3380 | 0.8399 | +0.1500 |
| checkpoint-26 | yes | yes | yes | 50000 | no | 80.4740 | 95.2480 | 0.8379 | +0.1100 |

原始摘要：

```text
Test: [distributed-summary]  Time: 10.194s  Loss: 0.8379  Acc@1: 80.4740  Acc@5: 95.2480  Samples: 50000
```

阶段结论：

1. `checkpoint-26` 从 `checkpoint-25` 的 `80.5140` 回落到 `80.4740`。
2. 当前最佳仍是 `checkpoint-25`，Top-1 `80.5140`，距离 `81.0` 还差 `0.4860`。
3. 截至 `checkpoint-26`，从 `checkpoint-10` 继续 16 个 resumed epoch 后仍未接近 `81.0`；训练已进入 epoch 26，继续等待 `checkpoint-27..checkpoint-30`。

## 2026-07-09 训练进度：checkpoint-27 已产出，显著刷新本 run 当前最佳但仍未到 81

epoch 26 已完成，训练保存 `checkpoint-27`，并自动做了 full ImageNet raw validation。

checkpoint 产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/checkpoint-27.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/last.pth.tar
```

训练摘要：

```text
TrainSummary: epoch=26 updates=2496 avg_step_time=0.223140s samples_per_step=512 samples_per_sec=2294.52
```

full-val 结果：

| checkpoint | strict W4A4 | OFQ public-family | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 checkpoint-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | yes | yes | 50000 | no | 80.3640 | 95.3140 | 0.8453 | baseline |
| checkpoint-11 | yes | yes | yes | 50000 | no | 80.3840 | 95.3220 | 0.8494 | +0.0200 |
| checkpoint-12 | yes | yes | yes | 50000 | no | 80.4540 | 95.3280 | 0.8444 | +0.0900 |
| checkpoint-13 | yes | yes | yes | 50000 | no | 80.2400 | 95.2960 | 0.8431 | -0.1240 |
| checkpoint-14 | yes | yes | yes | 50000 | no | 80.4500 | 95.2680 | 0.8431 | +0.0860 |
| checkpoint-15 | yes | yes | yes | 50000 | no | 80.2780 | 95.3400 | 0.8399 | -0.0860 |
| checkpoint-16 | yes | yes | yes | 50000 | no | 80.4480 | 95.2680 | 0.8435 | +0.0840 |
| checkpoint-17 | yes | yes | yes | 50000 | no | 80.4380 | 95.3080 | 0.8424 | +0.0740 |
| checkpoint-18 | yes | yes | yes | 50000 | no | 80.3180 | 95.2560 | 0.8391 | -0.0460 |
| checkpoint-19 | yes | yes | yes | 50000 | no | 80.3780 | 95.2760 | 0.8372 | +0.0140 |
| checkpoint-20 | yes | yes | yes | 50000 | no | 80.4100 | 95.3220 | 0.8372 | +0.0460 |
| checkpoint-21 | yes | yes | yes | 50000 | no | 80.3400 | 95.3480 | 0.8406 | -0.0240 |
| checkpoint-22 | yes | yes | yes | 50000 | no | 80.4220 | 95.2920 | 0.8409 | +0.0580 |
| checkpoint-23 | yes | yes | yes | 50000 | no | 80.4660 | 95.3020 | 0.8385 | +0.1020 |
| checkpoint-24 | yes | yes | yes | 50000 | no | 80.5080 | 95.3220 | 0.8371 | +0.1440 |
| checkpoint-25 | yes | yes | yes | 50000 | no | 80.5140 | 95.3380 | 0.8399 | +0.1500 |
| checkpoint-26 | yes | yes | yes | 50000 | no | 80.4740 | 95.2480 | 0.8379 | +0.1100 |
| checkpoint-27 | yes | yes | yes | 50000 | no | 80.5980 | 95.3560 | 0.8404 | +0.2340 |

原始摘要：

```text
Test: [distributed-summary]  Time: 10.142s  Loss: 0.8404  Acc@1: 80.5980  Acc@5: 95.3560  Samples: 50000
```

阶段结论：

1. `checkpoint-27` 显著刷新本次直接 resume run 的当前最佳，从 `checkpoint-25` 的 `80.5140` 提高到 `80.5980`。
2. 这个结果已经超过此前 3epoch 短更新 gate 的 `80.5540`，说明原版 OFQ 直接 resume 后段确实还能继续小幅爬升。
3. 但它距离 `81.0` 仍差 `0.4020`，尚未达到本 goal 成功标准；训练已进入 epoch 27，继续等待 `checkpoint-28..checkpoint-30`。

## 2026-07-09 训练进度：checkpoint-28 已产出，Top-1 回落但 loss 降低

epoch 27 已完成，训练保存 `checkpoint-28`，并自动做了 full ImageNet raw validation。

checkpoint 产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/checkpoint-28.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/last.pth.tar
```

训练摘要：

```text
TrainSummary: epoch=27 updates=2496 avg_step_time=0.223081s samples_per_step=512 samples_per_sec=2295.13
```

full-val 结果：

| checkpoint | strict W4A4 | OFQ public-family | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 checkpoint-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | yes | yes | 50000 | no | 80.3640 | 95.3140 | 0.8453 | baseline |
| checkpoint-11 | yes | yes | yes | 50000 | no | 80.3840 | 95.3220 | 0.8494 | +0.0200 |
| checkpoint-12 | yes | yes | yes | 50000 | no | 80.4540 | 95.3280 | 0.8444 | +0.0900 |
| checkpoint-13 | yes | yes | yes | 50000 | no | 80.2400 | 95.2960 | 0.8431 | -0.1240 |
| checkpoint-14 | yes | yes | yes | 50000 | no | 80.4500 | 95.2680 | 0.8431 | +0.0860 |
| checkpoint-15 | yes | yes | yes | 50000 | no | 80.2780 | 95.3400 | 0.8399 | -0.0860 |
| checkpoint-16 | yes | yes | yes | 50000 | no | 80.4480 | 95.2680 | 0.8435 | +0.0840 |
| checkpoint-17 | yes | yes | yes | 50000 | no | 80.4380 | 95.3080 | 0.8424 | +0.0740 |
| checkpoint-18 | yes | yes | yes | 50000 | no | 80.3180 | 95.2560 | 0.8391 | -0.0460 |
| checkpoint-19 | yes | yes | yes | 50000 | no | 80.3780 | 95.2760 | 0.8372 | +0.0140 |
| checkpoint-20 | yes | yes | yes | 50000 | no | 80.4100 | 95.3220 | 0.8372 | +0.0460 |
| checkpoint-21 | yes | yes | yes | 50000 | no | 80.3400 | 95.3480 | 0.8406 | -0.0240 |
| checkpoint-22 | yes | yes | yes | 50000 | no | 80.4220 | 95.2920 | 0.8409 | +0.0580 |
| checkpoint-23 | yes | yes | yes | 50000 | no | 80.4660 | 95.3020 | 0.8385 | +0.1020 |
| checkpoint-24 | yes | yes | yes | 50000 | no | 80.5080 | 95.3220 | 0.8371 | +0.1440 |
| checkpoint-25 | yes | yes | yes | 50000 | no | 80.5140 | 95.3380 | 0.8399 | +0.1500 |
| checkpoint-26 | yes | yes | yes | 50000 | no | 80.4740 | 95.2480 | 0.8379 | +0.1100 |
| checkpoint-27 | yes | yes | yes | 50000 | no | 80.5980 | 95.3560 | 0.8404 | +0.2340 |
| checkpoint-28 | yes | yes | yes | 50000 | no | 80.5380 | 95.3080 | 0.8342 | +0.1740 |

原始摘要：

```text
Test: [distributed-summary]  Time: 10.191s  Loss: 0.8342  Acc@1: 80.5380  Acc@5: 95.3080  Samples: 50000
```

阶段结论：

1. `checkpoint-28` 的 Top-1 从 `checkpoint-27` 的 `80.5980` 回落到 `80.5380`，未刷新 Top-1 最佳。
2. `checkpoint-28` 的 loss 降到 `0.8342`，是本 run 当前最低 loss，但本 goal 成功标准以 Top-1 >= `81.0` 为准。
3. 当前最佳仍是 `checkpoint-27`，Top-1 `80.5980`，距离 `81.0` 还差 `0.4020`；训练已进入 epoch 28，继续等待 `checkpoint-29..checkpoint-30`。

## 2026-07-09 训练进度：checkpoint-29 已产出，明显回落

epoch 28 已完成，训练保存 `checkpoint-29`，并自动做了 full ImageNet raw validation。

checkpoint 产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/checkpoint-29.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/last.pth.tar
```

训练摘要：

```text
TrainSummary: epoch=28 updates=2496 avg_step_time=0.222981s samples_per_step=512 samples_per_sec=2296.16
```

full-val 结果：

| checkpoint | strict W4A4 | OFQ public-family | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 checkpoint-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | yes | yes | 50000 | no | 80.3640 | 95.3140 | 0.8453 | baseline |
| checkpoint-11 | yes | yes | yes | 50000 | no | 80.3840 | 95.3220 | 0.8494 | +0.0200 |
| checkpoint-12 | yes | yes | yes | 50000 | no | 80.4540 | 95.3280 | 0.8444 | +0.0900 |
| checkpoint-13 | yes | yes | yes | 50000 | no | 80.2400 | 95.2960 | 0.8431 | -0.1240 |
| checkpoint-14 | yes | yes | yes | 50000 | no | 80.4500 | 95.2680 | 0.8431 | +0.0860 |
| checkpoint-15 | yes | yes | yes | 50000 | no | 80.2780 | 95.3400 | 0.8399 | -0.0860 |
| checkpoint-16 | yes | yes | yes | 50000 | no | 80.4480 | 95.2680 | 0.8435 | +0.0840 |
| checkpoint-17 | yes | yes | yes | 50000 | no | 80.4380 | 95.3080 | 0.8424 | +0.0740 |
| checkpoint-18 | yes | yes | yes | 50000 | no | 80.3180 | 95.2560 | 0.8391 | -0.0460 |
| checkpoint-19 | yes | yes | yes | 50000 | no | 80.3780 | 95.2760 | 0.8372 | +0.0140 |
| checkpoint-20 | yes | yes | yes | 50000 | no | 80.4100 | 95.3220 | 0.8372 | +0.0460 |
| checkpoint-21 | yes | yes | yes | 50000 | no | 80.3400 | 95.3480 | 0.8406 | -0.0240 |
| checkpoint-22 | yes | yes | yes | 50000 | no | 80.4220 | 95.2920 | 0.8409 | +0.0580 |
| checkpoint-23 | yes | yes | yes | 50000 | no | 80.4660 | 95.3020 | 0.8385 | +0.1020 |
| checkpoint-24 | yes | yes | yes | 50000 | no | 80.5080 | 95.3220 | 0.8371 | +0.1440 |
| checkpoint-25 | yes | yes | yes | 50000 | no | 80.5140 | 95.3380 | 0.8399 | +0.1500 |
| checkpoint-26 | yes | yes | yes | 50000 | no | 80.4740 | 95.2480 | 0.8379 | +0.1100 |
| checkpoint-27 | yes | yes | yes | 50000 | no | 80.5980 | 95.3560 | 0.8404 | +0.2340 |
| checkpoint-28 | yes | yes | yes | 50000 | no | 80.5380 | 95.3080 | 0.8342 | +0.1740 |
| checkpoint-29 | yes | yes | yes | 50000 | no | 80.3940 | 95.3540 | 0.8421 | +0.0300 |

原始摘要：

```text
Test: [distributed-summary]  Time: 10.179s  Loss: 0.8421  Acc@1: 80.3940  Acc@5: 95.3540  Samples: 50000
```

阶段结论：

1. `checkpoint-29` 从 `checkpoint-28` 的 `80.5380` 回落到 `80.3940`，明显低于当前最佳。
2. 当前最佳仍是 `checkpoint-27`，Top-1 `80.5980`，距离 `81.0` 还差 `0.4020`。
3. 训练已进入 epoch 29，继续等待最后一个目标 checkpoint：`checkpoint-30`。

## 2026-07-09 训练完成：checkpoint-30 已产出，20 epoch resume 未达到 81

epoch 29 已完成，训练保存 `checkpoint-30`，并自动做了 full ImageNet raw validation。本次 `checkpoint-10 -> checkpoint-30` 直接 resume 实验已经跑完。

checkpoint 产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/checkpoint-30.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_public_resume10_to30_20260709/last.pth.tar
```

训练摘要：

```text
TrainSummary: epoch=29 updates=2496 avg_step_time=0.223056s samples_per_step=512 samples_per_sec=2295.39
wall_seconds=11531
```

checkpoint-30 full-val 结果：

```text
Test: [distributed-summary]  Time: 10.208s  Loss: 0.8385  Acc@1: 80.4200  Acc@5: 95.3300  Samples: 50000
```

最终 full-val 汇总：

| checkpoint | strict W4A4 | OFQ public-family | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 checkpoint-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | yes | yes | 50000 | no | 80.3640 | 95.3140 | 0.8453 | baseline |
| checkpoint-11 | yes | yes | yes | 50000 | no | 80.3840 | 95.3220 | 0.8494 | +0.0200 |
| checkpoint-12 | yes | yes | yes | 50000 | no | 80.4540 | 95.3280 | 0.8444 | +0.0900 |
| checkpoint-13 | yes | yes | yes | 50000 | no | 80.2400 | 95.2960 | 0.8431 | -0.1240 |
| checkpoint-14 | yes | yes | yes | 50000 | no | 80.4500 | 95.2680 | 0.8431 | +0.0860 |
| checkpoint-15 | yes | yes | yes | 50000 | no | 80.2780 | 95.3400 | 0.8399 | -0.0860 |
| checkpoint-16 | yes | yes | yes | 50000 | no | 80.4480 | 95.2680 | 0.8435 | +0.0840 |
| checkpoint-17 | yes | yes | yes | 50000 | no | 80.4380 | 95.3080 | 0.8424 | +0.0740 |
| checkpoint-18 | yes | yes | yes | 50000 | no | 80.3180 | 95.2560 | 0.8391 | -0.0460 |
| checkpoint-19 | yes | yes | yes | 50000 | no | 80.3780 | 95.2760 | 0.8372 | +0.0140 |
| checkpoint-20 | yes | yes | yes | 50000 | no | 80.4100 | 95.3220 | 0.8372 | +0.0460 |
| checkpoint-21 | yes | yes | yes | 50000 | no | 80.3400 | 95.3480 | 0.8406 | -0.0240 |
| checkpoint-22 | yes | yes | yes | 50000 | no | 80.4220 | 95.2920 | 0.8409 | +0.0580 |
| checkpoint-23 | yes | yes | yes | 50000 | no | 80.4660 | 95.3020 | 0.8385 | +0.1020 |
| checkpoint-24 | yes | yes | yes | 50000 | no | 80.5080 | 95.3220 | 0.8371 | +0.1440 |
| checkpoint-25 | yes | yes | yes | 50000 | no | 80.5140 | 95.3380 | 0.8399 | +0.1500 |
| checkpoint-26 | yes | yes | yes | 50000 | no | 80.4740 | 95.2480 | 0.8379 | +0.1100 |
| checkpoint-27 | yes | yes | yes | 50000 | no | 80.5980 | 95.3560 | 0.8404 | +0.2340 |
| checkpoint-28 | yes | yes | yes | 50000 | no | 80.5380 | 95.3080 | 0.8342 | +0.1740 |
| checkpoint-29 | yes | yes | yes | 50000 | no | 80.3940 | 95.3540 | 0.8421 | +0.0300 |
| checkpoint-30 | yes | yes | yes | 50000 | no | 80.4200 | 95.3300 | 0.8385 | +0.0560 |

最终 best：

```text
Best Top-1: checkpoint-27, Top-1 80.5980, Top-5 95.3560, Loss 0.8404, Samples 50000
Best Loss: checkpoint-28, Loss 0.8342, Top-1 80.5380, Top-5 95.3080, Samples 50000
```

## completion audit

| 要求 | 证据 | 状态 |
|---|---|---|
| 使用 OFQ 有效链路 | `args.yaml` 和 train log 确认 `wq_mode: statsq`、`aq_mode: lsq`、`qk_reparam: true`、teacher KD、Swin-T W4A4 配置 | pass |
| 从指定 checkpoint-10 启动 | train log: `Strict resume: loaded model from .../recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar; missing=0, unexpected=0` | pass |
| 不从 checkpoint-300 启动 | 错误 checkpoint-300 实验已停止并记录为废弃；本 run 的 `args.yaml` 和 command 只指向 checkpoint-10 | pass |
| 直接跑 20 resumed epoch | train log 完成 epoch `10..29`，对应保存 `checkpoint-11..checkpoint-30` | pass |
| 每 epoch checkpoint | 输出目录存在 20 个 checkpoint：`checkpoint-11.pth.tar` 到 `checkpoint-30.pth.tar` | pass |
| full raw val | train log 有 20 条 `Test: [distributed-summary] ... Samples: 50000`，每个 checkpoint 对应一次 full ImageNet raw validation | pass |
| 单 checkpoint | 每条结果来自当 epoch 保存的单 checkpoint；没有 soup、checkpoint averaging、multi-checkpoint averaging 或 ensemble | pass |
| 成功标准 Top-1 >= 81.0 | 最佳单 checkpoint 是 `checkpoint-27`，Top-1 `80.5980`，低于 `81.0` | fail |

## 最终结论

本 goal 已完整验证完毕：用原版 OFQ / public-family OFQ 链路，从指定 `checkpoint-10` 的 `80.3640` Top-1 起点直接 resume 20 个 epoch 到 `checkpoint-30`，没有任何单 checkpoint 达到 `81.0`。

最佳结果为：

```text
checkpoint-27: Top-1 80.5980, Top-5 95.3560, Loss 0.8404, Samples 50000
```

因此，在当前口径下，“从 10epoch 80% checkpoint 直接 resume 20epoch 到 81%”这个任务定义没有被验证成立。它比早期 checkpoint 有小幅提升，也超过了此前 3epoch 短更新 gate 的 `80.5540`，但完整 20epoch 仍停在 `80.6` 以下。下一步应重新审视历史 `81%+` 结果是否属于同一 resume 起点、同一训练链路、同一评估方式、同一 checkpoint 状态，而不是继续把问题简单定义为“从这个 10epoch checkpoint 直接 resume 到 81”。
