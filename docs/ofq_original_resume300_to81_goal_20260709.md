# 原版 OFQ resume-to-81 任务定义验证记录

## 目标

本 goal 用原版 OFQ 链路验证“从已有 checkpoint resume 20 个 epoch 冲到 `81%+` Top-1”的任务定义是否成立。此前 clean AOQ-native / no-QKR / no-StatsQ 分支做了大量优化，但当前 best 只有 `80.2340`。因此本实验不再继续多版本优化，而是回到历史上已经验证过能达到 `81%+` 的 OFQ late-stage resume / finetune 链路，直接跑一条 20 epoch resume。

## 成功标准

必须同时满足：

1. 使用原版 OFQ 口径：`wq_mode=statsq`、`aq_mode=lsq`、`qk_reparam=true`、原始 OFQ config、原始 augmentation/KD 口径。
2. 从已有 checkpoint resume，不从头训练。
3. 直接训练 20 个 resumed epoch，不做多版本超参搜索。
4. 每个 epoch 保存 checkpoint。
5. 至少关键 checkpoint 做 full ImageNet raw validation，结果必须 `Samples=50000`。
6. 最终结论只认单 checkpoint，不使用 soup、checkpoint averaging、multi-checkpoint averaging 或 ensemble。
7. 命令、配置、checkpoint、full-val 结果和结论用中文记录。

## 历史证据与口径确认

本实验要验证的是 late-stage resume / finetune 任务，而不是从头 100 epoch 训练。历史 memory 记录的关键事实：

- OFQ resume regression 修复后，历史 late-stage 口径有 confirmed baseline `Top-1=81.38` 和 old high-point `Top-1=81.54`。
- 该类 `81%+` 结果属于 OFQ late-stage resume / finetune 语境，不应与 100 epoch from-scratch `78.4760` 直接混比。
- 当前稳定主线 checkpoint root：

```text
/mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_mainline_300ep_20260613/swin_t_w4a4_imagenet1k_8gpu_300ep_mainline
```

当前实物检查：

```text
checkpoint: /mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_mainline_300ep_20260613/swin_t_w4a4_imagenet1k_8gpu_300ep_mainline/checkpoint-300.pth.tar
size: 329M
checkpoint epoch metadata: 300
state_dict_tensors: 521
optimizer: present
```

原版 OFQ `args.yaml` 关键口径：

```text
aq_bitw: 4
aq_mode: lsq
wq_bitw: 4
wq_mode: statsq
qk_reparam: true
qk_reparam_type: 0
kd_hard_and_soft: 1
aa: rand-m9-mstd0.5-inc1
mixup: 0.8
cutmix: 1.0
reprob: 0.25
smoothing: 0.1
batch_size: 64
epoch_checkpoint_interval: 1
```

## 本次单实验设计

本次只跑一条实验，不做多版本搜索。

```text
experiment: ofq_original_resume300_to320_20260709
resume checkpoint: /mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_mainline_300ep_20260613/swin_t_w4a4_imagenet1k_8gpu_300ep_mainline/checkpoint-300.pth.tar
output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_original_resume300_to320_20260709
train log: /mlx_devbox/users/quyanyi/playground/train_ofq_original_resume300_to320_20260709.log
launcher: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_original_resume300_to320_20260709.sh
```

`checkpoint-300` 的 metadata `epoch=300`。`qat_launch.py` 的 strict resume 会返回该 epoch，并从 `range(start_epoch, epochs)` 进入训练。因此本实验设置：

```text
--resume checkpoint-300.pth.tar
--epochs 320
--scheduler-epochs 320
--epoch-checkpoint-interval 1
```

预期训练 epoch 为 `300..319`，对应 20 个 resumed epoch，保存 `checkpoint-301.pth.tar` 到 `checkpoint-320.pth.tar`。

启动命令：

```bash
NO_COLOR=1 TERM=dumb mlx worker login 984521

MASTER_PORT=31521 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_original_resume300_to320_20260709.sh
```

## 当前状态

截至本文创建时：

- 已完成：历史口径确认。
- 已完成：resume checkpoint 实物确认。
- 已完成：单实验启动脚本创建。
- 未完成：20 epoch 训练。
- 未完成：checkpoint 列表统计。
- 未完成：full-val 关键 checkpoint 评估。
- 未完成：最终结论。

## completion audit 模板

完成前必须逐项检查：

| 要求 | 证据 | 状态 |
|---|---|---|
| 原版 OFQ 口径 | args / log 中 `wq_mode=statsq`、`qk_reparam=true`、原始 aug/KD | pending |
| 从 checkpoint resume | log 中 strict resume from `checkpoint-300.pth.tar`，missing/unexpected 计数 | pending |
| 20 个 resumed epoch | log 中 epoch 300..319 完成，或中断时说明已保存 ckpt | pending |
| 每 epoch checkpoint | 输出目录存在 `checkpoint-301` 到 `checkpoint-320`，或中断时列出已有 ckpt | pending |
| full raw val | eval log `Test: [distributed-summary] ... Samples: 50000` | pending |
| 单 checkpoint | eval 命令只传单个 CKPT，无 soup/avg/ensemble | pending |
| 结论 | 达到或未达到 81，并判断任务定义是否成立 | pending |

