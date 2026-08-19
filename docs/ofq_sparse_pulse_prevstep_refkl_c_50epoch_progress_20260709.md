# OFQ sparse pulse prev-step refKL 方案 C 50epoch 长跑记录

## 目标

从已验证 `checkpoint-10` 起点继续训练 50 个 resumed epoch，保持 OFQ public-family 主链路，并加入方案 C：低频强脉冲 sparse prev-step refmodel attention KL。

目标：

```text
超过 baseline: 任意单 checkpoint Top-1 > 80.5980
达到最终目标: 任意单 checkpoint Top-1 >= 81.0
```

baseline：

```text
原版 OFQ direct resume best: checkpoint-27
Top-1: 80.5980
Top-5: 95.3560
Loss: 0.8404
Samples: 50000
```

## 实验信息

```text
script: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709.sh
experiment: ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709
log: /mlx_devbox/users/quyanyi/playground/train_ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709.log
output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709
resume checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
```

## 关键配置

OFQ public-family 主链路：

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
epochs: 60
scheduler_epochs: 60
batch_size: 64
epoch_checkpoint_interval: 1
checkpoint_hist: 60
```

方案 C sparse pulse prev-step refKL：

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

## 启动前状态

当前普通 shell 检查：

```text
/dev/nvidia0: not visible
```

说明：普通 shell 不可作为 NCCL/GPU 训练证据。训练需要在真实 `mlx worker login` 后的 GPU worker TTY 中启动。

## 2026-07-09 启动阻塞记录

本地准备已经完成：

```text
script exists: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709.sh
progress doc exists: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_sparse_pulse_prevstep_refkl_c_50epoch_progress_20260709.md
```

尝试进入历史 worker：

```text
NO_COLOR=1 TERM=dumb mlx worker login 984521
get worker ip port failed: failed to get worker: Not Found: "worker mlxlaboys4hkz16a081709-20260516070441-pit7oq-rp0a2w not found"
```

当前 worker 状态：

```text
mlx worker quota: 可查询，Public Workspace 有 H100-SXM-80GB 资源。
mlx worker list: 无现存 worker 记录。
current shell: no /dev/nvidia0
```

结论：

```text
训练尚未启动。
阻塞原因：没有可登录的现存 GPU worker；旧 worker 984521 已失效。
下一步：需要新的有效 worker ID / worker TTY，或用户明确允许启动新 worker。
```

拿到真实 GPU worker 后可直接执行：

```bash
cd /mlx_devbox/users/quyanyi/playground
MASTER_PORT=31561 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709.sh
```

## 结果表

| checkpoint | single checkpoint | raw val samples | RefW / pulse 状态 | Top-1 | Top-5 | Loss | 对比 80.5980 |
|---|---:|---:|---|---:|---:|---:|---:|
| checkpoint-10 起点 | yes | 50000 | 0 | 80.3640 | 95.3140 | 0.8453 | -0.2340 |
| checkpoint-11 | yes | 50000 | 0 | 80.4000 | 95.2460 | 0.8443 | -0.1980 |
| checkpoint-12 | yes | 50000 | 0 | 80.3740 | 95.2580 | 0.8452 | -0.2240 |
| checkpoint-13 | yes | 50000 | 0 | 80.4600 | 95.3180 | 0.8415 | -0.1380 |
| checkpoint-14 | yes | 50000 | 0 | 80.4020 | 95.3220 | 0.8441 | -0.1960 |
| checkpoint-15 | yes | 50000 | 0 | 80.4260 | 95.3120 | 0.8399 | -0.1720 |
| checkpoint-16 | yes | 50000 | 0 | 80.3260 | 95.3260 | 0.8426 | -0.2720 |
| checkpoint-17 | yes | 50000 | 0 | 80.5060 | 95.2760 | 0.8439 | -0.0920 |
| checkpoint-18 | yes | 50000 | 0 | 80.3600 | 95.3200 | 0.8415 | -0.2380 |
| checkpoint-19 | yes | 50000 | 0 | 80.4860 | 95.3280 | 0.8399 | -0.1120 |
| checkpoint-20 | yes | 50000 | 0 | 80.3460 | 95.3960 | 0.8364 | -0.2520 |
| checkpoint-21 | yes | 50000 | 0 | 80.3560 | 95.3240 | 0.8433 | -0.2420 |
| checkpoint-22 | yes | 50000 | 0 | 80.5140 | 95.3100 | 0.8420 | -0.0840 |
| checkpoint-23 | yes | 50000 | 0 | 80.4120 | 95.3520 | 0.8371 | -0.1860 |
| checkpoint-24 | yes | 50000 | 0 | 80.4200 | 95.3000 | 0.8359 | -0.1780 |
| checkpoint-25 | yes | 50000 | 0 | 80.4220 | 95.3460 | 0.8409 | -0.1760 |
| checkpoint-26 | yes | 50000 | 0 | 80.5160 | 95.3000 | 0.8446 | -0.0820 |
| checkpoint-27 | yes | 50000 | 0 | 80.5340 | 95.3280 | 0.8411 | -0.0640 |
| checkpoint-28 | yes | 50000 | pulse started after eval | 80.4160 | 95.3320 | 0.8351 | -0.1820 |
| checkpoint-29 | yes | 50000 | 3.000e-04 during epoch 28 | 80.4540 | 95.3400 | 0.8378 | -0.1440 |

## 原始 full-val 摘要

```text
checkpoint-10: Loss 0.8453  Acc@1 80.3640  Acc@5 95.3140  Samples 50000
checkpoint-11: Loss 0.8443  Acc@1 80.4000  Acc@5 95.2460  Samples 50000
checkpoint-12: Loss 0.8452  Acc@1 80.3740  Acc@5 95.2580  Samples 50000
checkpoint-13: Loss 0.8415  Acc@1 80.4600  Acc@5 95.3180  Samples 50000
checkpoint-14: Loss 0.8441  Acc@1 80.4020  Acc@5 95.3220  Samples 50000
checkpoint-15: Loss 0.8399  Acc@1 80.4260  Acc@5 95.3120  Samples 50000
checkpoint-16: Loss 0.8426  Acc@1 80.3260  Acc@5 95.3260  Samples 50000
checkpoint-17: Loss 0.8439  Acc@1 80.5060  Acc@5 95.2760  Samples 50000
checkpoint-18: Loss 0.8415  Acc@1 80.3600  Acc@5 95.3200  Samples 50000
checkpoint-19: Loss 0.8399  Acc@1 80.4860  Acc@5 95.3280  Samples 50000
checkpoint-20: Loss 0.8364  Acc@1 80.3460  Acc@5 95.3960  Samples 50000
checkpoint-21: Loss 0.8433  Acc@1 80.3560  Acc@5 95.3240  Samples 50000
checkpoint-22: Loss 0.8420  Acc@1 80.5140  Acc@5 95.3100  Samples 50000
checkpoint-23: Loss 0.8371  Acc@1 80.4120  Acc@5 95.3520  Samples 50000
checkpoint-24: Loss 0.8359  Acc@1 80.4200  Acc@5 95.3000  Samples 50000
checkpoint-25: Loss 0.8409  Acc@1 80.4220  Acc@5 95.3460  Samples 50000
checkpoint-26: Loss 0.8446  Acc@1 80.5160  Acc@5 95.3000  Samples 50000
checkpoint-27: Loss 0.8411  Acc@1 80.5340  Acc@5 95.3280  Samples 50000
checkpoint-28: Loss 0.8351  Acc@1 80.4160  Acc@5 95.3320  Samples 50000
checkpoint-29: Loss 0.8378  Acc@1 80.4540  Acc@5 95.3400  Samples 50000
```

## RefW 观察点

需要检查以下 epoch 是否出现非零 `RefW`：

```text
epoch 28: RefW 3.000e-04
epoch 29: RefW 3.000e-04
epoch 36: RefW 3.500e-04
epoch 37: RefW 3.500e-04
epoch 44: RefW 3.500e-04
epoch 45: RefW 3.500e-04
epoch 52: RefW 3.000e-04
epoch 53: RefW 3.000e-04
```

重点 checkpoint：

```text
checkpoint-30: 反映 epoch 28/29 pulse 后效果
checkpoint-38: 反映 epoch 36/37 pulse 后效果
checkpoint-46: 反映 epoch 44/45 pulse 后效果
checkpoint-54: 反映 epoch 52/53 pulse 后效果
checkpoint-60: 50epoch 长跑最终点
```

## 当前结论

训练已在 worker `990645` 上启动并持续运行，当前已完成到 `checkpoint-29`，并进入 epoch 29 训练。到目前为止所有 full-val 都是 50000 samples 的单 checkpoint 评估，不涉及 soup、平均或 ensemble。

当前最佳：

```text
checkpoint-27: Top-1 80.5340, Top-5 95.3280, Loss 0.8411
相对 baseline 80.5980: -0.0640
81.0 target: 未达到
```

阶段判断：

```text
checkpoint-29 已反映 epoch 28 sparse pulse 后效果，但未超过 baseline；checkpoint-29 保存后进入 epoch 29，RefW 继续保持 3.000e-04，RefAttnKL 继续非零。
第一个真正 sparse pulse 覆盖 epoch 28/29，关键观察点是 checkpoint-30。
当前训练速度和 GPU 利用率正常，应继续轮询 checkpoint-30，并重点判断双 pulse 后是否超过 baseline。
```

## 2026-07-09 16:54 UTC 复查

本轮继续推进前重新核对实际状态：

```text
active goal: 仍为方案 C checkpoint-10 -> checkpoint-60 长跑。
current shell: no /dev/nvidia0。
train log: /mlx_devbox/users/quyanyi/playground/train_ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709.log 不存在。
output dir: /mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709 不存在。
running process: 未发现 run_ofq_resume10_to60_sparse_pulse / train_ofq_resume10_to60_sparse / qat_launch / torchrun 相关训练进程。
```

实时 worker 查询：

```text
NO_COLOR=1 TERM=dumb mlx worker list
结果：只有表头，无现存 worker。

NO_COLOR=1 TERM=dumb mlx worker quota
结果：Public Workspace 有 H100-SXM-80GB 资源；其中至少有 8 GPU H100 host 可用。
```

历史 notebook worker 配置：

```text
/home/tiger/.merlin/notebook/workers.json 仍记录 id=984521, H100-SXM-80GB, 8 GPU。
但该 worker 不在当前 mlx worker list 中，并且此前 login 984521 已返回 worker not found。
结论：984521 是过期记录，不能作为可用 GPU worker 证据。
```

### 完成审计

目标拆解：

| 要求 | 当前证据 | 状态 |
|---|---|---|
| 按 goal doc 执行方案 C | goal doc、progress doc、launch script 已存在 | 部分完成 |
| 从 checkpoint-10 启动训练 | 尚未启动，无 log、无 output dir、无训练进程 | 未完成 |
| OFQ public-family 50epoch sparse pulse prev-step refKL 长跑 | 脚本参数已核对，训练未运行 | 未完成 |
| 持续轮询 checkpoint | output dir 不存在，无新 checkpoint | 未完成 |
| 持续轮询 full-val | 无新 full-val，只有起点 checkpoint-10 历史 full-val | 未完成 |
| 检查 RefW 生效 | 训练未启动，尚无 RefW 日志 | 未完成 |
| 更新中文进度文档 | 已更新启动阻塞、复查和审计记录 | 完成中 |
| 最终审计是否超过 80.5980 或达到 81.0 | 无新 checkpoint/full-val，不能判断 | 未完成 |

当前真实阻塞：

```text
没有可登录的现存 GPU worker。
当前 shell 无 GPU，不可启动 NCCL/GPU 训练。
根据既定边界，不能在未获得明确许可时擅自 mlx worker launch。
```

拿到有效 worker ID / worker TTY，或获得明确允许启动新 worker 后，继续执行：

```bash
cd /mlx_devbox/users/quyanyi/playground
MASTER_PORT=31561 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709.sh
```

## 2026-07-09 16:57 UTC 轮询工具和 preflight

新增方案 C 专用只读监控脚本：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/monitor_ofq_sparse_pulse_prevstep_refkl_c_20260709.sh
```

该脚本不启动训练、不改训练逻辑，只解析目标 log 和 output 目录，生成：

```text
summary: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_sparse_pulse_prevstep_refkl_c_50epoch_monitor_summary_20260709.txt
full-val table: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_sparse_pulse_prevstep_refkl_c_50epoch_status_20260709.tsv
RefW table: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_sparse_pulse_prevstep_refkl_c_50epoch_refw_20260709.tsv
```

当前 monitor 快照：

```text
timestamp=2026-07-09T16:57:00Z
log_exists=no
output_exists=no
checkpoint_count=0
latest_checkpoint=NA
best_fullval_line=NA
pulse_refw_nonzero_lines=0
```

启动前 preflight：

```text
launch script syntax: ok
monitor script syntax: ok
resume checkpoint-10: exists, 344191231 bytes
teacher checkpoint: exists, 113445839 bytes
ImageNet parquet: train_shards=294, validation_shards=14
方案 C 关键脚本参数: all_ok=True
```

已核对的关键参数：

```text
epochs=60
scheduler_epochs=60
resume=checkpoint-10
no_resume_opt=true
wq_mode=statsq
aq_mode=lsq
qk_reparam=true
kd_hard_and_soft=0
ref_update=prev_step
ref_head_mode=custom_subset:6:1,8:4,8:9,11:18,11:4
ref_attn_kl_weight_epoch_overrides=28:0.00030,29:0.00030,36:0.00035,37:0.00035,44:0.00035,45:0.00035,52:0.00030,53:0.00030
ref_attn_kl_drop_prob=0.50
```

实时资源状态：

```text
NO_COLOR=1 TERM=dumb mlx worker list: 只有表头，无现存 worker。
NO_COLOR=1 TERM=dumb mlx worker quota: Public Workspace 可见多个 8x H100-SXM-80GB host。
```

结论：

```text
训练依然尚未启动；不是脚本、数据或 checkpoint preflight 问题，而是缺少可登录的真实 GPU worker。
拿到 worker 后，先启动训练，再用 monitor 脚本轮询 full-val / checkpoint / RefW。
```

## 2026-07-09 17:18 UTC worker 和数据集准备

新 GPU worker：

```text
worker id: 990645
hostname: trial-301660525-trialrun-301660525-worker-0
gpu: 8 x NVIDIA H100 80GB HBM3
pod endpoint: fdbd:dccd:cdc2:1234:0:b8:::9801
```

真实 GPU worker TTY 内确认：

```text
/dev/nvidia0: visible
nvidia-smi: 8 张 H100 可见
workdir: /mlx_devbox/users/quyanyi/playground
```

数据集准备：

```text
目标路径: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
worker /tmp usage after sync: 714G used / 227G avail
```

同步方式说明：

```text
先尝试使用 Q-ViT/transfer_imnet_to_worker.sh，经 /mlx_devbox/users/quyanyi/imnet1k_relay 中转。
该路径因为 /mlx_devbox 系统盘 100% 满而失败，报错 No space left on device。
随后清理 imnet1k_relay 残留，系统盘从 100% 恢复到 94%。
最终改用 SSH tar stream 从 master /tmp 直传到 worker /tmp，只传 train/validation shards，不再占用 /mlx_devbox 中转空间。
```

下一步：

```bash
cd /mlx_devbox/users/quyanyi/playground
MASTER_PORT=31561 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709.sh
```

## 2026-07-09 17:27 UTC 启动成功

启动过程：

```text
17:19 UTC 首次在 worker TTY 内启动，参数和数据检查通过。
启动后发现 /mlx_devbox 系统盘只有 8.1G 可用，50 个 epoch checkpoint 直接写系统盘风险过高。
在首个 checkpoint 产生前中断首次启动，没有生成 checkpoint。
```

输出目录修正：

```text
原路径:
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709

现在为 symlink:
/mlx_devbox/users/quyanyi/playground/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709
  -> /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709

原因:
checkpoint 实际落 worker /tmp，避免 50epoch 长跑写满 /mlx_devbox 系统盘。
```

第二次启动：

```text
使用 MASTER_PORT=31561 后失败，根因是端口残留占用：
torch.distributed.DistNetworkError: EADDRINUSE, address already in use, port 31561

该次没有进入训练循环，没有生成 checkpoint。
```

第三次启动：

```text
MASTER_PORT=31637
启动方式: worker SSH nohup 后台启动
worker pid: 4541
log: /mlx_devbox/users/quyanyi/playground/train_ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709.log
remote output: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709
```

启动成功证据：

```text
Strict resume: loaded model from checkpoint-10; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
NCCL version 2.27.5+cuda12.9
Enabled EMA refmodel attention-KL scheme:
  ref_update=prev_step
  ref_update_interval=50
  head_mode=custom_subset:6:1,8:4,8:9,11:18,11:4
  selected_head_map={6: (1,), 8: (4, 9), 11: (4, 18)}
  warmup_epochs=28
Scheduled epochs: 60
Effective global batch: 512
Trainable parameter policy: epoch=10, policy=all
```

训练已经进入 epoch 10：

```text
Train: 10 [   0/2502 ...] LR: 1.433e-05 RefW: 0.000e+00
Train: 10 [  50/2502 ...] LR: 1.433e-05 RefW: 0.000e+00
Train: 10 [ 100/2502 ...] LR: 1.432e-05 RefW: 0.000e+00
Train: 10 [ 200/2502 ...] LR: 1.432e-05 RefW: 0.000e+00
```

当前 GPU 状态：

```text
8 x H100 全部约 28.4G 显存占用。
GPU utilization: 96% - 99%。
worker /tmp: 227G available。
/mlx_devbox 系统盘: 8.1G available。
```

`args.yaml` 已落盘，关键参数核对：

```text
epochs: 60
scheduler_epochs: 60
resume: checkpoint-10
no_resume_opt: true
wq_mode: statsq
aq_mode: lsq
qk_reparam: true
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
train_scheme: ema_ref_attn_kl
ref_update: prev_step
ref_update_interval: 50
ref_head_mode: custom_subset:6:1,8:4,8:9,11:18,11:4
ref_warmup_epochs: 28
ref_attn_kl_weight: 0.0
ref_attn_kl_drop_prob: 0.5
ref_attn_loss: kl_ref
checkpoint_hist: 60
epoch_checkpoint_interval: 1
```

当前状态：

```text
训练已成功启动并运行中。
尚未完成 epoch 10，因此还没有 checkpoint-11 或 full-val。
当前 RefW=0 符合设计，因为 warmup_epochs=28，pulse 从 epoch 28/29 开始。
下一步轮询 checkpoint-11、full-val 和后续 RefW。
```

## 2026-07-09 17:38 UTC checkpoint-11

epoch 10 已完成，并生成首个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-11.pth.tar
checkpoint size: 344196543 bytes
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=10 updates=2496 avg_step_time=0.232868s samples_per_step=512 samples_per_sec=2198.67
Test: [distributed-summary]  Time: 34.954s  Loss: 0.8443  Acc@1: 80.4000  Acc@5: 95.2460  Samples: 50000
```

对比：

```text
起点 checkpoint-10 Top-1: 80.3640
checkpoint-11 Top-1: 80.4000
相对起点: +0.0360
相对 baseline 80.5980: -0.1980
81.0 target: 未达到
```

RefW 状态：

```text
epoch 10: RefW=0.000e+00
epoch 11 start: RefW=0.000e+00
符合方案 C 设计，pulse 仍未开启；预期 epoch 28/29 才出现第一组非零 RefW。
```

运行状态：

```text
训练已进入 epoch 11，约 12% 进度。
8 x H100 仍为约 28.4G 显存占用，GPU utilization 约 97%-99%。
worker /tmp 可用空间约 227G，/mlx_devbox 系统盘约 8.1G 可用。
```

## 2026-07-09 17:48 UTC checkpoint-12

epoch 11 已完成，并生成第二个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-12.pth.tar
checkpoint size: 344196543 bytes
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=11 updates=2496 avg_step_time=0.232242s samples_per_step=512 samples_per_sec=2204.60
Test: [distributed-summary]  Time: 10.488s  Loss: 0.8452  Acc@1: 80.3740  Acc@5: 95.2580  Samples: 50000
```

对比：

```text
起点 checkpoint-10 Top-1: 80.3640
checkpoint-11 Top-1: 80.4000
checkpoint-12 Top-1: 80.3740
相对起点: +0.0100
相对 checkpoint-11: -0.0260
相对 baseline 80.5980: -0.2240
81.0 target: 未达到
```

RefW 状态：

```text
epoch 11: RefW=0.000e+00
epoch 12 start: RefW=0.000e+00
符合方案 C 设计，pulse 仍未开启；预期 epoch 28/29 才出现第一组非零 RefW。
```

运行状态：

```text
训练已进入 epoch 12，约 12% 进度。
8 x H100 仍为约 28.4G 显存占用，GPU utilization 约 90%-98%。
当前最佳仍为 checkpoint-11: Top-1 80.4000，尚未超过 80.5980 baseline。
```

## 2026-07-09 17:58 UTC checkpoint-13

epoch 12 已完成，并生成第三个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-13.pth.tar
checkpoint size: 344196543 bytes
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=12 updates=2496 avg_step_time=0.232712s samples_per_step=512 samples_per_sec=2200.14
Test: [distributed-summary]  Time: 10.412s  Loss: 0.8415  Acc@1: 80.4600  Acc@5: 95.3180  Samples: 50000
```

对比：

```text
起点 checkpoint-10 Top-1: 80.3640
checkpoint-11 Top-1: 80.4000
checkpoint-12 Top-1: 80.3740
checkpoint-13 Top-1: 80.4600
相对起点: +0.0960
相对 checkpoint-12: +0.0860
相对当前最佳 checkpoint-11: +0.0600
相对 baseline 80.5980: -0.1380
81.0 target: 未达到
```

RefW 状态：

```text
epoch 12: RefW=0.000e+00
epoch 13 start: RefW=0.000e+00
符合方案 C 设计，pulse 仍未开启；预期 epoch 28/29 才出现第一组非零 RefW。
```

运行状态：

```text
训练已进入 epoch 13，约 20% 进度。
8 x H100 仍为约 28.4G 显存占用，GPU utilization 约 97%-99%。
worker /tmp 可用空间约 226G，/mlx_devbox 系统盘约 8.1G 可用。
当前最佳更新为 checkpoint-13: Top-1 80.4600，但尚未超过 80.5980 baseline。
```

## 2026-07-09 18:07 UTC checkpoint-14

epoch 13 已完成，并生成第四个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-14.pth.tar
checkpoint size: 344196543 bytes
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=13 updates=2496 avg_step_time=0.232565s samples_per_step=512 samples_per_sec=2201.54
Test: [distributed-summary]  Time: 10.604s  Loss: 0.8441  Acc@1: 80.4020  Acc@5: 95.3220  Samples: 50000
```

对比：

```text
checkpoint-13 Top-1: 80.4600
checkpoint-14 Top-1: 80.4020
相对 checkpoint-13: -0.0580
相对 baseline 80.5980: -0.1960
81.0 target: 未达到
```

RefW 状态：

```text
epoch 13: RefW=0.000e+00
符合方案 C 设计，pulse 仍未开启；预期 epoch 28/29 才出现第一组非零 RefW。
```

运行状态：

```text
训练继续运行，进入 epoch 14。
当前最佳仍为 checkpoint-13: Top-1 80.4600，尚未超过 80.5980 baseline。
```

## 2026-07-09 18:18 UTC checkpoint-15

epoch 14 已完成，并生成第五个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-15.pth.tar
checkpoint size: 344196543 bytes
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=14 updates=2496 avg_step_time=0.232682s samples_per_step=512 samples_per_sec=2200.43
Test: [distributed-summary]  Time: 10.406s  Loss: 0.8399  Acc@1: 80.4260  Acc@5: 95.3120  Samples: 50000
```

对比：

```text
checkpoint-13 Top-1: 80.4600
checkpoint-14 Top-1: 80.4020
checkpoint-15 Top-1: 80.4260
相对 checkpoint-14: +0.0240
相对当前最佳 checkpoint-13: -0.0340
相对 baseline 80.5980: -0.1720
81.0 target: 未达到
```

RefW 状态：

```text
epoch 14: RefW=0.000e+00
epoch 15 start: RefW=0.000e+00
符合方案 C 设计，pulse 仍未开启；预期 epoch 28/29 才出现第一组非零 RefW。
```

运行状态：

```text
训练继续运行，进入 epoch 15，约 16% 进度。
8 x H100 仍为约 28.4G 显存占用，GPU utilization 约 97%-99%。
当前最佳仍为 checkpoint-13: Top-1 80.4600，尚未超过 80.5980 baseline。
```

## 2026-07-09 18:27 UTC checkpoint-16

epoch 15 已完成，并生成第六个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-16.pth.tar
checkpoint size: 344196543 bytes
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=15 updates=2496 avg_step_time=0.232472s samples_per_step=512 samples_per_sec=2202.42
Test: [distributed-summary]  Time: 10.374s  Loss: 0.8426  Acc@1: 80.3260  Acc@5: 95.3260  Samples: 50000
```

对比：

```text
checkpoint-13 Top-1: 80.4600
checkpoint-15 Top-1: 80.4260
checkpoint-16 Top-1: 80.3260
相对 checkpoint-15: -0.1000
相对当前最佳 checkpoint-13: -0.1340
相对 baseline 80.5980: -0.2720
81.0 target: 未达到
```

RefW 状态：

```text
epoch 15: RefW=0.000e+00
epoch 16 start: RefW=0.000e+00
符合方案 C 设计，pulse 仍未开启；预期 epoch 28/29 才出现第一组非零 RefW。
```

运行状态：

```text
训练继续运行，进入 epoch 16，约 6% 进度。
8 x H100 仍为约 28.4G 显存占用，GPU utilization 约 97%-99%。
当前最佳仍为 checkpoint-13: Top-1 80.4600，尚未超过 80.5980 baseline。
```

## 2026-07-09 18:38 UTC checkpoint-17

epoch 16 已完成，并生成第七个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-17.pth.tar
checkpoint size: 344196479 bytes
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=16 updates=2496 avg_step_time=0.232806s samples_per_step=512 samples_per_sec=2199.26
Test: [distributed-summary]  Time: 10.507s  Loss: 0.8439  Acc@1: 80.5060  Acc@5: 95.2760  Samples: 50000
```

对比：

```text
checkpoint-13 Top-1: 80.4600
checkpoint-16 Top-1: 80.3260
checkpoint-17 Top-1: 80.5060
相对 checkpoint-16: +0.1800
相对当前最佳 checkpoint-13: +0.0460
相对 baseline 80.5980: -0.0920
81.0 target: 未达到
```

RefW 状态：

```text
epoch 16: RefW=0.000e+00
epoch 17 start: RefW=0.000e+00
符合方案 C 设计，pulse 仍未开启；预期 epoch 28/29 才出现第一组非零 RefW。
```

运行状态：

```text
训练继续运行，进入 epoch 17，约 14% 进度。
8 x H100 仍为约 28.4G 显存占用，GPU utilization 约 97%-99%。
当前最佳更新为 checkpoint-17: Top-1 80.5060，但尚未超过 80.5980 baseline。
```

## 2026-07-09 18:48 UTC checkpoint-18

epoch 17 已完成，并生成第八个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-18.pth.tar
checkpoint size: 344196543 bytes
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=17 updates=2496 avg_step_time=0.232380s samples_per_step=512 samples_per_sec=2203.29
Test: [distributed-summary]  Time: 10.451s  Loss: 0.8415  Acc@1: 80.3600  Acc@5: 95.3200  Samples: 50000
```

对比：

```text
checkpoint-17 Top-1: 80.5060
checkpoint-18 Top-1: 80.3600
相对 checkpoint-17: -0.1460
相对 baseline 80.5980: -0.2380
81.0 target: 未达到
```

RefW 状态：

```text
epoch 17: RefW=0.000e+00
epoch 18 start: RefW=0.000e+00
符合方案 C 设计，pulse 仍未开启；预期 epoch 28/29 才出现第一组非零 RefW。
```

运行状态：

```text
训练继续运行，进入 epoch 18，约 14% 进度。
8 x H100 仍为约 28.4G 显存占用，GPU utilization 约 97%-99%。
当前最佳仍为 checkpoint-17: Top-1 80.5060，尚未超过 80.5980 baseline。
```

## 2026-07-09 18:56 UTC checkpoint-19

epoch 18 已完成，并生成第九个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-19.pth.tar
checkpoint size: 344196543 bytes
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=18 updates=2496 avg_step_time=0.232886s samples_per_step=512 samples_per_sec=2198.50
Test: [distributed-summary]  Time: 10.536s  Loss: 0.8399  Acc@1: 80.4860  Acc@5: 95.3280  Samples: 50000
```

对比：

```text
checkpoint-17 Top-1: 80.5060
checkpoint-18 Top-1: 80.3600
checkpoint-19 Top-1: 80.4860
相对 checkpoint-18: +0.1260
相对当前最佳 checkpoint-17: -0.0200
相对 baseline 80.5980: -0.1120
81.0 target: 未达到
```

RefW 状态：

```text
epoch 18: RefW=0.000e+00
epoch 19 start: RefW=0.000e+00
符合方案 C 设计，pulse 仍未开启；预期 epoch 28/29 才出现第一组非零 RefW。
```

运行状态：

```text
训练继续运行，进入 epoch 19，约 4% 进度。
8 x H100 仍为约 28.4G 显存占用，GPU utilization 约 96%-99%。
当前最佳仍为 checkpoint-17: Top-1 80.5060，尚未超过 80.5980 baseline。
下一关键节点：checkpoint-20 继续观察 warmup 段随机波动；checkpoint-30 才是第一组 sparse pulse 后的核心判据。
```

## 2026-07-09 19:08 UTC checkpoint-20

epoch 19 已完成，并生成第十个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-20.pth.tar
checkpoint size: 344196543 bytes
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=19 updates=2496 avg_step_time=0.232657s samples_per_step=512 samples_per_sec=2200.66
Test: [distributed-summary]  Time: 10.483s  Loss: 0.8364  Acc@1: 80.3460  Acc@5: 95.3960  Samples: 50000
```

对比：

```text
checkpoint-17 Top-1: 80.5060
checkpoint-19 Top-1: 80.4860
checkpoint-20 Top-1: 80.3460
相对 checkpoint-19: -0.1400
相对当前最佳 checkpoint-17: -0.1600
相对 baseline 80.5980: -0.2520
81.0 target: 未达到
```

RefW 状态：

```text
epoch 19: RefW=0.000e+00
epoch 20 start: RefW=0.000e+00
符合方案 C 设计，pulse 仍未开启；预期 epoch 28/29 才出现第一组非零 RefW。
```

运行状态：

```text
训练继续运行，进入 epoch 20，约 24% 进度。
8 x H100 仍为约 28.4G 显存占用，GPU utilization 约 96%-99%。
当前最佳仍为 checkpoint-17: Top-1 80.5060，尚未超过 80.5980 baseline。
下一关键节点：checkpoint-21 继续观察 warmup 段随机波动；checkpoint-30 才是第一组 sparse pulse 后的核心判据。
```

## 2026-07-09 19:16 UTC checkpoint-21

epoch 20 已完成，并生成第十一个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-21.pth.tar
checkpoint size: 344196543 bytes
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=20 updates=2496 avg_step_time=0.232587s samples_per_step=512 samples_per_sec=2201.33
Test: [distributed-summary]  Time: 10.501s  Loss: 0.8433  Acc@1: 80.3560  Acc@5: 95.3240  Samples: 50000
```

对比：

```text
checkpoint-17 Top-1: 80.5060
checkpoint-20 Top-1: 80.3460
checkpoint-21 Top-1: 80.3560
相对 checkpoint-20: +0.0100
相对当前最佳 checkpoint-17: -0.1500
相对 baseline 80.5980: -0.2420
81.0 target: 未达到
```

RefW 状态：

```text
epoch 20: RefW=0.000e+00
epoch 21 start: RefW=0.000e+00
符合方案 C 设计，pulse 仍未开启；预期 epoch 28/29 才出现第一组非零 RefW。
```

运行状态：

```text
训练继续运行，进入 epoch 21，约 6% 进度。
8 x H100 仍为约 28.4G 显存占用，GPU utilization 约 96%-99%。
当前最佳仍为 checkpoint-17: Top-1 80.5060，尚未超过 80.5980 baseline。
下一关键节点：checkpoint-22 继续观察 warmup 段随机波动；checkpoint-30 才是第一组 sparse pulse 后的核心判据。
```

## 2026-07-09 19:27 UTC checkpoint-22

epoch 21 已完成，并生成第十二个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-22.pth.tar
checkpoint size: 344196543 bytes
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=21 updates=2496 avg_step_time=0.232665s samples_per_step=512 samples_per_sec=2200.59
Test: [distributed-summary]  Time: 10.475s  Loss: 0.8420  Acc@1: 80.5140  Acc@5: 95.3100  Samples: 50000
```

对比：

```text
checkpoint-17 Top-1: 80.5060
checkpoint-21 Top-1: 80.3560
checkpoint-22 Top-1: 80.5140
相对 checkpoint-21: +0.1580
相对前最佳 checkpoint-17: +0.0080
相对 baseline 80.5980: -0.0840
81.0 target: 未达到
```

RefW 状态：

```text
epoch 21: RefW=0.000e+00
epoch 22 start: RefW=0.000e+00
符合方案 C 设计，pulse 仍未开启；预期 epoch 28/29 才出现第一组非零 RefW。
```

运行状态：

```text
训练继续运行，进入 epoch 22，约 18% 进度。
8 x H100 仍为约 28.4G 显存占用，GPU utilization 约 95%-98%。
当前最佳更新为 checkpoint-22: Top-1 80.5140，但尚未超过 80.5980 baseline。
下一关键节点：checkpoint-23 继续观察 warmup 段随机波动；checkpoint-30 才是第一组 sparse pulse 后的核心判据。
```

## 2026-07-09 19:35 UTC checkpoint-23

epoch 22 已完成，并生成第十三个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-23.pth.tar
checkpoint size: 344196543 bytes
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=22 updates=2496 avg_step_time=0.231870s samples_per_step=512 samples_per_sec=2208.14
Test: [distributed-summary]  Time: 10.413s  Loss: 0.8371  Acc@1: 80.4120  Acc@5: 95.3520  Samples: 50000
```

对比：

```text
checkpoint-22 Top-1: 80.5140
checkpoint-23 Top-1: 80.4120
相对 checkpoint-22: -0.1020
相对当前最佳 checkpoint-22: -0.1020
相对 baseline 80.5980: -0.1860
81.0 target: 未达到
```

RefW 状态：

```text
epoch 22: RefW=0.000e+00
epoch 23 start: RefW=0.000e+00
符合方案 C 设计，pulse 仍未开启；预期 epoch 28/29 才出现第一组非零 RefW。
```

运行状态：

```text
训练继续运行，进入 epoch 23，约 0% 进度。
8 x H100 仍为约 28.5G 显存占用，GPU utilization 约 92%-93%。
当前最佳仍为 checkpoint-22: Top-1 80.5140，尚未超过 80.5980 baseline。
下一关键节点：checkpoint-24 继续观察 warmup 段随机波动；checkpoint-30 才是第一组 sparse pulse 后的核心判据。
```

## 2026-07-09 19:46 UTC checkpoint-24

epoch 23 已完成，并生成第十四个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-24.pth.tar
checkpoint size: 344196543 bytes
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=23 updates=2496 avg_step_time=0.231977s samples_per_step=512 samples_per_sec=2207.11
Test: [distributed-summary]  Time: 10.418s  Loss: 0.8359  Acc@1: 80.4200  Acc@5: 95.3000  Samples: 50000
```

对比：

```text
checkpoint-22 Top-1: 80.5140
checkpoint-23 Top-1: 80.4120
checkpoint-24 Top-1: 80.4200
相对 checkpoint-23: +0.0080
相对当前最佳 checkpoint-22: -0.0940
相对 baseline 80.5980: -0.1780
81.0 target: 未达到
```

RefW 状态：

```text
epoch 23: RefW=0.000e+00
epoch 24 start: RefW=0.000e+00
符合方案 C 设计，pulse 仍未开启；预期 epoch 28/29 才出现第一组非零 RefW。
```

运行状态：

```text
训练继续运行，进入 epoch 24，约 16% 进度。
8 x H100 仍为约 28.4G 显存占用，GPU utilization 约 96%-98%。
当前最佳仍为 checkpoint-22: Top-1 80.5140，尚未超过 80.5980 baseline。
下一关键节点：checkpoint-25 继续观察 warmup 段随机波动；checkpoint-30 才是第一组 sparse pulse 后的核心判据。
```

## 2026-07-09 19:57 UTC checkpoint-25

epoch 24 已完成，并生成第十五个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-25.pth.tar
checkpoint size: 344196543 bytes
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=24 updates=2496 avg_step_time=0.232767s samples_per_step=512 samples_per_sec=2199.63
Test: [distributed-summary]  Time: 10.518s  Loss: 0.8409  Acc@1: 80.4220  Acc@5: 95.3460  Samples: 50000
```

对比：

```text
checkpoint-22 Top-1: 80.5140
checkpoint-24 Top-1: 80.4200
checkpoint-25 Top-1: 80.4220
相对 checkpoint-24: +0.0020
相对当前最佳 checkpoint-22: -0.0920
相对 baseline 80.5980: -0.1760
81.0 target: 未达到
```

RefW 状态：

```text
epoch 24: RefW=0.000e+00
epoch 25 start: RefW=0.000e+00
符合方案 C 设计，pulse 仍未开启；预期 epoch 28/29 才出现第一组非零 RefW。
```

运行状态：

```text
训练继续运行，进入 epoch 25，约 26% 进度。
8 x H100 仍为约 28.4G 显存占用，GPU utilization 约 92%-98%。
当前最佳仍为 checkpoint-22: Top-1 80.5140，尚未超过 80.5980 baseline。
下一关键节点：checkpoint-26 继续观察 warmup 段随机波动；checkpoint-30 才是第一组 sparse pulse 后的核心判据。
```

## 2026-07-09 20:05 UTC checkpoint-26

epoch 25 已完成，并生成第十六个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-26.pth.tar
checkpoint size: 344196543 bytes
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=25 updates=2496 avg_step_time=0.232223s samples_per_step=512 samples_per_sec=2204.78
Test: [distributed-summary]  Time: 10.392s  Loss: 0.8446  Acc@1: 80.5160  Acc@5: 95.3000  Samples: 50000
```

对比：

```text
checkpoint-22 Top-1: 80.5140
checkpoint-25 Top-1: 80.4220
checkpoint-26 Top-1: 80.5160
相对 checkpoint-25: +0.0940
相对前最佳 checkpoint-22: +0.0020
相对 baseline 80.5980: -0.0820
81.0 target: 未达到
```

RefW 状态：

```text
epoch 25: RefW=0.000e+00
epoch 26 start: RefW=0.000e+00
符合方案 C 设计，pulse 仍未开启；预期 epoch 28/29 才出现第一组非零 RefW。
```

运行状态：

```text
训练继续运行，进入 epoch 26，约 6% 进度。
8 x H100 仍为约 28.4G 显存占用，GPU utilization 约 97%-98%。
当前最佳更新为 checkpoint-26: Top-1 80.5160，但尚未超过 80.5980 baseline。
下一关键节点：checkpoint-27 继续观察 warmup 段随机波动；checkpoint-30 才是第一组 sparse pulse 后的核心判据。
```

## 2026-07-09 20:17 UTC checkpoint-27

epoch 26 已完成，并生成第十七个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-27.pth.tar
checkpoint size: 344196543 bytes
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=26 updates=2496 avg_step_time=0.232303s samples_per_step=512 samples_per_sec=2204.02
Test: [distributed-summary]  Time: 11.012s  Loss: 0.8411  Acc@1: 80.5340  Acc@5: 95.3280  Samples: 50000
```

对比：

```text
checkpoint-26 Top-1: 80.5160
checkpoint-27 Top-1: 80.5340
相对 checkpoint-26: +0.0180
相对前最佳 checkpoint-26: +0.0180
相对 baseline 80.5980: -0.0640
81.0 target: 未达到
```

RefW 状态：

```text
epoch 26: RefW=0.000e+00
epoch 27 start: RefW=0.000e+00
符合方案 C 设计，pulse 仍未开启；预期 epoch 28/29 才出现第一组非零 RefW。
```

运行状态：

```text
训练继续运行，进入 epoch 27，约 24% 进度。
8 x H100 仍为约 28.4G 显存占用，GPU utilization 约 97%-99%。
当前最佳更新为 checkpoint-27: Top-1 80.5340，但尚未超过 80.5980 baseline。
下一关键节点：checkpoint-28 继续观察 warmup 段末尾；checkpoint-30 才是第一组 sparse pulse 后的核心判据。
```

## 2026-07-09 20:25 UTC checkpoint-28

epoch 27 已完成，并生成第十八个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-28.pth.tar
checkpoint size: 344196543 bytes
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=27 updates=2496 avg_step_time=0.232172s samples_per_step=512 samples_per_sec=2205.26
Test: [distributed-summary]  Time: 10.410s  Loss: 0.8351  Acc@1: 80.4160  Acc@5: 95.3320  Samples: 50000
```

对比：

```text
checkpoint-27 Top-1: 80.5340
checkpoint-28 Top-1: 80.4160
相对 checkpoint-27: -0.1180
相对当前最佳 checkpoint-27: -0.1180
相对 baseline 80.5980: -0.1820
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 27: RefW=0.000e+00
epoch 28 start: RefW=3.000e-04
epoch 28 start: RefAttnKL=3.063e+01
epoch 28 update 50: RefW=3.000e-04, RefAttnKL=3.180e+01
```

结论：

```text
checkpoint-28 full-val 本身仍主要反映 pulse 开启前的 warmup 训练结果。
保存 checkpoint-28 后，epoch 28 已按方案 C 进入第一组 sparse pulse，RefW 和 RefAttnKL 均已非零。
下一关键判据是 checkpoint-29/30：checkpoint-29 反映 epoch 28 pulse 后效果，checkpoint-30 反映 epoch 28/29 双 pulse 后效果。
```

运行状态：

```text
训练继续运行，进入 epoch 28，约 4% 进度。
8 x H100 仍为约 31.3G 显存占用，GPU utilization 约 94%-99%。
当前最佳仍为 checkpoint-27: Top-1 80.5340，尚未超过 80.5980 baseline。
```

## 2026-07-09 20:38 UTC checkpoint-29

epoch 28 已完成，并生成第十九个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-29.pth.tar
checkpoint size: 344196543 bytes
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=28 updates=2496 avg_step_time=0.323886s samples_per_step=512 samples_per_sec=1580.80
Test: [distributed-summary]  Time: 10.480s  Loss: 0.8378  Acc@1: 80.4540  Acc@5: 95.3400  Samples: 50000
```

对比：

```text
checkpoint-27 Top-1: 80.5340
checkpoint-28 Top-1: 80.4160
checkpoint-29 Top-1: 80.4540
相对 checkpoint-28: +0.0380
相对当前最佳 checkpoint-27: -0.0800
相对 baseline 80.5980: -0.1440
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 28: RefW=3.000e-04 持续全 epoch
epoch 28 RefAttnKL: 多数日志点约 2e+01 到 5e+01，持续非零
epoch 29 start: RefW=3.000e-04
epoch 29 start: RefAttnKL=4.410e+01
```

结论：

```text
checkpoint-29 是第一组 sparse pulse 中 epoch 28 后的第一轮 full-val。
它从 checkpoint-28 的 80.4160 回升到 80.4540，但仍低于 checkpoint-27 的 80.5340，也低于 80.5980 baseline。
epoch 29 已继续进入第二个 pulse epoch，checkpoint-30 是判断 epoch 28/29 双 pulse 是否有效的关键点。
```

运行状态：

```text
训练继续运行，进入 epoch 29，约 0% 进度。
8 x H100 仍为约 31.4G 显存占用，GPU utilization 约 90%-99%。
当前最佳仍为 checkpoint-27: Top-1 80.5340，尚未超过 80.5980 baseline。
```

## 2026-07-09 20:52 UTC checkpoint-30

epoch 29 已完成，并生成第二十个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-30.pth.tar
checkpoint size: 344196543 bytes
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=29 updates=2496 avg_step_time=0.323669s samples_per_step=512 samples_per_sec=1581.86
Test: [distributed-summary]  Time: 10.572s  Loss: 0.8396  Acc@1: 80.4580  Acc@5: 95.3160  Samples: 50000
```

对比：

```text
checkpoint-27 Top-1: 80.5340
checkpoint-28 Top-1: 80.4160
checkpoint-29 Top-1: 80.4540
checkpoint-30 Top-1: 80.4580
相对 checkpoint-29: +0.0040
相对当前最佳 checkpoint-27: -0.0760
相对 baseline 80.5980: -0.1400
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 29: RefW=3.000e-04 持续全 epoch
epoch 29 RefAttnKL: 多数日志点约 1e+01 到 5e+01，持续非零
epoch 30 start: RefW=0.000e+00
epoch 30 start: RefAttnKL=0.000e+00
```

结论：

```text
checkpoint-30 是第一组 sparse pulse 中 epoch 28/29 双 pulse 后的关键 full-val。
结果只从 checkpoint-29 的 80.4540 小幅到 80.4580，没有超过 checkpoint-27 的 80.5340，也没有超过 direct-resume baseline 80.5980。
这说明 28/29 的 3.0e-4 双 pulse 没有带来正收益；下一组关键验证点是 36/37 的 3.5e-4 pulse。
```

运行状态：

```text
训练继续运行，进入 epoch 30，RefW 已按计划回到 0。
8 x H100 仍为约 28.4G 显存占用，GPU utilization 约 93%-98%。
当前最佳仍为 checkpoint-27: Top-1 80.5340，尚未超过 80.5980 baseline。
```

## 2026-07-09 21:02 UTC checkpoint-31

epoch 30 已完成，并生成第二十一个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-31.pth.tar
checkpoint_count: 21
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=30 updates=2496 avg_step_time=0.232947s samples_per_step=512 samples_per_sec=2197.93
Test: [distributed-summary]  Time: 10.551s  Loss: 0.8387  Acc@1: 80.4220  Acc@5: 95.3400  Samples: 50000
```

对比：

```text
checkpoint-27 Top-1: 80.5340
checkpoint-30 Top-1: 80.4580
checkpoint-31 Top-1: 80.4220
相对 checkpoint-30: -0.0360
相对当前最佳 checkpoint-27: -0.1120
相对 baseline 80.5980: -0.1760
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 30: RefW=0.000e+00 持续全 epoch
epoch 30 RefAttnKL=0.000e+00
epoch 31 start: RefW=0.000e+00
```

结论：

```text
checkpoint-31 是第一组 28/29 pulse 结束后的第一个非 pulse epoch。
结果从 checkpoint-30 的 80.4580 回落到 80.4220，没有出现 pulse 后恢复收益。
当前仍需继续观察 36/37 的第二组更强 pulse；到 checkpoint-31 为止，本 run 最佳仍是 pulse 前的 checkpoint-27。
```

运行状态：

```text
训练继续运行，进入 epoch 31，RefW 仍为 0。
当前最佳仍为 checkpoint-27: Top-1 80.5340，尚未超过 80.5980 baseline。
```

## 2026-07-09 21:12 UTC checkpoint-32

epoch 31 已完成，并生成第二十二个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-32.pth.tar
checkpoint_count: 22
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=31 updates=2496 avg_step_time=0.233229s samples_per_step=512 samples_per_sec=2195.27
Test: [distributed-summary]  Time: 10.580s  Loss: 0.8386  Acc@1: 80.4200  Acc@5: 95.2540  Samples: 50000
```

对比：

```text
checkpoint-27 Top-1: 80.5340
checkpoint-30 Top-1: 80.4580
checkpoint-31 Top-1: 80.4220
checkpoint-32 Top-1: 80.4200
相对 checkpoint-31: -0.0020
相对当前最佳 checkpoint-27: -0.1140
相对 baseline 80.5980: -0.1780
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 31: RefW=0.000e+00 持续全 epoch
epoch 31 RefAttnKL=0.000e+00
epoch 32 start: RefW=0.000e+00
```

结论：

```text
checkpoint-32 是第一组 28/29 pulse 结束后的第二个非 pulse epoch。
结果与 checkpoint-31 基本持平，仍未出现向 baseline 或 checkpoint-27 恢复的趋势。
到 checkpoint-32 为止，第一组 3.0e-4 pulse 未带来正收益；继续等待 36/37 的 3.5e-4 pulse 验证。
```

运行状态：

```text
训练继续运行，进入 epoch 32，RefW 仍为 0。
当前最佳仍为 checkpoint-27: Top-1 80.5340，尚未超过 80.5980 baseline。
```

## 2026-07-09 21:22 UTC checkpoint-33

epoch 32 已完成，并生成第二十三个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-33.pth.tar
checkpoint_count: 23
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=32 updates=2496 avg_step_time=0.232303s samples_per_step=512 samples_per_sec=2204.02
Test: [distributed-summary]  Time: 10.562s  Loss: 0.8359  Acc@1: 80.5520  Acc@5: 95.3520  Samples: 50000
```

对比：

```text
checkpoint-27 Top-1: 80.5340
checkpoint-30 Top-1: 80.4580
checkpoint-31 Top-1: 80.4220
checkpoint-32 Top-1: 80.4200
checkpoint-33 Top-1: 80.5520
相对 checkpoint-32: +0.1320
相对此前本 run 最佳 checkpoint-27: +0.0180
相对 baseline 80.5980: -0.0460
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 32: RefW=0.000e+00 持续全 epoch
epoch 32 RefAttnKL=0.000e+00
epoch 33 start: RefW=0.000e+00
```

结论：

```text
checkpoint-33 是第一组 28/29 pulse 结束后的第三个非 pulse epoch。
结果从 checkpoint-32 的 80.4200 回升到 80.5520，刷新本 run 最佳，但仍未超过 80.5980 direct-resume baseline。
这说明第一组 pulse 结束后存在延迟恢复，但到目前还不足以证明方案 C 超过 OFQ 基线；下一关键节点仍是 36/37 的 3.5e-4 pulse。
```

运行状态：

```text
训练继续运行，进入 epoch 33，RefW 仍为 0。
当前最佳更新为 checkpoint-33: Top-1 80.5520，尚未超过 80.5980 baseline。
```

## 2026-07-09 21:32 UTC checkpoint-34

epoch 33 已完成，并生成第二十四个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-34.pth.tar
checkpoint_count: 24
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=33 updates=2496 avg_step_time=0.232504s samples_per_step=512 samples_per_sec=2202.12
Test: [distributed-summary]  Time: 10.524s  Loss: 0.8376  Acc@1: 80.4920  Acc@5: 95.4220  Samples: 50000
```

对比：

```text
checkpoint-33 Top-1: 80.5520
checkpoint-34 Top-1: 80.4920
相对 checkpoint-33: -0.0600
相对当前最佳 checkpoint-33: -0.0600
相对 baseline 80.5980: -0.1060
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 33: RefW=0.000e+00 持续全 epoch
epoch 33 RefAttnKL=0.000e+00
epoch 34 start: RefW=0.000e+00
```

结论：

```text
checkpoint-34 是第一组 28/29 pulse 结束后的第四个非 pulse epoch。
结果从 checkpoint-33 的 80.5520 回落到 80.4920，没有稳定保持 80.55 高点。
到 checkpoint-34 为止，第一组 pulse 后的最好单点是 checkpoint-33，但仍未超过 80.5980 baseline；继续等待 36/37 的第二组 3.5e-4 pulse。
```

运行状态：

```text
训练继续运行，进入 epoch 34，RefW 仍为 0。
当前最佳仍为 checkpoint-33: Top-1 80.5520，尚未超过 80.5980 baseline。
```

## 2026-07-09 21:42 UTC checkpoint-35

epoch 34 已完成，并生成第二十五个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-35.pth.tar
checkpoint_count: 25
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=34 updates=2496 avg_step_time=0.232694s samples_per_step=512 samples_per_sec=2200.32
Test: [distributed-summary]  Time: 10.502s  Loss: 0.8360  Acc@1: 80.4180  Acc@5: 95.4100  Samples: 50000
```

对比：

```text
checkpoint-33 Top-1: 80.5520
checkpoint-34 Top-1: 80.4920
checkpoint-35 Top-1: 80.4180
相对 checkpoint-34: -0.0740
相对当前最佳 checkpoint-33: -0.1340
相对 baseline 80.5980: -0.1800
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 34: RefW=0.000e+00 持续全 epoch
epoch 34 RefAttnKL=0.000e+00
epoch 35 start: RefW=0.000e+00
```

结论：

```text
checkpoint-35 是第一组 28/29 pulse 后、第二组 36/37 pulse 前的非 pulse 观察点。
结果从 checkpoint-34 的 80.4920 继续回落到 80.4180，说明 checkpoint-33 的 80.5520 没有稳定延续。
到 checkpoint-35 为止，第一组 3.0e-4 pulse 未超过 80.5980 baseline；下一核心判据是 36/37 的 3.5e-4 pulse。
```

运行状态：

```text
训练继续运行，进入 epoch 35，RefW 仍为 0。
当前最佳仍为 checkpoint-33: Top-1 80.5520，尚未超过 80.5980 baseline。
```

## 2026-07-09 21:52 UTC checkpoint-36

epoch 35 已完成，并生成第二十六个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-36.pth.tar
checkpoint_count: 26
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=35 updates=2496 avg_step_time=0.232551s samples_per_step=512 samples_per_sec=2201.66
Test: [distributed-summary]  Time: 10.492s  Loss: 0.8311  Acc@1: 80.5040  Acc@5: 95.3420  Samples: 50000
```

对比：

```text
checkpoint-33 Top-1: 80.5520
checkpoint-35 Top-1: 80.4180
checkpoint-36 Top-1: 80.5040
相对 checkpoint-35: +0.0860
相对当前最佳 checkpoint-33: -0.0480
相对 baseline 80.5980: -0.0940
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 35: RefW=0.000e+00 持续全 epoch
epoch 36 start: RefW=3.500e-04
epoch 36 start: RefAttnKL=3.709e+01
epoch 36 update 50: RefW=3.500e-04, RefAttnKL=3.004e+01
```

结论：

```text
checkpoint-36 是第二组 36/37 pulse 开启前的最后一个 full-val，主要反映 epoch 35 非 pulse 训练结果。
结果从 checkpoint-35 的 80.4180 回升到 80.5040，但仍低于 checkpoint-33 的 80.5520，也低于 80.5980 baseline。
保存 checkpoint-36 后，epoch 36 已按方案 C 进入第二组 3.5e-4 sparse pulse，RefW 和 RefAttnKL 均已非零。
下一关键判据是 checkpoint-37/38：checkpoint-37 反映 epoch 36 pulse 后效果，checkpoint-38 反映 epoch 36/37 双 pulse 后效果。
```

运行状态：

```text
训练继续运行，进入 epoch 36，RefW 已切到 3.500e-04。
当前最佳仍为 checkpoint-33: Top-1 80.5520，尚未超过 80.5980 baseline。
```

## 2026-07-09 22:05 UTC checkpoint-37

epoch 36 已完成，并生成第二十七个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-37.pth.tar
checkpoint_count: 27
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=36 updates=2496 avg_step_time=0.323823s samples_per_step=512 samples_per_sec=1581.11
Test: [distributed-summary]  Time: 10.760s  Loss: 0.8341  Acc@1: 80.5620  Acc@5: 95.3280  Samples: 50000
```

对比：

```text
checkpoint-33 Top-1: 80.5520
checkpoint-36 Top-1: 80.5040
checkpoint-37 Top-1: 80.5620
相对 checkpoint-36: +0.0580
相对此前本 run 最佳 checkpoint-33: +0.0100
相对 baseline 80.5980: -0.0360
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 36: RefW=3.500e-04 持续全 epoch
epoch 36 RefAttnKL: 多数日志点约 1e+01 到 6e+01，持续非零
epoch 37 start: RefW=3.500e-04
epoch 37 start: RefAttnKL=3.097e+01
```

结论：

```text
checkpoint-37 是第二组 3.5e-4 sparse pulse 中 epoch 36 后的第一轮 full-val。
结果从 checkpoint-36 的 80.5040 回升到 80.5620，刷新本 run 最佳，但仍低于 80.5980 baseline 0.0360。
第二组 pulse 的第一轮效果强于第一组 28/29 的首轮结果；下一关键判据是 checkpoint-38，即 epoch 36/37 双 pulse 后是否能越过 baseline。
```

运行状态：

```text
训练继续运行，进入 epoch 37，RefW 仍为 3.500e-04。
当前最佳更新为 checkpoint-37: Top-1 80.5620，尚未超过 80.5980 baseline。
```

## 2026-07-09 22:19 UTC checkpoint-38

epoch 37 已完成，并生成第二十八个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-38.pth.tar
checkpoint_count: 28
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=37 updates=2496 avg_step_time=0.323919s samples_per_step=512 samples_per_sec=1580.64
Test: [distributed-summary]  Time: 10.522s  Loss: 0.8369  Acc@1: 80.4660  Acc@5: 95.3480  Samples: 50000
```

对比：

```text
checkpoint-36 Top-1: 80.5040
checkpoint-37 Top-1: 80.5620
checkpoint-38 Top-1: 80.4660
相对 checkpoint-37: -0.0960
相对当前最佳 checkpoint-37: -0.0960
相对 baseline 80.5980: -0.1320
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 37: RefW=3.500e-04 持续全 epoch
epoch 37 RefAttnKL: 多数日志点约 1e+01 到 5e+01，持续非零
epoch 38 start: RefW=0.000e+00
epoch 38 start: RefAttnKL=0.000e+00
```

结论：

```text
checkpoint-38 是第二组 36/37 双 pulse 后的关键 full-val。
结果从 checkpoint-37 的 80.5620 回落到 80.4660，没有超过 80.5980 baseline，也没有保持 checkpoint-37 的高点。
这说明第二组 3.5e-4 双 pulse 目前只产生单点提升，没有形成稳定越线；需要继续观察 epoch 38/39 非 pulse 恢复段是否有延迟收益。
```

运行状态：

```text
训练继续运行，进入 epoch 38，RefW 已按计划回到 0。
当前最佳仍为 checkpoint-37: Top-1 80.5620，尚未超过 80.5980 baseline。
```

## 2026-07-09 22:29 UTC checkpoint-39

epoch 38 已完成，并生成第二十九个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-39.pth.tar
checkpoint_count: 29
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=38 updates=2496 avg_step_time=0.233442s samples_per_step=512 samples_per_sec=2193.27
Test: [distributed-summary]  Time: 10.444s  Loss: 0.8366  Acc@1: 80.4860  Acc@5: 95.3140  Samples: 50000
```

对比：

```text
checkpoint-37 Top-1: 80.5620
checkpoint-38 Top-1: 80.4660
checkpoint-39 Top-1: 80.4860
相对 checkpoint-38: +0.0200
相对当前最佳 checkpoint-37: -0.0760
相对 baseline 80.5980: -0.1120
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 38: RefW=0.000e+00 持续全 epoch
epoch 38 RefAttnKL=0.000e+00
epoch 39 start: RefW=0.000e+00
```

结论：

```text
checkpoint-39 是第二组 36/37 pulse 后的第一个非 pulse 恢复点。
结果从 checkpoint-38 的 80.4660 小幅回升到 80.4860，但没有回到 checkpoint-37 的 80.5620，也没有超过 80.5980 baseline。
当前看，第二组 pulse 仍只有 checkpoint-37 单点接近 baseline；继续观察 checkpoint-40 的恢复段。
```

运行状态：

```text
训练继续运行，进入 epoch 39，RefW 仍为 0。
当前最佳仍为 checkpoint-37: Top-1 80.5620，尚未超过 80.5980 baseline。
```

## 2026-07-09 22:38 UTC checkpoint-40

epoch 39 已完成，并生成第三十个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-40.pth.tar
checkpoint_count: 30
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=39 updates=2496 avg_step_time=0.232441s samples_per_step=512 samples_per_sec=2202.71
Test: [distributed-summary]  Time: 10.457s  Loss: 0.8366  Acc@1: 80.5800  Acc@5: 95.3520  Samples: 50000
```

对比：

```text
checkpoint-37 Top-1: 80.5620
checkpoint-38 Top-1: 80.4660
checkpoint-39 Top-1: 80.4860
checkpoint-40 Top-1: 80.5800
相对 checkpoint-39: +0.0940
相对此前本 run 最佳 checkpoint-37: +0.0180
相对 baseline 80.5980: -0.0180
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 39: RefW=0.000e+00 持续全 epoch
epoch 39 RefAttnKL=0.000e+00
epoch 40 start: RefW=0.000e+00
```

结论：

```text
checkpoint-40 是第二组 36/37 pulse 后的第二个非 pulse 恢复点。
结果从 checkpoint-39 的 80.4860 回升到 80.5800，刷新本 run 最佳，距离 80.5980 baseline 只差 0.0180。
这说明第二组 pulse 后存在延迟恢复趋势，但仍未越过 baseline；继续观察 checkpoint-41/42 是否能够稳定过线。
```

运行状态：

```text
训练继续运行，进入 epoch 40，RefW 仍为 0。
当前最佳更新为 checkpoint-40: Top-1 80.5800，尚未超过 80.5980 baseline。
```

## 2026-07-09 22:48 UTC checkpoint-41

epoch 40 已完成，并生成第三十一个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-41.pth.tar
checkpoint_count: 31
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=40 updates=2496 avg_step_time=0.232492s samples_per_step=512 samples_per_sec=2202.23
Test: [distributed-summary]  Time: 10.440s  Loss: 0.8352  Acc@1: 80.4740  Acc@5: 95.3500  Samples: 50000
```

对比：

```text
checkpoint-40 Top-1: 80.5800
checkpoint-41 Top-1: 80.4740
相对 checkpoint-40: -0.1060
相对当前最佳 checkpoint-40: -0.1060
相对 baseline 80.5980: -0.1240
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 40: RefW=0.000e+00 持续全 epoch
epoch 40 RefAttnKL=0.000e+00
epoch 41 start: RefW=0.000e+00
```

结论：

```text
checkpoint-41 是第二组 36/37 pulse 后的第三个非 pulse 恢复点。
结果从 checkpoint-40 的 80.5800 回落到 80.4740，没有延续接近 baseline 的趋势。
当前最佳仍为 checkpoint-40，但仍低于 80.5980 baseline 0.0180；继续观察 checkpoint-42/43，下一组 pulse 在 epoch 44/45。
```

运行状态：

```text
训练继续运行，进入 epoch 41，RefW 仍为 0。
当前最佳仍为 checkpoint-40: Top-1 80.5800，尚未超过 80.5980 baseline。
```

## 2026-07-09 22:59 UTC checkpoint-42

epoch 41 已完成，并生成第三十二个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-42.pth.tar
checkpoint_count: 32
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=41 updates=2496 avg_step_time=0.233459s samples_per_step=512 samples_per_sec=2193.10
Test: [distributed-summary]  Time: 10.629s  Loss: 0.8353  Acc@1: 80.4940  Acc@5: 95.2620  Samples: 50000
```

对比：

```text
checkpoint-40 Top-1: 80.5800
checkpoint-41 Top-1: 80.4740
checkpoint-42 Top-1: 80.4940
相对 checkpoint-41: +0.0200
相对当前最佳 checkpoint-40: -0.0860
相对 baseline 80.5980: -0.1040
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 41: RefW=0.000e+00 持续全 epoch
epoch 41 RefAttnKL=0.000e+00
epoch 42 start: RefW=0.000e+00
```

结论：

```text
checkpoint-42 是第二组 36/37 pulse 后的第四个非 pulse 恢复点。
结果从 checkpoint-41 的 80.4740 小幅回升到 80.4940，但仍明显低于 checkpoint-40 的 80.5800 和 80.5980 baseline。
第二组 pulse 后的最好点仍是 checkpoint-40，尚未越线；下一组 3.5e-4 pulse 将在 epoch 44/45 开启。
```

运行状态：

```text
训练继续运行，进入 epoch 42，RefW 仍为 0。
当前最佳仍为 checkpoint-40: Top-1 80.5800，尚未超过 80.5980 baseline。
```

## 2026-07-09 23:09 UTC checkpoint-43

epoch 42 已完成，并生成第三十三个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-43.pth.tar
checkpoint_count: 33
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=42 updates=2496 avg_step_time=0.233420s samples_per_step=512 samples_per_sec=2193.47
Test: [distributed-summary]  Time: 10.543s  Loss: 0.8376  Acc@1: 80.4560  Acc@5: 95.3520  Samples: 50000
```

对比：

```text
checkpoint-40 Top-1: 80.5800
checkpoint-42 Top-1: 80.4940
checkpoint-43 Top-1: 80.4560
相对 checkpoint-42: -0.0380
相对当前最佳 checkpoint-40: -0.1240
相对 baseline 80.5980: -0.1420
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 42: RefW=0.000e+00 持续全 epoch
epoch 42 RefAttnKL=0.000e+00
epoch 43 start: RefW=0.000e+00
```

结论：

```text
checkpoint-43 是第三组 44/45 pulse 前的非 pulse 观察点。
结果从 checkpoint-42 的 80.4940 回落到 80.4560，没有延续 checkpoint-40 的接近 baseline 状态。
到 checkpoint-43 为止，本 run 最佳仍为 checkpoint-40 的 80.5800，距离 baseline 80.5980 仍差 0.0180；下一关键节点是 epoch 44/45 的第三组 3.5e-4 pulse。
```

运行状态：

```text
训练继续运行，进入 epoch 43，RefW 仍为 0。
当前最佳仍为 checkpoint-40: Top-1 80.5800，尚未超过 80.5980 baseline。
```

## 2026-07-09 23:18 UTC checkpoint-44

epoch 43 已完成，并生成第三十四个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-44.pth.tar
checkpoint_count: 34
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=43 updates=2496 avg_step_time=0.232912s samples_per_step=512 samples_per_sec=2198.26
Test: [distributed-summary]  Time: 10.464s  Loss: 0.8315  Acc@1: 80.5520  Acc@5: 95.3700  Samples: 50000
```

对比：

```text
checkpoint-40 Top-1: 80.5800
checkpoint-43 Top-1: 80.4560
checkpoint-44 Top-1: 80.5520
相对 checkpoint-43: +0.0960
相对当前最佳 checkpoint-40: -0.0280
相对 baseline 80.5980: -0.0460
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 43: RefW=0.000e+00 持续全 epoch
epoch 44 start: RefW=3.500e-04
epoch 44 start: RefAttnKL=3.612e+01
epoch 44 update 50: RefW=3.500e-04, RefAttnKL=3.742e+01
```

结论：

```text
checkpoint-44 是第三组 44/45 pulse 开启前的最后一个 full-val，主要反映 epoch 43 非 pulse 训练结果。
结果从 checkpoint-43 的 80.4560 回升到 80.5520，但仍低于 checkpoint-40 的 80.5800 和 baseline 80.5980。
保存 checkpoint-44 后，epoch 44 已按方案 C 进入第三组 3.5e-4 sparse pulse，RefW 和 RefAttnKL 均已非零。
下一关键判据是 checkpoint-45/46：checkpoint-45 反映 epoch 44 pulse 后效果，checkpoint-46 反映 epoch 44/45 双 pulse 后效果。
```

运行状态：

```text
训练继续运行，进入 epoch 44，RefW 已切到 3.500e-04。
当前最佳仍为 checkpoint-40: Top-1 80.5800，尚未超过 80.5980 baseline。
```

## 2026-07-09 23:32 UTC checkpoint-45

epoch 44 已完成，并生成第三十五个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-45.pth.tar
checkpoint_count: 35
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=44 updates=2496 avg_step_time=0.323943s samples_per_step=512 samples_per_sec=1580.52
Test: [distributed-summary]  Time: 10.638s  Loss: 0.8345  Acc@1: 80.5360  Acc@5: 95.3600  Samples: 50000
```

对比：

```text
checkpoint-40 Top-1: 80.5800
checkpoint-44 Top-1: 80.5520
checkpoint-45 Top-1: 80.5360
相对 checkpoint-44: -0.0160
相对当前最佳 checkpoint-40: -0.0440
相对 baseline 80.5980: -0.0620
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 44: RefW=3.500e-04 持续全 epoch
epoch 44 RefAttnKL: 多数日志点持续非零
epoch 45 start: RefW=3.500e-04
epoch 45 start: RefAttnKL=1.441e+01
```

结论：

```text
checkpoint-45 是第三组 3.5e-4 sparse pulse 中 epoch 44 后的第一轮 full-val。
结果从 checkpoint-44 的 80.5520 小幅回落到 80.5360，没有超过 checkpoint-40 的 80.5800，也没有超过 80.5980 baseline。
第三组 pulse 第一轮暂未显示超过前两组的收益；下一关键判据是 checkpoint-46，即 epoch 44/45 双 pulse 后效果。
```

运行状态：

```text
训练继续运行，进入 epoch 45，RefW 仍为 3.500e-04。
当前最佳仍为 checkpoint-40: Top-1 80.5800，尚未超过 80.5980 baseline。
```

## 2026-07-09 23:46 UTC checkpoint-46

epoch 45 已完成，并生成第三十六个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-46.pth.tar
checkpoint_count: 36
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=45 updates=2496 avg_step_time=0.323894s samples_per_step=512 samples_per_sec=1580.77
Test: [distributed-summary]  Time: 10.597s  Loss: 0.8374  Acc@1: 80.4980  Acc@5: 95.3400  Samples: 50000
```

对比：

```text
checkpoint-40 Top-1: 80.5800
checkpoint-44 Top-1: 80.5520
checkpoint-45 Top-1: 80.5360
checkpoint-46 Top-1: 80.4980
相对 checkpoint-45: -0.0380
相对当前最佳 checkpoint-40: -0.0820
相对 baseline 80.5980: -0.1000
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 45: RefW=3.500e-04 持续全 epoch
epoch 45 RefAttnKL: 持续非零
epoch 46 start: RefW=0.000e+00
epoch 46 start: RefAttnKL=0.000e+00
```

结论：

```text
checkpoint-46 是第三组 44/45 双 pulse 后的关键 full-val。
结果从 checkpoint-45 的 80.5360 继续回落到 80.4980，没有超过 checkpoint-40 的 80.5800，也没有超过 80.5980 baseline。
第三组 3.5e-4 双 pulse 目前没有形成正收益；继续观察 epoch 46/47 非 pulse 恢复段是否有延迟回升。
```

运行状态：

```text
训练继续运行，进入 epoch 46，RefW 已按计划回到 0。
当前最佳仍为 checkpoint-40: Top-1 80.5800，尚未超过 80.5980 baseline。
```

## 2026-07-09 23:55 UTC checkpoint-47

epoch 46 已完成，并生成第三十七个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-47.pth.tar
checkpoint_count: 37
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=46 updates=2496 avg_step_time=0.231734s samples_per_step=512 samples_per_sec=2209.43
Test: [distributed-summary]  Time: 10.614s  Loss: 0.8397  Acc@1: 80.4760  Acc@5: 95.3260  Samples: 50000
```

对比：

```text
checkpoint-40 Top-1: 80.5800
checkpoint-46 Top-1: 80.4980
checkpoint-47 Top-1: 80.4760
相对 checkpoint-46: -0.0220
相对当前最佳 checkpoint-40: -0.1040
相对 baseline 80.5980: -0.1220
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 46: RefW=0.000e+00 持续全 epoch
epoch 46 RefAttnKL=0.000e+00
epoch 47 start: RefW=0.000e+00
```

结论：

```text
checkpoint-47 是第三组 44/45 pulse 后的第一个非 pulse 恢复点。
结果从 checkpoint-46 的 80.4980 继续回落到 80.4760，没有出现延迟恢复。
到 checkpoint-47 为止，本 run 最佳仍为 checkpoint-40 的 80.5800，仍低于 80.5980 baseline 0.0180。
```

运行状态：

```text
训练继续运行，进入 epoch 47，RefW 仍为 0。
当前最佳仍为 checkpoint-40: Top-1 80.5800，尚未超过 80.5980 baseline。
```

## 2026-07-10 00:06 UTC checkpoint-48

epoch 47 已完成，并生成第三十八个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-48.pth.tar
checkpoint_count: 38
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=47 updates=2496 avg_step_time=0.231820s samples_per_step=512 samples_per_sec=2208.61
Test: [distributed-summary]  Time: 10.440s  Loss: 0.8341  Acc@1: 80.5540  Acc@5: 95.4080  Samples: 50000
```

对比：

```text
checkpoint-40 Top-1: 80.5800
checkpoint-46 Top-1: 80.4980
checkpoint-47 Top-1: 80.4760
checkpoint-48 Top-1: 80.5540
相对 checkpoint-47: +0.0780
相对当前最佳 checkpoint-40: -0.0260
相对 baseline 80.5980: -0.0440
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 47: RefW=0.000e+00 持续全 epoch
epoch 47 RefAttnKL=0.000e+00
epoch 48 start: RefW=0.000e+00
```

结论：

```text
checkpoint-48 是第三组 44/45 pulse 后的第二个非 pulse 恢复点。
结果从 checkpoint-47 的 80.4760 回升到 80.5540，但仍低于 checkpoint-40 的 80.5800 和 baseline 80.5980。
第三组 pulse 后有一定恢复，但目前仍未越线；继续观察 checkpoint-49/50，之后还有 epoch 52/53 最后一组 3.0e-4 pulse。
```

运行状态：

```text
训练继续运行，进入 epoch 48，RefW 仍为 0。
当前最佳仍为 checkpoint-40: Top-1 80.5800，尚未超过 80.5980 baseline。
```

## 2026-07-10 00:15 UTC checkpoint-49

epoch 48 已完成，并生成第三十九个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-49.pth.tar
checkpoint_count: 39
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=48 updates=2496 avg_step_time=0.232611s samples_per_step=512 samples_per_sec=2201.10
Test: [distributed-summary]  Time: 10.516s  Loss: 0.8338  Acc@1: 80.5000  Acc@5: 95.3600  Samples: 50000
```

对比：

```text
checkpoint-40 Top-1: 80.5800
checkpoint-48 Top-1: 80.5540
checkpoint-49 Top-1: 80.5000
相对 checkpoint-48: -0.0540
相对当前最佳 checkpoint-40: -0.0800
相对 baseline 80.5980: -0.0980
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 48: RefW=0.000e+00 持续全 epoch
epoch 48 RefAttnKL=0.000e+00
epoch 49 start: RefW=0.000e+00
```

结论：

```text
checkpoint-49 是第三组 44/45 pulse 后的第三个非 pulse 恢复点。
结果从 checkpoint-48 的 80.5540 回落到 80.5000，没有形成稳定恢复。
当前最佳仍为 checkpoint-40 的 80.5800，尚未超过 80.5980 baseline；继续观察 checkpoint-50/51。
```

运行状态：

```text
训练继续运行，进入 epoch 49，RefW 仍为 0。
当前最佳仍为 checkpoint-40: Top-1 80.5800，尚未超过 80.5980 baseline。
```

## 2026-07-10 00:26 UTC checkpoint-50

epoch 49 已完成，并生成第四十个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-50.pth.tar
checkpoint_count: 40
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=49 updates=2496 avg_step_time=0.232895s samples_per_step=512 samples_per_sec=2198.42
Test: [distributed-summary]  Time: 10.458s  Loss: 0.8315  Acc@1: 80.5940  Acc@5: 95.4040  Samples: 50000
```

对比：

```text
checkpoint-40 Top-1: 80.5800
checkpoint-48 Top-1: 80.5540
checkpoint-49 Top-1: 80.5000
checkpoint-50 Top-1: 80.5940
相对 checkpoint-49: +0.0940
相对此前本 run 最佳 checkpoint-40: +0.0140
相对 baseline 80.5980: -0.0040
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 49: RefW=0.000e+00 持续全 epoch
epoch 49 RefAttnKL=0.000e+00
epoch 50 start: RefW=0.000e+00
```

结论：

```text
checkpoint-50 是第三组 44/45 pulse 后的第四个非 pulse 恢复点。
结果从 checkpoint-49 的 80.5000 回升到 80.5940，刷新本 run 最佳，距离 80.5980 baseline 只差 0.0040。
这仍不算超过 baseline；继续观察 checkpoint-51/52，后面还有 epoch 52/53 最后一组 3.0e-4 pulse。
```

运行状态：

```text
训练继续运行，进入 epoch 50，RefW 仍为 0。
当前最佳更新为 checkpoint-50: Top-1 80.5940，尚未超过 80.5980 baseline。
```

## 2026-07-10 00:35 UTC checkpoint-51

epoch 50 已完成，并生成第四十一个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-51.pth.tar
checkpoint_count: 41
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=50 updates=2496 avg_step_time=0.231980s samples_per_step=512 samples_per_sec=2207.09
Test: [distributed-summary]  Time: 10.444s  Loss: 0.8341  Acc@1: 80.5120  Acc@5: 95.3260  Samples: 50000
```

对比：

```text
checkpoint-50 Top-1: 80.5940
checkpoint-51 Top-1: 80.5120
相对 checkpoint-50: -0.0820
相对当前最佳 checkpoint-50: -0.0820
相对 baseline 80.5980: -0.0860
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 50: RefW=0.000e+00 持续全 epoch
epoch 50 RefAttnKL=0.000e+00
epoch 51 start: RefW=0.000e+00
```

结论：

```text
checkpoint-51 是 epoch 52/53 最后一组 pulse 前的非 pulse 观察点。
结果从 checkpoint-50 的 80.5940 回落到 80.5120，没有超过 80.5980 baseline。
当前最佳仍为 checkpoint-50，距离 baseline 只差 0.0040；下一关键节点是 checkpoint-52 后 epoch 52 是否按计划进入最后一组 3.0e-4 pulse。
```

运行状态：

```text
训练继续运行，进入 epoch 51，RefW 仍为 0。
当前最佳仍为 checkpoint-50: Top-1 80.5940，尚未超过 80.5980 baseline。
```

## 2026-07-10 00:45 UTC checkpoint-52

epoch 51 已完成，并生成第四十二个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-52.pth.tar
checkpoint_count: 42
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=51 updates=2496 avg_step_time=0.231802s samples_per_step=512 samples_per_sec=2208.78
Test: [distributed-summary]  Time: 10.604s  Loss: 0.8386  Acc@1: 80.5340  Acc@5: 95.3760  Samples: 50000
```

对比：

```text
checkpoint-50 Top-1: 80.5940
checkpoint-51 Top-1: 80.5120
checkpoint-52 Top-1: 80.5340
相对 checkpoint-51: +0.0220
相对当前最佳 checkpoint-50: -0.0600
相对 baseline 80.5980: -0.0640
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 51: RefW=0.000e+00 持续全 epoch
epoch 52 start: RefW=3.000e-04
epoch 52 start: RefAttnKL=5.321e+01
epoch 52 update 50: RefW=3.000e-04, RefAttnKL=3.659e+01
```

结论：

```text
checkpoint-52 是最后一组 52/53 pulse 开启前的最后一个 full-val，主要反映 epoch 51 非 pulse 训练结果。
结果从 checkpoint-51 的 80.5120 小幅回升到 80.5340，但仍低于 checkpoint-50 的 80.5940 和 baseline 80.5980。
保存 checkpoint-52 后，epoch 52 已按方案 C 进入最后一组 3.0e-4 sparse pulse，RefW 和 RefAttnKL 均已非零。
下一关键判据是 checkpoint-53/54：checkpoint-53 反映 epoch 52 pulse 后效果，checkpoint-54 反映 epoch 52/53 双 pulse 后效果。
```

运行状态：

```text
训练继续运行，进入 epoch 52，RefW 已切到 3.000e-04。
当前最佳仍为 checkpoint-50: Top-1 80.5940，尚未超过 80.5980 baseline。
```

## 2026-07-10 00:59 UTC checkpoint-53

epoch 52 已完成，并生成第四十三个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-53.pth.tar
checkpoint_count: 43
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=52 updates=2496 avg_step_time=0.322432s samples_per_step=512 samples_per_sec=1587.93
Test: [distributed-summary]  Time: 10.700s  Loss: 0.8361  Acc@1: 80.5480  Acc@5: 95.3700  Samples: 50000
```

对比：

```text
checkpoint-50 Top-1: 80.5940
checkpoint-52 Top-1: 80.5340
checkpoint-53 Top-1: 80.5480
相对 checkpoint-52: +0.0140
相对当前最佳 checkpoint-50: -0.0460
相对 baseline 80.5980: -0.0500
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 52: RefW=3.000e-04 持续全 epoch
epoch 52 RefAttnKL: 持续非零
epoch 53 start: RefW=3.000e-04
epoch 53 start: RefAttnKL=2.282e+01
```

结论：

```text
checkpoint-53 是最后一组 3.0e-4 sparse pulse 中 epoch 52 后的第一轮 full-val。
结果从 checkpoint-52 的 80.5340 小幅回升到 80.5480，但仍低于 checkpoint-50 的 80.5940 和 80.5980 baseline。
下一关键判据是 checkpoint-54，即 epoch 52/53 双 pulse 后效果。
```

运行状态：

```text
训练继续运行，进入 epoch 53，RefW 仍为 3.000e-04。
当前最佳仍为 checkpoint-50: Top-1 80.5940，尚未超过 80.5980 baseline。
```

## 2026-07-10 01:13 UTC checkpoint-54

epoch 53 已完成，并生成第四十四个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-54.pth.tar
checkpoint_count: 44
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=53 updates=2496 avg_step_time=0.323183s samples_per_step=512 samples_per_sec=1584.24
Test: [distributed-summary]  Time: 10.563s  Loss: 0.8340  Acc@1: 80.6820  Acc@5: 95.3880  Samples: 50000
```

对比：

```text
checkpoint-50 Top-1: 80.5940
checkpoint-52 Top-1: 80.5340
checkpoint-53 Top-1: 80.5480
checkpoint-54 Top-1: 80.6820
相对 checkpoint-53: +0.1340
相对此前本 run 最佳 checkpoint-50: +0.0880
相对 baseline 80.5980: +0.0840
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 53: RefW=3.000e-04 持续全 epoch
epoch 53 RefAttnKL: 持续非零
epoch 54 start: RefW=0.000e+00
epoch 54 start: RefAttnKL=0.000e+00
```

结论：

```text
checkpoint-54 是最后一组 52/53 双 pulse 后的关键 full-val。
结果从 checkpoint-53 的 80.5480 跳升到 80.6820，首次超过 80.5980 baseline，超出 +0.0840。
这满足“超过 baseline”的单点成功判据，但尚未达到 81.0；训练仍需继续跑到 checkpoint-60 并做最终审计。
```

运行状态：

```text
训练继续运行，进入 epoch 54，RefW 已按计划回到 0。
当前最佳更新为 checkpoint-54: Top-1 80.6820，已超过 80.5980 baseline，但未达到 81.0。
```

## 2026-07-10 01:22 UTC checkpoint-55

epoch 54 已完成，并生成第四十五个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-55.pth.tar
checkpoint_count: 45
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=54 updates=2496 avg_step_time=0.231967s samples_per_step=512 samples_per_sec=2207.21
Test: [distributed-summary]  Time: 10.525s  Loss: 0.8360  Acc@1: 80.5120  Acc@5: 95.3220  Samples: 50000
```

对比：

```text
checkpoint-54 Top-1: 80.6820
checkpoint-55 Top-1: 80.5120
相对 checkpoint-54: -0.1700
相对当前最佳 checkpoint-54: -0.1700
相对 baseline 80.5980: -0.0860
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 54: RefW=0.000e+00 持续全 epoch
epoch 54 RefAttnKL=0.000e+00
epoch 55 start: RefW=0.000e+00
```

结论：

```text
checkpoint-55 是最后一组 52/53 pulse 后的第一个非 pulse 恢复点。
结果从 checkpoint-54 的 80.6820 回落到 80.5120，说明 checkpoint-54 是当前明确高点。
当前最佳仍为 checkpoint-54，已经超过 80.5980 baseline 0.0840，但尚未达到 81.0；继续跑到 checkpoint-60 做最终审计。
```

运行状态：

```text
训练继续运行，进入 epoch 55，RefW 仍为 0。
当前最佳仍为 checkpoint-54: Top-1 80.6820，已超过 80.5980 baseline，但未达到 81.0。
```

## 2026-07-10 01:32 UTC checkpoint-56

epoch 55 已完成，并生成第四十六个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-56.pth.tar
checkpoint_count: 46
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=55 updates=2496 avg_step_time=0.232045s samples_per_step=512 samples_per_sec=2206.47
Test: [distributed-summary]  Time: 10.468s  Loss: 0.8337  Acc@1: 80.5280  Acc@5: 95.4160  Samples: 50000
```

对比：

```text
checkpoint-54 Top-1: 80.6820
checkpoint-55 Top-1: 80.5120
checkpoint-56 Top-1: 80.5280
相对 checkpoint-55: +0.0160
相对当前最佳 checkpoint-54: -0.1540
相对 baseline 80.5980: -0.0700
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 55: RefW=0.000e+00 持续全 epoch
epoch 55 RefAttnKL=0.000e+00
epoch 56 start: RefW=0.000e+00
```

结论：

```text
checkpoint-56 是最后一组 52/53 pulse 后的第二个非 pulse 恢复点。
结果从 checkpoint-55 的 80.5120 小幅回升到 80.5280，但仍低于 checkpoint-54 的 80.6820 和 baseline 80.5980。
当前最佳仍为 checkpoint-54，已经超过 baseline 0.0840；继续跑到 checkpoint-60 做最终审计。
```

运行状态：

```text
训练继续运行，进入 epoch 56，RefW 仍为 0。
当前最佳仍为 checkpoint-54: Top-1 80.6820，已超过 80.5980 baseline，但未达到 81.0。
```

## 2026-07-10 01:42 UTC checkpoint-57

epoch 56 已完成，并生成第四十七个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-57.pth.tar
checkpoint_count: 47
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=56 updates=2496 avg_step_time=0.231995s samples_per_step=512 samples_per_sec=2206.94
Test: [distributed-summary]  Time: 10.442s  Loss: 0.8302  Acc@1: 80.5640  Acc@5: 95.3480  Samples: 50000
```

对比：

```text
checkpoint-54 Top-1: 80.6820
checkpoint-56 Top-1: 80.5280
checkpoint-57 Top-1: 80.5640
相对 checkpoint-56: +0.0360
相对当前最佳 checkpoint-54: -0.1180
相对 baseline 80.5980: -0.0340
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 56: RefW=0.000e+00 持续全 epoch
epoch 56 RefAttnKL=0.000e+00
epoch 57 start: RefW=0.000e+00
```

结论：

```text
checkpoint-57 是最后一组 52/53 pulse 后的第四个非 pulse 恢复点。
结果从 checkpoint-56 的 80.5280 回升到 80.5640，但仍低于 checkpoint-54 的 80.6820 和 baseline 80.5980。
当前最佳仍为 checkpoint-54，已经超过 baseline 0.0840；继续跑到 checkpoint-60 做最终审计。
```

运行状态：

```text
训练继续运行，进入 epoch 57，RefW 仍为 0。
当前最佳仍为 checkpoint-54: Top-1 80.6820，已超过 80.5980 baseline，但未达到 81.0。
```

## 2026-07-10 01:52 UTC checkpoint-58

epoch 57 已完成，并生成第四十八个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-58.pth.tar
checkpoint_count: 48
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=57 updates=2496 avg_step_time=0.232250s samples_per_step=512 samples_per_sec=2204.52
Test: [distributed-summary]  Time: 10.456s  Loss: 0.8365  Acc@1: 80.5140  Acc@5: 95.3960  Samples: 50000
```

对比：

```text
checkpoint-54 Top-1: 80.6820
checkpoint-57 Top-1: 80.5640
checkpoint-58 Top-1: 80.5140
相对 checkpoint-57: -0.0500
相对当前最佳 checkpoint-54: -0.1680
相对 baseline 80.5980: -0.0840
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 57: RefW=0.000e+00 持续全 epoch
epoch 57 RefAttnKL=0.000e+00
epoch 58 start: RefW=0.000e+00
```

结论：

```text
checkpoint-58 是后段非 pulse 收尾点。
结果从 checkpoint-57 的 80.5640 回落到 80.5140，未超过 checkpoint-54，也未达到 81.0。
当前最佳仍为 checkpoint-54: 80.6820；继续等待 checkpoint-59/60 后进行最终审计。
```

运行状态：

```text
训练继续运行，进入 epoch 58，RefW 仍为 0。
当前最佳仍为 checkpoint-54: Top-1 80.6820，已超过 80.5980 baseline，但未达到 81.0。
```

## 2026-07-10 02:02 UTC checkpoint-59

epoch 58 已完成，并生成第四十九个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-59.pth.tar
checkpoint_count: 49
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=58 updates=2496 avg_step_time=0.232266s samples_per_step=512 samples_per_sec=2204.37
Test: [distributed-summary]  Time: 10.521s  Loss: 0.8317  Acc@1: 80.5380  Acc@5: 95.3300  Samples: 50000
```

对比：

```text
checkpoint-54 Top-1: 80.6820
checkpoint-58 Top-1: 80.5140
checkpoint-59 Top-1: 80.5380
相对 checkpoint-58: +0.0240
相对当前最佳 checkpoint-54: -0.1440
相对 baseline 80.5980: -0.0600
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 58: RefW=0.000e+00 持续全 epoch
epoch 58 RefAttnKL=0.000e+00
epoch 59 start: RefW=0.000e+00
```

结论：

```text
checkpoint-59 是最后阶段非 pulse 收尾点。
结果从 checkpoint-58 的 80.5140 小幅回升到 80.5380，但仍低于 checkpoint-54 的 80.6820 和 80.5980 baseline。
当前最佳仍为 checkpoint-54；下一步等待 checkpoint-60 后做最终审计。
```

运行状态：

```text
训练继续运行，进入 epoch 59，RefW 仍为 0。
当前最佳仍为 checkpoint-54: Top-1 80.6820，已超过 80.5980 baseline，但未达到 81.0。
```

## 2026-07-10 02:12 UTC checkpoint-60

epoch 59 已完成，并生成第五十个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709/checkpoint-60.pth.tar
checkpoint_count: 50
last.pth.tar: 同步更新
```

full ImageNet validation：

```text
TrainSummary: epoch=59 updates=2496 avg_step_time=0.233131s samples_per_step=512 samples_per_sec=2196.19
Test: [distributed-summary]  Time: 10.519s  Loss: 0.8334  Acc@1: 80.5620  Acc@5: 95.3620  Samples: 50000
wall_seconds=31670
```

对比：

```text
checkpoint-54 Top-1: 80.6820
checkpoint-59 Top-1: 80.5380
checkpoint-60 Top-1: 80.5620
相对 checkpoint-59: +0.0240
相对当前最佳 checkpoint-54: -0.1200
相对 baseline 80.5980: -0.0360
81.0 target: 未达到
```

RefW / pulse 状态：

```text
epoch 59: RefW=0.000e+00 持续全 epoch
epoch 59 RefAttnKL=0.000e+00
训练结束，未再进入下一 epoch。
```

结论：

```text
checkpoint-60 是 50 个 resumed epoch 的最后一个 full-val。
最终 checkpoint 未超过 baseline；整轮最佳仍是 checkpoint-54: Top-1 80.6820。
```

## 2026-07-10 final audit

目标复述：

```text
按方案 C 从 checkpoint-10 启动 OFQ public-family 50 个 resumed epoch 长跑；
持续轮询 checkpoint / full-val / RefW 生效情况；
更新中文进度文档；
最终审计是否超过 80.5980 baseline 或达到 81.0。
```

prompt-to-artifact checklist：

```text
1. 方案文档存在：
   /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_sparse_pulse_prevstep_refkl_c_50epoch_goal_20260709.md

2. 启动脚本存在且参数匹配方案 C：
   /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709.sh

3. 训练输出目录存在：
   /tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709

4. resume 起点：
   /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar

5. 主链路配置：
   wq_mode=statsq
   aq_mode=lsq
   qk_reparam=true
   qk_reparam_type=0
   kd_hard_and_soft=0
   teacher_soft_temperature=2.75
   no_resume_opt=true
   batch_size=64
   epoch_checkpoint_interval=1
   checkpoint_hist=60

6. 方案 C refKL 配置：
   train_scheme=ema_ref_attn_kl
   ref_update=prev_step
   ref_update_interval=50
   ref_head_mode=custom_subset:6:1,8:4,8:9,11:18,11:4
   ref_warmup_epochs=28
   ref_attn_kl_weight=0.0
   ref_attn_kl_weight_epoch_overrides=28:0.00030,29:0.00030,36:0.00035,37:0.00035,44:0.00035,45:0.00035,52:0.00030,53:0.00030
   ref_attn_kl_drop_prob=0.50
   ref_attn_loss=kl_ref

7. checkpoint 数量：
   checkpoint-11 到 checkpoint-60，共 50 个 checkpoint。

8. full-val 数量：
   日志中 `Test: [distributed-summary] ... Samples: 50000` 共 50 条。

9. 机器可读结果表：
   /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_sparse_pulse_prevstep_refkl_c_50epoch_status_20260709.tsv
   rows=50

10. RefW 生效检查：
   epoch 28/29/36/37/44/45/52/53 均有非零 RefW 记录；
   每个 pulse epoch 记录到 50 条训练日志采样点。

11. 中文进度文档：
   本文件持续记录每个关键 checkpoint、阶段结论和最终审计。

12. 训练收尾：
   日志记录 wall_seconds=31670；
   checkpoint-60 已生成；
   训练进程结束，GPU 显存回落到 7 MiB 左右。
```

最终结果：

```text
baseline: 80.5980
best checkpoint: checkpoint-54
best Top-1: 80.6820
best Top-5: 95.3880
best Loss: 0.8340
Samples: 50000
delta_vs_baseline: +0.0840
above_baseline checkpoints: 1
81.0 target: 未达到
final checkpoint-60 Top-1: 80.5620
```

最终结论：

```text
方案 C 完整跑完 checkpoint-10 -> checkpoint-60 的 50 个 resumed epoch。
它在 checkpoint-54 达到 Top-1 80.6820，超过原版 OFQ direct-resume baseline 80.5980。
它没有达到 81.0。
最有效的增益来自最后一组 epoch 52/53 的 3.0e-4 sparse prev-step refKL 双 pulse；
此前 28/29、36/37、44/45 的 pulse 只产生过单点接近或回升，但没有稳定越线。
```
