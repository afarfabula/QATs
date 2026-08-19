# OFQ 原版 public-family resume10->60 对照实验进度

## 目标

运行一版 50epoch 原版 OFQ 对照实验，采用与方案 C 相同的 `checkpoint-10 -> checkpoint-60` resume 训练长度和主链路配置，但不启用任何 sparse pulse、prev-step refKL、refmodel attention KL、anchor/ref KL 等自研 KL 组件。

最终审计：

```text
原版 OFQ 50epoch 是否超过 direct-resume baseline 80.5980
原版 OFQ 50epoch 是否超过方案 C 当前最佳 checkpoint-54 Top-1=80.6820
原版 OFQ 50epoch 是否达到 81.0
```

## 固定配置

起点：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
```

实验：

```text
experiment: ofq_resume10_to60_original_ofq_public_20260710
log: /mlx_devbox/users/quyanyi/playground/train_ofq_resume10_to60_original_ofq_public_20260710.log
remote output: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710
script: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume10_to60_original_ofq_public_20260710.sh
monitor: /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/monitor_ofq_resume10_to60_original_ofq_public_20260710.sh
status tsv: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to60_original_ofq_public_status_20260710.tsv
refw tsv: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to60_original_ofq_public_refw_20260710.tsv
summary: /mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_resume10_to60_original_ofq_public_monitor_summary_20260710.txt
```

主链路：

```text
wq_mode=statsq
aq_mode=lsq
qk_reparam=true
qk_reparam_type=0
teacher KD enabled
kd_hard_and_soft=0
teacher_soft_temperature=2.75
no_resume_opt=true
batch_size=64
epoch_checkpoint_interval=1
checkpoint_hist=60
epochs=60
scheduler_epochs=60
```

禁止项：

```text
no soup
no checkpoint averaging
no multi-checkpoint averaging
no ensemble
no A8->A4
no sparse pulse / prev-step refKL / refmodel attention KL / anchor-ref KL
```

## 2026-07-10 preflight

环境检查：

```text
worker: fdbd:dccd:cdc2:1234:0:b8::, ssh port 9801
GPU: 8 x NVIDIA H100 80GB HBM3 visible
worker /tmp free: about 461G
worker root fs free: about 8.1G, so output must stay under /tmp
dataset train shards: 294
dataset validation shards: 14
checkpoint-10: exists
teacher checkpoint: exists
```

实现检查：

```text
启动脚本使用独立 experiment: ofq_resume10_to60_original_ofq_public_20260710
输出目录使用 worker-local /tmp/qat_public_repro
启动命令不包含 --train-scheme ema_ref_attn_kl
启动命令不包含 --ref-attn-kl-weight / --ref-attn-kl-weight-epoch-overrides
启动命令不包含 --ref-update / --ref-update-interval / --ref-head-mode / --ref-warmup-epochs
启动命令不包含 --anchor-ref-attn-kl-weight
```

下一步：

```text
在 worker 上后台启动训练。
启动后用 monitor 脚本持续轮询 checkpoint_count、latest_checkpoint、full-val rows、best line、RefW 是否保持 0。
```

## 2026-07-10 02:44 UTC 启动

启动命令：

```text
cd /mlx_devbox/users/quyanyi/playground
MASTER_PORT=31687 OUT=/tmp/qat_public_repro nohup bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume10_to60_original_ofq_public_20260710.sh >/tmp/ofq_resume10_to60_original_ofq_public_20260710.nohup 2>&1 &
```

启动日志确认：

```text
[QATs] command=/usr/bin/python3 train.py ... --experiment ofq_resume10_to60_original_ofq_public_20260710 ...
--resume checkpoint-10.pth.tar
--no-resume-opt
--epochs 60 --scheduler-epochs 60
--wq-mode statsq --aq-mode lsq
--qk_reparam --qk_reparam_type 0
--use-kd --kd_hard_and_soft 0
--teacher-soft-temperature 2.75
```

原版链路审计：

```text
启动命令不包含 --train-scheme ema_ref_attn_kl
启动命令不包含 --ref-attn-kl-weight
启动命令不包含 --ref-attn-kl-weight-epoch-overrides
启动命令不包含 --ref-update / --ref-head-mode / --ref-warmup-epochs
启动命令不包含 --anchor-ref-attn-kl-weight
args.yaml:
  train_scheme=baseline
  ref_attn_kl_weight=0.0
  ref_attn_kl_weight_epoch_overrides={}
  anchor_ref_attn_kl_weight=0.0
```

启动状态：

```text
训练已进入 epoch 10。
8 x H100 正常满载，显存约 28.3G / GPU。
RefW=0.000e+00。
```

## 2026-07-10 02:56 UTC checkpoint-11

epoch 10 已完成，并生成第一个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-11.pth.tar
checkpoint_count: 1
```

full ImageNet validation：

```text
TrainSummary: epoch=10 updates=2496 avg_step_time=0.223681s samples_per_step=512 samples_per_sec=2288.98
Test: [distributed-summary]  Time: 35.854s  Loss: 0.8443  Acc@1: 80.4000  Acc@5: 95.2460  Samples: 50000
```

对比：

```text
checkpoint-11 Top-1: 80.4000
相对 baseline 80.5980: -0.1980
相对方案 C checkpoint-54 80.6820: -0.2820
81.0 target: 未达到
```

RefW 状态：

```text
epoch 10: RefW=0.000e+00
epoch 11 start: RefW=0.000e+00
nonzero_refw_lines=0
```

运行状态：

```text
训练继续运行，进入 epoch 11。
当前最佳为 checkpoint-11: Top-1 80.4000。
```

## 2026-07-10 03:06 UTC checkpoint-12

epoch 11 已完成，并生成第二个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-12.pth.tar
checkpoint_count: 2
```

full ImageNet validation：

```text
TrainSummary: epoch=11 updates=2496 avg_step_time=0.223670s samples_per_step=512 samples_per_sec=2289.09
Test: [distributed-summary]  Time: 10.202s  Loss: 0.8452  Acc@1: 80.3740  Acc@5: 95.2580  Samples: 50000
```

对比：

```text
checkpoint-11 Top-1: 80.4000
checkpoint-12 Top-1: 80.3740
相对 checkpoint-11: -0.0260
相对 baseline 80.5980: -0.2240
相对方案 C checkpoint-54 80.6820: -0.3080
81.0 target: 未达到
```

RefW 状态：

```text
epoch 11: RefW=0.000e+00
epoch 12 start: RefW=0.000e+00
nonzero_refw_lines=0
```

运行状态：

```text
训练继续运行，进入 epoch 12。
当前最佳仍为 checkpoint-11: Top-1 80.4000。
```

## 2026-07-10 03:15 UTC checkpoint-13

epoch 12 已完成，并生成第三个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-13.pth.tar
checkpoint_count: 3
```

full ImageNet validation：

```text
TrainSummary: epoch=12 updates=2496 avg_step_time=0.223512s samples_per_step=512 samples_per_sec=2290.71
Test: [distributed-summary]  Time: 10.274s  Loss: 0.8415  Acc@1: 80.4600  Acc@5: 95.3180  Samples: 50000
```

对比：

```text
checkpoint-12 Top-1: 80.3740
checkpoint-13 Top-1: 80.4600
相对 checkpoint-12: +0.0860
相对 baseline 80.5980: -0.1380
相对方案 C checkpoint-54 80.6820: -0.2220
81.0 target: 未达到
```

RefW 状态：

```text
epoch 12: RefW=0.000e+00
epoch 13 start: RefW=0.000e+00
nonzero_refw_lines=0
```

运行状态：

```text
训练继续运行，进入 epoch 13。
当前最佳更新为 checkpoint-13: Top-1 80.4600。
```

## 2026-07-10 03:25 UTC checkpoint-14

epoch 13 已完成，并生成第四个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-14.pth.tar
checkpoint_count: 4
```

full ImageNet validation：

```text
TrainSummary: epoch=13 updates=2496 avg_step_time=0.223647s samples_per_step=512 samples_per_sec=2289.32
Test: [distributed-summary]  Time: 10.139s  Loss: 0.8441  Acc@1: 80.4020  Acc@5: 95.3220  Samples: 50000
```

对比：

```text
checkpoint-13 Top-1: 80.4600
checkpoint-14 Top-1: 80.4020
相对 checkpoint-13: -0.0580
相对 baseline 80.5980: -0.1960
相对方案 C checkpoint-54 80.6820: -0.2800
81.0 target: 未达到
```

RefW 状态：

```text
epoch 13: RefW=0.000e+00
epoch 14 start: RefW=0.000e+00
nonzero_refw_lines=0
```

运行状态：

```text
训练继续运行，进入 epoch 14。
当前最佳仍为 checkpoint-13: Top-1 80.4600。
```

## 2026-07-10 03:34 UTC checkpoint-15

epoch 14 已完成，并生成第五个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-15.pth.tar
checkpoint_count: 5
```

full ImageNet validation：

```text
TrainSummary: epoch=14 updates=2496 avg_step_time=0.223975s samples_per_step=512 samples_per_sec=2285.97
Test: [distributed-summary]  Time: 10.185s  Loss: 0.8399  Acc@1: 80.4260  Acc@5: 95.3120  Samples: 50000
```

对比：

```text
checkpoint-13 Top-1: 80.4600
checkpoint-14 Top-1: 80.4020
checkpoint-15 Top-1: 80.4260
相对 checkpoint-14: +0.0240
相对当前最佳 checkpoint-13: -0.0340
相对 baseline 80.5980: -0.1720
相对方案 C checkpoint-54 80.6820: -0.2560
81.0 target: 未达到
```

RefW 状态：

```text
epoch 14: RefW=0.000e+00
epoch 15 start: RefW=0.000e+00
nonzero_refw_lines=0
```

运行状态：

```text
训练继续运行，进入 epoch 15。
当前最佳仍为 checkpoint-13: Top-1 80.4600。
```

## 2026-07-10 03:44 UTC checkpoint-16

epoch 15 已完成，并生成第六个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-16.pth.tar
checkpoint_count: 6
```

full ImageNet validation：

```text
TrainSummary: epoch=15 updates=2496 avg_step_time=0.223850s samples_per_step=512 samples_per_sec=2287.25
Test: [distributed-summary]  Time: 10.161s  Loss: 0.8426  Acc@1: 80.3260  Acc@5: 95.3260  Samples: 50000
```

对比：

```text
checkpoint-13 Top-1: 80.4600
checkpoint-15 Top-1: 80.4260
checkpoint-16 Top-1: 80.3260
相对 checkpoint-15: -0.1000
相对当前最佳 checkpoint-13: -0.1340
相对 baseline 80.5980: -0.2720
相对方案 C checkpoint-54 80.6820: -0.3560
81.0 target: 未达到
```

RefW 状态：

```text
epoch 15: RefW=0.000e+00
epoch 16 start: RefW=0.000e+00
nonzero_refw_lines=0
```

运行状态：

```text
训练继续运行，进入 epoch 16。
当前最佳仍为 checkpoint-13: Top-1 80.4600。
```

## 2026-07-10 03:54 UTC checkpoint-17

epoch 16 已完成，并生成第七个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-17.pth.tar
checkpoint_count: 7
```

full ImageNet validation：

```text
TrainSummary: epoch=16 updates=2496 avg_step_time=0.223634s samples_per_step=512 samples_per_sec=2289.46
Test: [distributed-summary]  Time: 10.234s  Loss: 0.8439  Acc@1: 80.5060  Acc@5: 95.2760  Samples: 50000
```

对比：

```text
checkpoint-13 Top-1: 80.4600
checkpoint-16 Top-1: 80.3260
checkpoint-17 Top-1: 80.5060
相对 checkpoint-16: +0.1800
相对此前最佳 checkpoint-13: +0.0460
相对 baseline 80.5980: -0.0920
相对方案 C checkpoint-54 80.6820: -0.1760
81.0 target: 未达到
```

RefW 状态：

```text
epoch 16: RefW=0.000e+00
epoch 17 start: RefW=0.000e+00
nonzero_refw_lines=0
```

运行状态：

```text
训练继续运行，进入 epoch 17。
当前最佳更新为 checkpoint-17: Top-1 80.5060。
```

## 2026-07-10 04:03 UTC checkpoint-18

epoch 17 已完成，并生成第八个 resumed checkpoint：

```text
checkpoint: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-18.pth.tar
checkpoint_count: 8
```

full ImageNet validation：

```text
TrainSummary: epoch=17 updates=2496 avg_step_time=0.223465s samples_per_step=512 samples_per_sec=2291.19
Test: [distributed-summary]  Time: 10.188s  Loss: 0.8415  Acc@1: 80.3600  Acc@5: 95.3200  Samples: 50000
```

对比：

```text
checkpoint-17 Top-1: 80.5060
checkpoint-18 Top-1: 80.3600
相对 checkpoint-17: -0.1460
相对当前最佳 checkpoint-17: -0.1460
相对 baseline 80.5980: -0.2380
相对方案 C checkpoint-54 80.6820: -0.3220
81.0 target: 未达到
```

RefW 状态：

```text
epoch 17: RefW=0.000e+00
epoch 18 start: RefW=0.000e+00
nonzero_refw_lines=0
```

运行状态：

```text
训练继续运行，进入 epoch 18。
当前最佳仍为 checkpoint-17: Top-1 80.5060。
```

## 2026-07-10 04:13 UTC checkpoint-19/20

epoch 18 和 epoch 19 已完成：

```text
checkpoint-19: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-19.pth.tar
checkpoint-20: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-20.pth.tar
checkpoint_count: 10
```

full ImageNet validation：

```text
checkpoint-19:
TrainSummary: epoch=18 updates=2496 avg_step_time=0.223555s samples_per_step=512 samples_per_sec=2290.27
Test: [distributed-summary]  Time: 10.303s  Loss: 0.8399  Acc@1: 80.4860  Acc@5: 95.3280  Samples: 50000

checkpoint-20:
TrainSummary: epoch=19 updates=2496 avg_step_time=0.223559s samples_per_step=512 samples_per_sec=2290.23
Test: [distributed-summary]  Time: 10.217s  Loss: 0.8364  Acc@1: 80.3460  Acc@5: 95.3960  Samples: 50000
```

对比：

```text
checkpoint-17 Top-1: 80.5060
checkpoint-19 Top-1: 80.4860
checkpoint-20 Top-1: 80.3460
checkpoint-19 相对 baseline 80.5980: -0.1120
checkpoint-20 相对 baseline 80.5980: -0.2520
checkpoint-19 相对方案 C 80.6820: -0.1960
checkpoint-20 相对方案 C 80.6820: -0.3360
81.0 target: 未达到
```

RefW 状态：

```text
epoch 18/19: RefW=0.000e+00
epoch 20 start: RefW=0.000e+00
nonzero_refw_lines=0
```

运行状态：

```text
训练继续运行，进入 epoch 20。
当前最佳仍为 checkpoint-17: Top-1 80.5060。
```

## 2026-07-10 04:31 UTC checkpoint-21

epoch 20 已完成：

```text
checkpoint-21: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-21.pth.tar
checkpoint_count: 11
```

full ImageNet validation：

```text
TrainSummary: epoch=20 updates=2496 avg_step_time=0.223583s samples_per_step=512 samples_per_sec=2289.97
Test: [distributed-summary]  Time: 10.206s  Loss: 0.8433  Acc@1: 80.3560  Acc@5: 95.3240  Samples: 50000
```

对比：

```text
checkpoint-17 Top-1: 80.5060
checkpoint-21 Top-1: 80.3560
checkpoint-21 相对当前最佳 checkpoint-17: -0.1500
checkpoint-21 相对 baseline 80.5980: -0.2420
checkpoint-21 相对方案 C 80.6820: -0.3260
81.0 target: 未达到
```

RefW 状态：

```text
epoch 20/21: RefW=0.000e+00
nonzero_refw_lines=0
```

运行状态：

```text
训练继续运行，进入 epoch 21。
当前最佳仍为 checkpoint-17: Top-1 80.5060。
截至 checkpoint-21，原版 OFQ 还没有超过 80.5980 baseline。
```

## 2026-07-10 04:41 UTC checkpoint-22

epoch 21 已完成：

```text
checkpoint-22: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-22.pth.tar
checkpoint_count: 12
```

full ImageNet validation：

```text
TrainSummary: epoch=21 updates=2496 avg_step_time=0.223482s samples_per_step=512 samples_per_sec=2291.01
Test: [distributed-summary]  Time: 10.422s  Loss: 0.8420  Acc@1: 80.5140  Acc@5: 95.3100  Samples: 50000
```

对比：

```text
checkpoint-22 Top-1: 80.5140
相对 checkpoint-17 Top-1 80.5060: +0.0080
checkpoint-22 相对 baseline 80.5980: -0.0840
checkpoint-22 相对方案 C 80.6820: -0.1680
81.0 target: 未达到
```

RefW 状态：

```text
epoch 21/22: RefW=0.000e+00
nonzero_refw_lines=0
```

运行状态：

```text
训练继续运行，进入 epoch 22。
当前最佳更新为 checkpoint-22: Top-1 80.5140。
截至 checkpoint-22，原版 OFQ 仍未超过 80.5980 baseline，但与 baseline 的差距缩小到 -0.0840。
```

## 2026-07-10 04:51 UTC checkpoint-23

epoch 22 已完成：

```text
checkpoint-23: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-23.pth.tar
checkpoint_count: 13
```

full ImageNet validation：

```text
TrainSummary: epoch=22 updates=2496 avg_step_time=0.223557s samples_per_step=512 samples_per_sec=2290.24
Test: [distributed-summary]  Time: 10.232s  Loss: 0.8371  Acc@1: 80.4120  Acc@5: 95.3520  Samples: 50000
```

对比：

```text
checkpoint-23 Top-1: 80.4120
相对当前最佳 checkpoint-22 Top-1 80.5140: -0.1020
checkpoint-23 相对 baseline 80.5980: -0.1860
checkpoint-23 相对方案 C 80.6820: -0.2700
81.0 target: 未达到
```

RefW 状态：

```text
epoch 22/23: RefW=0.000e+00
nonzero_refw_lines=0
```

运行状态：

```text
训练继续运行，进入 epoch 23。
当前最佳仍为 checkpoint-22: Top-1 80.5140。
截至 checkpoint-23，原版 OFQ 13 条 full-val 结果均未超过 80.5980 baseline。
```

## 2026-07-10 05:00 UTC checkpoint-24

epoch 23 已完成：

```text
checkpoint-24: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-24.pth.tar
checkpoint_count: 14
```

full ImageNet validation：

```text
TrainSummary: epoch=23 updates=2496 avg_step_time=0.223519s samples_per_step=512 samples_per_sec=2290.64
Test: [distributed-summary]  Time: 10.216s  Loss: 0.8359  Acc@1: 80.4200  Acc@5: 95.3000  Samples: 50000
```

对比：

```text
checkpoint-24 Top-1: 80.4200
相对当前最佳 checkpoint-22 Top-1 80.5140: -0.0940
checkpoint-24 相对 baseline 80.5980: -0.1780
checkpoint-24 相对方案 C 80.6820: -0.2620
81.0 target: 未达到
```

RefW 状态：

```text
epoch 23/24: RefW=0.000e+00
nonzero_refw_lines=0
```

运行状态：

```text
训练继续运行，进入 epoch 24。
当前最佳仍为 checkpoint-22: Top-1 80.5140。
截至 checkpoint-24，原版 OFQ 14 条 full-val 结果均未超过 80.5980 baseline。
```

## 2026-07-10 05:09 UTC checkpoint-25

epoch 24 已完成：

```text
checkpoint-25: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-25.pth.tar
checkpoint_count: 15
```

full ImageNet validation：

```text
TrainSummary: epoch=24 updates=2496 avg_step_time=0.223536s samples_per_step=512 samples_per_sec=2290.46
Test: [distributed-summary]  Time: 10.215s  Loss: 0.8409  Acc@1: 80.4220  Acc@5: 95.3460  Samples: 50000
```

对比：

```text
checkpoint-25 Top-1: 80.4220
相对当前最佳 checkpoint-22 Top-1 80.5140: -0.0920
checkpoint-25 相对 baseline 80.5980: -0.1760
checkpoint-25 相对方案 C 80.6820: -0.2600
81.0 target: 未达到
```

RefW 状态：

```text
epoch 24/25: RefW=0.000e+00
nonzero_refw_lines=0
```

阶段性状态：

```text
训练继续运行，进入 epoch 25。
当前最佳仍为 checkpoint-22: Top-1 80.5140。
截至 checkpoint-25，原版 OFQ 已完成 15 条 resumed full-val 结果，全部 Samples=50000。
above_baseline_lines=0，above_scheme_c_lines=0。
原版 OFQ 前 15 个 checkpoint 尚未超过 80.5980 direct-resume baseline，也明显低于方案 C 80.6820。
```

## 2026-07-10 05:19 UTC checkpoint-26

epoch 25 已完成：

```text
checkpoint-26: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-26.pth.tar
checkpoint_count: 16
```

full ImageNet validation：

```text
TrainSummary: epoch=25 updates=2496 avg_step_time=0.223543s samples_per_step=512 samples_per_sec=2290.39
Test: [distributed-summary]  Time: 10.211s  Loss: 0.8446  Acc@1: 80.5160  Acc@5: 95.3000  Samples: 50000
```

对比：

```text
checkpoint-26 Top-1: 80.5160
相对 checkpoint-22 Top-1 80.5140: +0.0020
checkpoint-26 相对 baseline 80.5980: -0.0820
checkpoint-26 相对方案 C 80.6820: -0.1660
81.0 target: 未达到
```

RefW 状态：

```text
epoch 25/26: RefW=0.000e+00
nonzero_refw_lines=0
```

阶段性状态：

```text
训练继续运行，进入 epoch 26。
当前最佳更新为 checkpoint-26: Top-1 80.5160。
这是小幅刷新，幅度只有 +0.0020；截至 checkpoint-26，原版 OFQ 16 条 full-val 结果仍全部低于 80.5980 baseline。
```

## 2026-07-10 05:29 UTC checkpoint-27

epoch 26 已完成：

```text
checkpoint-27: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-27.pth.tar
checkpoint_count: 17
```

full ImageNet validation：

```text
TrainSummary: epoch=26 updates=2496 avg_step_time=0.223542s samples_per_step=512 samples_per_sec=2290.40
Test: [distributed-summary]  Time: 10.239s  Loss: 0.8411  Acc@1: 80.5340  Acc@5: 95.3280  Samples: 50000
```

对比：

```text
checkpoint-27 Top-1: 80.5340
相对 checkpoint-26 Top-1 80.5160: +0.0180
checkpoint-27 相对 baseline 80.5980: -0.0640
checkpoint-27 相对方案 C 80.6820: -0.1480
81.0 target: 未达到
```

RefW 状态：

```text
epoch 26/27: RefW=0.000e+00
nonzero_refw_lines=0
```

阶段性状态：

```text
训练继续运行，进入 epoch 27。
当前最佳更新为 checkpoint-27: Top-1 80.5340。
截至 checkpoint-27，原版 OFQ 17 条 full-val 结果仍全部低于 80.5980 baseline。
最近 checkpoint-26/27 连续小幅刷新 best，但还没有达到 baseline，更没有接近方案 C 80.6820。
```

## 2026-07-10 05:38 UTC checkpoint-28

epoch 27 已完成：

```text
checkpoint-28: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-28.pth.tar
checkpoint_count: 18
```

full ImageNet validation：

```text
TrainSummary: epoch=27 updates=2496 avg_step_time=0.223521s samples_per_step=512 samples_per_sec=2290.61
Test: [distributed-summary]  Time: 10.281s  Loss: 0.8351  Acc@1: 80.4160  Acc@5: 95.3320  Samples: 50000
```

对比：

```text
checkpoint-28 Top-1: 80.4160
相对当前最佳 checkpoint-27 Top-1 80.5340: -0.1180
checkpoint-28 相对 baseline 80.5980: -0.1820
checkpoint-28 相对方案 C 80.6820: -0.2660
81.0 target: 未达到
```

RefW 状态：

```text
epoch 27/28: RefW=0.000e+00
nonzero_refw_lines=0
```

阶段性状态：

```text
训练继续运行，进入 epoch 28。
当前最佳仍为 checkpoint-27: Top-1 80.5340。
checkpoint-28 没有延续 checkpoint-26/27 的小幅上移，重新回落到 80.4 区间。
截至 checkpoint-28，原版 OFQ 18 条 full-val 结果仍全部低于 80.5980 baseline。
```

## 2026-07-10 05:48 UTC checkpoint-29

epoch 28 已完成：

```text
checkpoint-29: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-29.pth.tar
checkpoint_count: 19
```

full ImageNet validation：

```text
TrainSummary: epoch=28 updates=2496 avg_step_time=0.223510s samples_per_step=512 samples_per_sec=2290.72
Test: [distributed-summary]  Time: 10.200s  Loss: 0.8385  Acc@1: 80.4360  Acc@5: 95.3460  Samples: 50000
```

对比：

```text
checkpoint-29 Top-1: 80.4360
相对当前最佳 checkpoint-27 Top-1 80.5340: -0.0980
checkpoint-29 相对 baseline 80.5980: -0.1620
checkpoint-29 相对方案 C 80.6820: -0.2460
81.0 target: 未达到
```

RefW 状态：

```text
epoch 28/29: RefW=0.000e+00
nonzero_refw_lines=0
```

阶段性状态：

```text
训练继续运行，进入 epoch 29。
当前最佳仍为 checkpoint-27: Top-1 80.5340。
截至 checkpoint-29，原版 OFQ 19 条 full-val 结果仍全部低于 80.5980 baseline。
checkpoint-28/29 连续回落，说明 checkpoint-26/27 的上移还不是稳定突破趋势。
```

## 2026-07-10 05:57 UTC checkpoint-30

epoch 29 已完成：

```text
checkpoint-30: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-30.pth.tar
checkpoint_count: 20
```

full ImageNet validation：

```text
TrainSummary: epoch=29 updates=2496 avg_step_time=0.223652s samples_per_step=512 samples_per_sec=2289.27
Test: [distributed-summary]  Time: 10.323s  Loss: 0.8418  Acc@1: 80.4300  Acc@5: 95.3320  Samples: 50000
```

对比：

```text
checkpoint-30 Top-1: 80.4300
相对当前最佳 checkpoint-27 Top-1 80.5340: -0.1040
checkpoint-30 相对 baseline 80.5980: -0.1680
checkpoint-30 相对方案 C 80.6820: -0.2520
81.0 target: 未达到
```

RefW 状态：

```text
epoch 29/30: RefW=0.000e+00
nonzero_refw_lines=0
```

半程阶段性状态：

```text
训练继续运行，进入 epoch 30。
当前最佳仍为 checkpoint-27: Top-1 80.5340。
截至 checkpoint-30，原版 OFQ 已完成 20 条 resumed full-val 结果，全部 Samples=50000。
above_baseline_lines=0，above_scheme_c_lines=0。
10 -> 30 半程结论：原版 OFQ 在这个设置下仍未超过 80.5980 baseline，也没有超过方案 C 80.6820。
```

## 2026-07-10 06:06 UTC checkpoint-31

epoch 30 已完成：

```text
checkpoint-31: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-31.pth.tar
checkpoint_count: 21
```

full ImageNet validation：

```text
TrainSummary: epoch=30 updates=2496 avg_step_time=0.223481s samples_per_step=512 samples_per_sec=2291.02
Test: [distributed-summary]  Time: 10.243s  Loss: 0.8394  Acc@1: 80.4240  Acc@5: 95.3200  Samples: 50000
```

对比：

```text
checkpoint-31 Top-1: 80.4240
相对当前最佳 checkpoint-27 Top-1 80.5340: -0.1100
checkpoint-31 相对 baseline 80.5980: -0.1740
checkpoint-31 相对方案 C 80.6820: -0.2580
81.0 target: 未达到
```

RefW 状态：

```text
epoch 30/31: RefW=0.000e+00
nonzero_refw_lines=0
```

阶段性状态：

```text
训练继续运行，进入 epoch 31。
当前最佳仍为 checkpoint-27: Top-1 80.5340。
截至 checkpoint-31，原版 OFQ 21 条 full-val 结果仍全部低于 80.5980 baseline。
半程后第一个 checkpoint 仍在 80.42 左右，没有自然爬升过 baseline。
```

## 2026-07-10 06:16 UTC checkpoint-32

epoch 31 已完成：

```text
checkpoint-32: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-32.pth.tar
checkpoint_count: 22
```

full ImageNet validation：

```text
TrainSummary: epoch=31 updates=2496 avg_step_time=0.223532s samples_per_step=512 samples_per_sec=2290.50
Test: [distributed-summary]  Time: 10.204s  Loss: 0.8389  Acc@1: 80.5420  Acc@5: 95.3440  Samples: 50000
```

对比：

```text
checkpoint-32 Top-1: 80.5420
相对 checkpoint-27 Top-1 80.5340: +0.0080
checkpoint-32 相对 baseline 80.5980: -0.0560
checkpoint-32 相对方案 C 80.6820: -0.1400
81.0 target: 未达到
```

RefW 状态：

```text
epoch 31/32: RefW=0.000e+00
nonzero_refw_lines=0
```

阶段性状态：

```text
训练继续运行，进入 epoch 32。
当前最佳更新为 checkpoint-32: Top-1 80.5420。
截至 checkpoint-32，原版 OFQ 22 条 full-val 结果仍全部低于 80.5980 baseline。
当前距 baseline 差距缩小到 -0.0560，但仍没有完成 baseline 突破。
```

## 2026-07-10 06:26 UTC checkpoint-33

epoch 32 已完成：

```text
checkpoint-33: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-33.pth.tar
checkpoint_count: 23
```

full ImageNet validation：

```text
TrainSummary: epoch=32 updates=2496 avg_step_time=0.223614s samples_per_step=512 samples_per_sec=2289.66
Test: [distributed-summary]  Time: 10.288s  Loss: 0.8376  Acc@1: 80.5260  Acc@5: 95.3220  Samples: 50000
```

对比：

```text
checkpoint-33 Top-1: 80.5260
相对当前最佳 checkpoint-32 Top-1 80.5420: -0.0160
checkpoint-33 相对 baseline 80.5980: -0.0720
checkpoint-33 相对方案 C 80.6820: -0.1560
81.0 target: 未达到
```

RefW 状态：

```text
epoch 32/33: RefW=0.000e+00
nonzero_refw_lines=0
```

阶段性状态：

```text
训练继续运行，进入 epoch 33。
当前最佳仍为 checkpoint-32: Top-1 80.5420。
截至 checkpoint-33，原版 OFQ 23 条 full-val 结果仍全部低于 80.5980 baseline。
checkpoint-33 低于 checkpoint-32，但仍处于当前较高区间；还需要继续观察是否能真正突破 baseline。
```

## 2026-07-10 06:35 UTC checkpoint-34

epoch 33 已完成：

```text
checkpoint-34: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-34.pth.tar
checkpoint_count: 24
```

full ImageNet validation：

```text
TrainSummary: epoch=33 updates=2496 avg_step_time=0.223481s samples_per_step=512 samples_per_sec=2291.03
Test: [distributed-summary]  Time: 10.307s  Loss: 0.8371  Acc@1: 80.4620  Acc@5: 95.3620  Samples: 50000
```

对比：

```text
checkpoint-34 Top-1: 80.4620
相对当前最佳 checkpoint-32 Top-1 80.5420: -0.0800
checkpoint-34 相对 baseline 80.5980: -0.1360
checkpoint-34 相对方案 C 80.6820: -0.2200
81.0 target: 未达到
```

RefW 状态：

```text
epoch 33/34: RefW=0.000e+00
nonzero_refw_lines=0
```

阶段性状态：

```text
训练继续运行，进入 epoch 34。
当前最佳仍为 checkpoint-32: Top-1 80.5420。
截至 checkpoint-34，原版 OFQ 24 条 full-val 结果仍全部低于 80.5980 baseline。
checkpoint-34 回落到 80.46，说明 checkpoint-32/33 的相对高点仍未形成稳定突破。
```

## 2026-07-10 06:45 UTC checkpoint-35

epoch 34 已完成：

```text
checkpoint-35: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-35.pth.tar
checkpoint_count: 25
```

full ImageNet validation：

```text
TrainSummary: epoch=34 updates=2496 avg_step_time=0.223677s samples_per_step=512 samples_per_sec=2289.02
Test: [distributed-summary]  Time: 10.273s  Loss: 0.8345  Acc@1: 80.4640  Acc@5: 95.3300  Samples: 50000
```

对比：

```text
checkpoint-35 Top-1: 80.4640
相对当前最佳 checkpoint-32 Top-1 80.5420: -0.0780
checkpoint-35 相对 baseline 80.5980: -0.1340
checkpoint-35 相对方案 C 80.6820: -0.2180
81.0 target: 未达到
```

RefW 状态：

```text
epoch 34/35: RefW=0.000e+00
nonzero_refw_lines=0
```

半数 checkpoint 阶段性状态：

```text
训练继续运行，进入 epoch 35。
当前最佳仍为 checkpoint-32: Top-1 80.5420。
截至 checkpoint-35，原版 OFQ 已完成 25 条 resumed full-val 结果，全部 Samples=50000。
above_baseline_lines=0，above_scheme_c_lines=0。
10 -> 60 的 50 个 resumed checkpoint 已完成一半；原版 OFQ best 仍低于 80.5980 baseline 0.0560，也低于方案 C 80.6820 0.1400。
```

## 2026-07-10 07:32 UTC checkpoint-36 到 checkpoint-40

epoch 35 到 epoch 39 已完成：

```text
checkpoint-36: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-36.pth.tar
checkpoint-37: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-37.pth.tar
checkpoint-38: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-38.pth.tar
checkpoint-39: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-39.pth.tar
checkpoint-40: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-40.pth.tar
checkpoint_count: 30
```

full ImageNet validation：

```text
checkpoint-36:
TrainSummary: epoch=35 updates=2496 avg_step_time=0.223552s samples_per_step=512 samples_per_sec=2290.29
Test: [distributed-summary]  Time: 10.399s  Loss: 0.8334  Acc@1: 80.5520  Acc@5: 95.3500  Samples: 50000

checkpoint-37:
TrainSummary: epoch=36 updates=2496 avg_step_time=0.223544s samples_per_step=512 samples_per_sec=2290.38
Test: [distributed-summary]  Time: 10.262s  Loss: 0.8349  Acc@1: 80.5740  Acc@5: 95.3460  Samples: 50000

checkpoint-38:
TrainSummary: epoch=37 updates=2496 avg_step_time=0.223551s samples_per_step=512 samples_per_sec=2290.30
Test: [distributed-summary]  Time: 10.564s  Loss: 0.8381  Acc@1: 80.4880  Acc@5: 95.3760  Samples: 50000

checkpoint-39:
TrainSummary: epoch=38 updates=2496 avg_step_time=0.223632s samples_per_step=512 samples_per_sec=2289.48
Test: [distributed-summary]  Time: 10.298s  Loss: 0.8341  Acc@1: 80.5040  Acc@5: 95.3720  Samples: 50000

checkpoint-40:
TrainSummary: epoch=39 updates=2496 avg_step_time=0.223802s samples_per_step=512 samples_per_sec=2287.74
Test: [distributed-summary]  Time: 10.233s  Loss: 0.8343  Acc@1: 80.4000  Acc@5: 95.3880  Samples: 50000
```

对比：

```text
checkpoint-36 Top-1: 80.5520，相对 baseline 80.5980: -0.0460，相对方案 C 80.6820: -0.1300
checkpoint-37 Top-1: 80.5740，相对 baseline 80.5980: -0.0240，相对方案 C 80.6820: -0.1080
checkpoint-38 Top-1: 80.4880，相对 baseline 80.5980: -0.1100，相对方案 C 80.6820: -0.1940
checkpoint-39 Top-1: 80.5040，相对 baseline 80.5980: -0.0940，相对方案 C 80.6820: -0.1780
checkpoint-40 Top-1: 80.4000，相对 baseline 80.5980: -0.1980，相对方案 C 80.6820: -0.2820
81.0 target: 未达到
```

RefW 状态：

```text
epoch 35-40: RefW=0.000e+00
nonzero_refw_lines=0
```

阶段性状态：

```text
训练继续运行，进入 epoch 40。
当前最佳更新为 checkpoint-37: Top-1 80.5740。
截至 checkpoint-40，原版 OFQ 已完成 30 条 resumed full-val 结果，全部 Samples=50000。
above_baseline_lines=0，above_scheme_c_lines=0。
checkpoint-36/37 继续逼近 80.5980 baseline，但 checkpoint-38/39/40 回落，尚不能认为原版 OFQ 已经稳定突破 direct-resume baseline。
```

## 2026-07-10 08:20 UTC checkpoint-41 到 checkpoint-45

epoch 40 到 epoch 44 已完成：

```text
checkpoint-41: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-41.pth.tar
checkpoint-42: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-42.pth.tar
checkpoint-43: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-43.pth.tar
checkpoint-44: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-44.pth.tar
checkpoint-45: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-45.pth.tar
checkpoint_count: 35
```

full ImageNet validation：

```text
checkpoint-41:
TrainSummary: epoch=40 updates=2496 avg_step_time=0.223620s samples_per_step=512 samples_per_sec=2289.59
Test: [distributed-summary]  Time: 10.265s  Loss: 0.8342  Acc@1: 80.5060  Acc@5: 95.3780  Samples: 50000

checkpoint-42:
TrainSummary: epoch=41 updates=2496 avg_step_time=0.223522s samples_per_step=512 samples_per_sec=2290.60
Test: [distributed-summary]  Time: 10.372s  Loss: 0.8355  Acc@1: 80.5340  Acc@5: 95.3180  Samples: 50000

checkpoint-43:
TrainSummary: epoch=42 updates=2496 avg_step_time=0.223621s samples_per_step=512 samples_per_sec=2289.58
Test: [distributed-summary]  Time: 10.291s  Loss: 0.8363  Acc@1: 80.5420  Acc@5: 95.3600  Samples: 50000

checkpoint-44:
TrainSummary: epoch=43 updates=2496 avg_step_time=0.223605s samples_per_step=512 samples_per_sec=2289.76
Test: [distributed-summary]  Time: 10.294s  Loss: 0.8312  Acc@1: 80.5500  Acc@5: 95.3380  Samples: 50000

checkpoint-45:
TrainSummary: epoch=44 updates=2496 avg_step_time=0.223492s samples_per_step=512 samples_per_sec=2290.91
Test: [distributed-summary]  Time: 10.282s  Loss: 0.8323  Acc@1: 80.5580  Acc@5: 95.3900  Samples: 50000
```

对比：

```text
checkpoint-41 Top-1: 80.5060，相对 baseline 80.5980: -0.0920，相对方案 C 80.6820: -0.1760
checkpoint-42 Top-1: 80.5340，相对 baseline 80.5980: -0.0640，相对方案 C 80.6820: -0.1480
checkpoint-43 Top-1: 80.5420，相对 baseline 80.5980: -0.0560，相对方案 C 80.6820: -0.1400
checkpoint-44 Top-1: 80.5500，相对 baseline 80.5980: -0.0480，相对方案 C 80.6820: -0.1320
checkpoint-45 Top-1: 80.5580，相对 baseline 80.5980: -0.0400，相对方案 C 80.6820: -0.1240
81.0 target: 未达到
```

RefW 状态：

```text
epoch 40-45: RefW=0.000e+00
nonzero_refw_lines=0
```

阶段性状态：

```text
训练继续运行，进入 epoch 45。
当前最佳仍为 checkpoint-37: Top-1 80.5740。
截至 checkpoint-45，原版 OFQ 已完成 35 条 resumed full-val 结果，全部 Samples=50000。
above_baseline_lines=0，above_scheme_c_lines=0。
checkpoint-41 到 checkpoint-45 呈现逐步贴近 baseline 的趋势，但仍没有超过 80.5980；当前最接近 baseline 的 checkpoint-37 仍差 -0.0240。
```

## 2026-07-10 09:08 UTC checkpoint-46 到 checkpoint-50

epoch 45 到 epoch 49 已完成：

```text
checkpoint-46: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-46.pth.tar
checkpoint-47: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-47.pth.tar
checkpoint-48: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-48.pth.tar
checkpoint-49: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-49.pth.tar
checkpoint-50: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-50.pth.tar
checkpoint_count: 40
```

full ImageNet validation：

```text
checkpoint-46:
TrainSummary: epoch=45 updates=2496 avg_step_time=0.223534s samples_per_step=512 samples_per_sec=2290.48
Test: [distributed-summary]  Time: 10.242s  Loss: 0.8365  Acc@1: 80.4620  Acc@5: 95.3780  Samples: 50000

checkpoint-47:
TrainSummary: epoch=46 updates=2496 avg_step_time=0.223974s samples_per_step=512 samples_per_sec=2285.98
Test: [distributed-summary]  Time: 10.241s  Loss: 0.8344  Acc@1: 80.5880  Acc@5: 95.3920  Samples: 50000

checkpoint-48:
TrainSummary: epoch=47 updates=2496 avg_step_time=0.223693s samples_per_step=512 samples_per_sec=2288.85
Test: [distributed-summary]  Time: 10.267s  Loss: 0.8347  Acc@1: 80.5980  Acc@5: 95.3740  Samples: 50000

checkpoint-49:
TrainSummary: epoch=48 updates=2496 avg_step_time=0.223598s samples_per_step=512 samples_per_sec=2289.82
Test: [distributed-summary]  Time: 10.286s  Loss: 0.8383  Acc@1: 80.5140  Acc@5: 95.3180  Samples: 50000

checkpoint-50:
TrainSummary: epoch=49 updates=2496 avg_step_time=0.223564s samples_per_step=512 samples_per_sec=2290.18
Test: [distributed-summary]  Time: 10.174s  Loss: 0.8341  Acc@1: 80.6300  Acc@5: 95.3600  Samples: 50000
```

对比：

```text
checkpoint-46 Top-1: 80.4620，相对 baseline 80.5980: -0.1360，相对方案 C 80.6820: -0.2200
checkpoint-47 Top-1: 80.5880，相对 baseline 80.5980: -0.0100，相对方案 C 80.6820: -0.0940
checkpoint-48 Top-1: 80.5980，相对 baseline 80.5980: +0.0000，相对方案 C 80.6820: -0.0840
checkpoint-49 Top-1: 80.5140，相对 baseline 80.5980: -0.0840，相对方案 C 80.6820: -0.1680
checkpoint-50 Top-1: 80.6300，相对 baseline 80.5980: +0.0320，相对方案 C 80.6820: -0.0520
81.0 target: 未达到
```

RefW 状态：

```text
epoch 45-50: RefW=0.000e+00
nonzero_refw_lines=0
```

关键阶段性状态：

```text
训练继续运行，进入 epoch 50。
当前最佳更新为 checkpoint-50: Top-1 80.6300。
截至 checkpoint-50，原版 OFQ 已完成 40 条 resumed full-val 结果，全部 Samples=50000。
above_baseline_lines=1，above_scheme_c_lines=0。
checkpoint-48 精确追平 80.5980 baseline；checkpoint-50 首次超过 baseline，幅度 +0.0320。
不过 checkpoint-50 仍低于方案 C 80.6820，差距 -0.0520，仍未达到 81.0。
```

## 2026-07-10 09:55 UTC checkpoint-51 到 checkpoint-55

epoch 50 到 epoch 54 已完成：

```text
checkpoint-51: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-51.pth.tar
checkpoint-52: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-52.pth.tar
checkpoint-53: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-53.pth.tar
checkpoint-54: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-54.pth.tar
checkpoint-55: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-55.pth.tar
checkpoint_count: 45
```

full ImageNet validation：

```text
checkpoint-51:
TrainSummary: epoch=50 updates=2496 avg_step_time=0.223650s samples_per_step=512 samples_per_sec=2289.29
Test: [distributed-summary]  Time: 10.247s  Loss: 0.8325  Acc@1: 80.6160  Acc@5: 95.3680  Samples: 50000

checkpoint-52:
TrainSummary: epoch=51 updates=2496 avg_step_time=0.223690s samples_per_step=512 samples_per_sec=2288.88
Test: [distributed-summary]  Time: 10.219s  Loss: 0.8369  Acc@1: 80.7240  Acc@5: 95.3400  Samples: 50000

checkpoint-53:
TrainSummary: epoch=52 updates=2496 avg_step_time=0.223562s samples_per_step=512 samples_per_sec=2290.19
Test: [distributed-summary]  Time: 10.292s  Loss: 0.8369  Acc@1: 80.6680  Acc@5: 95.3520  Samples: 50000

checkpoint-54:
TrainSummary: epoch=53 updates=2496 avg_step_time=0.223573s samples_per_step=512 samples_per_sec=2290.08
Test: [distributed-summary]  Time: 10.186s  Loss: 0.8321  Acc@1: 80.6460  Acc@5: 95.3740  Samples: 50000

checkpoint-55:
TrainSummary: epoch=54 updates=2496 avg_step_time=0.223483s samples_per_step=512 samples_per_sec=2291.00
Test: [distributed-summary]  Time: 10.181s  Loss: 0.8352  Acc@1: 80.5760  Acc@5: 95.3500  Samples: 50000
```

对比：

```text
checkpoint-51 Top-1: 80.6160，相对 baseline 80.5980: +0.0180，相对方案 C 80.6820: -0.0660
checkpoint-52 Top-1: 80.7240，相对 baseline 80.5980: +0.1260，相对方案 C 80.6820: +0.0420
checkpoint-53 Top-1: 80.6680，相对 baseline 80.5980: +0.0700，相对方案 C 80.6820: -0.0140
checkpoint-54 Top-1: 80.6460，相对 baseline 80.5980: +0.0480，相对方案 C 80.6820: -0.0360
checkpoint-55 Top-1: 80.5760，相对 baseline 80.5980: -0.0220，相对方案 C 80.6820: -0.1060
81.0 target: 未达到
```

RefW 状态：

```text
epoch 50-55: RefW=0.000e+00
nonzero_refw_lines=0
```

关键阶段性状态：

```text
训练继续运行，进入 epoch 55。
当前最佳更新为 checkpoint-52: Top-1 80.7240。
截至 checkpoint-55，原版 OFQ 已完成 45 条 resumed full-val 结果，全部 Samples=50000。
above_baseline_lines=5，above_scheme_c_lines=1。
checkpoint-52 首次超过方案 C 80.6820，幅度 +0.0420；但 checkpoint-55 回落到 baseline 下方，说明需要 checkpoint-56 到 checkpoint-60 判定这个突破是否稳定。
```

## 2026-07-10 10:43 UTC checkpoint-56 到 checkpoint-60

epoch 55 到 epoch 59 已完成：

```text
checkpoint-56: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-56.pth.tar
checkpoint-57: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-57.pth.tar
checkpoint-58: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-58.pth.tar
checkpoint-59: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-59.pth.tar
checkpoint-60: /tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710/checkpoint-60.pth.tar
checkpoint_count: 50
```

full ImageNet validation：

```text
checkpoint-56:
TrainSummary: epoch=55 updates=2496 avg_step_time=0.223437s samples_per_step=512 samples_per_sec=2291.47
Test: [distributed-summary]  Loss: 0.8338  Acc@1: 80.5360  Acc@5: 95.3460  Samples: 50000

checkpoint-57:
TrainSummary: epoch=56 updates=2496 avg_step_time=0.223447s samples_per_step=512 samples_per_sec=2291.38
Test: [distributed-summary]  Loss: 0.8301  Acc@1: 80.6600  Acc@5: 95.3880  Samples: 50000

checkpoint-58:
TrainSummary: epoch=57 updates=2496 avg_step_time=0.223447s samples_per_step=512 samples_per_sec=2291.38
Test: [distributed-summary]  Loss: 0.8390  Acc@1: 80.5620  Acc@5: 95.3340  Samples: 50000

checkpoint-59:
TrainSummary: epoch=58 updates=2496 avg_step_time=0.223447s samples_per_step=512 samples_per_sec=2291.38
Test: [distributed-summary]  Loss: 0.8306  Acc@1: 80.5620  Acc@5: 95.3500  Samples: 50000

checkpoint-60:
TrainSummary: epoch=59 updates=2496 avg_step_time=0.223650s samples_per_step=512 samples_per_sec=2289.29
Test: [distributed-summary]  Time: 10.227s  Loss: 0.8329  Acc@1: 80.5700  Acc@5: 95.3280  Samples: 50000
```

对比：

```text
checkpoint-56 Top-1: 80.5360，相对 baseline 80.5980: -0.0620，相对方案 C 80.6820: -0.1460
checkpoint-57 Top-1: 80.6600，相对 baseline 80.5980: +0.0620，相对方案 C 80.6820: -0.0220
checkpoint-58 Top-1: 80.5620，相对 baseline 80.5980: -0.0360，相对方案 C 80.6820: -0.1200
checkpoint-59 Top-1: 80.5620，相对 baseline 80.5980: -0.0360，相对方案 C 80.6820: -0.1200
checkpoint-60 Top-1: 80.5700，相对 baseline 80.5980: -0.0280，相对方案 C 80.6820: -0.1120
81.0 target: 未达到
```

RefW 状态：

```text
epoch 55-60: RefW=0.000e+00
nonzero_refw_lines=0
```

## 最终审计

完整性：

```text
checkpoint_count: 50
latest_checkpoint: checkpoint-60.pth.tar
full-val rows: 50
所有 full-val Samples: 50000
nonzero_refw_lines: 0
```

最终 Top-5 checkpoint：

```text
1. checkpoint-52: Top-1 80.7240, Top-5 95.3400, delta vs baseline +0.1260, delta vs 方案 C +0.0420
2. checkpoint-53: Top-1 80.6680, Top-5 95.3520, delta vs baseline +0.0700, delta vs 方案 C -0.0140
3. checkpoint-57: Top-1 80.6600, Top-5 95.3880, delta vs baseline +0.0620, delta vs 方案 C -0.0220
4. checkpoint-54: Top-1 80.6460, Top-5 95.3740, delta vs baseline +0.0480, delta vs 方案 C -0.0360
5. checkpoint-50: Top-1 80.6300, Top-5 95.3600, delta vs baseline +0.0320, delta vs 方案 C -0.0520
```

最终结论：

```text
原版 OFQ 10 -> 60 对照实验完成。
原版 OFQ best: checkpoint-52 Top-1 80.7240。
是否超过 direct-resume baseline 80.5980: 是，+0.1260。
是否超过方案 C best 80.6820: 是，+0.0420。
是否达到 81.0: 否，差 -0.2760。
超过 baseline 的 checkpoint 数: 6。
超过方案 C 的 checkpoint 数: 1。
```

解释：

```text
原版 OFQ 在 10 -> 60 的长跑设置里确实能超过 80.5980 baseline，并且出现一个超过方案 C 的峰值 checkpoint-52。
但后续 checkpoint-53 到 checkpoint-60 多数回落到方案 C 以下，说明这个提升不是稳定平台，而是一个较高峰值。
因此如果目标是单 checkpoint best，原版 OFQ 本次对照已经超过方案 C；如果目标是稳定后段平台，方案 C 仍有对照价值。
```
