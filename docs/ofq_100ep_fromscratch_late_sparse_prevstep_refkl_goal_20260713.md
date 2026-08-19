# OFQ 100epoch from-scratch late sparse prev-step KL goal

## 目标

设计并执行一版 Swin-T W4A4-family / OFQ public-family 的 100 epoch 重头训练实验，使用 late sparse prev-step attention KL，目标冲击 Top-1 81.0。

这里的“重头训练”指从 ImageNet pretrained / public OFQ 初始化开始，不从 `checkpoint-10` 或其他 QAT checkpoint resume；不是随机初始化训练。

## 历史依据

已有关键结论：

```text
original OFQ 10->110 best: 80.7520
dynamic sparse prev-step KL 10->110 best: 80.7600
late sparse prev-step KL 10->210 best: 80.8280
late sparse prev-step KL 10->210 target_81: 0
```

因此，prev-step KL 方向有效，但旧 controller 还不足以达到 81.0。新实验不再只是延长旧配置，而是面向 100 epoch from-scratch 重新设计训练范式。

## 主链路

保持 OFQ public-family：

```text
method=ofq
model=swin_t
dataset=/tmp/imagenet1k_full_parquet
wbits=4
abits=4
wq_mode=statsq
aq_mode=lsq
qk_reparam=true
qk_reparam_type=0
kd_hard_and_soft=0
teacher_soft_temperature=2.75
batch_size=64 per GPU
nproc_per_node=8
epoch_checkpoint_interval=1
checkpoint_hist>=100
```

禁止：

```text
soup
checkpoint averaging
ensemble
A8->A4
multi-checkpoint 拼接
```

## 训练范式

### Stage A: epoch 0-20

纯 OFQ public-family warmup，不启用有效 KL。

目的：

```text
让 quantizer、QKR、soft KD、activation scale 自然稳定；
避免 early attention relation 被坏 reference 约束。
```

### Stage B: epoch 20-50

启用 prev-step refmodel runtime，但 KL 权重仍为 0，只 observe。

关键配置：

```text
train_scheme=ema_ref_attn_kl
ref_update=prev_step
ref_update_interval=50
ref_attn_kl_weight=0.0
dynamic_sparse_prevstep_kl=true
dynamic_kl_start_epoch=51
dynamic_kl_observe_until_epoch=50
```

目的：

```text
保持 runtime 路径和后续一致；
建立 rolling best / drop / controller TSV 记录。
```

### Stage C: epoch 51-85

进入 late sparse prev-step KL 主阶段，使用 controller 对自然高点后的回落进行补救和保峰。

推荐静态 controller 配置：

```text
dynamic_kl_start_epoch=51
dynamic_kl_observe_until_epoch=50
dynamic_kl_primary_heads=8:4,5:7,4:11
dynamic_kl_secondary_heads=11:18,6:1
dynamic_kl_avoid_heads=6:6,7:7,4:1,2:4,10:13,11:4,6:7,11:16
dynamic_kl_drop_threshold=0.06
dynamic_kl_strong_drop_threshold=0.14
dynamic_kl_default_weight=1e-5
dynamic_kl_strong_weight=2e-5
dynamic_kl_max_weight=2e-5
dynamic_kl_cooldown_epochs=6
dynamic_kl_window_epochs=10
dynamic_kl_max_pulses_per_window=3
ref_attn_kl_clip=20.0
ref_attn_kl_drop_prob=0.5
ref_attn_loss=kl_ref
```

### Stage D: epoch 86-100

保峰 / polish 阶段。

如果不改代码，则沿用 Stage C 的静态配置跑完，并在日志中记录该限制。

如果实现成本低且安全，可以增加后段降权能力：

```text
drop_threshold=0.10
default_weight=5e-6
strong_weight=1e-5
max_weight=1e-5
ref_attn_kl_clip=10.0
```

优先级：

```text
不改代码静态跑完 > 为后段降权引入不稳定代码改动
```

## 成功标准

```text
最低通过: best Top-1 > 80.8280
有效通过: best >= 80.90，或至少 3 个 checkpoint > 80.8280，或 last20_avg >= 80.70
强通过: best Top-1 >= 81.0
失败判据: best <= 80.8280 且 last20_avg <= 80.6592
```

## 执行要求

1. 创建 run 脚本、monitor 脚本、中文进度文档。
2. 先 dry-run，确认参数无误，且无 soup / checkpoint averaging / A8->A4。
3. 在真实 worker GPU 环境启动训练。
4. 启动后核验 `args.yaml`、GPU、首条 `Train:`、`RefW`、controller TSV。
5. 持续轮询 checkpoint、full-val、RefW、controller、summary。
6. 按阶段写中文日志。
7. 最终做完整审计。

## 最终审计清单

```text
checkpoint-1 到 checkpoint-100 是否完整生成
full-val rows 是否完整且 Samples=50000
observe 阶段是否没有有效 KL pulse
dynamic 阶段 KL 是否只由 controller 触发
avoid heads 是否从未被选中
args.yaml 是否符合预期
best checkpoint / Top-1 / Top-5
超过 80.5980 / 80.6820 / 80.7240 / 80.7520 / 80.7600 / 80.8280 / 81.0 的 checkpoint 数量
last20_avg / last10_avg
controller 触发时机、head、weight、drop、window limit、cooldown
```

## 交付物

```text
训练输出目录
run script
monitor script
status TSV
refw TSV
controller TSV
monitor summary
中文进度文档
最终审计和结论
```
