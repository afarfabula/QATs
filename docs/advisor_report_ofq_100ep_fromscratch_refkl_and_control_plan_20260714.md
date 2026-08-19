# Swin-T W4A4-family 100epoch 从 public pretrained 起跑实验汇报

## 一句话结论

我们已经完成了一版从 ImageNet pretrained / public OFQ 初始化开始的 100 epoch Swin-T W4A4-family 训练实验，加入 late sparse prev-step attention KL 后，最终最好结果是：

```text
checkpoint-100
Top-1: 80.7720
Top-5: 95.4320
full validation samples: 50000
```

这版没有达到 81.0，也没有超过目前长跑中最好的 `resume10->210 late sparse prev-step KL` 的 `80.8280`，但它有明确的阶段性价值：证明 sparse prev-step KL controller 可以在从 public pretrained 起跑的 100epoch 设置里稳定工作，并且后段确实把曲线推到了 80.7+；主要瓶颈不是 controller 失效，而是前 50 epoch warmup 的自然高度不够。

## 这次实验到底验证了什么

本次实验不是从 `checkpoint-10`、`checkpoint-300` 或其他 QAT checkpoint resume，也不是 checkpoint soup / averaging / ensemble。它的定义是：

```text
ImageNet pretrained / public OFQ 初始化
+ OFQ public-family W4A4-family QAT 主链路
+ late sparse prev-step attention KL
+ 100 epoch
+ 每个 epoch 全量验证
```

因此它可以作为“从 public 初始化起跑的 100epoch KL 范式实验”向导师交代，但不能被表述成 strict W4A4 最终 SOTA。本项目当前这些 Swin-T 结果应称为 `W4A4-family`：主干是 W4/A4 量化链路，但需要继续按具体 `args.yaml` 区分 first/last layer 等是否严格 4bit。

## 已完成实验设置

实验名：

```text
ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713
```

关键配置：

```text
model: swin_t
method: ofq
dataset: ImageNet parquet, /tmp/imagenet1k_full_parquet
wbits / abits: 4 / 4
wq_mode / aq_mode: statsq / lsq
qk_reparam: true
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
batch_size: 64 per GPU
GPUs: 8 x H100
epochs: 100
scheduler_epochs: 100
lr: 2e-4
min_lr: 5e-6
weight_decay: 0.0
checkpoint interval: every epoch
```

KL 范式：

```text
train_scheme: ema_ref_attn_kl
ref_update: prev_step
base ref_attn_kl_weight: 0.0
dynamic_sparse_prevstep_kl: true
dynamic_kl_start_epoch: 51
dynamic_kl_observe_until_epoch: 50
primary heads: 8:4,5:7,4:11
secondary heads: 11:18,6:1
avoid heads: 6:6,7:7,4:1,2:4,10:13,11:4,6:7,11:16
default / strong KL weight: 1e-5 / 2e-5
clip: 20.0
drop_prob: 0.5
```

## 结果

完整性审计：

```text
checkpoint_count: 100
fullval_rows: 100
bad_sample_rows: 0
latest_checkpoint: checkpoint-100.pth.tar
best_checkpoint: checkpoint-100.pth.tar
best Top-1: 80.7720
best Top-5: 95.4320
target_81: 0
```

和历史结果的关系：

```text
original OFQ 10->110 best: 80.7520
dynamic sparse prev-step KL 10->110 best: 80.7600
late sparse prev-step KL 10->210 best: 80.8280
this 100ep from-pretrained KL best: 80.7720
```

这说明：

```text
1. 本实验略高于 10->110 原版 OFQ 和 10->110 dynamic KL 的最好点。
2. 本实验没有超过 10->210 late sparse KL 的 80.8280。
3. 本实验没有产生任何 81.0+ checkpoint。
```

后段稳定性：

```text
last20_avg: 80.6965
last10_avg: 80.7316
above original 10->110 best 80.7520: 3 checkpoints
above dynamic 10->110 best 80.7600: 2 checkpoints
above late 10->210 best 80.8280: 0 checkpoints
```

## 为什么没有到 81

主要原因是前期 warmup 高度不够，不是 KL controller 没有触发。

关键轨迹：

```text
checkpoint-20 best: 79.3600
checkpoint-31 best: 79.8120
checkpoint-40 best: 80.0900
checkpoint-51 best: 80.2140
checkpoint-80 best: 80.6500
checkpoint-91 best: 80.7240
checkpoint-100 best: 80.7720
```

解释：

```text
epoch 0-50 是 observe / no effective KL 阶段。
dynamic KL 从 epoch 51 后才开始有机会触发。
但进入 dynamic 阶段时 rolling best 只有 80.2140。
这给后段 KL 留下的提升空间太大，100epoch 内很难补到 81.0。
```

controller 审计：

```text
controller_rows: 100
controller_triggers: 10
observe_controller_triggers: 0
observe_nonzero_refw_lines: 0
selected avoid heads: 0
nonzero RefW epochs: 53,54,55,64,65,73,78,79,85,98
```

这个审计说明 controller 的边界是干净的：前 50 epoch 没有 KL pulse，后段 sparse pulse 正常触发，也没有误选 avoid heads。因此当前失败不是“KL 代码没跑起来”，而是“从 public pretrained 起跑的自然 OFQ 曲线在前半程没有足够高”。

## 给导师的建议表述

可以这样交代：

```text
我们把之前从 checkpoint resume 的调参，切换成了更干净的 100epoch 从 public pretrained 初始化起跑实验，目的是验证 sparse prev-step attention KL 是否可以作为完整训练范式的一部分，而不是只在 late resume 上做修补。

第一版 100epoch KL 实验已经完整跑完。结果 best Top-1 80.772，没有到 81，但超过了此前 10->110 原版 OFQ 的 80.752 和 10->110 dynamic KL 的 80.760。审计显示 KL controller 正常工作，后段有 10 次 sparse pulse，并且没有污染 observe 阶段，也没有选到我们标记为不稳定的 heads。

目前主要问题是从 public pretrained 起跑的前期 warmup 不够快：到 epoch 51 才 80.214，后段 KL 虽然把结果推到 80.772，但高度不够。因此下一步需要做严格的 no-KL 100epoch 对照，判断这 80.772 是自然 OFQ 长跑带来的，还是 sparse prev-step KL 在后段带来的增益。
```

## 第二个实验应该怎么设计

我同意下一步最适合做“完全相同 100epoch 但不启用 KL”的对照实验。这是交差意义最强的第二个实验，因为它回答的是最核心的消融问题：

```text
在相同 public pretrained 初始化、相同 OFQ public-family 主链路、相同数据、相同训练长度下，
late sparse prev-step KL 到底有没有带来可测收益？
```

推荐实验名：

```text
ofq_100ep_fromscratch_original_ofq_public_control_20260714
```

唯一应改变的变量：

```text
关闭 train_scheme=ema_ref_attn_kl
关闭 ref_update / refmodel
关闭 ref_attn_kl_weight
关闭 dynamic_sparse_prevstep_kl controller
不生成 controller pulse
```

其余保持一致：

```text
ImageNet pretrained / public OFQ 初始化
不 resume QAT checkpoint
method=ofq
model=swin_t
data=/tmp/imagenet1k_full_parquet
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
epochs=100
scheduler_epochs=100
lr=2e-4
min_lr=5e-6
weight_decay=0.0
epoch_checkpoint_interval=1
checkpoint_hist=100
same augmentation flags
same seed=42
same full validation every epoch
```

禁止项也保持一致：

```text
不使用 soup
不使用 checkpoint averaging
不使用 ensemble
不使用 A8->A4
不做 multi-checkpoint 拼接
不从 checkpoint-10 resume
```

## 第二个实验的成功/解释标准

这个实验不是为了“再赌一次 81”，而是为了形成可解释的消融证据。

建议用以下解释矩阵：

| no-KL 100ep 结果 | 解释 | 下一步 |
| --- | --- | --- |
| best 明显低于 80.772，例如低 0.05-0.15 | sparse prev-step KL 对后段保峰有正贡献；第一版 KL 没到 81 的主因是 early warmup 不足 | 保留 sparse KL，下一步优化前 50 epoch warmup / init |
| best 接近 80.772，例如差距在 ±0.03 | KL 增益很小，80.77 主要来自 OFQ 自然长跑 | 需要重新设计 KL 触发机制或 head 检测，不能只沿用当前 controller |
| best 高于 80.772 | 当前 KL 可能干扰了自然收敛，虽然后段有保峰但整体不划算 | 降低 KL 频率/权重，或只在更晚阶段极稀疏启用 |
| no-KL 也接近或超过 80.828 | 从 public pretrained 起跑 100epoch 本身有潜力，KL 应作为 polish 而不是主训练范式 | 先固定更强 OFQ baseline，再做 late-only KL |
| no-KL 也到 81 | 问题转为 OFQ baseline 复现和稳定性；KL 不再是达到 81 的必要条件 | 对 OFQ baseline 做多 seed / 稳定性验证 |
| no-KL 明显低于 80.7 | KL 的后段保峰价值成立，但 81 需要 warmup 改造 | 做 warmup 强化实验，例如前 50 epoch LR/scheduler 或 quantizer 稳定化 |

最低需要报告的指标：

```text
best checkpoint / Top-1 / Top-5
checkpoint-40, checkpoint-51, checkpoint-80, checkpoint-100 Top-1
last20_avg
last10_avg
above_80.598 / above_80.682 / above_80.724 / above_80.752 / above_80.760 / above_80.772 / above_80.828 / target_81 counts
fullval rows 是否 100
Samples 是否全为 50000
RefW 是否始终为 0
args.yaml 是否没有 train_scheme/ref/dynamic KL 启用
```

## 第二个实验的预期价值

这组对照能让汇报从“我们又跑了一版没到 81”变成一个完整实验闭环：

```text
实验 1：100epoch + sparse prev-step KL
结果：80.772
结论：KL 后段能工作，但 early warmup 不够，没到 81。

实验 2：100epoch + no-KL strict control
目的：隔离 KL 的真实贡献，判断 80.772 是 OFQ 自然收敛还是 KL 带来的。
```

如果 no-KL 明显低于 80.772，就可以对导师说：方向是有效的，但第一版训练范式把 KL 放在了后段，受限于前期 warmup；下一阶段应该优化前 50 epoch，而不是否定 attention relation KL。

如果 no-KL 和 80.772 持平甚至更高，就应该诚实调整方向：当前 sparse prev-step KL 对从 public pretrained 起跑的 100epoch 设置贡献不足，需要重新设计动态 head 检测、触发时机或 KL 只作为极晚期 polish。

## 我建议的下一步

下一步就跑这个严格 no-KL 100epoch 对照。不要同时改 LR、batch、augmentation、scheduler、head list 或 warmup，否则无法回答 KL 是否有效。

完成第二个实验后，再决定第三个实验：

```text
如果 no-KL < KL：第三个实验做 warmup 强化 + sparse KL。
如果 no-KL ~= KL：第三个实验做更强动态 head 检测，而不是固定 head list。
如果 no-KL > KL：第三个实验做 no-KL baseline 延长或 late-only ultra-sparse KL。
```

