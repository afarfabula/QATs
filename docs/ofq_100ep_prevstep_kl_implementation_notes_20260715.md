# OFQ 100epoch prev-step KL 添加方式与实现细节

## 范围

本文只解释这次 100epoch 重跑：

```text
experiment: ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713
目标: 从 ImageNet pretrained / public OFQ 初始化开始，跑 Swin-T W4A4-family 100epoch，并在后段加入 sparse prev-step attention KL。
```

相关产物：

```text
run script:
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713.sh

progress:
/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_100ep_fromscratch_late_sparse_prevstep_refkl_progress_20260713.md

controller TSV:
/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_100ep_fromscratch_late_sparse_prevstep_refkl_controller_20260713.tsv

status TSV:
/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_100ep_fromscratch_late_sparse_prevstep_refkl_status_20260713.tsv

train log:
/mlx_devbox/users/quyanyi/playground/train_ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713.log
```

## 一句话说明

这次 KL 不是改 OFQ 基础 CE/KD loss，也不是全程固定加一个 KL。它是在 `qat_launch.py` 里启用 `ema_ref_attn_kl` 训练方案，创建一个 `prev_step` reference model，然后用 `DynamicSparsePrevStepKLController` 根据每个 epoch 的 full-val Top-1 回落决定下一轮是否对某个指定 attention head 加一个很小的 KL pulse。

最终加到 loss 里的形式是：

```text
loss = base_loss + current_ref_attn_kl_weight * ref_attn_kl_loss
```

其中本实验大多数时间 `current_ref_attn_kl_weight=0`，只有 controller 触发的少数 epoch 才变成 `1e-5`。

## 启动参数

run script 里的 KL 相关参数是：

```text
--train-scheme ema_ref_attn_kl
--ref-update prev_step
--ref-update-interval 50
--ref-attn-loss kl_ref
--ref-attn-kl-weight 0.0
--ref-head-mode custom_subset:8:4
--ref-warmup-epochs 0
--ref-attn-kl-drop-prob 0.50
--ref-attn-kl-clip 20.0

--dynamic-sparse-prevstep-kl
--dynamic-kl-start-epoch 51
--dynamic-kl-observe-until-epoch 50
--dynamic-kl-primary-heads 8:4,5:7,4:11
--dynamic-kl-secondary-heads 11:18,6:1
--dynamic-kl-avoid-heads 6:6,7:7,4:1,2:4,10:13,11:4,6:7,11:16
--dynamic-kl-drop-threshold 0.06
--dynamic-kl-strong-drop-threshold 0.14
--dynamic-kl-default-weight 0.00001
--dynamic-kl-strong-weight 0.00002
--dynamic-kl-max-weight 0.00002
--dynamic-kl-cooldown-epochs 6
--dynamic-kl-window-epochs 10
--dynamic-kl-max-pulses-per-window 3
--dynamic-kl-controller-tsv ...
--dynamic-kl-prior-source ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713_static_controller
```

注意几个关键点：

```text
1. ref_attn_kl_weight=0.0，说明默认不加 KL。
2. 真正的 KL 权重由 dynamic controller 在特定 epoch 改写。
3. dynamic_kl_start_epoch=51，前 50 epoch 只 observe。
4. ref_attn_kl_drop_prob=0.50，说明即使某个 epoch RefW 非零，也只有约一半 batch 实际应用 KL。
5. ref_attn_kl_clip=20.0，用来截断单次 head loss 的极端值。
```

## 代码入口

### 参数传递

`qat_launch.py` 负责把 CLI 参数传给 OFQ runtime。相关参数在命令构造和 parser/defaults 中都有映射：

```text
QATs/qat_launch.py:572-579
  --train-scheme
  --ref-update
  --ref-update-interval
  --ref-attn-loss
  --ref-attn-kl-weight
  --ref-attn-kl-drop-prob
  --ref-attn-kl-drop-scale
  --ref-attn-kl-clip

QATs/qat_launch.py:720
  --dynamic-sparse-prevstep-kl

QATs/qat_launch.py:8182-8189
  OFQ 训练方案、ref update 和 ref attention KL 参数定义

QATs/qat_launch.py:8332
  dynamic sparse prev-step KL controller 开关定义
```

### 参数合法性检查

`dynamic_sparse_prevstep_kl` 有硬约束：

```text
QATs/qat_launch.py:2321-2327
```

要求：

```text
1. train_scheme 必须是 ema_ref_attn_kl。
2. ref_update 必须是 prev_step。
3. 不能和 ref_attn_kl_weight_epoch_overrides 同时使用。
```

这保证了 dynamic controller 的语义是“prev-step refmodel 的稀疏 KL pulse”，而不是和其它权重覆盖机制混在一起。

## ref model 是怎么来的

当 `train_scheme=ema_ref_attn_kl` 时，runtime 会创建 `ref_model`。本实验使用：

```text
ref_update=prev_step
ref_update_interval=50
```

训练时每隔 `ref_update_interval` 个 optimizer update，把当前 model 拷贝到 ref_model：

```text
QATs/qat_launch.py:6871-6879

if update_step
  and train_scheme == "ema_ref_attn_kl"
  and ref_model is not None
  and ref_update == "prev_step"
  and local_update_count % ref_update_interval == 0:
      update_ref_model(model, ref_model, 0.0)
```

这里 `update_ref_model(model, ref_model, 0.0)` 可以理解为硬拷贝，而不是 EMA 平滑。因此 reference 是最近一步模型快照，目标是约束 attention relation 不要在量化训练中产生过大的 step-to-step 震荡。

## attention KL 怎么算

### 基础 pair loss

核心函数：

```text
QATs/qat_launch.py:4999-5020
attention_kl_pair_loss(student_attn, ref_attn, loss_type)
```

本实验使用 `loss_type=kl_ref`，公式对应：

```python
student_prob = student_attn.clamp_min(1e-8)
ref_prob = ref_attn.clamp_min(1e-8)
loss = F.kl_div(torch.log(student_prob), ref_prob, reduction="batchmean")
```

支持但本次未使用的类型：

```text
cosine
centered_cosine
symmetric_kl
js
```

### clip

核心函数：

```text
QATs/qat_launch.py:5023-5026
maybe_clip_ref_loss(loss, clip_value)
```

本次设置：

```text
ref_attn_kl_clip=20.0
```

含义：

```text
如果单个 attention KL loss 超过 20.0，就 clamp 到 20.0，避免极端 batch/head 把训练拉偏。
```

### head 级 consistency loss

核心函数：

```text
QATs/qat_launch.py:5032-5118
attention_kl_consistency_loss(student_attn_info, ref_attn_info, head_mode, loss_type, clip_value)
```

它会从 student/ref 的 attention info 中取出 attention probability list，然后按 `head_mode` 选择 layer/head。

本次 dynamic controller 每次 pulse 最终会将 `head_mode` 改成：

```text
custom_subset:layer:head
```

例如：

```text
custom_subset:8:4
custom_subset:5:7
custom_subset:4:11
```

这意味着每次 pulse 只对一个 head 的 attention 分布做 KL，而不是所有 attention head 都加 KL。

## KL 在 loss 里在哪里相加

真正加 loss 的代码在：

```text
QATs/qat_launch.py:6538-6584
```

关键逻辑：

```python
current_ref_attn_kl_weight = ref_attn_kl_weight

use_ref_scheme = (
    runtime_args.train_scheme == "ema_ref_attn_kl"
    and ref_model is not None
    and epoch >= runtime_args.ref_warmup_epochs
    and local_update_count >= runtime_args.ref_warmup_updates
    and (current_ref_attn_kl_weight > 0 or current_ref_logit_kl_weight > 0)
)

if use_ref_scheme:
    with torch.no_grad():
        ref_logits, ref_attn_info = ref_model(input)

    if current_ref_attn_kl_weight > 0:
        ref_attn_kl_loss = attention_kl_consistency_loss(...)

        if runtime_args.ref_attn_kl_drop_prob < 1.0:
            kl_gate = Bernoulli(ref_attn_kl_drop_prob)
            ref_attn_kl_loss = ref_attn_kl_loss * kl_gate

        loss = loss + current_ref_attn_kl_weight * ref_attn_kl_loss
```

所以这次 KL 的实际加入条件是：

```text
1. train_scheme=ema_ref_attn_kl
2. ref_model 存在
3. 当前 epoch/update 已过 warmup
4. current_ref_attn_kl_weight > 0
5. batch 级 kl_gate 通过
```

由于本实验 base `ref_attn_kl_weight=0.0`，所以绝大多数 epoch 不会满足第 4 条。只有 dynamic controller 给某个 epoch 分配了非零权重时，才会真正把 KL 加进 loss。

## dynamic controller 怎么控制 KL

controller 类：

```text
QATs/qat_launch.py:4780-4928
DynamicSparsePrevStepKLController
```

它的状态包括：

```text
enabled
start_epoch
observe_until_epoch
primary_heads
secondary_heads
avoid_heads
drop_threshold
strong_drop_threshold
default_weight
strong_weight
max_weight
cooldown_epochs
window_epochs
max_pulses_per_window
rolling_best
next_head
next_weight
cooldown_until
pulse_epochs
tsv_path
```

### 每个 epoch 开始前如何应用

代码位置：

```text
QATs/qat_launch.py:7587-7630
```

每个 epoch 开始前，rank 0 调用：

```python
dynamic_kl_controller.decision_for_epoch(epoch)
```

返回：

```text
epoch_ref_head_mode
epoch_ref_attn_kl_weight
epoch_dynamic_head
epoch_dynamic_spike_score
epoch_dynamic_reason
```

然后广播给所有 rank，并写回：

```python
runtime_args.ref_head_mode = epoch_ref_head_mode
runtime_args.ref_attn_kl_weight = epoch_ref_attn_kl_weight
```

如果当轮没有 pulse：

```text
ref_attn_kl_weight = 0.0
```

如果当轮有 pulse：

```text
ref_head_mode = custom_subset:L:H
ref_attn_kl_weight = 1e-5 或 2e-5
```

### 每个 epoch 验证后如何决定下一轮

代码位置：

```text
QATs/qat_launch.py:4852-4904
update_after_validation(...)
```

full-val 后读取：

```text
top1
top5
samples
```

维护：

```text
rolling_best
drop = previous_rolling_best - current_top1
```

如果当前 Top-1 刷新 rolling best，drop 就是 0。

### 触发条件

代码位置：

```text
QATs/qat_launch.py:4906-4928
_choose_next(...)
```

触发逻辑：

```text
1. epoch < start_epoch: 不触发
2. drop < drop_threshold: 不触发
3. 最近 window_epochs 内 pulse 数 >= max_pulses_per_window: 不触发
4. 依次扫描 primary_heads + secondary_heads
5. 跳过 avoid_heads
6. 跳过 cooldown 中的 head
7. 选中第一个可用 head
8. drop >= strong_drop_threshold 时用 strong_weight，否则 default_weight
9. weight 不超过 max_weight
```

本实验的具体阈值： 

```text
start_epoch=51
observe_until_epoch=50
drop_threshold=0.06
strong_drop_threshold=0.14
default_weight=1e-5
strong_weight=2e-5
max_weight=2e-5
cooldown_epochs=6
window_epochs=10
max_pulses_per_window=3
```

## 本次实际 KL 触发记录

最终 monitor summary： 

```text
controller_rows=100
controller_triggers=10
controller_observe_triggers=0
controller_selected_avoid=0
observe_nonzero_refw_lines=0
nonzero_refw_epochs=53,54,55,64,65,73,78,79,85,98
selected heads=8:4,5:7,4:11
```

### 第一组 pulse: epoch 52-55

```text
epoch 52:
  top1=80.2280
  rolling_best=80.3180
  drop=0.0900
  next_head=8:4
  next_weight=1e-5

epoch 53:
  applied_head=8:4
  applied_weight=1e-5
  next_head=5:7

epoch 54:
  applied_head=5:7
  applied_weight=1e-5
  next_head=4:11

epoch 55:
  applied_head=4:11
  applied_weight=1e-5
```

这组 pulse 后的同点对照： 

```text
checkpoint-54:
  KL    = 80.1920
  no-KL = 80.2880

checkpoint-55:
  KL    = 80.2120
  no-KL = 80.3180

checkpoint-60:
  KL    = 80.4480
  no-KL = 80.5260
```

说明第一组 pulse 没有带来正向拉升，no-KL 反而更高。

### 第二组 pulse: epoch 63-65

```text
epoch 63:
  drop=0.1200
  next_head=8:4
  next_weight=1e-5

epoch 64:
  applied_head=8:4
  next_head=5:7

epoch 65:
  applied_head=5:7
```

同期对照： 

```text
checkpoint-64:
  KL    = 80.3500
  no-KL = 80.4700

checkpoint-65:
  KL    = 80.3780
  no-KL = 80.4600
```

### 第三组 pulse: epoch 72-73

```text
epoch 72:
  drop=0.0840
  next_head=8:4
  next_weight=1e-5

epoch 73:
  applied_head=8:4
  no further trigger due to window_limit
```

### 第四组 pulse: epoch 77-79

```text
epoch 77:
  drop=0.0720
  next_head=5:7

epoch 78:
  applied_head=5:7
  next_head=8:4

epoch 79:
  applied_head=8:4
```

这段 KL 在 checkpoint-80 有局部高点：

```text
checkpoint-80:
  KL    = 80.6500
  no-KL = 80.5840
```

但 no-KL 在 checkpoint-77 已经达到 80.6360，所以 KL 的局部优势只有 0.014 的 best 差距。

### 第五组 pulse: epoch 84-85

```text
epoch 84:
  drop=0.1380
  next_head=8:4

epoch 85:
  applied_head=8:4
```

但 no-KL 在 checkpoint-82 达到 80.7920，已经超过 KL 全程 best。

### 最后一组 pulse: epoch 97-98

```text
epoch 97:
  drop=0.0600
  next_head=8:4

epoch 98:
  applied_head=8:4
```

最后结果：

```text
checkpoint-98:
  KL    = 80.7040
  no-KL = 80.7760

checkpoint-100:
  KL    = 80.7720
  no-KL = 80.6780
```

KL 在最终点更高，但 no-KL 全程 best 更高。

## 这次 KL 添加的关键特点

### 特点一：不是全程 KL，而是 sparse pulse

base weight 是 0： 

```text
ref_attn_kl_weight=0.0
```

只有 controller 触发时才临时把某个 epoch 的 `RefW` 变成 `1e-5`。

### 特点二：不是全头 KL，而是单 head KL

每次 pulse 用： 

```text
custom_subset:L:H
```

因此本次实际只约束过：

```text
8:4
5:7
4:11
```

### 特点三：reference 是 prev-step，不是 teacher

本次 attention KL 的 reference 是 `ref_model`，由当前模型周期性硬拷贝得到，不是 FP teacher。

FP teacher 仍用于 soft-KD 主链路，但不是这次 `ref_attn_kl_loss` 的 reference。

### 特点四：有 batch 级 dropout

```text
ref_attn_kl_drop_prob=0.5
```

因此即便 `RefW=1e-5`，也不是每个 batch 都加 KL。

### 特点五：controller 只根据 full-val Top-1 drop 触发

它不直接在线计算当前 head 的真实震荡强度，只使用静态 prior head list，然后根据 full-val Top-1 相对 rolling best 的 drop 触发。

这也是本次效果不强的一个重要原因：head 选择和触发时机可能不够精确。

## 本次实现带来的结论

从实现角度看，KL 机制是正确接入并实际运行了的： 

```text
1. args.yaml 启用了 ema_ref_attn_kl + prev_step。
2. controller TSV 有 100 行。
3. observe 阶段没有 RefW 非零。
4. dynamic 阶段有 10 次 trigger。
5. RefW 非零 epoch 与 controller 决策一致。
6. no-KL 对照全程 RefW=0，controller artifact absent。
```

但从效果看： 

```text
1. KL 版本 best=80.7720。
2. no-KL 对照 best=80.7920。
3. no-KL 在 checkpoint-82 已超过 KL 全程 best。
4. KL 的局部优势主要出现在 checkpoint-80 和 checkpoint-100，不能形成稳定 best 提升。
```

所以这次 100epoch 重跑里，KL 的添加方式可以总结为：

```text
工程上：KL 已正确添加，controller 正常触发，loss 路径生效。
算法上：当前固定-head sparse prev-step KL 没有证明有效，不能解释 80.7+ 的主要收益。
```

## 后续如果要改 KL，应该改哪里

优先改动点： 

```text
1. DynamicSparsePrevStepKLController 的 head 选择逻辑。
   当前是静态 primary/secondary/avoid list。
   应改为动态检测当前 attention relation oscillation 的 head。

2. _choose_next 的触发信号。
   当前只看 full-val Top-1 drop。
   可以引入 head-level oscillation score、attention relation delta、teacher/ref agreement。

3. KL 启动时间。
   当前 epoch 51 开始。
   从 no-KL 对照看，51-80 自然曲线并不弱，可能应该更晚启用，只做 ultra-sparse polish。

4. KL 权重和 dropout。
   当前 1e-5 + drop_prob 0.5 很轻。
   如果改强，需要同时有更严格的触发条件，避免扰乱自然收敛。

5. head pool。
   当前固定 8:4,5:7,4:11 未证明有效。
   应按当前 run 的在线诊断重新选，而不是复用历史 prior。
```

不建议继续做： 

```text
直接原样加长当前 fixed-head sparse prev-step KL。
```

原因：no-KL 严格对照已经证明当前版本不是 80.7+ 的主要来源。

