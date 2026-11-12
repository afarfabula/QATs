# DeiT-Tiny W2A2 20 epoch objective alignment

日期：2026-11-12

## 1. 问题

最近一批 DeiT-Tiny W2A2 20 epoch 实验使用了纯软 KD：

```text
L = T^2 * KL(p_teacher || p_student), T=2.75
```

启动参数为 `kd_hard_and_soft=0`。因此训练目标没有 ground-truth CE。

这和官方 OFQ DeiT-T 训练脚本不一致。官方脚本使用：

```text
--use-kd --kd_hard_and_soft 1
```

`KDLossSoftandHard` 的实现是：

```text
L = CE(student_cls_logits, ground_truth)
  + KL(student_dist_logits, teacher_logits)
```

两项等权，官方 `KLLossSoft` 默认温度 `T=1.0`。官方 W2A2 DeiT-T
配方同时关闭 MixUp/CutMix，并训练 300 epoch 后再做 CGA。

因此，最近这批结果可以作为纯软 KD 条件下的内部消融，但不能作为
OFQ 论文目标的复现。

## 2. 2026-11-12 前的 20 epoch 结果

共同条件：DeiT-Tiny distilled、主 block W2A2、StatsQ/LSQ、QK reparam、
patch/head W8A8、seed 42、每轮 50k full-val。

| 实验 | 目标 | Best Top-1 | Final Top-1 | Final Top-5 |
|---|---|---:|---:|---:|
| 纯 soft KD baseline | `T^2 * KL` | 47.736 @ep18 | 47.660 | 74.502 |
| teacher attention rank | soft KD + `0.3 * rank` | 47.446 @ep19 | 47.446 | 74.458 |
| block 8-11 teacher KL | soft KD + `0.0047 * KL` | 47.148 @ep19 | 47.148 | 74.048 |
| block 0-3 teacher KL | soft KD + `0.0074 * KL` | 47.074 @ep19 | 47.074 | 74.126 |

rank 和 attention KL 都没有超过纯 soft KD 基线。固定 KL 权重只保证
运行初期约 15% loss 占比；原始 KL 随训练增大后，后四层占比约为 16.6%，
前四层约为 23.7%。这说明 loss 占比不是稳定的梯度预算。

## 3. 官方目标对齐

`qat_launch.py` 新增 `--ofq-official-kd-heads`：

- 默认关闭，保持历史实验行为；
- 与 `kd_hard_and_soft=1` 一起使用时，保留 DeiT 双头输出；
- CE 使用 `cls` head 和 ground truth；
- KL 使用 `dist` head 和冻结 FP teacher。

官方对齐实验入口：

```bash
bash tmp_scripts/run_deit_tiny_w2a2_official_cekld_t1_noaug_20ep_4gpu_20261112.sh
```

该实验使用：

```text
CE(cls, GT) + KL(dist, teacher)
T = 1.0
MixUp = 0
CutMix = 0
20 epoch
```

其余本地 W2A2 配置保持不变，便于和已有 20 epoch 结果对照。

## 4. 结论

在官方 CE+KD 对照完成前，不应继续根据纯 soft KD 曲线判断注意力
rank/KL 是否有价值。后续辅助目标实验应建立在 CE+KD 主目标上，并优先
按任务 margin 或实际离散 code flip 归因，而不是按量化保真度或抖动大小。
