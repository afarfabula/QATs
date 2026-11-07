# DeiT-Tiny W4A4-family 100epoch baseline

日期：2026-11-07

## 目标

把当前 `qat_launch.py` OFQ 训练链路正式适配到
`deit_tiny_distilled_patch16_224`，并在本机仅剩的两张卡 `4,7` 上运行
100epoch、每轮 50k ImageNet 全量验证的纯 KD 基线。

## 配方

| 项 | 值 |
|---|---|
| 学生 | `deit_tiny_distilled_patch16_224` |
| 教师 | 同一 FP 预训练模型 |
| 量化 | 主 block W4A4-family；StatsQ weights + LSQ activations |
| QK | reparameterized，type 0 |
| 首尾层 | patch embedding 与两个 classifier head 为 W8A8 |
| 目标 | `T^2 * KD(T=2.75)`，pure soft KD，无 CE/rank 辅助项 |
| 更新 batch | 64/GPU x 2 ranks x accum 4 = 512 images |
| 训练 | 100 epoch，5 epoch warmup，cosine LR |
| 数据 | `/datadisk2/linyichen/OFQ/ImageNet-1K` (folder) |
| GPU | `4,7` |

DeiT 蒸馏模型 training 模式返回 `(cls_logits, dist_logits)`，eval 模式返回
两个 head 的平均 logits。统一入口现在在训练时先将两个 head 平均，再计算
KD；这样既与 eval 语义一致，也让 `head_dist` 参与反向传播，避免 DDP
报 unused parameter。

## 启动

```bash
cd /home_ext/quyanyi/tiger/resume_repos/QATs
setsid nohup bash \
  tmp_scripts/run_deit_tiny_w4a4_baseline_100ep_2gpu_20261107.sh \
  > /datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_baseline_100ep_2gpu_20261107.runner.log \
  2>&1 < /dev/null &
```

短跑检查：

```bash
MAX_UPDATES=20 SKIP_VALIDATE=1 CHECKPOINT_HIST=0 \
  EXP=deit_tiny_w4a4_baseline_smoke_20261107 \
  bash tmp_scripts/run_deit_tiny_w4a4_baseline_100ep_2gpu_20261107.sh
```

## Smoke 验证

第二次 smoke 在修复 DeiT 双头梯度后通过：

- 2 ranks / GPU `4,7`
- `exit_status=0`
- 20 optimizer updates
- 墙钟 73 秒
- 稳态约 `0.19s` / micro-step，约 650 images/s
- `Train: 0 [100/10009]`，loss 正常下降
- `head_dist` 缺梯度错误消失

## 运行产物

- 日志：
  `/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_baseline_100ep_2gpu_20261107.log`
- 输出：
  `/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_baseline_100ep_2gpu_20261107/`
- 每 epoch 全量验证 50,000 张
- `checkpoint_hist=2`

## 状态

正式 100epoch 已于 2026-11-07 启动并进入稳定训练。最终 Top-1 结果待 100
个 epoch 完成后补入本文件。
