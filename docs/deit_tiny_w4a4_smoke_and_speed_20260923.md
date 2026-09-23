# DeiT-Tiny W4A4 QAT 冒烟与速度对比

日期: 2026-09-23
机器: 8×RTX 3090 节点, GPU 4

## 0. 结论

1. **DeiT-Tiny 的 OFQ/QAT 链路跑通**：`W4A4 + statsq/lsq + QK-reparam + KD + bf16 + folder ImageNet`，400 步无 NaN，
   `Stopped early after 400 optimizer updates`（顺带验证了 `--max_train_updates` 现在按 optimizer update 计数）。
2. **迭代确实更快，值得上**：同一张卡、同样设置（bs32）下稳态单步 **0.144s vs Swin-T 0.331s**，
   吞吐 **222 vs 97 img/s（2.3×）**；整段平均（含启动/数据停顿）**169 vs 94 img/s（1.8×）**。
3. DeiT-Tiny 还能吃更大 batch（bs64 ≈ 339 img/s、bs128 稳态 ≈ 401 img/s），而 **Swin-T 在 24GB 的 3090 上 bs64 直接 OOM**。
4. 代价：数据流水线的周期性停顿把整段平均拖低（bs128 稳态 401 img/s 但整段只有 171 img/s），这个开销两个模型都要付。

## 1. 目的

- 确认 `deit_tiny_distilled_patch16_224` 能走通现有 QAT 主链路（后面要用它做 attention-relation 约束的实验）。
- 量化"换成 DeiT 能省多少"：single-GPU step time / 吞吐 / 单 epoch 时长，和 Swin-T 在同等设置下对比。
- 速度不够快就不值得上 DeiT，所以这是决策依据，不做精度结论。

## 2. 设置

```text
入口        qat_launch.py --method ofq --stage train (统一 OFQ 路径, 单进程)
模型/教师   deit_tiny_distilled_patch16_224 / swin_t（同架构做 teacher, 随机权重, 只测速）
配置        configs/deit_default_imagent.attn_q.yml / configs/swin_t_imagenet.attn_q.yml
量化        W4A4, --wq-mode statsq --aq-mode lsq, per-channel, clip learnable, --qk-reparam --qk-reparam-type 0
训练        KD (--use-kd --kd-hard-and-soft 1), amp bf16, grad-accum 1, seed 42, --skip_validate
数据        /datadisk2/linyichen/OFQ/ImageNet-1K (folder, 只读)
日志        /tmp/qat_smoke_20260923/（/tmp 会被清理，长期保留请看本文件里的表格）
脚本        tmp_scripts/smoke_deit_vs_swin_speed_20260923.sh
```

速度对比统一用 `--max_train_updates 400`（= 400 个 optimizer update），`--log-interval 25`。

## 3. 结果

### 3.1 单步时间与吞吐（同一张 GPU 4，同一时间段，背靠背跑）

| 模型 | batch | updates | 稳态 median s/step | 稳态 img/s | 整段 avg_step_time | 整段 img/s |
|---|---:|---:|---:|---:|---:|---:|
| DeiT-Tiny | 32 | 400 | **0.144** | **222** | 0.189 | 169 |
| DeiT-Tiny | 32 | 400（复跑） | 0.135 | 238 | 0.191 | 168 |
| Swin-T | 32 | 400 | 0.331 | 97 | 0.339 | 95 |
| DeiT-Tiny | 64 | 200 | 0.189 | 339 | 0.399 | 160 |
| DeiT-Tiny | 128 | 400 | 0.319 | 401 | 0.750 | 171 |
| Swin-T | 64 | 200 | — | — | — | **OOM** |

说明：

- "稳态 median"：跳过前 2 个 log 区间后的中位数，代表"数据一到位就是多少"（和运行手册里 Swin-T 0.33s 的基准一致）。
- "整段 avg_step_time"：`TrainSummary` 的整段平均，含首步预热与数据停顿，短跑里偏悲观。
- 数据停顿实测：`Data` 多数区间 0.004s，但个别区间 2–13s（bs128 那次最大 12.8s），与运行手册第 9 节的结论一致。

### 3.2 单 epoch 估算（1×3090，train 集 1,281,167 张）

| 配置 | 稳态推算 | 按整段平均推算 |
|---|---:|---:|
| DeiT-Tiny bs32 | 1.6 h | 2.1 h |
| DeiT-Tiny bs64 | 1.05 h | 2.2 h |
| DeiT-Tiny bs128 | 0.89 h | 2.1 h |
| Swin-T bs32 | 3.7 h | 3.8 h |

稳态口径下 DeiT bs64 ≈ Swin-T bs32 的 **3.5×**；整段口径下也是 **1.8×** 以上。
（数据停顿主要由共用节点的 CPU/磁盘争用造成，两个模型同等承受，所以稳态口径更能反映算力差别。）

## 4. 为跑通做的改动

| 改动 | 位置 | 原因 |
|---|---|---|
| 新增 `--num-aug-repeats`，并在 `world_size<=1` 时把非 0 值强制置 0 并打印警告 | `qat_launch.py`（parser + `build_ofq_runtime_config` 归一化段） | DeiT 官方 recipe 里 `num_aug_repeats: 3`，而 timm 的 `create_loader` 只在分布式下支持非 0；单卡冒烟会直接 `AssertionError` |
| 新增冒烟/测速脚本 | `tmp_scripts/smoke_deit_vs_swin_speed_20260923.sh` | 一条命令跑 DeiT 与 Swin 的同设置对比 |

必须显式传 `--model-type deit`（和 `--teacher-type deit`）：`build_ofq_runtime_config` 的默认值里 `model_type="swin"`
优先于 `infer_ofq_model_type`，不传会拿 Swin 的量化包装去包 DeiT，报 `KeyError: Conv2d`。

## 5. 权重与其余准备

- `deit_tiny_distilled_patch16_224-b40b3cf7.pth`（23.7MB）已下载并校验 sha256=`b40b3cf7…`，
  放到 `$TORCH_HOME/hub/checkpoints/`（即 `/datadisk2/quyanyi/cache/torch/hub/checkpoints/`）和 `/home_ext/quyanyi/qat_weights/`。
  用本仓库模型 `load_state_dict(..., strict=True)` 验证：**missing=0, unexpected=0**。
  所以后续带 `--pretrained --teacher-pretrained` 的运行不再需要联网。
- 离线 attention 震荡分析脚本（`third_party/OFQ/tools/offline_attention_oscillation.py` 与
  `tmp_scripts/analyze_attn_relation_oscillation_*.py`）是 Swin 专用（`SWIN_STAGE_SPECS`），DeiT 要用需另写 probe。

## 6. 上 attention 约束之前还差的两处代码

1. `third_party/OFQ/src/quantization/modules/attention.py` 里四个 DeiT 量化注意力（含 standard 的
   `QAttention_qkreparam`）forward 都是 `return x, None`，**不返回注意力矩阵**，因此 DeiT 上任何 attention-KL /
   attention-ranking 目前都会静默为 0。需要照 `QAttention_swin` 的写法，在 `qqkkvv` / `collect_attention` 时返回
   softmax 后的注意力（并按需支持 head 子集）。
2. DeiT 的 FP `Attention`（`src/deit_vision_transformer.py`）只有 `qqkkvv` 开关，没有 `collect_attention` /
   `collect_attention_head_indices`；上游 `set_selected_attention_heads()` 对 DeiT 无效（只在 loss 层切片生效）。

另外注意 DeiT 的注意力形状是 `[B, 3, 198, 198]`（每行 198 个 key），比 Swin 窗口内的 49 大 4 倍，
做 attention-ranking 时成对张量规模要按这个量级估算。

## 7. 复现

```bash
cd /home_ext/quyanyi/tiger/resume_repos/QATs
# DeiT-Tiny 与 Swin-T 各 400 步, 同一张卡背靠背
GPU=4 STEPS=400 BATCH=32 LOGINTERVAL=25 TAG=speed RUNS=deit,swin \
  bash tmp_scripts/smoke_deit_vs_swin_speed_20260923.sh

# 只看 DeiT, 换 batch
GPU=4 STEPS=400 BATCH=128 LOGINTERVAL=25 TAG=bs128 RUNS=deit \
  bash tmp_scripts/smoke_deit_vs_swin_speed_20260923.sh
```

日志落在 `/tmp/qat_smoke_20260923/`，输出（checkpoint）落在 `/tmp/qat_runs_smoke_20260923/`，两者都在 `/tmp`，会随清理消失。
