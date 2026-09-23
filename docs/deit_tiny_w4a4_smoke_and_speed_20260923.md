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

## 6. DeiT 注意力采集 + attention-relation ranking（本次实现）

两处改动，都很小：

1. **DeiT 注意力可被采集**：`src/deit_vision_transformer.py` 的 `Attention` 增加 `collect_attention` 开关与
   `select_attention_heads()`（支持 `collect_attention_head_indices` 的 head 子集），`Block.forward` 在开关打开时把
   注意力一起返回；`src/quantization/modules/attention.py` 的四个量化注意力（含 `QAttention_qkreparam`）在
   `collect_attention` 打开时返回量化后的 softmax 注意力，不再恒为 `None`。
   `qat_launch.py` 里 `is_attention_module()` / `set_selected_attention_heads()` / `enable_attention_collection()`
   原先只认 Swin，现在也认 DeiT。
2. **新增 attention-relation ranking 损失**（`attention_relation_ranking_loss()`）：教师在每个 (head, query) 行内取
   top-k 个 key，与所有教师分数严格更低的 key 组成有向对，用 `softplus(-(log s_c - log s_d))` 要求学生保持同样顺序；
   log 概率差等于 softmax 前的分数差，所以等价于约束 QK 分数的大小关系。教师只提供顺序，不提供数值。

用法（默认关闭，`--attn-rank-weight` 为 0 时完全不生效）：

```bash
--attn-rank-weight 1.0              # >0 才会自动打开 student/teacher 的注意力采集
--attn-rank-topk 1                  # 每行排序起点个数，默认 1
--ref-head-mode custom_subset:0:0,1:0,2:0,3:0,4:0   # 复用已有开关限制层/head
```

训练日志会打印一次 `Attention-relation ranking debug: ... valid_pairs=...`（用来确认真的拿到注意力、不是静默 0），
每个 log 区间里多一列 `AttnRank: x (avg)`。

### 6.1 速度影响（DeiT-Tiny bs32，400 updates，同一张卡）

| 配置 | 稳态 median s/step | 稳态 img/s | 整段 avg_step_time |
|---|---:|---:|---:|
| 不开 ranking（本次改动前） | 0.144 | 222 | 0.189 |
| 不开 ranking（本次改动后） | 0.137 | 234 | 0.195 |
| ranking，全部 36 个 head | 0.179 | 178 | 0.202 |
| ranking，5 个 head（`custom_subset`） | 0.150 | 213 | 0.191 |

结论：默认关闭时开销可忽略；全 head 打开约 +30% 单步（仍是 Swin-T bs32 的 1.85×），限定 head 后约 +9%。
DeiT 注意力形状是 `[B, 3, 198, 198]`（每行 198 个 key），比 Swin 窗口内的 49 大 4 倍，所以 head/layer 要按需限制。

## 7. 复现

```bash
cd /home_ext/quyanyi/tiger/resume_repos/QATs
# DeiT-Tiny 与 Swin-T 各 400 步, 同一张卡背靠背
GPU=4 STEPS=400 BATCH=32 LOGINTERVAL=25 TAG=speed RUNS=deit,swin \
  bash tmp_scripts/smoke_deit_vs_swin_speed_20260923.sh

# 只看 DeiT, 换 batch
GPU=4 STEPS=400 BATCH=128 LOGINTERVAL=25 TAG=bs128 RUNS=deit \
  bash tmp_scripts/smoke_deit_vs_swin_speed_20260923.sh

# 带 attention-relation ranking 的短跑（EXTRA 里的参数原样追加给 qat_launch.py）
GPU=4 STEPS=400 BATCH=32 LOGINTERVAL=25 TAG=rank RUNS=deit \
  EXTRA="--attn-rank-weight 1.0" bash tmp_scripts/smoke_deit_vs_swin_speed_20260923.sh
```

日志落在 `/tmp/qat_smoke_20260923/`，输出（checkpoint）落在 `/tmp/qat_runs_smoke_20260923/`，两者都在 `/tmp`，会随清理消失。
