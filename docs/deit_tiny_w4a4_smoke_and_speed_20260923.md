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

## 6. Attention-relation ranking（模块化，DeiT 与 Swin 共用）

### 6.1 模块

损失本体在 `third_party/OFQ/src/attn_relation_ranking.py`，**与模型无关**，只要求 attention module 在
`collect_attention` 打开时返回 softmax 后的注意力 `[*, heads, tokens, tokens]`：

```python
attention_relation_ranking_loss(student_attn_info, teacher_attn_info, heads=None, topk=1)
# heads=None 用全部 (layer, head)，或传 ((layer_idx, head_idx), ...)
# 返回 (loss, 有效对数)
```

规则：教师在每个 (layer, head, query) 行内取 top-k 个 key，与所有教师分数严格更低的 key 组成有向对，
用 `softplus(-(log s_c - log s_d))` 要求学生保持同样顺序。log 概率差等于 softmax 前的分数差，所以等价于
约束 QK 分数的大小关系；教师只给顺序不给数值，教师值 ≤ 0（mask / dropout）与并列的对都排除。

模型侧只需保证注意力能被采集：

- **Swin**：本来就有（`ShiftedWindowAttention` / `QAttention_swin*` 的 `collect_attention`），未改。
- **DeiT**：本次补上。`src/deit_vision_transformer.py` 的 `Attention` 增加 `collect_attention` 与
  `select_attention_heads()`，`Block.forward` 采集时返回注意力；`src/quantization/modules/attention.py` 的四个量化
  注意力（含 `QAttention_qkreparam`）在 `collect_attention` 打开时返回量化后的 softmax 注意力。
  `qat_launch.py` 的 `is_attention_module()` / `set_selected_attention_heads()` / `enable_attention_collection()`
  原先只认 Swin，现在也认 DeiT。

### 6.2 用法

默认关闭，`--attn-rank-weight` 为 0 时完全不生效（也不会打开注意力采集）：

```bash
--attn-rank-weight 1.0              # >0 才会自动打开 student/teacher 的注意力采集
--attn-rank-topk 1                  # 每行排序起点个数，默认 1
--ref-head-mode custom_subset:0:0,1:0,2:0,3:0,4:0   # 复用已有开关限制层/head
```

训练日志会打印一次 `Attention-relation ranking debug: ... valid_pairs=...`（用来确认真的拿到注意力、不是静默 0），
每个 log 区间里多一列 `AttnRank: x (avg)`。

### 6.3 速度影响（bs32，400 updates，同一张卡）

| 模型 | 配置 | 稳态 median s/step | 稳态 img/s | 有效对/step |
|---|---|---:|---:|---:|
| DeiT-Tiny | 不开 ranking | 0.137 | 234 | — |
| DeiT-Tiny | ranking，全 head | 0.179（+30%） | 178 | 44,928,813 |
| DeiT-Tiny | ranking，5 个 head | 0.150（+9%） | 213 | 6,240,301 |
| Swin-T | 不开 ranking | 0.331 | 97 | — |
| Swin-T | ranking，全 head | 0.500（+51%） | 64 | 68,640,466 |
| Swin-T | ranking，5 个 head | 0.338（+2%） | 95 | 1,505,267 |

结论：默认关闭时开销可忽略；全 head 打开比较贵（DeiT +30%、Swin +51%），用 `--ref-head-mode custom_subset:...`
限定 5 个 head 后基本回到基线（+9% / +2%）。DeiT 的注意力是 `[B, 3, 198, 198]`（每行 198 个 key），Swin 是
`[B*num_windows, heads, 49, 49]`，两者成对规模都随 head 数线性变化，所以 head 选择是这个损失唯一需要调的性能旋钮。

一个已知差异：DeiT 返回 dropout 之前的注意力，Swin 沿用既有实现返回 dropout 之后的（既有 KL 路径一直如此）。

## 7. 分类 logits ranking（最朴素版，先做这个）

模块 `third_party/OFQ/src/logits_ranking.py`：

```python
logits_ranking_loss(student_logits, teacher_logits, topk=5)   # -> (loss, 有效对数)
```

规则和 `rank_idea/` 文档里那套一致：取教师分数最高的 top-k 个**类别**，与所有教师分数严格更低的类别组成
有向对，用 `softplus(-(s_c - s_d))` 要求学生在同样两个类别上保持同样顺序；并列的对不计入；先按样本对有效对
取平均，再对样本取平均。DeiT 蒸馏模型训练时返回 `(cls, dist)`，这里取 `cls`（与仓库既有 KD 辅助函数一致）。

```bash
--logit-rank-weight 1.0     # 需要 KD（teacher 才有 logits），权重为 0 时完全不生效
--logit-rank-topk 5         # 默认 5
```

日志多一列 `LogitRank: x (avg)`，并打印一次 `Logits-ranking debug: ... valid_pairs=...`。

验证：

- 单元级：同序 0.151 / 反序 2.484；C=6 时 top-5 每样本 15 对、top-1 每样本 5 对、top-2 每样本 9 对；并列被排除；
  梯度对高排名类别为负、低排名为正；传 `(cls, dist)` 元组也正常。
- 端到端：DeiT 与 Swin 各 400 步（bs32，`--logit-rank-weight 1.0`）都跑通，
  `valid_pairs=159,510`（= 4,985 对/样本 × 32）与 `159,509`（Swin 有极少数并列），
  `LogitRank` 分别从 0.777→0.484、0.700→0.241 下降。
- 开销：同一时段 A/B（DeiT bs32）基线 0.228 vs +ranking 0.231 s/step = **+1.3%**；Swin 0.331 vs 0.336 = +1.5%。
  张量只有 `[B, k, C] ≈ 1.6e5` 个元素，可以认为几乎免费。
  （注意：跨时段比较不能用于判开销——同配置基线在同一天里从 0.137 漂到 0.228 s/step，是共用节点被别人占满导致的。）

## 8. 100 epoch 时长估算

口径：ImageNet-1k train 1,281,167 张/epoch；本机单卡实测（bs32/卡、KD、W4A4、QK-reparam、bf16）
DeiT-Tiny 0.137 s/step = 234 img/s、Swin-T 0.331 s/step = 97 img/s；多卡按本机实测的 DDP 近线性
（2 卡 1.9×，运行手册 §9）外推；不含每 epoch 的 50k 全量验证（8 卡下约 +30 min/100ep，单卡约 +3.5 h/100ep）。

| 卡数 | DeiT 单 epoch | DeiT 100 epoch | Swin 单 epoch | Swin 100 epoch |
|---:|---:|---:|---:|---:|
| 1 | 1.5 h | **6.4 天** | 3.7 h | **15.3 天** |
| 2 | 48 min | 3.3 天 | 1.9 h | 8.1 天 |
| 4 | 25 min | 1.7 天 | 1.0 h | 4.1 天 |
| 8 | 12 min | **20 h** | 30 min | **50 h** |

注意这是"算力口径"。本机是多人共用的 8×3090 节点，实测长期平均会被数据流水线停顿拉高：运行手册 §9 里
4 卡的 2-epoch 复现是 2.2 h/epoch（同口径算力估计只有 1.0 h/epoch），即本机现实系数约 2×。
按这个系数，8 卡 100 epoch 大约是 DeiT 30~44 h、Swin 75~110 h；单卡则分别约 6~12 天和 15~30 天。

同一张卡上同配置的相对波动可以很大：今天从空闲时的 0.137 s/step（DeiT bs32）漂到别人把 2/5/6/7 号卡占满时的
0.228 s/step（+66%）。所以下面这张表按"能拿到空闲卡"来读，抢不到卡时按 1.5~2× 折算。

### 8.1 四卡 Swin-T W4A4 + logits ranking 实测（2026-09-24）

4 卡（GPU 4,5,6,7）、bs32/卡、KD + `--logit-rank-weight 1.0`、400 步：

| 指标 | 实测 |
|---|---|
| 每 rank micro-step（稳态中位） | 0.362 s（15 个区间的中位数） |
| 聚合吞吐 | 128 / 0.362 = **354 img/s**（单卡 97 img/s 的 3.65×，DDP 近线性） |
| 每 epoch（10,009 micro-step，global batch 128） | **约 1.0 h**（含启动/小停顿按 0.373 s/step 算是 1.04 h） |
| 100 epoch 训练 | **约 4.2~4.3 天** |
| 加上每 epoch 50k 全量验证 | 再 +约 1.7 h（100 epoch 合计） |

对应启动脚本：`tmp_scripts/run_swin_w4a4_100ep_logitrank_4gpu_20260924.sh`
（bs32/卡 × 4 卡 × accum 4 = 512 图/优化步，与历史 8×H100 对照一致；`RANK_W` 控制 ranking 权重）。

### 8.2 logits ranking 权重定标（2026-09-24）

方法沿用 `rank_idea` 文档的 ρ 思路，但共享张量换成学生 logits（两个损失都作用在它上面）：

1. **梯度范数探针**（`--extra-arg=--logit-rank-probe`，只跑一次，用真实配方 bs32 + 预训练学生/教师 + KD T=2.75）：

   ```text
   LogitRank grad probe: ||dKD/ds||=1.5586e-02  ||dRank/ds||=3.9782e-02  ratio=0.392
   lambda(rho=0.03)=0.0118   lambda(rho=0.1)=0.0392   lambda(rho=0.3)=0.1175
   ```

   ρ=0.1（文档取值）⇒ **λ ≈ 0.04**。注意 KD 在初始化时梯度很小（量化学生与 FP 教师非常接近），
   所以按"损失数值比"直觉给权（比如 1.0）会过强。

2. **λ 扫描校验**（4 卡并行，每档 700 micro-step = 44 个优化步、有效 batch 512、同种子同数据顺序）：

   | λ | BaseLoss（≥700 步均值） | LogitRank（≥700 步均值） |
   |---:|---:|---:|
   | 0（对照） | 52.2579 | — |
   | **0.04** | 52.2566 | 0.6328 |
   | 0.12 | 52.2536 | 0.6188 |
   | 0.4 | 52.2446 | 0.5712 |

   结论：KD 轨迹在各 λ 下几乎完全重合（λ=0.4 也只看第 3 位小数），说明该约束不扰动主目标；
   ranking loss 在 λ=0.04 就已明显下降，再放大 10 倍只有边际收益，因此取 **λ=0.04**。
   扫描脚本：`tmp_scripts/sweep_swin_w4a4_logitrank_weight_4gpu_20260924.sh`。

   注意 `--max_train_updates` 现在按**优化步**计数（不是 micro-step），扫描里传 1500 实际会跑 1500 个优化步。

DeiT 还能吃更大 batch 换吞吐（单卡 bs64 实测 339 img/s、bs128 稳定段 401 img/s，Swin bs64 在 24GB 上 OOM），
所以 8 卡 bs64 时 DeiT 的 100 epoch 算力口径可以压到 ~14 h。

## 9. 复现

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
