# QAT 目标函数、注意力震荡约束与 STE 的结构性审计

> 日期：2026-10-22
> 范围：`third_party/OFQ` + `qat_launch.py` 的 Swin-T W4A4 训练栈；2026-09-23 至 2026-10-20 的 rank-loss 实验
> 目的：回答三个问题——(1) 这套 QAT 和普通训练到底差在哪；(2) 我们是否只是用一堆 trick 在优化蒸馏；(3) “约束量化后的注意力震荡”这件事是否真的做到了，STE 是否有结构性缺陷。

---

## 0. 结论摘要

1. **训练目标不是任务损失，而是纯软蒸馏。** 2026-07 以后的全部 100/200/300ep OFQ 实验和全部 rank-loss 实验都用 `kd_hard_and_soft=0`，此时 loss 只剩 `T²·KL(p_teacher || p_student)`，`CrossEntropy` 分支根本不会被执行（例外：两条早期的 300ep/50ep 归档用 `kd_hard_and_soft=1`，即 soft+hard）。仓库里不存在一条“CE-only 训练的 W4A4 Swin-T 基线”。
2. **当前的 attention-relation ranking 不是震荡约束，而是注意力蒸馏。** 正在跑的 100ep 实验用 `attn_rank_source=teacher`，约束学生量化注意力与 FP 教师注意力的大小关系一致；真正对齐“时间维度震荡”定义的只有 `source=ref` + `prev_step`，那条路线在 10/08 已被判定失败。
3. **即使换成 prev-step，这个损失也没有在度量它声称要度量的东西。** `log(clamp_min(post_quant_attn, 1e-8))` 使“尾部被 round 成 0”成为白送的零损失；代码注释里已经明确写了这一点。损失全程稳定在 ~0.34，而 Top-1 可以从 69% 崩到 6%，说明这个标量与目标现象脱钩。
4. **STE 不是写错了，而是用错了场合。** 实现是标准 LSQ STE；问题在于目标函数（量化后注意力）对预量化参数是分段常数函数，STE 用恒等梯度把它线性化，只提供有偏的替代方向，还带来 clamp 死区和步长 `s` 的梯度耦合。这是“loss 降了、精度不涨、偶发崩盘”的结构性来源。
5. **100 epoch 长跑给出否定结果。** teacher attention-rank（w=0.3）最终 best **80.8720 @ep92**，逐 epoch 配对平均比本地 logit-rank λ=0.04 曲线低 **0.098 分**，100 轮里只赢 17 轮。同路径 λ=0 对照在第 10 轮反而最高（79.6640）。目前没有任何一条辅助损失被证明有正收益。
6. **最需要补的不是新 trick，而是三组测量**：量化水平误差 vs 时间抖动的量级对比、逐量化器的精度归因、损失空洞性（零化比例/退化配对比例）。在这三组数出来之前，任何新机制都只能算猜测。

---

## 1. 训练目标：事实核查

### 1.1 代码路径

`kd_hard_and_soft=0` 时，[qat_launch.py:6277](../qat_launch.py) 走的是：

```python
loss = teacher_soft_kd_with_temperature(
    student_logit, teacher_logit, temperature=runtime_args.teacher_soft_temperature
)
```

而 [qat_launch.py:5395](../qat_launch.py) 的实现是：

```python
per_sample_loss = -(teacher_prob * student_log_prob).sum(dim=1) * (temp * temp)
```

即 `T² · CE(p_teacher, p_student) = T² · [KL(p_teacher||p_student) + H(p_teacher)]`。教师的熵是常数，所以优化目标等价于**纯 KL 蒸馏**。

`loss_fn` 的构造在 [qat_launch.py:5717](../qat_launch.py)：`kd_hard_and_soft=0` 返回 `KLLossSoft()`，但它只在非 KD 或 `kd_hard_and_soft ∈ {1,2,3}` 的分支里被调用；在当前配置下它是死参数。

### 1.2 当前 100ep 实验的实际配置

取自 `/datadisk2/quyanyi/qat_runs/swin100ep_20261014_teacherrank_w03_v2/args.yaml`：

| 字段 | 值 | 含义 |
|---|---|---|
| `use_kd` | `true` | 蒸馏开启 |
| `kd_hard_and_soft` | `0` | 纯软 KD，无 hard/CE 项 |
| `teacher_soft_temperature` | `2.75` | KD 温度 |
| `clean_start_target_loss_weight` | `0.0` | 没有额外 CE 项 |
| `teacher_confidence_band_kd_weight` | `0.0` | 没有置信度带 KD 项 |
| `smoothing` | `0.1` | 有效，但只影响未被调用的 CE 构造 |
| `attn_rank_weight` | `0.3` | 注意力排序损失 |
| `attn_rank_source` | `teacher` | 参考是 FP 教师 |
| `attn_rank_target` | `post_quant` | 学生侧取量化后注意力 |
| `freeze_attn_softmax_scale` | `false` | 步长 s 可被损失更新 |

### 1.3 归档实验的目标函数分布

对 `experiment_logs/` 下所有带命令行记录的 run 做了逐条检查，结果分三类：

| 实验群 | 目标函数 | 代表 |
|---|---|---|
| 2026-07 起的 100/200/300ep from-scratch / resume / rank-loss | `--use-kd --kd_hard_and_soft 0`（纯软 KD） | `playground__train_ofq_100ep_fromscratch_original_ofq_public_control_20260714.log` |
| 两条早期 300ep/50ep 归档 | `--use-kd --kd_hard_and_soft 1`（soft + hard） | `qats_logs__swin_t_w4a4_ofq_mainline_300ep_20260613_rerun2.log` |
| 完全不用 KD（CE-only） | **不存在** | — |

也就是说：本项目的量化训练从 7 月起就统一用纯软 KD；`kd_hard_and_soft=1` 的两条早期归档是仅有的带数据锚点的记录，但它们既不是同栈、也不是同配方。

**推论：**

- 仓库里的“QAT baseline”（100ep 组）其实是“FP 教师蒸馏 + 量化架构约束”的结果，不是“任务训练 + 量化约束”的结果。
- 任何新增损失都是 KD 的同级竞争者，没有数据锚点把它拉回真实标签。
- 因此“这个 trick 让 loss 降了”与“量化网络精度变好”之间没有必然联系，报告里反复出现的“BaseLoss 几乎不动、辅助 loss 下降、Top-1 不动”是这一结构的结果。

---

## 2. 这套 QAT 与普通训练的实际差别

除去“目标函数换成了 KD”，架构层面只有四块差异。

### 2.1 量化器插点

`QAttention_swin_qkreparam`（[swin_attention_and_mlp.py:285](../third_party/OFQ/src/quantization/modules/swin_attention_and_mlp.py)）中：

| 位置 | 量化器 | 位宽 | 粒度 |
|---|---|---|---|
| QK 重参数化后的乘积权重 | `StatsQuantizer` | 4 | per-channel |
| 进入 QK 的激活 `quant_x` | `quant_x_4_qkv` | 4 | per-channel |
| 中间激活 `qkx = Wqk·x` | `quan_a_qkx_fn` (LSQ) | 4 | per-channel |
| V 激活 | `quan_a_v_fn` (LSQ) | 4 | per-channel |
| **softmax 后的注意力概率** | `quan_a_softmax_fn` (LSQ) | 4 | per query 位置 |
| 输出投影 | `QLinear` | 4 | per-channel |

注意：注意力概率是被**直接量化后送入 `attn.matmul(v)`**（[swin_attention_and_mlp.py:480](../third_party/OFQ/src/quantization/modules/swin_attention_and_mlp.py) 附近），量化后不做行归一化，因此 `Σ q(attn) ≠ 1` 是这套设计的常态。

### 2.2 QK 重参数化

注意力分数不是 `q·k`，而是：

```text
Wqk       = Wqᵀ·Wk                    # 每个 head 一个 C×C 矩阵
qkx       = Wqk_quant · xᵀ
qkx_quant = Q(qkx)
score     = x · qkx_quant
```

也就是说 **Q 与 K 的量化被合并到了“乘积矩阵 + 中间激活”上**。这改变了误差结构：量化误差不再等价于分别量化 q 和 k，而是一个稠密的 C×C 矩阵量化，再乘两次激活。做归因实验时不能沿用“量化 q/k”的直觉。

### 2.3 全精度旁路

`move_qkx_b4` / `move_qkx_aft` 都是 `LearnableBias`（[qbias.py:5](../third_party/OFQ/src/quantization/modules/qbias.py)），本质是可学习加性偏置：

```text
qkx  →  +b_b4  →  Q(qkx)  →  +b_aft  →  score
```

`b_aft` 加在量化器**之后**，且与 `quant_x` 相乘后进入分数，因此它不是常数偏移，而是给分数加入了一个**不经过量化的秩一修正**；`q_bias`/`k_bias` 的交叉项同样在 FP 中计算。严格说，“量化后的注意力分数”里始终混着全精度成分。

### 2.4 可学习量化步长

`LsqQuantizer` 的 `s`（[lsq.py:571](../third_party/OFQ/src/quantization/quantizer/lsq.py)）是可学习参数；对注意力概率，per-channel 维度取 `x.shape[-2]`，即 **每个 query 位置一个步长，跨 batch、head、window 共享**。Swin-T 的 window 是 7×7，所以每层只有 49 个自由度来决定整层注意力的量化粗细。

### 2.5 OFQ 的震荡机制并未启用

`TrackOscillation`（[lsq.py:139](../third_party/OFQ/src/quantization/quantizer/lsq.py)）实现了 OFQ 的整数域翻转检测与权重冻结；它只被 `LsqQuantizerWeight_iterative_freezing`（`lsq.py:230`）使用，而后者在 `qat_launch.py`、`qlinear.py`、注意力模块中**没有任何实例化点**。**本项目名字里的 OFQ 主线机制在所有实验里都是死代码**，实际生效的只有量化器 + QK 重参数 + STE。

---

## 3. “注意力震荡”这个问题的证据强度

### 3.1 已有的测量

2026-07 的分析（[attn_relation_oscillation_analysis_20260710](attn_relation_oscillation_analysis_20260710/analysis_report.md)）用相邻 checkpoint 的注意力 JS 散度定义震荡，`head_oscillation_summary.tsv` 的逐 head `mean_js` 大致落在 `1e-6 ~ 1e-4`。按 stage 汇总的中位数：

| stage | global blocks | 中位 JS |
|---|---|---|
| 0 | 0–1 | 2.3e-6 |
| 1 | 2–3 | 2.1e-5 |
| 2 | 4–9 | 1.2e-4 |
| 3 | 10–11 | 4.3e-4 |

同一张表里，震荡指标与精度变化的相关系数只有 `±0.3` 左右（`acc_up_corr` / `acc_down_corr` 列），11 个 checkpoint 对。

### 3.2 与“量化水平误差”的量级对比（推算，待测量验证）

初始化时，softmax 量化器的步长为

```text
s ≈ 4·mean(p) / sqrt(15) ≈ 4·(1/49)/3.873 ≈ 0.021
```

半步长 ≈ 0.0105，即一行 49 个 key 中，概率低于约 **1%** 的 key 会被直接 round 成 0。对于大多数 attention 行，这类 key 是多数，因此量化会成片抹掉尾部，并让峰值吸收这部分质量。

由此得到的**推算**是：量化造成的分布变化（归一化后的 JS）在 `1e-2 ~ 1e-1` 量级，而相邻 epoch 的时间抖动在 `1e-4 ~ 1e-6`。若该推算成立，则我们花一个月约束的量比真正的误差源小 **2~4 个数量级**。

> 这是全文最重要、也最需要先验证的一条。验证方法见 §8.1，不需要训练，只需要已有 checkpoint。

### 3.3 一个已知的反证

10/08 的反作弊实验记录：λ=3 时 Top-1 从 69.4 掉到 34.5，λ=30 掉到 6.43，而 `AttnRank` 损失在三档 λ 下都稳定在 ~0.35，训练 KD 只动 +0.2%。任何一个真正度量“注意力排序保真度”的指标，都不可能在模型近乎崩溃时保持不变。

---

## 4. 当前 attention-relation ranking 实际约束了什么

### 4.1 参考系决定语义

| `attn_rank_source` | 参考 | 实际语义 |
|---|---|---|
| `teacher` | 固定 FP 教师 | **注意力蒸馏**：让学生（量化）的注意力排序靠近 FP |
| `ref` (`prev_step`) | 50 步前的学生权重 | **时间平滑**：惩罚相对上一时刻的变化 |

当前 100ep 跑的是第一行。它和“抑制量化引起的注意力抖动”不是一回事：FP 教师本身也有它自己的注意力形态，学生完全可能被拉向一个与量化误差无关的方向。

### 4.2 学生侧取的是哪一份注意力

`collected_attention_view`（[swin_attention_and_mlp.py:65](../third_party/OFQ/src/quantization/modules/swin_attention_and_mlp.py)）：

| `attn_rank_target` | 返回 |
|---|---|
| `post_quant`（当前） | softmax 后再过 4bit 量化器的概率 |
| `post_quant_detached_scale` | 数值相同，但切断到 `s` 的梯度 |
| `pre_quant` | 未经量化的 softmax 概率 |

当前实验用的是 `post_quant`，同时 `freeze_attn_softmax_scale=false`。也就是说，**已知的“通过动 s 降低损失”的通道是开着的**。

### 4.3 损失形式与退化通道

[attn_relation_ranking.py:95](../third_party/OFQ/src/attn_relation_ranking.py)：

```python
delta = (
    torch.log(student_head.gather(-1, top_indices).clamp_min(eps)).unsqueeze(-1)
    - torch.log(student_head.clamp_min(eps)).unsqueeze(-2)
)
pair_term = F.softplus(-delta)
```

失效链条：

1. 4bit + 每行一个步长下，尾部概率大量 round 成**精确的 0**；
2. `clamp_min(1e-8)` 把 0 映射成全局最小概率，于是 `log(0) → log(eps) = -18.42`；
3. 与任何非零 top-k 配对都会得到 `Δ ≫ 0`，`softplus(-Δ) ≈ 0`，**损失白送**；
4. 同时 `clamp_min` 对 0 输入的梯度为 0，这些配对的梯度**也被切断**；
5. 有效配对里真正贡献梯度的只剩“学生侧仍非零”的那一小部分。

代码作者对此已有明确认知，[lsq.py:660](../third_party/OFQ/src/quantization/quantizer/lsq.py) 的注释写道：

> 让某些损失……不能通过缩小量化步长 `s` 把小于 `s/2` 的注意力尾部 round 成精确的 0 来“作弊”（0 会被损失里的 `clamp_min(eps)` 变成白送的零损失）。

这解释了三个此前无法解释的观察：

- 损失值始终“健康”（~0.34）而精度可以崩溃；
- 冻结 `s`、冻结 `move_qkx_*` 只能把 38.48 挪到 38.92（噪声级），其余旁路继续生效；
- 损失对 `head` 选择、`topk`、`min_attn` 等超参高度敏感（因为它们直接改变“有多少配对处于退化区”）。

### 4.4 结论

当前实验**没有**实现对“量化后注意力震荡”的约束，原因是三层：

1. 参考系是教师，不是时间；
2. 损失度量的是排序关系，而排序在量化后的尾部根本不可表示；
3. 损失可被零化通道满足，且相关作弊旋钮处于开启状态。

---

## 5. STE 的结构性问题

### 5.1 实现本身

`round_pass`（[lsq.py:11](../third_party/OFQ/src/quantization/quantizer/lsq.py)）：

```python
def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad
```

前向是 `round`，反向是恒等。`clip` 用 `x - x.detach() + x_clip.detach()` 把越界梯度置零。步长用 `grad_scale` 乘 `1/sqrt(Qp·N)`。这是**教科书式 LSQ**，没有实现错误。

### 5.2 问题一：目标函数是分段常数，STE 假装它光滑

部署的模型是 `round(x/s)·s`。对预量化参数 $\theta$ 而言，`L(θ)` 在绝大多数位置是常数（同一个量化格子内），只在跨过舍入边界时跳变。STE 给出的“梯度”是把这个函数线性化后的替代方向，它：

- 在格子内部提供非零梯度，但前向完全不变 → 优化器可以在不改变模型的情况下走很远；
- 不提供“距离边界还有多远”的信息 → 无法判断一次更新会不会翻转某个元素；
- 在少数元素决定结果的目标上（softmax 尾部排序、top-k）偏差是系统性的，而不是可以被平均掉的。

分类目标（CE 或 KD）上，STE 的偏差被百万级 token/样本平均掉了；注意力排序上，单个 head 的 49 个 key 就是全部统计量，平均化不成立。

### 5.3 问题二：clamp 死区 + 步长耦合

`LsqQuantizer.forward` 的顺序是 `x/s → clamp → round → ×s`。超出 `[-Qn·s, Qp·s]` 的元素，clamp 的梯度为 0；它们唯一能影响损失的方式是通过 `s`。于是：

- 越界元素越多，`s` 承担的更新就越多；
- `s` 变成不受权重更新约束的“自由旋钮”，这正是它成为作弊杠杆的结构性原因；
- 反过来，冻结 `s` 会让越界元素彻底失联，而不是迫使权重修正。

### 5.4 问题三：归一化尺度错配

`s` 的梯度被 `1/sqrt(Qp·N)` 缩放，`N` 是该张量元素数。KD 的 `N` 是百万级，注意力排序的有效元素只有几百到几千。两套损失的梯度尺度天然差几个数量级，因此 **λ 没有物理含义**，只能经验扫描。10/14 报告里的 λ 扫描、梯度探针、`lambda cliff`，本质上都是在补这个定义域缺口。

### 5.5 问题四：注意力路径存在全精度旁路

§2.3 的 `b_aft`、`q_bias/k_bias` 交叉项都在量化之外。一个“约束量化后注意力”的损失，其梯度可以部分作用在这些全精度参数上，于是：

- 损失下降不能证明量化路径变好；
- 冻结单个旋钮只会让优化器换一个旁路（10/08 记录中的“打地鼠”）。

---

## 6. 100 epoch 结果复核（截至 2026-10-20）

### 6.1 实验完成情况

| 项目 | teacher attention-rank |
|---|---|
| run id | `swin100ep_20261014_teacherrank_w03_v2` |
| 起止 | 2026-10-14 16:13 → 2026-10-20 14:37 |
| 墙钟 | 512,674 s ≈ 5.93 天（4×3090，GPU 1/2/4/6） |
| 状态 | `exit_status=0`，100 个逐 epoch 50k 全量验证点 |
| 配置 | teacher-rank w=0.3，topk=5，layer 8–11 全 head，logit_rank=0，无 KL 项 |

### 6.2 三条曲线的最终数字

| 曲线 | best | @ep | last | 最后 10 轮均值 |
|---|---:|---:|---:|---:|
| λ=0 同路径对照（仅 10ep） | 79.6640 | 10 | 79.6640 | — |
| logit-rank λ=0.04，100ep | **80.9380** | 94 | 80.8400 | 80.8562 |
| teacher attention-rank w=0.3，100ep | 80.8720 | 92 | 80.7800 | 80.7778 |

逐 epoch 配对（teacher − logit，n=100）：

| 区间 | 平均差 | teacher 胜出 |
|---|---:|---:|
| ep1–10 | −0.0598 | 3/10 |
| ep11–30 | −0.1116 | 4/20 |
| ep31–60 | −0.1095 | 5/30 |
| ep61–90 | −0.0961 | 3/30 |
| ep91–100 | −0.0784 | 2/10 |
| **全程** | **−0.0978**（sd 0.0985） | **17/100** |

### 6.3 读法

- 100 轮里 teacher-rank 只有 17 轮高于 logit-rank，且没有任何一个区间转正；**没有“后期翻盘”**。
- 但两条曲线的差（~0.1）小于 50k val 的单点标准误（~0.18），所以严格说这是“没有证据显示它更好”，不是“显著更差”。
- 更重要的是：**logit-rank 曲线本身也不能当基线**。同路径 10ep 对照（λ=0）在 ep10 是 79.6640，高于 logit-rank 的 79.5620。也就是说现有唯一一条 100ep 本地曲线里仍然含着一个未被证明有效的 rank 项。
- 10/21 的绘图脚本 `plot_swin_w4a4_100ep_new_baseline_vs_three_paths_20261021.py` 把 Sep-24 的 logit-rank λ=0.04 曲线标成“新基线”，这是一个**标注错误**：该 run 的 `args.yaml` 明确写着 `logit_rank_weight: 0.04`。本文档与该脚本的标签已按实际配置修正。

### 6.4 可以/不可以声称的结论

| 结论 | 状态 |
|---|---|
| teacher-anchored attention rank 在 100ep 尺度上没有超过 logit-rank 曲线 | 可以（配对 100 轮、同栈、同评测） |
| attention rank 优于纯基线 | **不可以**（纯基线 100ep 不存在） |
| attention rank 显著更差 | 不可以（差值 < val 标准误） |
| 两条 rank 路线都没有可复现正收益 | 可以（10/14 报告 + 本次 100ep） |

---

## 7. 综合判断

### 7.1 已由代码/数据确认

1. 训练目标 = 纯软 KD，无 CE；2026-07 起的 100ep 及以上实验群与全部 rank-loss 实验相同。
2. 当前 100ep 的 attention 项 = 教师注意力排序蒸馏，不是时间震荡约束。
3. 排序损失存在“零化白送”的退化通道，且代码注释已承认。
4. 当前 run 允许 `s` 被损失更新（未冻结、未 detach）。
5. OFQ 的 `TrackOscillation` 机制未启用。
6. 100ep 结果：teacher-rank 80.8720 @ep92，配对比 logit-rank 低 0.098，17/100 胜。

### 7.2 高度可疑、需要用测量确认

1. 量化造成的注意力水平误差比相邻 epoch 的时间抖动大 2–4 个数量级（§3.2 推算）。
2. 现有 rank 损失的有效梯度主要集中在少数“仍非零”的尾部元素上，其余配对处于 clamp 退化区。
3. 精度差距主要来自 `qkx` 激活量化与 `Wqk` 权重量化，而不是注意力概率量化（需要 §8.2 的逐个关闭实验）。

### 7.3 目前不知道

1. 把 CE 加回来会给 W4A4 Swin-T 带来多少收益（仓库里没有这条数据）。
2. 量化注意力是否真的在时间维度“来回跳”，以及跳变与精度损失的因果关系。
3. 在没有 STE 软约束的前提下，是否存在更合理的量化注意力控制手段（如 AdaRound 式舍入搜索、逐 head 步长控制）。

---

## 8. 下一步实验（按顺序，全部先小成本）

### 8.1 前提审计：量化误差 vs 时间抖动

目标：验证 §3.2 的推算。

在固定 val batch（建议 512–1024 张，固定 seed 与预处理）上，对同一套 checkpoint 计算：

| 指标 | 定义 | 用途 |
|---|---|---|
| `ΔA_quant` | `JS(A_post_quant, A_pre_quant)`，同一权重、同一样本 | 量化把注意力改了多少 |
| `ΔA_time` | `JS(A_t, A_{t+1})`，同一样本、相邻 epoch | 训练过程中的抖动 |
| `flip_rate` | 同一激活在相邻 checkpoint 下量化整数级不同的比例 | 离散域是否真的在翻动 |

现有 checkpoint 足够：λ=0 对照的 9/10、3ep 三个 run 的 1–3、teacher-rank 的 98–100、logit-rank 的 99–100（各自保留最近几个 + `last.pth.tar`）。

**决策规则**：若 `ΔA_quant / ΔA_time > 10²`，则“震荡”不是主要矛盾，停止在该方向继续加约束；若两者同量级，再进入 §8.2。

### 8.2 归因审计：哪一层量化在吃精度

eval-only，用已训练 checkpoint，不改训练：

| 实验 | 操作 | 回答的问题 |
|---|---|---|
| A | 关闭 softmax 概率量化器（恒等） | 注意力概率量化值多少分 |
| B | 关闭 `quan_a_qkx_fn` | QK 中间激活量化值多少分 |
| C | 关闭 `qk_quant`（Wqk 用 FP） | QK 权重量化值多少分 |
| D | 关闭 `b_aft`（置零） | 全精度旁路值多少分 |
| E | 全部关闭 | 这套 W4A4 的总损失 |

### 8.3 损失空洞性审计

单 batch、单步，统计：

1. 学生 `post_quant` 注意力中恰为 0 的元素比例；
2. 排序配对中 `Δ` 落在 clamp 退化值（`log(1/eps)`）的比例；
3. 梯度分别落到 `quan_a_softmax_fn.s`、QK 权重、`move_qkx_*`、`q/k_bias` 上的范数占比。

若 (1) 或 (2) 占多数，则该损失在数学上无法约束目标现象，应直接废弃，而不是继续调 λ。

### 8.4 目标函数审计：把 CE 加回来

用同一配方跑 10 epoch 的 `kd_hard_and_soft=1`（或给 `clean_start_target_loss_weight` 一个非零值），与 λ=0 的 10ep 对照直接比较。

这是整个清单里最便宜、信息量最大的一格：如果纯 KD → CE+KD 的差异就有 0.3 分以上，那么过去一个月优化的对象（蒸馏保真度）本来就不是精度的主要来源。

### 8.5 只有在 8.1–8.4 之后才考虑机制

如果确认问题确实是“量化注意力的水平误差”，合理方向是：

- 舍入感知的目标（AdaRound / BRECQ 式局部重构），而不是 STE 上的软排序；
- 对注意力概率量化器做逐 head / 逐 query 的步长控制或局部搜索；
- 以“跨边界元素比例”作为可微代理（它才直接对应离散域的变化量）；
- 约束“同一权重下 post-quant 与 pre-quant 的差”，而不是约束到教师或上一版自己。

---

## 9. 附录

### 9.1 关键代码位置

| 内容 | 位置 |
|---|---|
| KD-only 分支 | `qat_launch.py:6277` |
| 软 KD 公式 | `qat_launch.py:5395` |
| attention 收集目标语义 | `qat_launch.py:3143` |
| attention rank 加入总损失 | `qat_launch.py:6937` |
| 排序损失实现 | `third_party/OFQ/src/attn_relation_ranking.py:31` |
| 零化退化点 | `attn_relation_ranking.py:95` |
| STE `round_pass` | `third_party/OFQ/src/quantization/quantizer/lsq.py:11` |
| 步长梯度缩放 | `lsq.py:92` |
| LSQ 激活量化器 | `lsq.py:571` |
| `forward_detached_scale` 注释 | `lsq.py:660` |
| `TrackOscillation`（未启用） | `lsq.py:139` |
| QK 重参数前向 | `swin_attention_and_mlp.py:285`（类）、`365–515`（forward） |
| 注意力概率量化 | `swin_attention_and_mlp.py:480` |
| 全精度旁路 bias | `third_party/OFQ/src/quantization/modules/qbias.py:5` |

### 9.2 日志与产物

| 内容 | 路径 |
|---|---|
| teacher-rank 100ep 日志 | `/datadisk2/quyanyi/qat_runs/swin100ep_20261014_teacherrank_w03_v2.log` |
| 同上 nohup | `/datadisk2/quyanyi/qat_runs/swin100ep_20261014_teacherrank_w03_v2.nohup.log` |
| logit-rank 100ep 日志 | `/datadisk2/quyanyi/qat_runs/ofq_swin_t_w4a4_100ep_logitrank_20260924.log` |
| λ=0 10ep 对照 | `/datadisk2/quyanyi/qat_runs/ofq_swin_t_w4a4_lambda0_control_10ep_20260929.log` |
| 3ep 参考系消融 | `/datadisk2/quyanyi/qat_runs/swin3ep_20261009_{baseline,teacherkl,teacherrank}.log` |
| 震荡分析 | `docs/attn_relation_oscillation_analysis_20260710/` |
| 100ep 对照图（已修正标签） | `docs/figures/swin_t_w4a4_100ep_new_baseline_vs_three_paths_top1.png` |

### 9.3 复现取数命令

```bash
# 逐 epoch 50k 全量验证
grep "Test: \[distributed-summary\]" <log> | grep -o "Acc@1:[ ]*[0-9.]*"

# 三条曲线的配对统计
python3 - <<'PY'
import re, statistics
def curve(p):
    return [float(re.search(r"Acc@1:\s*([0-9.]+)", l).group(1))
            for l in open(p) if "Test: [distributed-summary]" in l]
L = curve(".../ofq_swin_t_w4a4_100ep_logitrank_20260924.log")
T = curve(".../swin100ep_20261014_teacherrank_w03_v2.log")
d = [t - l for t, l in zip(T, L)]
print(len(d), statistics.mean(d), max(T), T.index(max(T)) + 1)
PY
```

### 9.4 统计口径

- 所有 Top-1 来自 ImageNet-1K val 全量 50,000 张，单点二项标准误 ≈ 0.18 分。
- 全部为单 seed（seed=42）轨迹；相邻 epoch 强自相关，配对差值的朴素 t 值不能直接当显著性。
- “胜出轮数”是描述性统计，不是假设检验。
