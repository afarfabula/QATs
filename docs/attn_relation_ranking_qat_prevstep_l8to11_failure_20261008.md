# QAT 里 attention-relation ranking(prev-step ref)的实现与失效分析

日期: 2026-10-08
相关提交: 本文件所在提交
相关脚本:

- `tmp_scripts/run_swin_w4a4_attnrank_prevstep_l8to11_4gpu_20261008.sh`(主实验启动脚本)
- `tmp_scripts/sweep_swin_w4a4_attnrank_weight_20261008.sh`(λ 扫描)
- `tmp_scripts/analyze_attnrank_parameter_gradients_20261008.py`(逐张量梯度分解)
- `tmp_scripts/analyze_attnrank_failure_checkpoints_20261008.py`(checkpoint 对比)

---

## 0. 结论摘要

1. 目标是用 attention 的**视觉 token 排序**约束去抑制注意力震荡,替换之前的 KL 路径;参考模型用
   **prev-step 自参考**(学生自己 50 个 optimizer step 前的硬拷贝),作用层选 global block **8/9/10/11 的全部 head**,
   全程常开、权重恒定。
2. 实现完成并验证正确:损失按 rank_idea 文档的成对规则实现,SW-MSA mask 按**模型真实结构**处理,
   梯度确实到达了这些层的 q/k 权重与激活量化 scale。
3. **但是 λ 区间比预期低 2~3 个数量级,而且是一条悬崖**:λ=0.3 开始掉点,λ=3 直接把
   Top-1 从 69.4% 打到 34.5%,λ=30 打到 6.4%。而 train 的 KD loss 只从 52.12 变成 52.23(+0.2%)。
4. 根因已经定位:损失最省力的下降路径是**把 `attn.quan_a_softmax_fn.s`(softmax 的 LSQ 量化步长)压小**,
   让 4-bit 量化后的注意力尾部变成 0;`log(0.clamp_min(eps))` 使配对的 Δ 变得极大,损失趋近 0。
   实测 λ=30 时该 scale 掉到 λ=0 的 **0.51~0.68 倍**,注意力熵从 2.6~2.7 塌到 1.7。
5. 这是一个**损失形式与场景不匹配**的问题,不是实现 bug:`softplus(-Δ)` 没有目标 margin,
   而这里没有"钉住注意力输出"的项兜底(PTQ 文档里那个场景有 block 重构项,所以不会退化)。

---

## 1. 本次要解决的问题与设计决定

### 1.1 背景

- 之前抑制注意力震荡一直走 KL 路线(teacher attn-KL / self-ref KL / prev-step ref KL)。
  复盘三条已归档 100epoch 轨迹后发现这些 KL 都没有真正起作用:

| 信号 | teacher attn-KL(2026-08-04 100ep) | self/prev-step ref-KL(2026-07-13 100ep) |
|---|---|---|
| 作用 head | 3 个(`8:4,11:18,6:1`),ep30–70 扩到 5 个 | 3 个 primary(`8:4,5:7,4:11`) |
| 占全部 head-slot | 3/138 ≈ 2% | 3/138 ≈ 2% |
| 权重 | ep5–30: 1e-6,ep30–70: 2e-6,ep≥90: 0 | 1e-5,**且只有 10/100 个 epoch 生效** |
| 相对 base loss | ≈1e-6 量级 | ≈2e-6 量级 |
| loss 列实况 | `TeacherAttnKL` 全程恒等于 `2.000e+01`(= `--ref-attn-kl-clip 20.0`),原始 KL 全程 ≥20,约束从未接近满足 | `RefAttnKL` 只在 106/9606 个 log 点非零 |
| 额外削弱 | — | `--ref-attn-kl-drop-prob 0.50` 又随机丢一半;控制器是"精度掉了才补一针"的反应式脉冲 |
| 三条 100ep best | 80.8180 | 80.7720(no-KL 80.7920) |

  结论:之前不是"KL 这个形式不行",而是"三个数量级偏小 × 覆盖 2% head × 开一会儿关一会儿"。
  所以本轮的三条设计原则是:**扩大 head 覆盖、全程常开、换成 rank 形式**。

### 1.2 选层依据(去掉先验)

用 `docs/attn_relation_oscillation_analysis_20260710/head_oscillation_summary.tsv`(两个 run 合计 138 个 head-slot)
按 global block 汇总 JS 震荡的质量份额:

| global block | stage | heads | JS 占比 | 累计 |
|---:|---:|---:|---:|---:|
| 0–3 | 0–1 | 3/6 | 0.9% | 0.9% |
| 4–7 | 2 | 12 | 11.1% | 12.0% |
| 8 | 2 | 12 | **12.6%** | 24.6% |
| 9 | 2 | 12 | 5.3% | 29.9% |
| **10** | 3 | 24 | **41.8%** | 71.7% |
| **11** | 3 | 24 | **28.3%** | 100% |

按 stage 单调递增:stage0 `2.3e-6` → stage1 `2.1e-5` → stage2 `1.2e-4` → stage3 `4.3e-4`。
block10/11 是整层一起震(中位数≈均值),所以"整层全 head"是合理选择。

另一个白捡的好处:最后两层采集注意力的成本最低(Swin-T 每张图的 attention 矩阵数:
stage0 `64窗×3头=192`、stage1 `16×6=96`、stage2 `4×12=48`、stage3 `1×24=24`)。
block 8+9+10+11 全 head 只有 `48+48+24+24=144` / 全网络 `912` ≈ 16%。

**因此最终选择:global block 8/9/10/11,全部 head,共 72 个 head-slot(占 138 的 52%)。**

### 1.3 参考模型的选择

三种候选:

- **teacher(FP)** — 固定锚点。但 teacher 就是学生初始化权重,而且只锚到"FP 的排序",不是针对时间维度的震荡。
- **self / prev-step ref** — 直接惩罚"相对上一时刻的变化",和"震荡"的定义对齐。
- **EMA ref** — 同上,但时间常数更平滑。

本轮选 **prev-step ref**(`--train-scheme ema_ref_attn_kl --ref-update prev_step --ref-update-interval 50`),
即每 50 个 optimizer step 把参考模型硬拷贝成学生当前权重。

---

## 2. 代码改动

### 2.1 损失:token 有效性判据(关键修正)

文件:`third_party/OFQ/src/attn_relation_ranking.py`

原来的判据是 `teacher_keys > 0`。实测发现 **SW-MSA 被 mask 的 key 在 softmax 之后不是干净的 0,
而是 0 或 ~1e-30 量级的 denormal**,`> 0` 会把这批残值当成有效 key,给 `valid_pairs` 灌水并破坏逐行归一化。

改为阈值判据(新增 `min_attn`,默认 `1e-4`):

```python
teacher_top = top_values.unsqueeze(-1)
valid = (teacher_top > teacher_keys) & (teacher_keys > min_attn) & (teacher_top > min_attn)
```

即"排序起点"和"排序目标 key"都必须超过阈值。

### 2.2 和模型结构对齐的 mask 事实(实测)

用真实 `swin_t @ 224` 跑一遍,看每个 query 行有多少个非零 attention 的 key:

| block | stage | 是否 shift | 每行有效 key 数 |
|---:|---:|---|---|
| 8 | 2 | 否 | **49** |
| **9** | 2 | **是(14×14, shift 3)** | **15~49** |
| 10 | 3 | 否 | **49** |
| 11 | 3 | 名义 shifted,实际禁用 | **49** |

block 11 之所以没有 mask,是 timm 的 `SwinTransformerBlock.__init__` 里有:

```python
if min(self.input_resolution) <= self.window_size:
    self.shift_size = 0
    self.window_size = min(self.input_resolution)
```

stage3 是 7×7、window 也是 7,shift 直接被关掉。

**所以这四层里只有 block 9 需要做 mask 处理。**

另外注意:统计脚本 `third_party/OFQ/tools/offline_attention_oscillation.py` 里的
`make_valid_attention_mask` 用的是"移位后属于同一原始窗口"的定义,对 stage2-shifted 给出 9/12/16、
对 stage3-shifted 给出 49;而真实模型是 15~49 和"无 shift"。**两者不一致,本实现没有复用它,
而是直接在模型输出上加阈值**,按约定以模型结构为准。

### 2.3 训练入口新增参数

文件:`qat_launch.py`

| 参数 | 默认 | 说明 |
|---|---|---|
| `--attn-rank-source {teacher,ref}` | `teacher` | ranking 的参考来源。`ref` 用 `ema_ref_attn_kl` 那套参考模型 |
| `--attn-rank-min-attn` | `1e-4` | 判定 key 为有效 token 的注意力阈值 |
| `--attn-rank-probe` | off | 一次性梯度范数探针(定权重用) |

其他改动:

- `use_ref_scheme` 现在在"rank 需要 ref"时也会触发参考模型 forward,
  即使 `--ref-attn-kl-weight` / `--ref-logit-kl-weight` 都是 0。
- `source=ref` 时不再强制打开 teacher 的注意力采集(省一份开销)。
- 调试行增加 `source` / `min_attn`,并打印 `ref_layers`。

### 2.4 启动脚本

`tmp_scripts/run_swin_w4a4_attnrank_prevstep_l8to11_4gpu_20261008.sh`

```bash
RANK_W=0.03 DEVICES=4,5,6,7 bash tmp_scripts/run_swin_w4a4_attnrank_prevstep_l8to11_4gpu_20261008.sh
```

固定配置:8/9/10/11 全 head(`--ref-head-mode custom_subset:8:0,...,11:23`,72 项)、
`--train-scheme ema_ref_attn_kl --ref-update prev_step --ref-update-interval 50`、
`--ref-attn-kl-weight 0 --ref-logit-kl-weight 0`、`--attn-rank-source ref --attn-rank-topk 5 --attn-rank-min-attn 1e-4`。
其余和 2026-09-24 的 logits-ranking 100ep 配方一致(bs32/GPU、accum=16、有效 batch 512、
lr 2e-4→5e-6、W4A4、qk-reparam、bf16、KD T=2.75、seed 42)。

---

## 3. 正确性验证

### 3.1 单元级

- 合成张量:把最后一个 key 设成 `1e-30` 的 mask 残值后,有效对从 72 降到 54,masked key 上的梯度**正好为 0**,
  排序起点上的梯度为负(方向正确)。
- `heads=((layer,head),...)` 选择正常;`min_attn=0` 时 masked key 会被算进来(loss 从 0.218 变 0.164),
  证明阈值确实在起作用。

### 3.2 梯度确实到参数上

单独建模型跑一次 forward/backward,按张量分解 `||g_rank||` 与 `||g_kd||`:

```
 block  params   ||g_rank||     ||g_kd||    ratio  constrained
     8      23   3.1602e-02   6.7462e+00   0.0047  YES
     9      23   2.4765e-02   5.2646e+00   0.0047  YES
    10      23   2.7370e-02   1.1648e+01   0.0023  YES
    11      23   9.9698e-03   9.3105e+00   0.0011  YES
```

block 10 注意力模块内部的细分:

```
                 bias n= 18  ||g_rank||=1.03e-02
       input_quant_fn.s n= 4  ||g_rank||=4.10e-06   (激活量化 scale)
       quan_a_qkx_fn.s n= 1  ||g_rank||=1.13e-06
     quan_a_softmax_fn.s n= 1  ||g_rank||=3.18e-05   <-- 后面证明这是"后门"
           quan_a_v_fn.s n= 1  ||g_rank||=1.68e-06
 relative_position_bias_table n=1 ||g_rank||=3.50e-03
                 weight n= 8  ||g_rank||=3.48e-02
```

即:权重、bias、rel-pos bias、**激活量化 scale** 都拿到了非零 rank 梯度;
只有结构上的下游(最后一块的 V/proj、最后的 norm/head)是 0,符合预期。

### 3.3 探针口径的修正

最初写的探针是"在共享的注意力张量上比较两个损失的梯度",实测 `||dBase/dA|| = 0`。
原因是:开了 head 子集之后,注意力模块返回的是

```python
attn = attn[:, head_indices]      # advanced indexing -> 副本
return x, attn                    # x 用的是索引之前的 attn
```

也就是说返回的 attn 是**副本,不是 base loss 的祖先节点**,`dBase/dA` 结构上恒为 0。
探针已改为在**模型参数**上比较两个损失的梯度范数。

---

## 4. 失效现象:λ 是一条悬崖

口径:每次 100 个 optimizer step(= 1600 micro-step = epoch 0 的 2%),4 卡各跑一个 λ(单卡 bs32×accum16=512,
同 seed 42、同数据顺序),结束后跑 50k 全量验证。

### 4.1 高 λ 段

| λ | val Top-1 | val CE | train KD(BaseLoss) | s/micro-step(单卡) |
|---:|---:|---:|---:|---:|
| 0 | **69.3640** | 1.7161 | 52.1206 | 0.334 |
| 3 | **34.4520** | 4.6223 | 52.2000 | 0.541 |
| 10 | 10.8540 | 6.0520 | 52.2219 | 0.541 |
| 30 | 6.4320 | 6.3022 | 52.2337 | 0.545 |

### 4.2 低 λ 段

| λ | val Top-1 | val CE | train KD(BaseLoss) |
|---:|---:|---:|---:|
| 0.01 | 69.4820 | 1.7055 | 52.1530 |
| 0.03 | **69.6500** | 1.7364 | 52.1525 |
| 0.1 | 69.4760 | 1.7532 | 52.1510 |
| 0.3 | 68.7480 | 1.7530 | 52.1489 |

(0.01–0.3 那四行的 0.54 s/micro 与高 λ 段同条件;它们的 BaseLoss 与 λ=0 的 52.12 只差 0.03,属于运行间波动。)

### 4.3 现象总结

1. **可用区间 λ ≲ 0.1**,0.3 已经开始掉点;0.3 → 3 之间是悬崖,不是缓坡。
2. **train 的 KD loss 完全没有报警**:52.12 → 52.23(+0.2%),而 Top-1 从 69.4% 掉到 6.4%。
   也就是说 KD-loss 的变化**不能**当作"有没有伤害"的指标。
3. 训练日志里的 `AttnRank` 也没有区分度:λ=3/10/30 都停在 ~0.35,一模一样。

---

## 5. 根因分析

### 5.1 损失的形状:对 Δ 没有上界目标

对一对 key `(c,d)`:

```
ℓ = softplus(-Δ),   Δ = log s_c - log s_d
∂ℓ/∂Δ = -σ(-Δ)
```

softmax 的归一化项在相减时约掉,所以 **Δ 就是 softmax 之前的分数差**(QK + rel-pos bias)。
而 `softplus(-Δ)` 在 Δ→+∞ 时趋近 0、梯度也趋近 0 但**永远不为 0**,即这个损失**没有目标 margin**,
最小化它等价于把 Δ 推向无穷。PTQ 文档里其实也写了这一点("顺序已经正确但差距较小时,损失仍推动差距增大")。

而且成对结构会放大这个倾向:top-5 起点 × 所有更低的 key,49 个有效 key 时每行约 **210 对**;
top-5 只当 5 次"c",却有 ~44 个 key 各当几十次"d",
**净效果是"顶上更尖、其余更平"——纯锐化**。

### 5.2 为什么 PTQ 里没事

PTQ 场景里,同一块上还有一个 Fisher / Hessian 的**块输出重构项**直接钉住块输出,softmax 想塌也塌不动。
本仓库的 base loss 只在**最终 logits** 上做 T=2.75 的 soft-KD,对注意力的约束是间接且弱的。

### 5.3 全局梯度范数比是骗人的

```
global  ||g_rank||/||g_kd|| = 0.0039      <- 500 多个张量的平均
```

按这个比例配 ρ=0.1 会得到 λ≈25,正是灾难区间。按张量看,真正被 rank 主导的是:

```
 6.016  block8  attn.relative_position_bias_table   rank=3.50e-03  kd=5.82e-04
 3.732  block11 attn.relative_position_bias_table   rank=3.92e-03  kd=1.05e-03
 3.618  block10 attn.relative_position_bias_table   rank=4.43e-03  kd=1.23e-03
 2.440  block11 attn.move_qkx_b4.bias
 0.779  block11 attn.quan_a_softmax_fn.s            rank=2.01e-04  kd=2.58e-04
 0.620  block10 attn.quan_a_softmax_fn.s            rank=1.88e-04  kd=3.03e-04
 0.497  block11 attn.q.weight
```

### 5.4 为什么是悬崖:Adam 的逐参数归一化

Adam 对每个张量做自己的归一化,更新方向基本由**该张量自己的梯度方向**决定,与全局梯度大小无关。
所以在这些张量上,只要 `λ·g_rank` 超过 `g_kd`,该张量的更新方向就"翻"过去了。
把这条对上实测:

| λ | block8 rel-pos 上的主导倍数(6.0×λ) | val Top-1 |
|---:|---:|---:|
| 0.03 | 0.18× | 69.65 |
| 0.1 | 0.60× | 69.48 |
| 0.3 | 1.80× | 68.75 |
| 3 | 18× | 34.45 |
| 10 | 60× | 10.85 |
| 30 | 180× | 6.43 |

掉点精确地发生在"主导倍数跨过 1"的位置。

### 5.5 真正被利用的是"softmax 量化步长"这个后门

同 seed、只改 λ,跑到 step 100(110 个 optimizer step)后取 checkpoint 对比:

| | val Top-1(6400 张) | H(block8) | H(block10) | H(block11) | train KD |
|---|---:|---:|---:|---:|---:|
| λ=0 | **73.58%** | 2.7311 | 2.6325 | 2.5786 | 52.04 |
| λ=30 | **6.89%** | 1.8232 | 1.6812 | 1.6902 | 52.23 |

**注意力熵从 2.6~2.7 塌到 1.7**,即 softmax 明显变尖。

再逐张量比参数变化(step 100,`||·||` 的 λ30/λ0 比值):

| block | q.weight | k.weight | v.weight | proj.weight | qkx scale | **softmax scale** | v scale |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.008 | 1.009 | 0.999 | 0.998 | 0.979 | **0.533** | 0.980 |
| 9 | 0.997 | 0.997 | 1.000 | 0.999 | 0.934 | **0.683** | 0.979 |
| 10 | 1.008 | 1.007 | 0.999 | 0.999 | 0.967 | **0.650** | 0.992 |
| 11 | 0.999 | 0.997 | 1.000 | 0.999 | 0.955 | **0.509** | 0.998 |

rel-pos bias 也几乎没动(block8 `0.56853→0.57089`、block10 `0.60216→0.60381`、block11 `0.57585→0.57896`)。

**唯一显著变化的量就是 `attn.quan_a_softmax_fn.s`**——注意力经过 softmax 之后那个 4-bit LSQ 量化器的步长,
λ=30 时被压到 λ=0 的 **0.51~0.68 倍**。

机制:

1. 该 scale 变小 ⇒ 4-bit 可表示范围 `[-8s, 7s]` 变窄、远处的量化台阶变密;
2. 注意力尾部的小概率被量化到 **0**;
3. 损失里 `student_head.clamp_min(eps)` 用的是 `eps=1e-8`,于是
   `Δ = log s_c - log(1e-8) ≈ 18.4 + log s_c`,**极大**;
4. `softplus(-Δ) ≈ 0`——把这些对"白送"到零损失。

也就是说,**这个损失有一个平凡退化解:把量化后的注意力尾部打成 0**。
而实现这个解最省力的旋钮就是 `quan_a_softmax_fn.s`,且它在 KD 上的梯度只有同量级(ratio 0.6~0.8),
λ 稍大就会被 rank 项完全接管。

### 5.6 为什么训练日志完全没有报警

- **BaseLoss 是 T=2.75 的软分布匹配**,数值 ~52、很平滑,对注意力退化只表现为 +0.2%;
- **AttnRank 自己也不报警**:reference 是 50 步前的自己,它跟着一起退化,这个损失测的是
  "相对上一时刻的变化速率"而不是"累计漂移量",稳态下就是常数(λ=3/10/30 全都是 ~0.35)。

所以"AttnRank 在降、BaseLoss 没变"是完全会骗人的信号组合。

---

## 6. 成本

同条件(单卡)对比:加入 ref forward + rank 计算后 micro-step 从 **0.334 s → 0.541 s(+62%)**;
4 卡实测稳态中位 **0.585 s/micro-step**(对照 2026-09-24 logits-ranking 的 0.362 s,但那是空闲机器,
这次机器上还有别人的任务,数值只能当量级参考)。

按 +60% 估,100 epoch 从 ~4.2 天变成 **~6.8 天**。

---

## 7. 结论与后续修法

### 7.1 这不是实现 bug

- 损失按 `rank_idea/logits-ranking-ptq.md` 的成对规则实现;
- mask 按模型真实结构处理(已核对只有 block 9 有 SW-MSA mask);
- 梯度路径验证通过(权重 / rel-pos bias / 激活量化 scale 都有非零梯度);
- 是**损失形式与本场景不匹配**:`softplus(-Δ)` 是纯 margin-maximization,
  而这里没有任何"钉住注意力输出"的项兜底。

### 7.2 建议的修法(尚未实施)

1. **换成带 margin 的 hinge**:`relu(m - Δ)`。满足 `m` 之后梯度正好为 0,锐化压力从根上消失。
2. **堵住量化后门**:把 ranking 目标放在**量化之前**的 softmax(或直接用 softmax 前的 QK 分数)上;
   或者给 `quan_a_softmax_fn.s` 设下界 / 从 rank 梯度里排除;
   或者把损失里的 `clamp_min(eps)` 提高到与量化步长同量级,让"打成 0"不再白送损失。
3. **定标口径换成"关键张量的最大梯度比"**,而不是 500 个张量的全局平均;按这个口径 λ 应落在 0.01~0.1。
4. 只把 λ 压到 0.03 也能跑(实测 69.65 vs 69.36,无区别),但那是治标,后门还在。

### 7.3 另一个尚未解决的问题:参考模型的时间尺度

prev-step ref 每 50 个 optimizer step 拷贝一次(= epoch 的 2%),所以这个损失**只看得见比 2% epoch 更快的漂移**;
而 7 月统计出来的"震荡"是相邻 checkpoint(1 个 epoch)之间的变化。更根本的是,
自一致性对**系统性漂移完全免疫**(参考模型跟着一起漂)。

也就是说当前这个配置大概率**看不见我们真正想抑制的目标现象**,却仍然付出了 +62% 的算力。
下一步如果要继续走 rank 这条路,建议同时把 ref lag 拉长到 ~1 epoch 或改用长 EMA,再谈 λ。

---

## 8. 复现清单

```bash
# 1. λ 扫描(4 卡各一个 λ,100 个 optimizer step)
bash tmp_scripts/sweep_swin_w4a4_attnrank_weight_20261008.sh

# 2. 逐张量梯度分解
/datadisk2/quyanyi/envs/qat_env/bin/python \
  tmp_scripts/analyze_attnrank_parameter_gradients_20261008.py

# 3. checkpoint 对比(val / 注意力熵 / 逐张量参数变化)
/datadisk2/quyanyi/envs/qat_env/bin/python \
  tmp_scripts/analyze_attnrank_failure_checkpoints_20261008.py
```

原始日志:

| 内容 | 路径 |
|---|---|
| λ 高段扫描 | `/datadisk2/quyanyi/qat_runs/attnrank_sweep_w{0,3,10,30}_20261008.log` |
| λ 低段扫描 | `/datadisk2/quyanyi/qat_runs/attnrank_sweep_lo_w{0.01,0.03,0.1,0.3}_20261008.log` |
| step-100 checkpoint 对照 | `/datadisk2/quyanyi/qat_runs/attnrank_ckpt_w{0,30}_20261008/step_checkpoints/step_0100.pth.tar` |
| 4 卡成本实测 | `/datadisk2/quyanyi/qat_runs/attnrank_4gpu_costcheck_20261008.log` |
