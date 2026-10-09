# QAT attention-relation ranking:反作弊实验记录(2026-10-08 第二轮)

配套文档:`docs/attn_relation_ranking_qat_prevstep_l8to11_failure_20261008.md`(第一轮:λ 悬崖 + 根因)
本文只记录**第二轮**:针对"rank loss 通过某个旋钮作弊"这个问题做的所有干预实验与结论。

---

## 0. 一句话结论

**"让 rank loss 别去更新 `s`" 这件事能做、也确实做到了,但堵住它没有用。**

把两个最大的作弊杠杆(softmax 的 LSQ 量化步长 `s`、qk-reparam 的加性偏置 `move_qkx_*`)都**粘性冻结**之后,
λ=3 的 Top-1 只从 38.5% 挪到 **38.9%(等于噪声)**;换成 hinge 损失可以关掉"锐化"这条路
(λ=3 从 38% 改善到 58%),但它同时打开了"压平造并列"这条新路,λ≥30 照样崩。

根本原因不是某一个旋钮,而是 **损失形式 + 自参考** 的组合:参考模型是滞后 50 个 optimizer step 的自己,
它跟着学生一起漂,于是这个损失只约束"变化速率"、不约束"目标",任何能让变化看起来变小、
或者让有效对变少的捷径(变尖、变平、缩量化步长)都会被它利用。

**后续两节(第 6、7 节)把这个判断做实了 —— 换参考系才是关键:**

| 参考 | 损失 | 可用权重 | 最好点(6400 val) |
|---|---|---:|---:|
| 无约束 | — | — | 73.58% |
| prev-step 自己 | naive KL | ≤0.01 | 72.22% |
| prev-step 自己 | rank / softplus | ≤0.3 | 38.5% @λ=3(崩) |
| **FP teacher(固定锚)** | **naive KL** | **0.003~0.1** | **74.84%** |
| **FP teacher(固定锚)** | **rank / softplus** | **0.3** | **75.03%** |

即:**同样的损失、同样的层/head/常开方式,只把参考从"prev-step 自己"换成"固定 teacher",
可用权重范围大 3~10 倍,而且最好点从"崩"变成略高于基线。prev-step 是这轮的主要问题来源。**
(注意 +1.3~1.5 分是 epoch 0 的 2% 处的早期信号,还不能当最终收益。)

---

## 1. 问题是什么

第一轮已经确认:在 Swin-T W4A4、prev-step ref、8/9/10/11 全 head、softplus 排序损失下,
权重 λ 有一条悬崖(100 个 optimizer step、50k 全量验证):

| λ | Top-1 | val CE | train KD(BaseLoss) |
|---:|---:|---:|---:|
| 0 | **69.3640** | 1.7161 | 52.1206 |
| 0.01 | 69.4820 | 1.7055 | 52.1530 |
| 0.03 | 69.6500 | 1.7364 | 52.1525 |
| 0.1 | 69.4760 | 1.7532 | 52.1510 |
| 0.3 | 68.7480 | 1.7530 | 52.1489 |
| **3** | **34.4520** | 4.6223 | 52.2000 |
| 10 | 10.8540 | 6.0520 | 52.2219 |
| 30 | 6.4320 | 6.3022 | 52.2337 |

train 的 KD loss 只动了 0.11(+0.2%),Top-1 却从 69% 掉到 6%,说明训练日志完全不会报警。

第一轮的观察是:λ=30 时注意力熵从 2.6~2.7 塌到 1.7,同时
`attn.quan_a_softmax_fn.s`(softmax 输出的 4-bit LSQ 量化步长)缩到 λ=0 的 0.51~0.68 倍。
由此提出假设:**rank loss 通过缩小 `s` 把注意力尾部 round 成 0 来"作弊"**,于是有了本轮问题:

> 能不能让 rank loss 不去更新这个 `s`?

---

## 2. `s` 到底是什么

`quan_a_softmax_fn` 是套在**注意力概率**上的激活量化器(`LsqQuantizer`,`bit=4`,`all_positive`,
`per_channel`,`learnable` 由 `--aq-clip-learnable` 打开)。键 `s` 就是 LSQ 的**量化步长**
(反量化时 `x ≈ q * s`),前向是:

```python
alpha = s.unsqueeze(-1)
s_scale = grad_scale(clip(alpha, 1e-5), s_grad_scale)
x = clamp(x / s_scale, 0, 15)
x = round(x)
x = x * s_scale
```

所以:

- **`s` 可以学,也不是"必须学"**;冻结它完全可行,只是失去对当前激活分布的自适应标定。
- 对注意力概率而言,`s` 越小 ⇒ "死区"越大:**小于 `s/2` 的概率会被 round 成精确的 0**。
  49 个 key 的均匀值才 0.0204,`s` 从 ~0.19 缩到 ~0.095 时尾部几乎全被打成 0。
- 损失里用的是 `student_head.clamp_min(eps)`,而 `eps=1e-8`,于是
  `Δ = log s_c - log(1e-8) ≈ 18.4 + log s_c`,**极大**,`softplus(-Δ) ≈ 0` —— 这些配对白送零损失。

这就是"作弊路径"的完整链条。

---

## 3. 实现了哪些"不让 rank loss 更新 s"的方式

全部已实现并验证:

| 方式 | 开关 | 做法与验证 |
|---|---|---|
| 数值相同但切断到 `s` 的梯度 | `--attn-rank-target post_quant_detached_scale` | 收集器返回 `quantizer.forward_detached_scale(attn_prob)`;实测与 post-quant **数值差 0.0**,有效对完全相同(6,946,259),但没有任何到 `s` 的 autograd 路径 |
| rank 直接看量化前的 softmax | `--attn-rank-target pre_quant` | 收集器返回 softmax 之后、激活量化之前的概率;与 post-quant 数值差最大 1.054e-2,有效对 10,774,904(因为量化前没有被打成 0 的 key) |
| 粘性冻结 | `--freeze-attn-softmax-scale` / `--freeze-param-suffix a,b` | 按参数名子串把 `requires_grad` 置 False,并在每次 policy 变更后重新施加 |

### 3.1 梯度路径的直接验证

单模型一次 forward/backward,统计**所有** `quan_a_softmax_fn.s` 上的梯度:

```
target=post_quant                 AttnRank=0.38510  非零 softmax 步长梯度 = 12/12
   最大: features.7.1 = 1.461e-3, features.7.0 = 1.256e-3, features.5.4 = 1.821e-4
target=post_quant_detached_scale  AttnRank=0.38510  非零 softmax 步长梯度 = 11/12
   最大: features.7.0 = 2.126e-4, features.5.1 = 5.695e-5, features.5.0 = 5.852e-5
target=pre_quant                  AttnRank=0.37730  非零 softmax 步长梯度 = 11/12
   最大: features.7.0 = 1.493e-4, features.5.0 = 4.841e-5, features.5.1 = 5.082e-5
```

注意 `detached` / `pre_quant` 下**不是 0/12**:剩下的 11 个是从**特征链**过去的
(上游 block 的 softmax 步长 → 改变它的输出特征 → 影响下游被约束层的注意力),
这不是同层作弊,是真实的间接通路。**最后一个被约束 block(block 11)自己的 softmax 步长是唯一减到 0 的那个。**

### 3.2 踩到的坑(重要,以后别再踩)

第一次做 `--freeze-attn-softmax-scale` 时打印 `count=0`,而且对照与冻结两组 checkpoint
**545 个张量逐位完全相同**。两个原因:

1. **LSQ 的 `s` 是在首个 forward 里才创建的**。冻结代码放在 `setup_alpha` 之前,`s` 还是 `None`,
   `isinstance(s, nn.Parameter)` 判 False,一个都没冻上。
2. **训练循环第一步会调用 `set_trainable_policy(model, "all")**`(见 `qat_launch.py` 里
   `trainable_policy_update_mode=requires_grad` 的分支),它会对**所有**参数执行
   `param.requires_grad_(True)`,把外部冻结**静默覆盖掉**。

修正:冻结必须在 `setup_alpha` 之后施加,并且存成 `runtime_args._sticky_freeze_suffixes`,
在每次 `set_trainable_policy` 之后**重新施加**。修正后打印 `count=12 / 24 / 36`,并已核实冻结真的保持
(见 4.3)。

---

## 4. 实验与结果

### 口径

- 全部在 **同一台 4×3090 机器、同 seed 42、同数据顺序**下跑单个 GPU(bs32 × accum16 ⇒ 有效 batch 512);
- 每组跑 **110 个 optimizer step**(= 1760 micro-step ≈ epoch 0 的 2%),按 step 存 checkpoint;
- 评测:**同一份 6400 张 ImageNet val 图**,同一份离线评测代码(不跑 50k 全量,以节省时间);
- 各组差异只有 λ / collect target / 冻结集合 / hinge,其余完全一致。

### 4.1 第一批:证实"堵 s"的三种方式都能生效,但都不救模型

| 配置 | Top-1 | H(block8) | H(block10) | H(block11) |
|---|---:|---:|---:|---:|
| λ=0(参照) | **73.58%** | 2.7311 | 2.6325 | 2.5786 |
| λ=3,softplus(原始) | 38.48% | 2.0977 | 1.9929 | 2.1568 |
| λ=3,rank 走 pre-quant softmax | 41.17% | 1.9992 | 2.0475 | 2.2111 |
| λ=3,rank 走 scale-detached post-quant | 38.41% | 2.1631 | 2.0135 | 2.1298 |

(同一批里还跑了一个 `--freeze-attn-softmax-scale`,但它当时因为 3.2 的坑没生效,结果与原始组逐位相同,已作废。)

**三种方式都没有把 38% 拉回来。**

### 4.2 第二批:修好冻结之后,结果依旧

| 配置(λ=3) | Top-1 | H(block8) | H(block10) | H(block11) |
|---|---:|---:|---:|---:|
| λ=0 | **73.58%** | 2.7311 | 2.6325 | 2.5786 |
| 粘性冻结 softmax 步长 `s` | 39.39% | 2.1708 | 2.0529 | 2.2455 |
| 粘性冻结 `move_qkx_b4/aft.bias` | 39.34% | 2.0968 | 2.0005 | 2.1564 |
| **两个都冻** | **38.92%** | 2.1761 | 2.0539 | 2.2354 |
| λ=10,两个都冻 | 9.78% | 1.9735 | 1.7621 | 1.8674 |

**把两个最大的杠杆都冻死,38.48% → 38.92%,就是噪声。**

### 4.3 冻结确实生效(核实)

step 50 → step 100 的参数范数:

| run | s(block8) | s(block11) | move_qkx_b4(block8) | move_qkx_aft(block11) |
|---|---:|---:|---:|---:|
| `b3_w3_freezeS`(冻 s) | **0.1475 → 0.1475** | **0.1475 → 0.1475** | 0.1691 → 0.4311 | 0.2360 → 0.2605 |
| `b3_w3_freezeMOVE`(冻 bias) | 0.0902 → 0.1012 | 0.0919 → 0.0823 | **0.0000 → 0.0000** | **0.0000 → 0.0000** |
| `b3_w3_freezeBOTH` | **0.1475 → 0.1475** | **0.1475 → 0.1475** | **0.0000 → 0.0000** | **0.0000 → 0.0000** |
| 对照(不冻) | 0.0902 → 0.1012 | 0.0920 → 0.0824 | 0.1746 → 0.4438 | 0.2225 → 0.2451 |

### 4.4 真正的大杠杆是 qk-reparam 的加性偏置,不是 `s`

λ=0 → λ=3,按张量范数相对变化排序(取前几名):

```
 +123.4%  0.2062 -> 0.4607   features.5.5.attn.move_qkx_b4.bias     (block 9)
 +121.0%  0.1109 -> 0.2451   features.7.1.attn.move_qkx_aft.bias    (block 11)
 +114.1%  0.2073 -> 0.4438   features.5.4.attn.move_qkx_b4.bias     (block 8)
 +112.1%  0.2147 -> 0.4554   features.7.1.attn.move_qkx_b4.bias     (block 11)
 +102.4%  0.1771 -> 0.3583   features.7.0.attn.move_qkx_b4.bias     (block 10)
  +86.8%  0.5945 -> 1.1104   features.0.0.move_b4.bias / move_aft.bias  (patch embed)
  -52.1%  0.2114 -> 0.1012   features.5.4.attn.quan_a_softmax_fn.s
  -56.0%  0.1870 -> 0.0824   features.7.1.attn.quan_a_softmax_fn.s
```

λ=0 → λ=30 更明显(注意 `s` 反而没有 λ=3 掉得狠,可见它并不是因果主因):

```
 +235.5%  0.0413 -> 0.1387   features.7.0.attn.quant_x_4_qkv.move_aft.bias
 +228.4%  0.0417 -> 0.1371   features.7.0.attn.quant_x_4_qkv.move_b4.bias
 +203.7%  0.2147 -> 0.6519   features.7.1.attn.move_qkx_b4.bias
 +193.3%  0.1771 -> 0.5194   features.7.0.attn.move_qkx_b4.bias
 +156.8%  0.2062 -> 0.5295   features.5.5.attn.move_qkx_b4.bias
 +153.3%  0.2073 -> 0.5252   features.5.4.attn.move_qkx_b4.bias
  -46.7%  0.2114 -> 0.1127   features.5.4.attn.quan_a_softmax_fn.s
  -49.1%  0.1870 -> 0.0953   features.7.1.attn.quan_a_softmax_fn.s
```

`move_qkx_b4/aft` 是 `LearnableBias`,直接加在 qk 分数向量上(`qkx = self.move_qkx_b4(qkx)`,
softmax 之前),所以**它直接改 softmax 前的分数**——把偏置整体抬大就等于把注意力变尖。
这解释了第一轮看到的"熵塌陷"。

### 4.5 换成 hinge 损失:关掉了"变尖",但打开了"变平"

新增 `--attn-rank-hinge --attn-rank-margin m`,损失由 `softplus(-Δ)` 换成 `relu(m - Δ)`
(Δ ≥ m 时损失与梯度同时为 0)。`margin=0` 是最纯粹的"别把顺序弄反":
参考与学生在同一状态时损失恰好为 0(日志里第一步 `AttnRank` 确实是 `0.000e+00`,随后随漂移升到 ~1.8e-2)。

| 配置(margin=0) | Top-1 | H(block8) | H(block10) | H(block11) |
|---|---:|---:|---:|---:|
| λ=0 | **73.58%** | 2.7311 | 2.6325 | 2.5786 |
| λ=3 | 58.39% | 3.3297 | 3.5053 | 3.5783 |
| λ=30 | 9.72% | 3.5189 | 3.6342 | 3.6488 |
| λ=300 | 2.72% | 3.5431 | 3.6367 | 3.6497 |

两个观察:

1. **hinge 确实关掉了"锐化"**:注意力熵不再下降,反而升到 3.3~3.6(比 λ=0 还平);
   λ=3 的 Top-1 从 38.48% 改善到 **58.39%**。所以"变尖"这条捷径是可修的。
2. **但它打开了另一条退化路:压平。** 成对样本是按**参考**的严格顺序定义的
   (`valid = top_values > teacher_keys`),而参考是滞后的自己;学生压平 → 参考也压平 →
   大量并列 → 并列被排除 → 有效对变少 → 损失照样趋 0。λ≥30 依旧崩,只是崩法从"越来越尖"变成"越来越平"。

---

## 5. 结论

1. **"让 rank loss 别更新 `s`" 可以做,已经做了三种方式并逐一验证生效**;
2. **但堵住它没用**:冻结 `s` 得 39.39%,再冻结真正的大杠杆 `move_qkx_*` 得 38.92%,
   对照组 38.48% —— 全是噪声。这是"打地鼠",损失总能换一个旋钮;
3. 主因不是 `s`,而是 **qk-reparam 的加性偏置**(直接改 softmax 前的分数)以及更上游的整条特征链;
4. 更深一层的原因是**自参考**:参考是滞后 50 个 optimizer step 的自己,它跟着学生一起漂,
   所以损失只约束**变化速率**、不约束**目标**;
5. 两条"消掉损失"的捷径都毁模型:
   - `softplus` 无界 → **变尖**是捷径(熵 2.6 → 1.7);
   - `hinge` 有界但并列被排除 → **压平**是捷径(熵 2.6 → 3.5);
6. 因此"先堵住作弊再看 acc 收益"这条路,实验上不成立。

---

## 6. 对照实验:把损失换成老的 naive attention-KL(2026-10-08 晚)

**目的**:隔离变量。设置完全不变(prev-step ref、interval 50、global block 8/9/10/11 全 head、
全程常开、无 clip、无 drop),**只把损失从 rank 换成老的 naive attention-KL**
(`--ref-attn-loss kl_ref`,`--ref-attn-kl-clip 0`,`--ref-attn-kl-drop-prob 1.0`,`--attn-rank-weight 0`)。

如果这样不崩 ⇒ 崩的是 rank 这个损失形式;如果照样崩 ⇒ 问题在 prev-step 自参考(或"四层常开"本身)。

### 6.1 权重怎么定

KL 的数值量级远大于 rank:它对每个 head 的 (Q×K) 求和再按 batch 归一,49 个 token 时单个 head 就是
几到几十,72 个 head 取平均后仍在 **50~95** 量级。先按第一轮的方法做了梯度定标(init 时
`||dKD/dtheta||=2.577e+01`,`||dKL/dtheta||=5.153e-01`,ratio≈50 ⇒ λ(rho=0.1)≈5),
**但实测 λ=0.3 时该项已占 loss 的 ~37%,四档全部偏大**,于是换到更低的区间。

### 6.2 结果(110 个 optimizer step = epoch 0 的 2%,同一批 6400 张 val)

| 配置 | Top-1 | H(block8) | H(block10) | H(block11) | λ·KL(同一 log 区间) | 占 BaseLoss 比例 |
|---|---:|---:|---:|---:|---:|---:|
| λ=0(参照) | **73.58%** | 2.7311 | 2.6325 | 2.5786 | — | — |
| naive KL,λ=0.01 | **72.22%** | 2.7533 | 2.9465 | 3.3064 | 0.20 | 0.4% |
| naive KL,λ=0.03 | 59.47% | 2.4879 | 2.5746 | 2.7918 | 0.65 | 1.2% |
| naive KL,λ=0.1 | 12.14% | 2.3272 | 2.5128 | 2.5179 | 2.70 | 5% |
| naive KL,λ=0.3 | 1.47% | 2.4551 | 2.7729 | 2.8266 | 10.9 | 21% |

(四档同 run 条件:同一 log 区间里 `BaseLoss` 全都是 52.23,说明 KL 并没有把 KD 目标拉坏;
变的是模型本身。)

### 6.3 读法

1. **λ=0.01 基本无害**:72.22% vs 73.58%(−1.4 分)。也就是说
   **"prev-step ref + 最后四层全 head + 全程常开"这套设置本身并不是坏的**——
   换成朴素的 KL 之后不会出现 rank 那种"熵塌陷 + Top-1 掉到 6%"的灾难。
   所以第一轮的崩,**主要是 rank 这个损失形式的问题**(它有自己的退化捷径),不是 prev-step 单独造成。
2. **但可用窗口极窄**:λ=0.03 就已经 −14 分,λ≥0.1 直接崩。
   按"占 base loss 的比例"看,KL 的悬崖在 **~1%~5%**,而 rank 的悬崖在 **~2.7%**(λ=3)——两者其实同一量级。
3. 这说明 **prev-step 自参考确实有自己的问题,但不是"作弊",而是"粘性"**:
   参考是 50 步前的自己,任何"强到能起作用"的约束都在跟"模型本来要往前走"这件事对抗;
   所以能容忍的权重被压到只占主 loss 的百分之一量级,而这个量级下它自己也接近无效。
4. 注意力熵在 KL 下并不会塌(λ=0.01 时 2.75/2.95/3.31,甚至比参照略平),
   进一步佐证"熵塌陷"是 rank 独有的退化方向。

### 6.4 结论(供决策)

- **不是 prev-step 单方面"坏掉"**:同一套设置 + KL 在 λ=0.01 下能跑。
- **但 prev-step 自参考把可用的约束强度限制得非常小**(≤1% 量级),而这个强度下约束几乎不起作用;
  想让它真正起作用就得加大权重,一加大就掉点。所以"用 prev-step 做震荡抑制"这条路,
  即便换成 KL,也很难既强又稳。
- 下一步若要继续:要么**把参考系换掉**(固定锚/长 lag),
  要么**放弃自参考、改用别的机制**(例如只在被约束层上做权重层面的稳定化,而不是注意力分布层面的约束)。

---

## 7. 对照实验二:把参考换成 FP teacher(固定锚,2026-10-08 晚)

第 6 节把"损失"从 rank 换成了 KL,结论是 prev-step KL 也只能承受 λ≤0.01。
本节做**真正的单变量替换**:损失形式、层、head、常开方式全部不动,**只把参考模型从
"50 步前的自己"换成"FP teacher"**(固定锚)。

### 7.1 teacher-KL

用现成的 `--teacher-attn-kl-weight`(它走的就是同一个 `attention_kl_consistency_loss`,
`head_mode` 和 `ref_attn_kl_clip` 也共用),warmup=0、无 clip、无 drop、全程常开。

先量量级:init 时 **teacher-KL = 95.92**(prev-step 是 0,因为 ref 就是学生自己),
`||dKL_teacher|| = 20.34` vs `||dKD|| = 25.77`,ratio ≈ **1.27**(prev-step 是 0.0039),
也就是**同样的 λ 下 teacher-KL 的梯度大约强 40 倍**。所以按"占主 loss 的比例"对齐,区间取到千分位。

| 配置 | Top-1 | H(block8) | H(block10) | H(block11) | λ·KL | 占 BaseLoss |
|---|---:|---:|---:|---:|---:|---:|
| λ=0(参照) | 73.58% | 2.7311 | 2.6325 | 2.5786 | — | — |
| prev-step KL,λ=0.01(第 6 节) | 72.22% | 2.7533 | 2.9465 | 3.3064 | 0.20 | 0.4% |
| prev-step KL,λ=0.03 | 59.47% | 2.4879 | 2.5746 | 2.7918 | 0.65 | 1.2% |
| **teacher KL,λ=0.003** | **74.30%** | 2.5199 | 2.2336 | 2.2579 | 0.27 | 0.5% |
| **teacher KL,λ=0.01** | **74.84%** | 2.3544 | 2.0584 | 2.0865 | 0.90 | 1.7% |
| **teacher KL,λ=0.03** | **74.59%** | 2.2948 | 1.8948 | 1.9205 | 3.02 | 5.8% |
| teacher KL,λ=0.1 | 72.66% | 2.2086 | 1.8613 | 1.8694 | 13.8 | 26% |

读法:

1. teacher 固定锚下,λ 一路加到 **0.1**(比 prev-step 已经崩掉的 0.03 还大 3 倍)都只是轻微掉点;
2. λ=0.003~0.03 三档全部**高于 λ=0 参照**(74.3~74.8 vs 73.58),最高 +1.26 分;
3. 注意力熵的变化方向完全不同:teacher-KL 让后层熵**下降**(2.63→1.86),即学生的注意力
   **向教师靠拢**;而 prev-step KL 让熵**上升**(2.58→3.31),因为约束的是"别动"而不是"像谁"。

### 7.2 teacher-rank(softplus,topk=5,min_attn=1e-4)

用 `--attn-rank-source teacher`,权重区间跟第一轮 prev-step rank 的扫描**完全对齐**,便于直接比。

| 配置 | Top-1 | H(block8) | H(block10) | H(block11) | AttnRank |
|---|---:|---:|---:|---:|---:|
| λ=0(参照) | 73.58% | 2.7311 | 2.6325 | 2.5786 | — |
| prev-step rank,λ=3(第一轮) | 38.48% | 2.0977 | 1.9929 | 2.1568 | ~0.35 |
| **teacher rank,λ=0.3** | **75.03%** | 2.4883 | 2.2286 | 2.1655 | ~1.08 |
| teacher rank,λ=3 | 41.64% | 1.2035 | **0.0301** | 0.0428 | ~1.10 |
| teacher rank,λ=10 | 4.52% | 0.2621 | 0.0259 | 0.0097 | ~1.10 |
| teacher rank,λ=30 | 1.00% | 0.2392 | 0.0174 | 0.0123 | ~1.08 |

注意 λ=3 那一行的熵:**block10/11 掉到 0.03 / 0.04**。
49 个 key 的最大熵是 3.89,0.03 意味着注意力几乎是**严格的 one-hot**(全部质量压在一个 key 上)。

也就是说:

- 固定锚把"压平造并列"那条捷径彻底堵死了(对集合由教师固定,学生变平不会减少有效对);
- 于是无界的 `softplus(-Δ)` 只能往**另一个极端**走 —— **把注意力锐化成 one-hot**,
  这样就一劳永逸地满足所有 Δ>0。
- 之前在 prev-step 下只看到熵掉到 ~2.0(没到 one-hot),是因为参考跟着一起漂、把退化"摊薄"了。

### 7.3 汇总(同一口径:110 个 optimizer step = epoch 0 的 2%,6400 张 val)

| 参考 | 损失 | 可用 λ | 该点 Top-1 | 崩溃方向 |
|---|---|---|---:|---|
| 无(λ=0) | — | — | 73.58% | — |
| prev-step | rank / softplus | ≤0.3 | 68.75%(50k val) | 中度锐化(熵 2.6→2.0) |
| prev-step | rank / hinge | ≤3 | 58.39% | 压平造并列(熵 2.6→3.5) |
| prev-step | KL | ≤0.01 | 72.22% | 与学习对抗(约束"别动") |
| **teacher** | **KL** | **0.003~0.1** | **74.3~74.8%** | 未观察到 |
| **teacher** | **rank / softplus** | **0.3** | **75.03%** | 锐化成 one-hot(熵→0.03) |

### 7.4 结论

1. **prev-step 确实是这一轮最大的问题来源。** 同样的损失形式、同样的层/head/常开方式,
   只把参考换成固定 teacher:KL 的可用权重放大 3~10 倍、最好点从 72.2% 提到 **74.84%**;
   rank 从 λ=3 就崩变成 λ=0.3 给 **75.03%**。
2. **但 rank 这个损失形式本身仍然有退化方向**(无界 softplus),只是从"压平"换成了"锐化成 one-hot"。
   所以只有 KL,或带 margin 的 hinge,在固定锚下才是安全的。
3. **+1.3~1.5 分只是 epoch 0 的 2% 处的早期信号**,还不能当作最终收益;能确定的是
   **固定锚下存在一个又宽又稳的权重区间**,而 prev-step 几乎没有。

---

## 8. 代码改动清单

| 文件 | 改动 |
|---|---|
| `third_party/OFQ/src/quantization/quantizer/lsq.py` | 新增 `LsqQuantizer.forward_detached_scale(x)`:数值与 `forward(x)` 完全相同,但切断到 `s` 的梯度路径 |
| `third_party/OFQ/src/quantization/modules/swin_attention_and_mlp.py` | 新增 `collected_attention_view()` 与 `module.collect_attn_target`(post_quant / post_quant_detached_scale / pre_quant),三个注意力的收集分支统一走它 |
| `third_party/OFQ/src/attn_relation_ranking.py` | 新增 `hinge` / `margin` 参数,可用 `relu(margin - Δ)` 替代 `softplus(-Δ)` |
| `qat_launch.py` | 新增 `--attn-rank-target`、`--attn-rank-hinge`、`--attn-rank-margin`、`--freeze-attn-softmax-scale`、`--freeze-param-suffix`;实现**粘性冻结**(`_sticky_freeze_suffixes`,在 `setup_alpha` 之后施加、并在每次 `set_trainable_policy` 之后重新施加) |
| `tmp_scripts/run_..._4gpu_20261008.sh` | 新增 `RANK_TARGET` / `FREEZE_S` / `FREEZE_SUFFIX` / `HINGE` / `MARGIN` / `SAVE_STEPS` 等开关 |
| `tmp_scripts/analyze_attnrank_failure_checkpoints_20261008.py` | 新增 `--runs` 批量汇总模式(一次评估多组,打印 Top-1 + 注意力熵) |
| `tmp_scripts/run_..._4gpu_20261008.sh` | 新增 `REF_KL_W` / `REF_KL_CLIP` / `TEACHER_KL_W` / `RANK_SOURCE` 开关,用于第 6、7 节的参考系对照 |

---

## 9. 复现

```bash
QATS=/home/quyanyi/tiger/resume_repos/QATs
PY=/datadisk2/quyanyi/envs/qat_env/bin/python
RUN=$QATS/tmp_scripts/run_swin_w4a4_attnrank_prevstep_l8to11_4gpu_20261008.sh

# 某一组(单卡,110 个 optimizer step,存 step checkpoint)
RANK_W=3 RANK_TARGET=post_quant FREEZE_S=1 \
  FREEZE_SUFFIX="move_qkx_b4.bias,move_qkx_aft.bias" \
  MAX_UPDATES=110 LOG_INTERVAL=50 SAVE_STEPS=1 STEP_INTERVAL=50 \
  DEVICES=4 NPROC=1 EXP=demo LOG=/datadisk2/quyanyi/qat_runs/demo.log bash $RUN

# hinge 版
RANK_W=3 HINGE=1 MARGIN=0 MAX_UPDATES=110 SAVE_STEPS=1 DEVICES=4 NPROC=1 \
  EXP=demo_hinge LOG=/datadisk2/quyanyi/qat_runs/demo_hinge.log bash $RUN

# 离线汇总评估(同一份 6400 张 val)
CUDA_VISIBLE_DEVICES=4 $PY $QATS/tmp_scripts/analyze_attnrank_failure_checkpoints_20261008.py \
  --max-samples 6400 --runs /datadisk2/quyanyi/qat_runs/<run1> /datadisk2/quyanyi/qat_runs/<run2> ...

# 逐张量梯度分解(看 rank/kd 在哪些张量上谁主导)
CUDA_VISIBLE_DEVICES=4 $PY $QATS/tmp_scripts/analyze_attnrank_parameter_gradients_20261008.py
```

> **运维提醒**:启动脚本用的是 `setsid nohup`,所以只 kill 外面那个 bash 包装进程**不会**杀掉
> 真正的 python 训练进程,GPU 显存仍然被占着(本轮就因为这一点撞了一次 OOM)。
> 收尾时用 `ps -eo pid,cmd | awk '/[q]at_launch\.py/{print $1}' | xargs -r kill -9` 清干净。

原始日志与 checkpoint:

| 批次 | 路径 |
|---|---|
| λ 扫描(高段,50k val) | `/datadisk2/quyanyi/qat_runs/attnrank_sweep_w{0,3,10,30}_20261008.log` |
| λ 扫描(低段,50k val) | `/datadisk2/quyanyi/qat_runs/attnrank_sweep_lo_w{0.01,0.03,0.1,0.3}_20261008.log` |
| 第一批(pre-quant / detached / 无效冻结) | `/datadisk2/quyanyi/qat_runs/anticheat_b1_{base,freeze,detached,prequant}_w3.log` |
| 第二批(λ=0 参照 + 无效冻结复现) | `/datadisk2/quyanyi/qat_runs/anticheat_b2_*.log` |
| 第三批(粘性冻结) | `/datadisk2/quyanyi/qat_runs/b3_*.log` |
| 第四批(hinge) | `/datadisk2/quyanyi/qat_runs/b4_*.log` |
| 第五批(prev-step naive KL 对照) | `/datadisk2/quyanyi/qat_runs/klnaive_w{001,003,01,03}.log`(以及作废的高权重档 `kl_w*.log`) |
| 第六批(FP teacher KL) | `/datadisk2/quyanyi/qat_runs/tkl_w{0003,001,003,01}.log` |
| 第七批(FP teacher rank) | `/datadisk2/quyanyi/qat_runs/trank_w{03,3,10,30}.log` |
| checkpoint | 各 run 目录下 `step_checkpoints/step_{0050,0100}.pth.tar` |
