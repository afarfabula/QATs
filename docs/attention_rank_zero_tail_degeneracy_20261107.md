# Attention-rank 的零值尾部退化

> 日期：2026-11-07  
> 范围：Swin-T W4A4 QAT 中的 post-softmax attention 4-bit 概率量化、attention-relation ranking loss 及其修复方向  
> 结论：当前 attention-rank 存在明确的“零值尾部退化 + 梯度断开”通道，足以否定当前 pairwise log-rank 目标的测量有效性；但它是否是当前整体精度损失的主要来源，尚未通过直接测量确认。

---

## 0. 结论摘要

1. 问题不应称为 affine quantization 中的 “zero-point 问题”。更准确的名称是 **zero-tail degeneracy**，即：

   > 4-bit all-positive 概率量化把大量很小的注意力概率 round 成精确 0；attention-rank loss 又通过 `clamp_min(eps)` 把这些 0 伪装成 `1e-8`。

2. 教师认为应该排在后面的 key 一旦被量化成 0，排序差距会被伪造成：

   \[
   \Delta \approx \log(a_{\mathrm{anchor}})-\log(10^{-8})\gg 0
   \]

   于是 `softplus(-Delta)` 接近 0，损失看起来“排序正确”，但学生实际已经失去了这个注意力位置。

3. `clamp_min` 在输入为 0 时梯度为 0。因此这些 pair 不只是损失白送，连把概率从 0 拉回来的梯度也没有。

4. 当前 `valid` mask 和有效 pair 数由教师注意力决定，没有检查学生侧概率是否仍可表示。学生已经退化的 pair 仍会进入分母，进一步稀释有效梯度。

5. 另一个独立问题是 `softplus` 会把 log probability gap 持续推向正无穷，鼓励 attention 越来越尖。即使没有零值退化，也会造成 attention entropy collapse。

6. 这个问题对 attention-rank 路线是 critical blocker；但对整个 QAT 主线是否是主要精度瓶颈，目前只能列为未证实。必须先测 post-quant 零值比例、教师 top-k mass 保留率和量化误差/时间抖动比。

7. 不建议继续靠调 `lambda`、换 top-k、冻结 `s`、冻结 `move_qkx_*` 或单独切换 `pre_quant` 解决。现有反作弊实验已经证明这些单点修补不够。

8. 推荐方向：

   - 时间稳定性约束放在 dense 的 pre-quant attention 上；
   - 量化保真度单独约束 post-quant 与 pre-quant 的分布差异；
   - 如果保留 pairwise rank，则必须只统计学生侧真正可表示的 pair，并额外加入 top-k 概率质量保留项；
   - 用 1-3 epoch 或 10 epoch gate 验证，不再直接启动新的 100 epoch。

---

## 1. 术语澄清

### 1.1 不是 zero-point

在 asymmetric affine quantization 中，zero-point 通常指映射实数 0 到整数域的偏移量：

\[
q=\operatorname{round}(x/s)+z
\]

本文讨论的 `z` 无关。当前 attention 概率 quantizer 是 all-positive LSQ，整数范围是 `0..15`，问题来自小概率经 round 后变成整数 0。

### 1.2 本文使用“零值尾部退化”

定义为：

```text
pre-quant attention 中一个非零、且在教师参考中有意义的概率
在 post-quant attention 中被量化成精确 0，
随后被 loss 中的 clamp_min(eps) 替换为 1e-8。
```

它同时造成：

- 前向分布质量损失；
- 排序关系信息损失；
- 反向梯度断开；
- loss 数值失真。

---

## 2. 当前代码路径

### 2.1 注意力概率量化

相关位置：

- `third_party/OFQ/src/quantization/modules/swin_attention_and_mlp.py:481`
- `third_party/OFQ/src/quantization/quantizer/lsq.py:627`

Swin attention 在 softmax 后直接执行：

```python
attn = F.softmax(attn, dim=-1)
attn_prob = attn
attn = self.quan_a_softmax_fn(attn)
attn = F.dropout(attn, p=self.attention_dropout)
x = attn.matmul(v)
```

`quan_a_softmax_fn` 是 all-positive 4-bit LSQ：

```python
x = x / s
x = torch.clamp(x, 0, 15)
x = round_pass(x)
x = x * s
```

因此任意满足：

\[
a < s/2
\]

的注意力概率都会被 round 成精确 0。

### 2.2 ranking loss

相关位置：

- `third_party/OFQ/src/attn_relation_ranking.py:88`
- `third_party/OFQ/src/attn_relation_ranking.py:94`

核心计算是：

```python
delta = (
    torch.log(student_head.gather(-1, top_indices).clamp_min(eps)).unsqueeze(-1)
    - torch.log(student_head.clamp_min(eps)).unsqueeze(-2)
)
pair_term = F.softplus(-delta)
```

其中 `eps=1e-8`。

---

## 3. 失效链

### 3.1 第一步：尾部概率被量化成 0

Swin 的 window 通常包含 \(K=7\times7=49\) 个 key。若注意力完全均匀：

\[
a_i=\frac{1}{49}\approx0.0204
\]

当量化步长 \(s\) 较大时，许多低于 \(s/2\) 的概率都会变成 0。

因此 4-bit attention 的自然形态通常不是“49 个都保留了很小的概率”，而是：

```text
少数非零峰 + 大量精确 0 的尾部
```

这不是偶发数值问题，而是低比特概率量化的结构特征。

### 3.2 第二步：教师侧有效但学生侧不可表示

当前 `valid` mask 只检查教师：

```python
valid = (
    (teacher_top > teacher_keys)
    & (teacher_keys > min_attn)
    & (teacher_top > min_attn)
)
```

它保证：

- 教师认为 anchor 高于 target；
- 两个教师概率都高于 `min_attn`。

它没有保证：

- 学生 anchor 仍大于 0；
- 学生 target 仍大于 0；
- 学生的两个概率仍能体现 teacher 的排序。

因此一个 pair 可以满足：

```text
teacher anchor > teacher target > min_attn
student anchor > 0
student target = 0
```

但仍被计入 loss。

### 3.3 第三步：`clamp_min(eps)` 把消失变成“正确排序”

对于 pair \((c,d)\)：

\[
\Delta=\log a_c-\log a_d
\]

\[
L_{c,d}=\operatorname{softplus}(-\Delta)
\]

若 \(a_c>a_d\)，则 \(\Delta>0\)，loss 较小；若 \(a_d=0\)，clamp 后：

\[
a_d\leftarrow 10^{-8}
\]

于是：

\[
\Delta=\log\left(\frac{a_c}{10^{-8}}\right)\gg0
\]

\[
L_{c,d}\approx0
\]

从 loss 的视角看，这个 pair 已经“排序完全正确”；从部署模型看，target key 已经彻底消失。

### 3.4 数值例子

假设：

```text
教师：anchor = 0.30，target = 0.05
学生量化前：anchor = 0.29，target = 0.02
学生量化后：anchor = 0.29，target = 0.00
```

如果 target 没有被量化为 0：

\[
\Delta=\log\frac{0.29}{0.02}\approx2.674
\]

\[
L=\operatorname{softplus}(-2.674)\approx0.0644
\]

如果 target 被量化为 0：

\[
\Delta=\log\frac{0.29}{10^{-8}}\approx17.191
\]

\[
L=\operatorname{softplus}(-17.191)\approx3.4\times10^{-8}
\]

于是从 `0.0644` 降到 `3.4e-8`。损失下降不是模型更好地保持了注意力关系，而是把需要约束的概率位置删掉了。

### 3.5 第四步：梯度在 0 处同时被切断

`clamp_min(eps)` 在输入小于等于 0 时的导数为 0：

\[
\frac{\partial}{\partial a_d}\operatorname{clamp\_min}(a_d,10^{-8})=0,
\quad a_d=0
\]

因此：

- 前向把 0 伪装成 \(10^{-8}\)；
- 反向仍然把它当成断开的常量；
- target 概率没有动力从 0 恢复；
- 排序目标无法再把该位置纳入有效优化。

### 3.6 有效 pair 数会误导

`pair_count` 的分母来自教师侧 `valid`：

```python
pair_count = valid.sum(dim=(-1, -2))
per_row = (pair_term * valid).sum(dim=(-1, -2)) / pair_count.clamp_min(1)
```

因此即使大量 student target 已经为 0：

- 它们仍被计入 `valid_pairs`；
- 它们仍进入分母；
- 它们对 loss 的梯度贡献近似为 0；
- 日志里的有效 pair 数会明显高于真正有梯度的 pair 数。

这会同时造成两个后果：

1. 报告出来的 `valid_pairs` 不能代表有效监督量；
2. 非退化 pair 的平均梯度被大量退化 pair 稀释。

---

## 4. 第二个问题：softplus 会鼓励注意力无限变尖

即使没有零值退化，当前 softplus 仍有问题。

\[
\operatorname{softplus}(-\Delta)
\]

只有在：

\[
\Delta\rightarrow+\infty
\]

时才接近 0。

而：

\[
\Delta=\log a_c-\log a_d
\]

因此 softplus 并不只是要求：

\[
a_c>a_d
\]

它还会不断鼓励：

\[
\frac{a_c}{a_d}\rightarrow+\infty
\]

这会造成 attention entropy collapse，使注意力越来越接近 one-hot。

已有实验中可以观察到：

| 配置 | Top-1 | H(block10) |
|---|---:|---:|
| lambda=0 | 73.58% | 2.6325 |
| lambda=3 | 38.48% | 1.9929 |
| lambda=30 | 6.89% | 1.6812 |

hinge 可以缓解第二个问题，因为达到目标 margin 后梯度归零：

\[
\operatorname{relu}(m-\Delta)
\]

但 hinge 不能修复零值尾部退化。实验中 lambda=3 加 hinge 后，Top-1 从 38.48% 回到 58.39%，改善显著，但仍远低于基线。

因此这是两个独立问题：

1. **zero-tail degeneracy**：目标概率消失后得到免费零损失；
2. **softplus over-sharpening**：损失持续鼓励拉大 log gap。

---

## 5. 已有实验证据

### 5.1 loss 与精度脱钩

反作弊实验中，attention-rank loss 在多组 lambda 下都大致停留在 0.35 附近，但 Top-1 可以从：

```text
69.4%
```

降到：

```text
34.5% / 10.9% / 6.4%
```

一个真正度量“注意力排序是否保持”的目标，不应在模型接近崩溃时保持看似稳定。

### 5.2 单点堵漏无效

同一短跑口径下：

| 修复方式 | lambda=3 Top-1 |
|---|---:|
| 原始 post-quant softplus | 38.48% |
| 在 pre-quant softmax 上算 rank | 41.17% |
| post-quant 数值 + detached scale | 38.41% |
| 冻结 attention softmax 的 `s` | 39.39% |
| 冻结 `move_qkx_*` bias | 39.34% |
| 同时冻结 `s` 和 move bias | 38.92% |
| softplus 改 hinge，margin=0 | 58.39% |
| lambda=0 基线 | 73.58% |

这说明：

- `s` 是一个杠杆，但不是唯一原因；
- 全精度 `move_qkx_*` 旁路也是杠杆；
- `pre_quant` 能避免部分零化，但没有恢复有效收益；
- hinge 能缓解锐化，但不能让目标变得可靠。

### 5.3 100 epoch 长跑也没有证明收益

teacher attention-rank w=0.3 的 100 epoch 运行：

- best：`80.8720 @ep92`；
- 同栈 logit-rank 曲线 best：`80.9380 @ep94`；
- 逐 epoch 配对平均：teacher-rank 低 `0.098`；
- 100 轮只赢 17 轮。

由于 logit-rank 曲线本身也不是纯 `lambda=0` 基线，这不能证明 attention-rank 显著更差；但它明确说明没有“100 epoch 后翻盘”的证据。

---

## 6. 这个问题有多关键

### 6.1 对 attention-rank 方法本身：critical blocker

当前无法严谨声称：

```text
attention-rank loss 下降
= 学生更好地保持了注意力关系
= 模型更稳或精度更高
```

因为 loss 下降可能只是：

- 将尾部概率量化为 0；
- 拉大 log gap；
- 缩减量化步长；
- 利用 QK reparameterization 的全精度 bias；
- 让有效监督集中在少数可表示的峰上。

这是目标函数和评价指标脱钩，不是普通数值误差。

### 6.2 对当前整体 QAT 主线：重要但未证明为主因

当前最好的 200 epoch 结果并不依赖 attention-rank 获得稳定正收益。现有证据支持：

```text
attention-rank 路线无效或不可归因
```

但尚不能推出：

```text
当前主模型的主要精度损失来自 zero-tail degeneracy
```

在没有任何直接测量前，正确判断是：

- 它是 attention-rank 研究的阻断项；
- 它可能是低比特 attention 部署质量的重要误差来源；
- 它对整体 Top-1 的贡献大小未知。

建议用以下优先级表示：

| 问题 | 重要性 |
|---|---:|
| 当前 attention-rank loss 是否可信 | 9/10 |
| 是否应继续调当前 rank 的 lambda / top-k | 9/10，应先停 |
| zero-tail 是否是当前整体 QAT 的主要精度瓶颈 | 4/10，尚待测量 |
| 是否值得先做 zero-tail / mass-loss 诊断 | 10/10 |
| 是否应立即再跑新的 100 epoch | 2/10，不建议 |

---

## 7. 必做的无损诊断

在所有改动之前，先在固定 val batch 上做一次只读诊断。建议每层、每 head、每 epoch 统计以下指标。

### 7.1 Post-quant 零值比例

\[
\mathrm{zero\_rate}
=
\frac{\#\{A_{q}=0\}}{\#A_{q}}
\]

应分别报告：

- 全部 attention；
- 被 rank 约束的 layer/head；
- 教师 top-k 内部；
- 教师 top-k 外部；
- anchor 和被比较 target 分开统计。

### 7.2 教师概率质量保留率

对每个 query row：

\[
M_{k}^{ref}
=
\sum_{c\in\mathrm{TopK}(A_{ref})}A_{ref}(c)
\]

\[
M_{k}^{q}
=
\sum_{c\in\mathrm{TopK}(A_{ref})}A_{q}(c)
\]

统计：

\[
\frac{M_{k}^{q}}{M_{k}^{ref}}
\]

如果 top-k 的 mass 从接近 1 掉到很低，说明不是只丢无用长尾，而是真正吃掉了重要注意力。

### 7.3 退化 pair 分类

对所有教师有效 pair，分别统计：

| 类型 | 学生 anchor | 学生 target |
|---|---|---|
| 正常 | >0 | >0 |
| target 消失 | >0 | =0 |
| anchor 消失 | =0 | >0 |
| 两边都消失 | =0 | =0 |

只有第一类 pair 可能提供正常的 log-rank 比较。

### 7.4 数值与梯度覆盖

同时记录：

- 原始 teacher-valid pair 数；
- student-effective pair 数；
- 两类 pair 比例；
- rank 对 weight、`s`、`move_qkx_*`、bias 的梯度范数；
- 最后一层 attention softmax scale 的梯度。

### 7.5 水平量化误差 vs 时间抖动

在同一个固定 val batch 上计算：

\[
\Delta A_{\mathrm{quant}}
=
\operatorname{JSD}(A_{q},A_{\mathrm{pre}})
\]

\[
\Delta A_{\mathrm{time}}
=
\operatorname{JSD}(A_{t},A_{t-1})
\]

分别对 pre-quant 和 post-quant 统计。

关键是比较：

\[
\frac{\Delta A_{\mathrm{quant}}}{\Delta A_{\mathrm{time}}}
\]

如果该比值达到 \(10^2\) 以上，说明主要问题是当前量化的水平误差，而不是相邻 epoch 的时间震荡。此时继续优化时间平滑方向的收益上限很低。

---

## 8. 推荐的目标函数重构

### 8.1 首选：拆成“时间约束”和“量化保真”

概念目标：

\[
L
=
L_{\mathrm{KD}}
+\lambda_t L_{\mathrm{temporal}}
+\lambda_q L_{\mathrm{quantization}}
\]

#### 时间约束放在 pre-quant

pre-quant attention 是 dense distribution，不会因为 4-bit round 产生精确 0。

\[
L_{\mathrm{temporal}}
=
\operatorname{JSD}
\left(
A_t^{\mathrm{pre}},
A_{t-1}^{\mathrm{pre}}
\right)
\]

它回答：

```text
模型自己的注意力是否在时间上漂移？
```

#### 量化保真单独约束

先归一化概率分布：

\[
\hat A_q
=
\frac{A_q}{\sum A_q+\epsilon}
\]

\[
\hat A_p
=
\frac{A_p}{\sum A_p+\epsilon}
\]

分布形状损失：

\[
L_{\mathrm{shape}}
=
\operatorname{JSD}
\left(
\hat A_q,
\operatorname{stopgrad}(\hat A_p)
\right)
\]

当前 forward 中 quantized attention 没有重新归一化，因此还必须显式约束行 mass：

\[
L_{\mathrm{mass}}
=
\left(
\sum A_q-\sum A_p
\right)^2
\]

最终：

\[
L_{\mathrm{quantization}}
=
L_{\mathrm{shape}}
+\alpha L_{\mathrm{mass}}
\]

它回答：

```text
4-bit 量化是否改变了当前注意力的形状和质量？
```

优点：

- 不需要对零概率做 `log(0)`；
- 不再依赖单个尾部 key 的 pairwise 排序；
- 可以直接区分时间漂移和水平量化误差；
- 教师锚定和时间自参考可复用同一个分布损失。

### 8.2 次选：保留 pairwise rank，但补上两个保护

如果必须保留排序形式，建议使用：

\[
L
=
L_{\mathrm{rank}}^{\mathrm{effective}}
+\beta L_{\mathrm{topk\_mass}}
\]

#### 只统计学生有效 pair

增加：

```python
student_valid = (
    student_anchor > student_eps
) & (
    student_target > student_eps
)

effective = teacher_valid & student_valid
```

注意：

- student 侧退化的 pair 不能被当作“已满足”；
- 它们只能从 rank loss 中排除；
- 排除后必须由 mass loss 补上约束，否则就是放弃这些位置。

#### 加 top-k mass loss

\[
L_{\mathrm{topk\_mass}}
=
\left(
\sum_{c\in\mathrm{TopK}(A_{ref})}A_q(c)
-
\sum_{c\in\mathrm{TopK}(A_{ref})}A_{ref}(c)
\right)^2
\]

这个项能让精确 0 重新获得梯度通道。原因是 LSQ 的 STE 仍然允许梯度从 rounded output 回到 pre-quant 输入，而 mass loss 不依赖 `log(clamp_min(0))`。

#### 使用 hinge

建议：

\[
L_{\mathrm{rank}}
=
\operatorname{relu}(m-\Delta)
\]

而不是 softplus。达到目标 margin 后 loss 和梯度同时为 0，避免持续鼓励 attention 锐化。

### 8.3 不建议的方案

#### 只把 eps 放大

只是换一个更大的假概率，仍然不反映真实分布。

#### 只把 `clamp_min(q, eps)` 改成 `q + eps`

梯度在 0 处不再严格为 0，但量级约为 \(1/\epsilon=10^8\)，病态且容易梯度爆炸。

#### 只去掉 clamp

会产生 `log(0)`、NaN 或 Inf。

#### 只切换到 pre-quant

可以避免零值，但约束对象不再是部署时实际使用的 post-quant attention。

#### 只冻结 scale 或 move bias

已有实验证明只能带来噪声级变化，而且会切换作弊杠杆。

#### 只用 hinge

只解决过度锐化，不解决概率质量丢失和梯度断开。

---

## 9. 量化器和部署层面的可选方向

这些方向的改动大于损失函数修补，应先做 eval-only 或短跑 gate。

### 9.1 更好的 per-row scale calibration

当前 softmax quantizer 的 `s` 是 per-channel，在 attention 上按 query 位置组织。可以选择让 scale 同时优化：

- JSD(post-quant, pre-quant)；
- top-k mass retention；
- 行 mass error；
- clipping rate。

目标不是简单最小化 MSE，而是保留决定注意力行为的前若干概率。

### 9.2 Post-quant row renormalization

执行：

\[
A_q\leftarrow\frac{A_q}{\sum A_q}
\]

优点：

- 保证行概率质量；
- 减少“峰吸收尾部质量”导致的整体注意力幅值变化。

风险：

- 当前 forward 没有这一步，会改变模型语义和已有 checkpoint；
- 需要检查目标硬件是否支持相应归一化；
- 不能解决精确 0 pair 的排序信息缺失。

### 9.3 Log-domain / companding quantization

对概率先做：

\[
y=\log(1+a/\tau)
\]

再执行 4-bit 量化，可以更好保留尾部相对关系。

风险：

- 非均匀量化可能不符合目标硬件；
- 反量化后仍需归一化；
- 与 strict W4A4 的部署语义可能不一致。

### 9.4 训练期软量化辅助

可以增加一个只用于辅助损失的 soft/expected quantized attention，使尾部梯度更稳定。

但必须同时保留真实 hard post-quant 指标，否则会重新出现：

```text
训练代理很好，部署模型仍然很差
```

---

## 10. 建议的实验顺序

不要直接启动新的 100 epoch。

### Gate 0：只读诊断

- 使用固定 512-1024 张 val；
- 统计 zero rate、top-k mass、退化 pair 分类；
- 统计 `Delta A_quant / Delta A_time`；
- 统计 rank 梯度落点。

决策：

- 若 zero rate 只发生在教师无关的长尾，且 top-k mass 基本无损：zero-tail 不是主因；
- 若 top-k mass 显著下降、退化 pair 占多数：当前 rank loss 基本失效；
- 若 `Delta A_quant / Delta A_time > 100`：优先修水平量化误差，不再围绕时间震荡加约束。

### Gate 1：10 epoch 同路径对照

至少包含：

1. 纯 KD；
2. CE + KD；
3. 新的 pre-quant temporal JSD + post/pre quant fidelity；
4. 原 attention-rank 作为历史对照。

全部使用：

- 同入口；
- 同 seed；
- 同数据；
- 同 effective batch；
- 每 epoch 50k full validation。

### Gate 2：只有前 3-5 epoch 有明确趋势时才扩长

如果新目标在 3-5 epoch 内：

- top-k mass 明显改善；
- quant/pre JS 下降；
- full-val 不劣于同路径 KD；

再考虑 10 epoch，之后才考虑长跑。

---

## 11. 最终判断

当前 attention-rank 不是普通的“loss 写法可以再调调”的问题。

它同时存在：

1. 概率尾部被 round 成 0；
2. loss 将 0 伪装成 \(10^{-8}\)；
3. clamp 在 0 处切断梯度；
4. pair 分母仍包含退化 pair；
5. softplus 鼓励注意力无限变尖；
6. 全精度 QK 旁路提供替代作弊通道；
7. STE 对离散排序目标存在系统性偏差。

因此：

- **对当前 attention-rank 路线**：这是阻断性问题，应停止继续调当前形式；
- **对整个 QAT 主线**：它是重要嫌疑，但在完成直接测量前，不应宣称为主要精度瓶颈；
- **最合理的下一步**：先测量零值比例、top-k mass 和量化/时间误差比，再把目标改成分离的“时间分布稳定性 + 量化保真度”双目标。

一句话：

> 当前损失会把“把注意力尾部删掉”误认为“把排序做对了”。在这个通道关闭之前，attention-rank 的 loss 数值不能作为方法有效的证据。

