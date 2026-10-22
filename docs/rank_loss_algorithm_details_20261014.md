# Rank Loss：算法与实验结果

## 1. 分类 logits ranking

### 1.1 方法

在 KD 训练上增加一个排序辅助损失：教师只提供“谁排在谁前面”，学生保持同样的相对顺序。

对一对类别 $(c,d)$，$c$ 是教师认为排名更高的类别，$d$ 是排名更低的类别。记学生 logits 的差值 $\Delta_{c,d} = s_c - s_d$，用 softplus 要求学生保持 $\Delta_{c,d} > 0$：

$$
\ell_{c,d}
= \operatorname{softplus}(-\Delta_{c,d})
= \log\left(1 + e^{-\Delta_{c,d}}\right)
$$

对每个样本 $i$，取教师 logits 的 top-k 类别作为 anchor，与所有教师分数严格更低的类别配对，并列不计入；实验中取 **topk = 5**。分类 logits ranking 损失为

$$
\mathcal{L}_{\mathrm{logit\text{-}rank}}
=
\frac{1}{|\mathcal{I}|}
\sum_{i \in \mathcal{I}}
\frac{1}{|\mathcal{P}_i|}
\sum_{(c,d) \in \mathcal{P}_i}
\operatorname{softplus}\!\left(-(s_{i,c} - s_{i,d})\right)
$$

其中 $\mathcal{I}$ 是至少有一个有效对的样本集合，$\mathcal{P}_i$ 是样本 $i$ 的有效对集合。先按样本内的有效对取平均，再对样本取平均。总损失为 $\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{KD}} + \lambda_{\mathrm{logit}}\mathcal{L}_{\mathrm{logit\text{-}rank}}$。

### 1.2 100 epoch 结果（红线为最新logit rankloss）

![](https://secure2.wostatic.cn/static/kRgAzq5BHpHRG59dr8XU6K/image.png?auth_key=1791954097-tAnztDGgXiD92ZHWfFp4mW-0-10a82b0e7888cf222d5c82142f4dd10a)

|实验|Top-1|
|-|-|
|**logits-ranking λ=0.04，100 epoch**|**80.9380**（best @ep94）|
|归档 no-KL baseline|80.7920|
|归档 FP teacher KL|80.8180|
|归档 self-ref KL|80.7720|


logits-ranking 的 best 比归档基线高约 **0.15 pp**，看上去有收益。



### 1.3 同栈 10 epoch 对照：收益来自换栈

两组的逐轮 50k 全量验证：

|epoch|λ=0 对照|logits-ranking λ=0.04|差值（对照 − rank）|
|-|-|-|-|
|1|78.4180|78.3120|+0.1060|
|2|78.9160|78.7800|+0.1360|
|3|79.0740|79.1800|−0.1060|
|4|79.3260|79.2960|+0.0300|
|5|79.2760|79.4260|−0.1500|
|6|79.3400|79.5080|−0.1680|
|7|79.5480|79.4780|+0.0700|
|8|79.5640|79.4920|+0.0720|
|9|79.4460|79.3720|+0.0740|
|**10**|**79.6640**|**79.5620**|**+0.1020**|


![](https://secure2.wostatic.cn/static/qBvKNmywoTwYMZDwDwSMhE/image.png?auth_key=1791954252-23XByEj26GH3RTDMD6wbz3-0-e75b9da3340c5340f69a889e67584892)

但这个 100 epoch 对比不是单变量：新跑的 logits-ranking 用的是，torch 版本、数据格式、硬件都不同。

把入口、torch 版本、数据、卡数、seed 全部对齐后，只跑 10 epoch：

|epoch 10|Top-1|
|-|-|
|λ=0 对照（同栈）|**79.6640**|
|logits-ranking λ=0.04|79.5620|


同栈下 λ=0 对照反而更高，说明之前的收益主要来自**换栈**，而不是 logits ranking 本身：

- torch 版本不同（当前 2.5.1，历史环境 2.9.1）；
- 硬件不同；
- 单跑复现显示：old `train.py` 路径 epoch 0 = **76.97**，`qat_launch` 路径 epoch 0 = **78.418**，栈本身差 **+1.45 pp**。

### 1.4 为什么 logits ranking 没有作用

因为 baseline 里本来就有 KD。KD 不是只在最后做一个硬标签 CE，而是直接把学生的 logits 软分布拉向教师分布：

$$
\mathcal{L}_{\mathrm{KD}}
=
\frac{T^2}{B}
\sum_{i=1}^{B}
\sum_{c=1}^{C}
p^{T}_{i,c}
\left(\log p^{T}_{i,c} - \log p^{S}_{i,c}\right)
$$

其中 $p^{T}=\operatorname{softmax}(t/T)$，$p^{S}=\operatorname{softmax}(s/T)$。它对每个学生 logit 的梯度是

$$
\frac{\partial \mathcal{L}_{\mathrm{KD}}}{\partial s_{i,c}}
=
\frac{T}{B}\left(p^{S}_{i,c} - p^{T}_{i,c}\right)
$$

所以只要教师认为类别 $c$ 排在 $d$ 前面（$p^{T}_{c} > p^{T}_{d}$），KD 的梯度就自动把 $s_c$ 相对 $s_d$ 往上推。也就是说，**KD 本身已经包含了“保持教师类别顺序”的监督，而且是全类别、带软权重的版本**。

logits ranking 只是在同一批学生 logits 上再加一个 top-5 的 pairwise 项，方向和 KD 完全相同：教师认为谁更高，KD 已经推谁更高。它没有约束 KD 没约束到的东西，也没有引入新的监督信息。

PTQ 文档里 rank loss 有用，是因为那里的基础损失是 block reconstruction，不是最终 logits 的 CE/KD；重构损失不直接约束类别顺序，所以 rank 提供了新信息。这里基础损失已经是 KD，相当于在最终 logits 上做全类别软 CE，rank loss 的边际信息量就非常小。

实验现象也一致：λ 扫描里 BaseLoss 在 λ=0/0.04/0.12/0.4 下几乎不变（52.2579/52.2566/52.2536/52.2446）；同栈 10 epoch 对照 λ=0 是 79.664，logits-ranking 是 79.562。rank loss 的数值可以下降，但那只是它自己拟合了 KD 已经产生的顺序，并不代表主任务多获得了监督。

所以第一部分结论是：logits ranking 不破坏主目标、开销很低，但同栈 10 epoch 对照没有复现收益，**100 epoch 的表面提升不能归因给 rank loss**。

---

## 2. Attention-relation ranking

### 2.1 约束范围与做法

为了清晰观察“对 attention 做 rank”这个变量的作用，本次直接约束**最后四层（global block 8/9/10/11）的全部 head**。

最后四个 block 按 head 平均的 JS 变化最大，其中 block 10/11 最严重；因此这一轮把约束范围放在最后四层全部 head。这里一个 block 对应一个 Swin Transformer block，最后四个 block 即 global block 8/9/10/11（stage 2 的最后两个 block + stage 3 的两个 block）。

对每个 `(layer, head, query q)`，排序的对象是这个 query 对所有 key 的 **pre-softmax QK 分数**。记第 $c$ 个 key 对 query $q$ 的分数为

$$
z_{q,c} = \frac{q_q \cdot k_c}{\sqrt{d}} + b_{q,c}
$$

其中 $q_q \cdot k_c$ 是 QK 点积，$b_{q,c}$ 是 relative position bias 等加性项。把这个分数在 key 维做 softmax，就得到注意力概率

$$
A_{q,c} = \frac{\exp(z_{q,c})}{\sum_{j=1}^{K}\exp(z_{q,j})}
$$

下面用 $A^{S}$ 表示学生的 attention，用 $A^{T}$ 表示教师/参考模型的 attention；每一行都满足 $\sum_c A_{q,c}=1$。教师给出这一行的顺序：

1. 取教师 $A^{T}_{q,\cdot}$（等价地取 $z^{T}_{q,\cdot}$）的 top-k key 作为 anchor $c$；softmax 单调，所以两种取法完全一致；
2. 与所有满足 $A^{T}_{q,d} < A^{T}_{q,c}$（等价于 $z^{T}_{q,d} < z^{T}_{q,c}$）的 key $d$ 配对；
3. $c$ 和 $d$ 的教师注意力都要超过阈值 $\varepsilon_{\mathrm{attn}}$，用来排除 mask / padding / 量化残值；
4. 学生保持 $A^{S}_{q,c} > A^{S}_{q,d}$（等价于 $z^{S}_{q,c} > z^{S}_{q,d}$）。

也就是说，排序的单位是“**每个 query 内部的一行 key 分数**”：对同一个 query，key 之间的 QK 分数大小关系要跟教师一致。不同 query 的分数不在同一个 softmax 里，不能跨 query 比较。

实验中取 **topk = 5**（`--attn-rank-topk 5`）、**min_attn = 1e-4**。`min_attn` 的具体作用是：Swin 的 SW-MSA 会把不同区域拼进同一个窗口，被 mask 的 key 在 softmax 之后不是干净的 0，而是 0 或约 $10^{-30}$ 量级的 denormal；如果只用 `>0` 判断有效，这些无效 key 会被当成真实的排序目标，而且它们进入 log 后会被 clamp 到 $10^{-8}$，产生极大的 $\log A$ 差值（等价于极大的 $z$ 差），把损失和有效对数都带偏。因此 `min_attn` 要求 anchor key 和目标 key 的教师注意力都大于 $10^{-4}$ 才允许配对，从而把 mask / padding / 已被量化成 0 的 key 排除掉。

因为对同一个 query 行，softmax 的分母相同，所以

$$
\log A^{S}_{q,c} - \log A^{S}_{q,d} = z^{S}_{q,c} - z^{S}_{q,d}
$$

也就是说：排序 $A^{S}$ 和排序 $z^{S}$ 在这一行内完全等价。模型 forward 返回的是 softmax 之后的 $A$，所以代码里直接用 $\log A^{S}$ 的差来计算 $z^{S}$ 的差。attention-relation ranking 损失为

$$
\mathcal{L}_{\mathrm{attn\text{-}rank}}
=
\frac{1}{|\mathcal{H}|}
\sum_{h \in \mathcal{H}}
\frac{1}{|\mathcal{Q}_h|}
\sum_{q \in \mathcal{Q}_h}
\frac{1}{|\mathcal{P}_{h,q}|}
\sum_{(c,d) \in \mathcal{P}_{h,q}}
\operatorname{softplus}\!\left(-\left(\log A^{S}_{h,q,c} - \log A^{S}_{h,q,d}\right)\right)
$$

其中 $\mathcal{H}$ 是被约束的 `(layer, head)` 集合，$\mathcal{Q}_h$ 是至少有一个有效对的 query 行集合，$\mathcal{P}_{h,q}$ 是该行的有效对集合。每行先按有效对数归一化，再对行和 head 取平均。总损失为 $\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{KD}} + \lambda_{\mathrm{attn}}\mathcal{L}_{\mathrm{attn\text{-}rank}}$。

参考模型有两种：

- **自参考（prev-step self）**：每 50 个 optimizer step 把参考模型硬拷贝成学生当前权重；
- **固定 FP teacher**：始终用全精度教师，不随学生漂移。

### 2.2 自参考：早期就崩坏，且没有超过基线

**最好的 λ=0.03 是 69.65% vs 基线 69.36%，只高 0.29 pp**

![](https://secure2.wostatic.cn/static/9dSiSM6nbzx22aTYy9MM9D/image.png?auth_key=1791954318-2KZnEeKCDB5KNJUvxCsRr7-0-f655390bb36da7cae8641c64acb628b2)

自参考在训练早期就出现了严重问题。λ 扫描（100 个 optimizer step = epoch 0 的 2%，50k 全量验证）：

|λ|0|0.03|0.1|0.3|3|10|30|
|-|-|-|-|-|-|-|-|
|Top-1 (%)|69.36|**69.65**|69.48|68.75|**34.45**|10.85|**6.43**|


原因是自参考是“50 步前的学生自己”，它跟着学生一起漂移，所以这个损失只约束“变化速率”，不约束目标，模型可以通过把注意力变得极尖来把损失做小：

- λ=30 时注意力熵从 2.6~2.7 掉到 1.7，注意力明显锐化；
- 在更大权重下，最后两层甚至接近 **one-hot**（block 10/11 的熵降到 0.03 / 0.04）；
- 同时量化步长 ss 被压小、`move_qkx_*` 加性偏置被抬大，注意力尾部被量化成 0，这些配对于是“白送”零损失；
- 把 ss 和 `move_qkx_*` 都冻结后，λ=3 只从 38.48% 挪到 38.92%，没有修好；换成 hinge 可以止住锐化，但会把注意力压平、制造并列，λ≥30 依旧崩掉。

**在可用区间里也没有收益**：**最好的 λ=0.03 是 69.65% vs 基线 69.36%，只高 0.29 pp；**λ=0.3 已经开始掉点。所以自参考这条路没有继续。

### 2.3 改成固定 FP teacher 监督

把参考模型换成固定的 FP teacher，其余设置不变（最后四层全 head、全程常开），分别跑旧的 attention KL 和新的 attention rank。

2% epoch 短跑精度没有崩（**这里基线和2.2不一样是因为验证集用的子集**）：

|配置|Top-1 (%)|
|-|-|
|λ=0 基线|73.58|
|prev-step KL λ=0.01|72.22|
|teacher KL λ=0.01|**74.84**|
|**teacher rank λ=0.3**|**75.03**|
|teacher rank λ=3|41.64|


但把训练拉长到 3 epoch（每轮 50k 全量验证）后：



![](https://secure2.wostatic.cn/static/qWi9UBLGsFsp17jGzCkrHX/image.png?auth_key=1791954426-2GMxuPZgzvcxRRqw7EauRk-0-7ba0cf6c9febe12fb244286ce43604a0)

|epoch|baseline|teacher KL λ=0.01|teacher rank λ=0.3|
|-|-|-|-|
|1|78.4180|78.1860|78.5040|
|2|78.9160|78.5840|78.7680|
|**3**|**79.0740**|78.7660|79.0320|


3 epoch 时两者都没有超过基线，但 rank 的差距（−0.042 pp）比 KL（−0.308 pp）小得多。由于目前只是少量 epoch 的短跑，**固定 teacher + rank 仍然只是一个有希望、但未证实收益的方向**。

### 2.4 下一步：训练后期再施加 attention 约束

目前的计划是把 attention 约束从“全程常开”改成“只在训练后期施加”。

动机是：训练早期学生本身变化很快，自参考会跟着漂移，固定 teacher 的约束又容易和早期拟合主目标冲突；无论是 KL 还是 rank，早期持续施加都没有拿到超过基线的稳定收益。后期模型表示已经基本稳定时再约束 attention，更有可能在不影响主目标的情况下抑制注意力漂移。

---

## 3. 当前小结

1. 分类 logits ranking：100 epoch 表面有收益，但同栈 10 epoch 对照没有复现，收益主要来自换栈；
2. attention-relation ranking（自参考）：早期就出现 λ 悬崖和注意力锐化，没有超过基线；
3. attention-relation ranking（固定 FP teacher）：短跑里 rank 好于 KL，但 3 epoch 时都没有超过基线；
4. 下一步：探索只在训练后期施加 attention 约束。

*文档日期：2026-10-14。*
