# 逐块量化中的教师 Top-5 Logits Ranking

本文说明已在 FIMA-Q 与 LS-ViT 的 DeiT-Tiny 量化实验中使用的 ranking 做法。它是在**原逐块重构目标上增加教师类别顺序监督**，不是对最终 logits 做一次后处理，也不是重新训练全模型。

## 1. 方法概览

传统逐块重构主要要求量化块的输出接近全精度块的输出。这里额外要求：当前量化块的输出经过剩余网络后，分类 logits 应保留全精度教师认为重要的类别顺序。

每次只优化当前块的 AdaRound 舍入参数与原方法启用的激活量化尺度。后续网络的参数不更新，但参与前向计算和反向传播，以便最终类别排序的梯度能够回到当前块。原重构项、舍入正则、QDrop、曲率估计、优化器和学习率规则均保留。

新增部分只有两项：

1. 以教师 Top-5 类别为起点的成对排序损失。
2. 在每个块开始优化前，用四个固定批次的**块输出梯度范数**定标一个固定损失权重。

训练结束后不保留教师、排序损失或后缀反传路径。Ranking 本身不增加部署模型的参数或前向算子。

## 2. 记号、数据与逐块计算图

设分类任务有 $C$ 个类别，批大小为 $B$。全精度教师为 $f_{\mathrm{FP}}$，其参数始终固定。对于第 $i$ 个已经完成训练增强的输入 $x_i$，教师原始 logits 为

$$
t_i=f_{\mathrm{FP}}(x_i)\in\mathbb{R}^{C}.
$$

重构第 $b$ 个块时，记：

- $\theta_b=(\alpha_b,a_b)$ 为可学习的量化参数。其中 $\alpha_b$ 控制权重的软舍入，$a_b$ 表示原流程允许优化的激活尺度集合。
- $u_{b,i}$ 为送入当前块的缓存输入。它按原重构流程，从全精度前缀特征、已量化前缀特征或二者的 QDrop 混合得到。
- $v^{\mathrm{FP}}_{b,i}$ 为同一输入的全精度块输出，用作原重构目标。
- $y_{b,i}=Q_b(u_{b,i};\theta_b)$ 为当前量化块输出。
- $G_b$ 为当前块之后到分类 logits 的真实剩余网络，本文称为后缀网络。

排序损失使用的学生 logits 是

$$
s_{b,i}=G_b(y_{b,i})\in\mathbb{R}^{C}.
$$

这里的学生是**当前逐块重构状态下的混合精度路径**，不要求尚未重构的后缀提前变成最终低比特网络。后缀保持原流程的 raw 模式；已重构前缀的量化影响通过缓存输入进入当前块。当前块仍按原流程使用软舍入和 QDrop。

### 数据配对与缓存

同一张图片的一次训练增强结果只生成一次并缓存。教师 logits、全精度块特征、量化前缀特征按同一索引配对。不能让教师与学生分别重新抽样随机增强。

教师 logits 可在原全精度特征收集过程中缓存。固定前缀特征按原逐块流程收集并复用，不在每次优化步骤重新运行教师或固定前缀。学生 logits 不能跨优化步骤缓存，因为它们必须反映当前量化参数，并保留到当前块的梯度链。

## 3. 教师 Top-5 成对排序损失

### 3.1 排序对的定义

对每个样本，先从教师 logits 选出最大的五个类别索引：

$$
\mathcal{T}_i=\operatorname{TopKIndices}(t_i,5).
$$

以这些类别为起点，与所有教师分数严格更低的类别构成有向对：

$$
\mathcal{P}_i=
\left\{(c,d):c\in\mathcal{T}_i,\ d\in\{1,\ldots,C\},\ t_{i,c}>t_{i,d}\right\}.
$$

因此，这不是只约束 Top-5 内部的顺序，也不是只比较 Top-5 与其余类别：两种比较都包含在内。教师相等分数的类别对和自比较均被排除。学生必须使用**教师给出的类别索引**，不能把自己的 logits 排序后按名次匹配。

例如，教师分数为 $(5,4,3,2,1,0)$ 时，五个起点分别与 $5,4,3,2,1$ 个更低类别比较，共有 15 对。对于无 ties 的 1000 类任务，每个样本有

$$
|\mathcal{P}_i|=\sum_{r=1}^{5}(1000-r)=4985
$$

个有效对。若某个样本不存在严格排序对，例如所有教师 logits 相等，当前实现直接报错，不把该样本静默记为零损失。

### 3.2 损失与归约

对有效对 $(c,d)$，希望学生也满足 $s_{b,i,c}>s_{b,i,d}$。定义学生的成对 logit 差为 $\Delta_{i,c,d}=s_{b,i,c}-s_{b,i,d}$，使用平滑的 logistic 排序损失：

$$
\ell(\Delta)=\operatorname{softplus}(-\Delta)
=\log(1+\exp(-\Delta)).
$$

先对每个样本的有效对取平均，再对批内样本取平均：

$$
\mathcal{L}_{\mathrm{rank},b}
=\frac{1}{B}\sum_{i=1}^{B}
\frac{1}{|\mathcal{P}_i|}
\sum_{(c,d)\in\mathcal{P}_i}
\operatorname{softplus}\!\left(-(s_{b,i,c}-s_{b,i,d})\right).
$$

该归约使每个样本具有相同权重。存在 ties 时，各样本的有效对数可能不同，因此不能用整个批次的有效对总数直接替代逐样本分母，也不能把被 mask 的位置当作零值后对完整张量直接 `mean()`。

实际设定为温度 1、无显式 margin、无温度平方补偿。教师只提供类别索引和严格大小关系，不用教师 logit 差作为软标签或样本对权重。

### 3.3 梯度的作用

单个有效对的梯度为

$$
\frac{\partial\ell}{\partial s_c}=-\sigma(-\Delta),
\qquad
\frac{\partial\ell}{\partial s_d}=\sigma(-\Delta),
$$

其中 $\sigma$ 是 sigmoid。梯度下降倾向于提高教师优先类别相对于较低类别的学生分数。顺序错误时作用较强；顺序已经正确但差距较小时，损失仍推动差距增大。因此它不是仅在错序时非零的损失，也不以精确匹配教师 margin 为目标。

排序损失对学生所有类别 logits 的共同平移不敏感，但不对共同缩放不敏感。它约束的是以教师 Top-5 为起点的部分顺序，并不恢复完整的类别排序。

成对张量的形状为 $B\times5\times C$，不构造 $B\times C\times C$ 张量。选出 Top-5 后，成对损失的计算量与临时存储量均为 $O(B\cdot5C)$。这不包含后缀网络的前向和反向成本；后者仍需每一步实际执行。

## 4. 原目标与逐块权重定标

### 4.1 完整优化目标

令 $\mathcal{L}_{\mathrm{rec},b}$ 表示原方法的块重构项，$\mathcal{R}_{\mathrm{round},b}$ 表示已包含原系数和调度规则的 AdaRound 正则。完整目标为

$$
\mathcal{L}_{b}
=\mathcal{L}_{\mathrm{rec},b}
+\mathcal{R}_{\mathrm{round},b}
+\lambda_b\mathcal{L}_{\mathrm{rank},b}.
$$

$\mathcal{L}_{\mathrm{rec},b}$ 是原实现实际使用的标量，包括其原有的归一化，不能随意换成普通 MSE。FIMA-Q 保留原 Fisher 重构项及更新规则；LS-ViT 保留原最小二乘 Hessian 重构项。分类头仍使用原 KL 重构项，不用 ranking 替换 KL。本文把原量化方法视为基础算法，ranking 是可附加于它的目标项。

### 4.2 在同一个块输出上比较梯度

不同块的输出维度、后缀深度和原损失尺度不同，因此不对所有块使用同一个手设权重。每块在第一次参数更新之前，使用四个按固定 seed 预定的批次进行定标。先按原流程准备该块的初始曲率信息和损失归一化锚点。

对第 $j$ 个探测批次，记当前块的**完整批输出张量**为 $Y_b^{(j)}$，计算

$$
g_{\mathrm{rec},b}^{(j)}
=\left\|\nabla_{Y_b^{(j)}}\mathcal{L}_{\mathrm{rec},b}^{(j)}\right\|_2,
\qquad
g_{\mathrm{rank},b}^{(j)}
=\left\|\nabla_{Y_b^{(j)}}\mathcal{L}_{\mathrm{rank},b}^{(j)}\right\|_2.
$$

这里的范数覆盖批维和所有特征维，相当于展平整个输出梯度后取欧氏范数。不是先计算各样本范数再平均，也不是量化参数梯度的范数。

固定目标比例 $\rho=0.1$，设置

$$
\lambda_b
=\rho\,
\frac{\sum_{j=1}^{4}g_{\mathrm{rec},b}^{(j)}}
     {\sum_{j=1}^{4}g_{\mathrm{rank},b}^{(j)}}.
$$

这是“范数之和的比值”，不是“四个比值的平均”。在这四次探测上，加权 ranking 输出梯度的范数之和等于原重构输出梯度范数之和的 10%。这不意味着两个损失数值之比为 10%，也不保证后续参数梯度或实际更新量保持该比例。

定标时排除舍入正则：需要比较的是两个数据目标对同一输出张量的作用，而不是将直接作用于舍入参数的正则混入其中。得到的 $\lambda_b$ 转为常数并在整个块的重构过程中冻结，后续曲率更新也不重新定标。

### 4.3 探测的状态约束

四个探测批次使用独立、确定性的采样随机源；探测区域保存并恢复 Python、NumPy、PyTorch CPU 与 CUDA 的 RNG 状态。探测不执行 optimizer step，不推进学习率、正则调度或损失计数器，也不更新重构项的归一化锚点。

第一步的正式重构批次应按原流程的时序预先取出并保留，原方法需要的首次曲率准备只执行一次。探测完成后，正式第一步复用该批次。原方法的后续曲率与归一化更新仍按原规则执行。

每个探测批次的两种输出梯度范数都必须有限且严格为正。遇到零范数或非有限值时，应检查梯度断链、无有效排序对、损失退化等原因，而不是在分母加 epsilon 后继续运行。

## 5. 梯度路径与可学习参数

Ranking 对量化参数的梯度经过两段雅可比：

$$
\nabla_{\theta_b}\mathcal{L}_{\mathrm{rank},b}
=
\left(\frac{\partial Y_b}{\partial\theta_b}\right)^\top
\left(\frac{\partial S_b}{\partial Y_b}\right)^\top
\nabla_{S_b}\mathcal{L}_{\mathrm{rank},b}.
$$

$Y_b$ 和 $S_b$ 分别表示整批块输出与学生 logits，上式按展平后的张量理解。后缀参数固定只表示不对它们求参数更新，并不消除后缀对输入的雅可比。因此：

- 教师 logits、重构目标和缓存的固定前缀输入应当 detach。
- 当前块输出与学生 logits 不能 detach；后缀不能放入 `torch.no_grad()`。
- 先冻结学生的所有参数，再只启用当前块的 alpha 与允许学习的激活尺度。FP 权重、bias、LayerNorm 参数及其他块的量化参数不更新。
- 定标只用输出梯度。参数梯度另作路径验证：在早期普通 Transformer block、中间 block 和 head 上，分别检查 alpha 组与激活尺度组的有限非零梯度。该检查不能替代定标公式，也不要求组内每个参数在每个批次都有非零梯度。

对于 DeiT-Tiny，重构单元依次为 patch embedding、12 个 Transformer blocks 和分类头，共 14 个。patch embedding 之后的后缀必须保留 class token、位置编码、dropout/预归一化、剩余 blocks、最终 norm 与模型真实的分类头路径；不能只把剩余 Transformer blocks 串起来。当前单元是分类头时，块输出已经是 logits，后缀是恒等映射。

## 6. PyTorch 风格伪代码

下面把 ranking 核心写为可执行的张量函数，把既有量化器、曲率估计及重构状态机保留为基础算法接口。它展示新增算法的完整控制关系，不另行实现 FIMA-Q 或 LS-ViT。

### 6.1 排序损失、随机状态保护与权重定标

`rec_only(y, batch)` 必须返回原方法的纯重构项，读取当前曲率与锚点，但不修改任何状态。`make_probe_batches()` 每次产生四个固定批次；批次包含 `inputs`、配对的 `teacher_logits`，以及 `rec_only` 所需的原重构目标和曲率辅助信息。两者均由原重构流程提供。

```python
import math
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F


def ranking_loss(student_logits, teacher_logits):
    # teacher/student 均为 [B, C]，并共享图片与增强索引。
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("Expected paired [B, C] logits")
    if student_logits.shape[1] < 5:
        raise ValueError("Teacher Top-5 requires at least five classes")

    teacher = teacher_logits.detach()
    top_values, top_ids = teacher.topk(5, dim=-1)
    student_top = student_logits.gather(1, top_ids)
    valid = top_values[:, :, None] > teacher[:, None, :]  # [B, 5, C]
    counts = valid.sum(dim=(1, 2))                       # [B]
    if (counts == 0).any().item():
        raise RuntimeError("A sample has no strict teacher ordering pair")

    delta = student_top[:, :, None] - student_logits[:, None, :]
    per_sample = (F.softplus(-delta) * valid).sum(dim=(1, 2)) / counts
    loss = per_sample.mean()
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite ranking loss")
    return loss


@contextmanager
def preserve_rng():
    py_state = random.getstate()
    np_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def calibrate_lambda(block, suffix, rec_only, make_probe_batches):
    # 调用前已准备曲率/锚点，且只启用当前块量化参数的梯度。
    observations = []
    with preserve_rng():
        for batch in make_probe_batches():
            y = block(batch["inputs"])
            rec = rec_only(y, batch)  # 不含舍入正则，不推进计数/锚点。
            rank = ranking_loss(suffix(y), batch["teacher_logits"])
            g_rec, = torch.autograd.grad(rec, y, retain_graph=True)
            g_rank, = torch.autograd.grad(rank, y)
            nr = g_rec.detach().float().square().sum().sqrt().item()
            nk = g_rank.detach().float().square().sum().sqrt().item()
            if not (math.isfinite(nr) and math.isfinite(nk) and nr > 0 and nk > 0):
                raise RuntimeError(f"Invalid output-gradient norms: rec={nr}, rank={nk}")
            observations.append((nr, nk))

    if len(observations) != 4:
        raise RuntimeError("Exactly four probe batches are required")
    lam = 0.1 * sum(x[0] for x in observations) / sum(x[1] for x in observations)
    if not math.isfinite(lam) or lam <= 0:
        raise RuntimeError(f"Invalid ranking weight: {lam}")
    return lam, observations  # Python float，不参与后续自动微分。
```

这里只向 `autograd.grad` 请求块输出梯度，不将探测梯度累积到参数的 `.grad` 中。`rec_only` 的只读语义与独立批次采样是接口要求，不能用有计数器副作用的原总损失直接代替。

### 6.2 DeiT 后缀

该函数对应本次 DeiT 实现；其他模型应按其真实 forward 定义后缀，并先验证数值等价。

```python
def deit_suffix(model, block_name, y):
    if block_name == "head":
        return y
    if block_name == "patch_embed":
        x = model._pos_embed(y)  # 包含模型自己的 class/position token 逻辑。
        x = model.patch_drop(x)
        x = model.norm_pre(x)
        start = 0
    elif block_name.startswith("blocks."):
        x = y
        start = int(block_name.split(".")[1]) + 1
    else:
        raise ValueError(f"Unsupported reconstruction unit: {block_name}")

    for j in range(start, len(model.blocks)):
        x = model.blocks[j](x)
    x = model.norm(x)
    return model.forward_head(x)  # 保留真实 pooling / norm / head 路径。
```

### 6.3 完整逐块流程

`base` 表示原量化方法的适配接口，其职责如下。接口名用于表达算法，不是要求仓库存在同名类。

| 接口 | 必须完成的工作 |
|---|---|
| `initialize_from_fp` | 用全精度模型和初始化集执行原量化初始化与校准。 |
| `units` | 按原顺序返回 14 个重构单元。 |
| `collect_paired_cache` | 对固定增强输入缓存当前块所需的 FP 输入/目标、量化前缀输入，以及按索引配对的教师 logits。 |
| `prepare_stage` | 设置当前块软量化和 QDrop；冻结其他参数；建立原优化器/调度器；按原时序取出首批、准备首次曲率和锚点。返回该块的重构状态。 |
| `stage.make_probe_batches` | 用独立固定随机源产生四个配对探测批次，保持正式采样序列。 |
| `stage.rec_only` | 使用已准备的曲率和锚点，计算不含舍入正则的只读重构项。 |
| `stage.begin_step` | 第零步复用保留的首批；其他步先按原规则采样，再按原规则更新曲率；不重复首次曲率准备。 |
| `stage.original_objective` | 每个正式步骤只调用一次，计算原重构项和原舍入正则，并正常推进原损失状态。 |
| `stage.step_optimizers_and_schedulers` | 仅更新当前块 alpha/激活尺度，并执行原调度。 |
| `finish_stage` / `export_hardened` | 按原方法完成该块的模式切换、后续缓存准备及最终硬舍入导出。 |

```python
def quantize_with_ranking(fp_model, init_images, optim_images, base, seed):
    # 两个输入集合都已按各自原训练预处理增强一次并缓存。
    student = base.initialize_from_fp(fp_model, init_images)

    for name, block in base.units(student):
        cache = base.collect_paired_cache(student, fp_model, name, optim_images)
        stage = base.prepare_stage(
            student, name, cache, seed=seed,
            steps=20000, quant_act=True, qdrop_probability=0.5,
        )
        suffix = lambda y: deit_suffix(student, name, y)
        lam, probes = calibrate_lambda(
            block, suffix, stage.rec_only, stage.make_probe_batches,
        )
        stage.record_lambda(lam, probes)

        for step in range(20000):
            batch = stage.begin_step(step)
            stage.zero_grad()
            y = block(batch["inputs"])
            original = stage.original_objective(y, batch)
            rank = ranking_loss(suffix(y), batch["teacher_logits"])
            loss = original + lam * rank
            loss.backward()
            stage.step_optimizers_and_schedulers()

        base.finish_stage(student, name, stage)

    return base.export_hardened(student)
```

`stage.zero_grad` 清除当前块优化器的梯度，`stage.record_lambda` 记录四组范数和固定权重；它们不改变算法目标。上述后缀调用始终保留 autograd。正式运行需显式使用指定 CUDA 设备和 FP32，不采用不可用时自动回退 CPU 的设备选择。

## 7. 本次配置与 HC3 的关系

| 项目 | 本次设置 |
|---|---|
| 模型与基础方法 | DeiT-Tiny；官方 FIMA-Q / LS-ViT。 |
| 量化设置 | W3A3、W2A3，保留原方法的各层位宽策略。 |
| 重构单元与步数 | 14 个单元，各 20,000 步。 |
| 初始化与优化图片 | 按各 seed 归档的 128 / 1024 张图片。 |
| 优化批大小与 QDrop | 32；概率 0.5。 |
| Ranking 配置 | 教师 Top-5、温度 1、无 margin；四批输出梯度定标，目标比例 0.1。 |
| FIMA-Q 特有配置 | rank / k 为 15，`dis_mode=q`，`p1=p2=1.0`。 |
| LS-ViT 特有配置 | `metric=lsvit`，保留原最小二乘 Hessian 规则。 |
| Seeds | 3407、1201、2024。 |
| 评测 | 最终硬化检查点；ImageNet 验证集 50,000 张，batch 32、workers 1、FP32 CUDA。 |

Ranking 量化结束后的模型记为 C。若继续进行 HC3-ORS-InvAR 校正，应在 **C 的新 logits** 上用 1024 张优化集图片重新拟合，再得到 D；不能复用针对旧量化模型拟合的校正系数。HC3 不参与上述 ranking 损失、梯度定标或块重构。

因此，四组比较的含义是：A 为历史原量化模型，B 为历史原量化模型经 HC3 校正的结果，C 为加入 ranking 后重新量化的模型，D 为 C 重新拟合 HC3 后的结果。历史完整增强张量不可用，A/B 与 C/D 不能宣称逐增强严格配对；新 ranking 运行内部的教师—学生增强配对是另一个独立要求，必须满足。

## 8. 实现核对与来源

复现时应核对以下行为，而不仅检查损失能否下降：

1. 向量化排序损失与按定义枚举有效对的结果及梯度相同，ties 的归约正确。
2. 同一全精度块输出送入后缀后，与原模型相应完整 forward 的 logits 一致。
3. Ranking 梯度能到达当前块的 alpha 与激活尺度；教师及非当前块参数保持不变。
4. 四次定标使用块输出梯度；不推进优化器、损失计数、锚点或正式 RNG 序列。
5. 关闭 ranking 的路径跳过额外探测，并在短程行为对照中退化为原重构流程。
6. 最终导出使用硬舍入，保存重载后评测行为一致。

本说明依据以下实际实现，而非重新设想的 ranking 变体：

- FIMA-Q worktree：`/Users/shawn/projects/2026/PTQ/FIMA-Q-ranking-20260918-193442`。
- LS-ViT worktree：`/Users/shawn/projects/2026/PTQ/LS-ViT-ranking-20260918-193442`。
- 两者的 `utils/ranking.py`：排序损失、后缀与输出梯度定标。
- 两者的 `utils/block_recon.py`：配对批次、曲率与锚点准备、参数冻结和逐块优化。
- 两者的 `ranking_experiment.py`：增强缓存、正式运行参数、硬化重载与评测。
- 实验记录位于各 worktree 的 `docs/experiments/logits-ranking/EXPERIMENT.md`。

这些路径用于核对实现来源；理解排序目标、权重定标与训练流程不依赖访问它们。
