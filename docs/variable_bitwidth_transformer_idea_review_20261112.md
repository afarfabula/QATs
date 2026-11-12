# Variable-Bit-Width Transformer：可行性与相关工作审计

日期：2026-11-12

状态：初步文献审计，不是最终 novelty 证明。

## 1. 结论先行

这个 idea 可以做，但需要收紧表述。

如果方案只是“首尾高 bit、中间低 bit”，它已经被多个方向覆盖：

- 当前 QATs Swin-T/DeiT-T 主实验本身就是 patch embedding/head W8A8、
  Transformer blocks W4A4 或 W2A2。
- VVTQ 明确把 patch embedding 和 classifier head 固定为 8-bit，主体为 4-bit。
- Q-ViT 的可学习 bit-width 自动把 first/last layer 学到高精度。
- LRP-QViT、Mix-QViT、MPTQ-ViT、LampQ 等工作已经做 layer-wise 或
  module-wise mixed precision allocation。
- Q-PACE 已经在 QAT 训练中动态分配 per-layer precision。
- VBQ 已经把 precision 定义为固定 bit budget 下的可学习非均匀资源。

如果研究问题是下面这个完整组合，则仍有空间：

> 在固定平均 bit/bit-operation 预算和固定模型宽度下，把 depth-wise precision
> profile 作为一阶研究对象，系统比较 uniform、H-L-H、L-H-L、单调变化等形状，
> 研究 ViT/Swin QAT 的最优 precision shape，以及它是否随模型大小和深度呈现
> 可预测的 scaling law。

初步检索没有找到与这个组合完全一致的 ViT/Swin QAT 论文。但不能因此宣称
“没人做过”；更准确的结论是“组成部件已广泛存在，形状层面的系统研究可能是
空白”。

## 2. 参考论文

### 2.1 ICLR 2027 submission

`Variable-Width Transformers`，Zhaofeng Wu et al.：

- arXiv: <https://arxiv.org/abs/2606.18246>
- OpenReview: <https://openreview.net/forum?id=6pbC57phxh>
- 状态：ICLR 2027 Conference Submission，under review，尚未 accepted。

论文提出 `×`-shaped 的 `> <former`：early/late layers 较宽，middle layers
较窄，并用 parameter-free residual resizing 处理相邻层宽度变化。

论文的关键不仅是“中间窄”，还包括：

- block parameter 数近似随宽度平方增长，attention FLOPs/KV cache 近似随宽度
  线性增长。
- 在 parameter-matched 条件下，非均匀宽度可以降低 average layer width 和
  attention 成本。
- 在 200M-2B dense 和 3B/1B-active MoE 上比较统一宽度与变宽度模型。
- 消融显示 fixed residual + carry-forward 比 zero padding 或 learned projection
  更好。

因此它研究的是模型容量、残差表示和计算量之间的 depth-wise allocation。

## 3. “用 bit 代替 width”是否成立

作为类比成立，作为严格等价不成立。

### 3.1 可以借鉴的部分

两者都可以看成固定预算下沿 depth 分配资源：

```text
variable width: 每一层的 hidden dimension d_l 不同
variable bits:  每一层的 W/A precision b_l^W, b_l^A 不同
```

都可以产生 high-low-high、low-high-low、单调等 profile，也都需要回答：

- 哪些层更需要容量或精度？
- 这种需求是否与 depth 有稳定关系？
- profile 的收益是否随模型规模变化？

### 3.2 不能直接照搬的部分

width 和 bit-width 的成本结构不同：

```text
linear-layer parameter count ≈ K d_l^2
attention FLOPs / KV cache ≈ N^2 d_l
weight storage in bits ≈ parameters × b_l
bit-operation cost ≈ MACs × f(b_l^W, b_l^A)
```

所以 variable-width 在 parameter-matched 时会自然降低 attention 成本；而
在固定 average bit 的情况下，mixed precision 的存储/bit-op 总量基本是线性
再分配，不会仅凭 H-L-H 形状自动获得 variable-width 那种结构性节省。

同时，bit-width 改变的是 numeric resolution，不会让一部分 hidden dimensions
绕过某个 block，也不产生 variable-width 论文中的 residual carry-forward。
它影响的是量化误差传播和数值表示，不是 residual rank/bottleneck。

因此更准确的题目是：

> Depth-wise precision allocation as a capacity axis in quantized Transformers

或：

> Do hourglass-shaped precision profiles improve QAT under a fixed bit budget?

不建议直接写“用 bit width 完全替代 model width”。

## 4. 必须定义的预算

“首尾高、中间低”至少可能指三种不同设置：

1. patch embedding 和 classifier head 高 bit，所有 Transformer blocks 低 bit。
2. 前几个和最后几个 Transformer blocks 高 bit，中间 blocks 低 bit。
3. 每层内部只让部分模块高 bit，如 V/proj/MLP 高 bit，Q/K 低 bit。

这三种不能混称为同一个 H-L-H profile。

比较时至少要固定一种成本口径：

```text
B_weight = Σ_i N_i b_i^W
B_act    = Σ_i M_i b_i^A
B_op     = Σ_i MAC_i · cost(b_i^W, b_i^A)
```

其中 `N_i`/`M_i` 应按真实 tensor element 数加权。Swin 不同 stage 的 token 数和
模块大小不同，不能对 12 个 block 做简单算术平均。

一个简单例子：若 12 个同尺寸 blocks 中首尾两个用 8-bit、其他用 4-bit，则
平均 block precision 是：

```text
(2 × 8 + 10 × 4) / 12 = 4.667 bit
```

它不能直接和 uniform W4A4 比较“同预算”。若目标是 average 4-bit，需要把中间
层进一步降到约 3.2-bit，或者采用 3/4-bit 混合、不同 W/A bit、weight-only
mixed precision 等方式补偿预算。

## 5. 相关工作对照

| 工作 | 设置 | 与 idea 的关系 | 是否已覆盖 |
|---|---|---|---|
| Variable-Width Transformers, 2026 | LLM dense/MoE | 固定参数预算下的 depth-wise width shape | 提供研究范式，但不是 bit-width |
| VVTQ / Quantization Variation, 2023 | ViT/Swin QAT | patch embedding 和 head 固定 8-bit，主体 W4A4 | 首尾高精度已明确做过 |
| Q-ViT, 2022 | ViT QAT | scales 和 bit-widths 可学习；first/last 学到高 bit，head-wise bit allocation | 可学习非均匀 precision 已做过 |
| LRP-QViT, 2024 | ViT PTQ | LRP 层重要性做 layer-wise bit allocation；消融中前两个 block 加高 bit 有收益 | layer importance + high early blocks 已做过 |
| Mix-QViT, 2025 | ViT PTQ/QAT | LRP importance + quantization sensitivity，在 2-bit mixed QAT 上验证 | 已做 layer-wise mixed-precision QAT |
| LampQ, AAAI 2026 | ViT PTQ | type-aware Fisher sensitivity + ILP + iterative bit allocation | layer-wise 最优分配已做得很系统 |
| MPTQ-ViT, 2024 | ViT PTQ | Greedy MP layer-wise bit assignment | 已做 |
| QBitOpt, 2023 | CNN QAT | QAT 中在严格资源约束下重分配 bit-width | 已做 QAT 约束下的 bit allocation |
| Q-PACE, 2026 | LLM QAT | 训练中周期估计 per-layer curvature，动态重分配 precision | 与“QAT 中改层 bit”重叠最强之一 |
| VBQ, 2026 | LLM from-scratch | 固定 bit budget 下学习 per-group precision；明确“不均匀花 bit 优于均匀” | 在概念上与“bit-width as resource”重叠最强 |
| Scaling Laws for Precision, 2024 | LM training/inference | precision-aware scaling law，并讨论模型不同部分不同 precision | 已有 precision scaling，但未系统解决 depth-wise profile 形状 |
| MixFrag, 2026 | ViT PTQ | fragility-guided bit allocation under target bit budget | 已做固定 bit budget 的 ViT layer-wise MPQ |

文献链接见本文末尾。

## 6. 当前 QATs 仓库状态

当前项目已经有：

- patch embedding/head W8A8。
- 主 Transformer blocks W4A4 或 W2A2。
- QKR、StatsQ 和 LSQ fake-quant。
- `--progressive-bit-schedule`，可以按 epoch 从 W8A8 降到 W6A6/W4A4。
- 按 layer-name 选择训练参数、reconstruction 和量化诊断的基础设施。

当前项目已经新增首版 block-wise profile 支持，入口见
`docs/deit_tiny_depth_profile_qat_plan_20261112.md` 和
`tmp_scripts/run_deit_tiny_qat_depth_profile_hlh_20ep_4gpu_20261112.sh`。

当前项目还没有：

- fixed average bit/BOPs 下的 profile 搜索。
- H-L-H profile 与 uniform baseline 的系统对照。
- 不同模型大小/深度下的 precision profile scaling law。

首版实现把 `build_ofq_qconfigs` 扩展为按 Transformer block 顺序读取
`--depth-bit-profile`。旧的 `set_fake_quant_bits` 仍是全局 epoch-wise 切换，
并且首版禁止它与 depth profile 同时使用。per-layer precision 目前覆盖：

- QLinear `statsq_fn` / `lsqw_fn`
- `input_quant_fn`
- QKR 的 `qk_quant`、`v_quant`
- `quant_x_4_qkv`
- `quan_a_qkx_fn`
- `quan_a_v_fn`
- `quan_a_softmax_fn`

patch embedding/head 继续固定 W8A8。

## 7. 推荐的 novelty 边界

### 7.1 不建议作为主要贡献

- 仅提出 `[8, 4, ..., 4, 8]` 的手工 profile。
- 仅说明“敏感层用高 bit、不敏感层用低 bit”。
- 仅做 layer-wise mixed-precision search。

这些点已有直接或近邻工作，审稿人很容易用 Q-ViT、LampQ、Mix-QViT、Q-PACE、
VBQ 反驳。

### 7.2 可形成独立问题的方向

建议把主贡献放在以下任一组合：

1. **Precision shape 假设检验**：在固定 average bit 和 bit-op budget 下，
   系统比较 constant、monotonic、H-L-H、L-H-L 等 profile，确认 H-L-H 是否真的
   优于自动 sensitivity allocation。
2. **Shape scaling law**：最优 bottleneck position/depth ratio 如何随层数、宽度、
   数据量和 QAT token 数变化。
3. **Static vs dynamic profile**：Q-PACE 已做 LLM 动态分配；可研究 ViT 分类中
   profile 是否在训练早期锁定，以及 early/late high precision 的时序课程。
4. **Error propagation mechanism**：验证 early-layer quantization error 是否被
   middle self-heal、是否必须在 late layers 重新提高 precision。VBQ 在 LLM 中
   报告 deeper layers self-heal early error，而 Q-PACE 在另一设置中报告 later
   blocks generally more sensitive；这说明“高-低-高”不能先验假定，值得做机制研究。
5. **QKR-specific precision profile**：QKR 后 Q/K branch 的量化误差结构改变，
   可专门研究 joint-QK path 是否应使用与 V/proj 不同的 bit。这个方向与当前仓库
   的 QKR 现象直接相连，辨识度更高。

## 8. 最小可行实验

### Phase 0：layer sensitivity

使用已训练 checkpoint，对每层逐一切换 `4→3/2`、`4→8`，记录：

- Top-1/Top-5
- attention-output KL
- block feature KL/MSE
- logit KL
- 相对 FP teacher 的误差

不要只按 weight reconstruction error 排序。近期工作已指出 reconstruction error
不一定预测 end-to-end quantization sensitivity。

### Phase 1：固定预算 profile sweep

在同一 DeiT-Tiny 或 Swin-T checkpoint、同一最终 average bits 下比较：

```text
Uniform:  [4, 4, 4, 4, 4, 4]
∧:        [8, 6, 4, 4, 3, 3]
∨:        [3, 3, 4, 4, 6, 8]
H-L-H:    [8, 6, 3, 3, 6, 8]
L-H-L:    [3, 4, 8, 8, 4, 3]
Sensitivity allocation: 由 Phase 0 做 DP/knapsack
```

每项都要报告 realized bits、BOPs、峰值显存和吞吐，不能只报名义 bit。

### Phase 2：QAT validation

短跑只用于筛 profile；至少需要和仓库现有 20/100 epoch 设置对齐，避免把
1-2 epoch 噪声当成形状效应。主对照建议：

- 当前 QKR + uniform W4A4-family。
- 当前 patch/head W8 + blocks W4A4。
- H-L-H 且平均 bit 与 uniform 相同。
- sensitivity-optimized profile 且预算相同。

### Phase 3：scaling

在 DeiT-Tiny/DeiT-Small 和 Swin-T 上改变深度或宽度，拟合类似：

```text
Loss(profile, model_size, bit_budget)
```

检验最优 bottleneck position 是否近似 `r_l × L`，以及 H-L-H 的收益是否随模型
规模扩大。

## 9. 建议的项目定位

最稳妥的表述不是“我们把 variable-width Transformer 改成了 variable-bit
Transformer”，而是：

> 我们研究量化 Transformer 中 depth-wise precision shape 是否存在与
> variable-width architecture 类似的结构性效应；在严格固定 bit/BOP budget
> 下，比较人工 shape、sensitivity allocation 和动态 QAT allocation，并分析
> 首尾高精度是否来自误差传播或输出对齐，而不是简单假定。

如果后续实验表明 H-L-H 在相同预算下稳定优于 uniform 和自动 sensitivity
allocation，并且最优 shape 可跨模型规模迁移，那么这条线有机会形成独立贡献。
如果只是手工 8-4-8 比 4-4-4 好，这更像现有 mixed-precision 的工程复现，创新
强度不足。

## 10. 文献链接

- Variable-Width Transformers: <https://arxiv.org/abs/2606.18246>
- ICLR 2027 OpenReview: <https://openreview.net/forum?id=6pbC57phxh>
- Scaling Laws for Precision: <https://arxiv.org/abs/2411.04330>
- Q-ViT: <https://arxiv.org/abs/2201.07703>
- LRP-QViT: <https://arxiv.org/abs/2401.11243>
- MPTQ-ViT: <https://arxiv.org/abs/2401.14895>
- Mix-QViT: <https://arxiv.org/abs/2501.06357>
- LampQ: <https://arxiv.org/abs/2511.10004>
- Q-PACE: <https://arxiv.org/abs/2610.09183>
- VBQ: <https://arxiv.org/abs/2607.02893>
- MixFrag: <https://arxiv.org/abs/2607.28589>
- VVTQ / Quantization Variation: <https://arxiv.org/abs/2307.00331>
- QBitOpt: <https://arxiv.org/abs/2307.04535>
- GAMMA: <https://arxiv.org/abs/2605.18475>
