# QK reparameterization、fake-quant 次数与实际计算图

日期：2026-11-12

## 1. 结论

当前 QATs/OFQ 主链路确实使用了 query-key reparameterization（QKR），但不是
`qat_launch.py` 的全局默认行为，而是主流启动脚本显式打开的可选路径。

QKR 的主要作用不是减少量化次数，而是改变 Q/K 量化误差进入 attention score
的位置和方式。用无量化算术看，它只是矩阵乘法结合律；用 fake-quant 算术看，
它确实改变了实际计算图、梯度落点和中间张量。

三个结论需要分开表述：

1. 浮点代数：标准 attention 与 QKR 的 pre-softmax score 等价。
2. QAT fake-quant：二者一般不等价，因为量化函数不能穿过重新结合后的矩阵乘法。
3. 项目实现：当前训练与评测实际执行 QKR 图，未发现自动折回标准 Q/K 图的导出路径。

## 2. 当前项目是否启用

`qat_launch.py` 的运行时默认配置是 `qk_reparam=False`，见
[qat_launch.py](../qat_launch.py)。但是当前文档和主流实验脚本都显式启用
`qk_reparam_type=0`，例如：

- DeiT-Tiny W4A4 100 epoch 脚本。
- Swin-T W4A4 100/200 epoch 主脚本。
- `run_task.sh` 的 OFQ 分支。

因此应区分两种情况：

- 使用当前归档的主实验脚本：启用了 QKR。
- 直接调用 `qat_launch.py` 且不传 `--qk-reparam`：没有启用 QKR。

模块替换入口位于
[utils.py](../third_party/OFQ/src/quantization/modules/utils.py)，DeiT/Swin
分别映射到 `QAttention_qkreparam` 和 `QAttention_swin_qkreparam`。

## 3. 标准 attention 与 QKR

令一个 head 的输入为：

```text
X ∈ R^(N × C)
```

Q/K 投影权重为：

```text
Wq_h, Wk_h ∈ R^(d_h × C)
```

标准 attention 的 pre-softmax score 为：

```text
S_h = (X Wq_h^T)(X Wk_h^T)^T
```

对其重新结合：

```text
S_h = X (Wq_h^T Wk_h) X^T
    = X M_h X^T
M_h = Wq_h^T Wk_h
```

QKR 使用这个形式，先算 `M_h`，再算 `Z_h = M_h X^T`，最后得到
`S_h = X Z_h`。DeiT 实现见
[attention.py](../third_party/OFQ/src/quantization/modules/attention.py) 的
`QAttention_qkreparam.forward`；Swin 实现见
[swin_attention_and_mlp.py](../third_party/OFQ/src/quantization/modules/swin_attention_and_mlp.py)。

softmax 和 `attention_prob @ V` 没有做数学变换。被重新结合的是
`softmax` 之前的 QK score 计算。

## 4. Fake-quant 后为什么不等价

记 fake-quant 算子为 `Q(·)`。标准路径近似执行：

```text
Q̂ = Q(X Wq^T)
K̂ = Q(X Wk^T)
Ŝ_std = Q̂ K̂^T
```

QKR 路径执行：

```text
X̂ = Q(X)
M̂ = Q(Wq^T Wk)
Ẑ = Q(M̂ X̂^T)
Ŝ_qkr = X̂ Ẑ
```

`Q(·)` 包含 scale、clip、round 和 straight-through estimator，是非线性的，
也不与矩阵乘法交换。因此：

```text
Q(X Wq^T) Q(X Wk^T)^T
≠
Q(X) Q(Q(Wq^T Wk) Q(X)^T)
```

QKR 的目标是缓解 Wq/Wk 分别量化的误差通过乘积互相耦合所产生的权重振荡和
梯度误估，不是让两条量化路径数值完全一致。

## 5. 量化次数是否减少

不能笼统说 QKR 减少了量化次数。以一层 DeiT attention 的 fake-quant 调用为例：

| 位置 | 标准 attention | QKR |
|---|---:|---:|
| 输入 `X` | 1 | 1 |
| Q/K 激活 | Q、K 各一次 | 合并后的 joint-QK 激活一次 |
| V 激活 | 1 | 1 |
| QKV 权重 | fused QKV 一次 | QK composed weight 一次 + V 一次 |
| softmax 概率 | 1 | 1 |
| proj 输入/权重 | 2 | 2 |

局部看，Q/K 的两个独立激活量化变成 joint-QK 的一个量化；全局看，fused QKV
权重被拆成 QK 和 V，又多了一次 V 权重量化，因此总 quantizer 调用数基本不变。

更重要的是，调用次数不是计算成本的全部：

1. QKR 显式构造 per-head 的 `M_h = Wq_h^T Wk_h`。
2. `M_h` 的理论秩不超过 `d_h`，但当前代码仍实例化并量化完整的 `C × C` 矩阵。
3. 前向多了 `M_h X^T` 和 `X Z_h` 两次矩阵乘。
4. joint-QK 中间张量很大，仓库记录过 QKR 对 batch/memory 的敏感性。

因此 QKR 是一次计算图重排和量化误差重参数化，不是普遍意义上的量化次数削减。

## 6. 实际部署图的含义

在项目内部，训练构造和评测构造都按同一个 `qk_reparam` 参数选择模块：

- 训练入口在 `get_ofq_qat_model` 中替换 attention。
- OFQ 评测入口也重新按 `--qk_reparam` 构造模块。

当前仓库没有发现把训练后的 QKR 图自动折叠回标准 `Q_hat K_hat^T` 的导出或
fold 流程。因此：

- 在本仓库做训练、validation 或 eval 时，QKR 是实际执行的 simulation graph。
- 若外部部署工具改用原始 Q/K 线性层和标准 attention，它是另一张图，不能假设
  与 QKR checkpoint 数值一致。

## 7. 实现层面的两个注意点

### 7.1 Swin Q/K bias

Swin QKR 路径保存了原始 Q/K bias，并在 joint-QK score 上补了
`Q·b_k`、`K·b_q` 和 `b_q·b_k` 三个 bias correction 项。

### 7.2 DeiT Q/K bias

DeiT QKR wrapper 创建 `self.q` 和 `self.k` 时使用 `bias=False`，没有类似
Swin 的 bias correction，只复制了 V bias。如果原 DeiT attention 的 Q/K bias
非零，则即使不量化，这条包装路径也不严格等于标准 attention。使用 DeiT QKR
做严格等价性论证前，需要先修复或显式消融这一点。

## 8. 对后续实验的影响

任何比较 QKR 与标准 attention 的实验，都不应只标注“量化位宽相同”，还应
固定或明确以下因素：

- Q/K 激活 quantizer 的数量和位置。
- fused QKV 权重与拆分 Q/K/V 权重的量化粒度差异。
- QKR joint-QK 中间张量的 scale/clip。
- Q/K bias correction。
- 是否把 QKR 图直接用于部署，还是另做标准 Q/K 导出。

QKR 不是“同一个图少量化几次”，而是“换了一个同样经过 QAT、但量化误差结构
完全不同的 attention 图”。
