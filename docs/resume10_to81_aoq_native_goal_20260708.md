# Swin-T 严格 W4A4 AOQ 原生 QAT 训练范式冲击 81

## 中文目标

在以下目录继续优化 Swin-T 严格 W4A4 QAT 的继续训练 / 微调范式：

```text
/mlx_devbox/users/quyanyi/playground/QATs
```

目标保持不变：

```text
在最多 20 个额外 epoch 内，
让任一严格 W4A4 单模型 checkpoint
在完整 ImageNet raw validation 上达到 Top-1 >= 81.0，
且 Samples=50000。
```

大的方法切换：

```text
不再继续围绕 OFQ-family 的局部 patch、小超参扫描、QKR/StatsQ/damping/anchor/freeze 做增量修补。
改为构建 AOQ 原生的严格 W4A4 Swin-T 训练范式。
```

新范式参考以下论文思想：

```text
Allowing Oscillation Quantization: Overcoming Solution Space Limitation in Low Bit-Width Quantization
```

核心假设：

```text
当前严格 W4A4 Swin-T 已经卡在一个很窄的量化局部 basin。
继续全程抑制 oscillation 会限制离散解空间探索。
新范式应该在前期允许或诱导受控的 weight-bin crossing，
扩大低 bit QAT 的离散解空间；
中后期再用 delayed dampening / bin-center stabilization 收敛。
```

## 中文实验日志要求

本 goal 的实验记录必须保留中文日志。具体要求：

- `resume10_to81_goal_progress_20260706.md` 中每个新实验条目必须用中文写清楚：
  - 实验动机
  - 方法设计
  - 关键命令
  - 关键 `args.yaml` 证据
  - 严格 resume / 严格 init 证据
  - full-val 结果
  - 是否超过当前 best
  - 失败或成功原因
  - 下一步判断
- 终端日志文件可以保留训练脚本原始英文输出，但进度文档里的人工总结、结论、实验决策必须是中文。
- 如果新增诊断 TSV/JSON，字段名可以用英文以便脚本处理，但必须在 progress doc 中提供中文解释。
- 如果新增 run script 或代码注释，允许英文变量名和参数名；但实验记录和 handoff 说明必须保留中文。
- 每次 full-val 结果必须用中文明确标注是否满足：
  - 严格 W4A4
  - 单 checkpoint
  - 完整 ImageNet raw validation
  - `Samples=50000`
  - 是否使用了 soup / averaging / ensemble

## 提交 goal 时使用的中文目标摘要

提交 goal 时只使用下面这段中文文本；后续实验日志也必须继续保留中文人工记录：

```text
完成 /mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_to81_aoq_native_goal_20260708.md 中定义的 AOQ 原生 Swin-T 严格 W4A4 QAT 训练范式目标。

核心目标是在最多 20 个额外 epoch 内，让任一严格 W4A4 单模型 checkpoint 在完整 ImageNet raw validation 上达到 Top-1 >= 81.0，Samples=50000。

方法上需要从已经耗尽的 OFQ-family 局部 patch 路线切换到 AOQ 原生范式：尽量丢弃 QKR、StatsQ 等 OFQ-specific innovation，围绕 weight-bin crossing 诊断、前期受控 oscillation / 离散解空间探索、中后期 delayed dampening / bin-center stabilization 设计新的严格 W4A4 Swin-T 训练流程。

必须在真实 GPU worker TTY 中训练和验证；不允许 soup、checkpoint averaging、multi-checkpoint averaging 或 ensemble。每个实验都必须把中文实验日志写入 /mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_to81_goal_progress_20260706.md，包含实验动机、方法设计、命令、关键 args、严格 resume/init 证据、full-val 结果、是否超过当前 best、失败或成功原因、结论和下一步判断。终端训练日志可以保留脚本原始英文输出，但进度文档里的人工总结、结论、实验决策必须是中文。

目标未达到前不要调用 update_goal complete。只有 completion audit 确认严格 W4A4、单 checkpoint、完整 ImageNet raw validation、Top-1 >= 81.0、Samples=50000、无 soup/averaging/ensemble，并且中文实验日志完整记录后，才算完成。
```

## 中文实现参考

以下内容是这个 goal 的中文实现参考。上面的中文目标和中文实验日志要求是验收时的最高优先级。

## 目标

在以下目录继续优化 Swin-T 严格 W4A4 QAT 的继续训练 / 微调范式：

```text
/mlx_devbox/users/quyanyi/playground/QATs
```

目标保持不变：

```text
在最多 20 个额外 epoch 内，
让任一严格 W4A4 单 checkpoint
在完整 ImageNet raw validation 上达到 Top-1 >= 81.0。
```

大的方法切换是离开 OFQ-family 的局部 patch 邻域，构建受下面论文启发的 AOQ-native 训练范式：

```text
Allowing Oscillation Quantization: Overcoming Solution Space Limitation in Low Bit-Width Quantization
```

工作假设：

```text
当前严格 W4A4 Swin-T 卡在较窄的量化局部 basin。
新范式不应从头到尾压制 oscillation，
而应允许前期受控 weight-bin 探索，再做延迟稳定化。
```

## 固定路径与基线

基线数据：

```text
/tmp/imagenet1k_full_parquet
```

中文进度日志：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_to81_goal_progress_20260706.md
```

fixed-QKR public-family epoch10 checkpoint：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
```

fixed start 的独立 full-val 结果：

```text
Top-1 80.3640
Samples 50000
```

当前 best strict W4A4 单 checkpoint：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540
Top-5 95.3060
Loss 0.8387
Samples 50000
```

## 硬约束

只有下面这种结果计入目标：

```text
strict W4A4
single checkpoint
full ImageNet raw validation
Samples=50000
```

以下方法不允许使用：

```text
soup
checkpoint averaging
multi-checkpoint averaging
ensemble
```

除非 strict W4A4 单 checkpoint 达到下面结果，否则不要调用 `update_goal complete`：

```text
Top-1 >= 81.0
Samples=50000
```

每个实验必须记录到：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_to81_goal_progress_20260706.md
```

每条中文实验记录必须包含：

- 实验动机
- 方法设计
- 命令
- 关键 `args.yaml` 证据
- strict resume 或 strict init 证据
- full-val 结果
- 是否超过当前 best
- 失败或成功原因
- 中文结论
- 下一步判断

## GPU 执行要求

分布式训练和 full validation 必须在真实 GPU worker TTY 中运行，不能在 Jupyter/master shell 中运行。

默认 worker 进入方式：

```bash
NO_COLOR=1 TERM=dumb mlx worker login
```

进入后先验证 GPU 可见性：

```bash
test -e /dev/nvidia0 && echo gpu-device-present
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

优先复用已有 worker；除非明确需要，不主动 launch 新 worker。

如果 shell 显示：

```text
NVIDIA_VISIBLE_DEVICES=none
torch.cuda.is_available()=False
no /dev/nvidia0
```

则它不是有效 GPU 训练 shell。不要把这种环境里的 NCCL/CUDA 失败解释为模型代码失败。

## 当前历史状态

resume10 目标附近已知本地分支：

- `checkpoint-10` 独立 full-val：Top-1 `80.3640`
- ordinary continuation / structure / qk relation / anchor-ref 分支：持平或更差
- low-LR full-param best：约 `80.4580`
- quant-only 分支：低收益
- no-feature auxiliary 分支：低于 best
- pre-QAT feature reconstruction 100 updates 得到过约 `80.5220` 的强 source checkpoint，但后续 epoch 回落
- Phase 2Z 是旧分支 current best：`80.5540`
- 从 Phase 2S 做 5-epoch continuous full late-block recipe 得到：

```text
80.5180 -> 80.4400 -> 80.5460 -> 80.4680 -> 80.4720
```

这条 5-epoch 曲线说明，Phase 2W/2Z 附近的局部 recipe 只是继续延长，并不会自然爬到 81。

## 为什么换范式

已有工作大多处在 OFQ-family 局部修补范式中：

- QKR
- StatsQ
- attention/ref KL
- direction anchors
- activation scale calibration
- dampening
- freeze or partial freeze
- local teacher feature-output auxiliaries
- selective bin regularization

这些机制给出了一个很窄的高点，但没有形成稳定通向 81 的路径。

AOQ 给出一个不同解释：

```text
low-bit QAT 可能因为 weights 很少跨过 quantization thresholds 而卡住。
如果 bin assignments 几乎不变，QAT 实际上只在初始量化解附近的小邻域搜索。
```

因此，从一开始就抑制所有 oscillation 可能是有害的。新的目标是测试更大的范式：

```text
前期允许受控 weight-bin exploration
后期再使用 delayed dampening / bin-center stabilization
```

## 要迁移的 AOQ 思想

AOQ 可以抽象为三个阶段。

### 阶段 1：诱导 oscillation

AOQ 在训练前期缩小 quantization thresholds 和 quantization levels 之间的间隔。

目的：

```text
增加 weight threshold crossing
扩大量化解空间探索
避免被锁定在初始 bin assignment 中
```

### 阶段 2：学习 quantization levels

AOQ 固定 threshold interval，同时学习 quantization level interval。

目的：

```text
让 quantized value representation 在探索后自适应
随着 level spacing 增大，自然减少 noisy oscillation
```

### 阶段 3：延迟 dampening oscillation

AOQ 只在训练后段施加 delayed oscillation dampening。

目的：

```text
稳定已经发现的 bin assignment
把 weights 从 threshold boundaries 附近推开
改善收敛
```

相对前面局部修补工作的关键变化：

```text
不要全程抑制 oscillation。
把 oscillation 作为前期探索机制使用，然后再稳定化。
```

## 主要新方向

不要继续做单纯标量超参扫描。

要构建并评估 AOQ-native strict W4A4 Swin-T 训练范式。

新范式要主动远离 OFQ-specific 机制：

- 不把 QKR 当作必需项
- 不把 StatsQ 当作必需项
- 不依赖 OFQ 的 oscillation-free 假设作为核心机制
- 不继续给 Phase 2W/2Z 加另一个局部 anchor、freeze 或 damping term

预期方向：

```text
clean Swin-T strict W4A4 QAT
AOQ-style weight-bin exploration
delayed stabilization
minimal OFQ-specific assumptions
```

fixed-QKR checkpoint 仍可以作为：

- weight source
- evaluation baseline
- compatibility bridge
- initialization artifact

但新方法要按“是否能在不依赖 QKR/StatsQ 作为核心解释的情况下工作”来评估。

## 必做工作流 A：AOQ-compatible bin-crossing 诊断

在启动大实验前，先建立能度量 AOQ-relevant behavior 的诊断。

诊断需要比较关键 checkpoints，例如：

- fixed start `checkpoint-10`
- Phase 2S source checkpoint
- Phase 2W source checkpoint
- Phase 2Z best checkpoint
- any new AOQ-native gate checkpoints

对选定 Swin-T modules 计算：

- quantized bin assignment
- threshold distance
- bin crossing count between checkpoints
- fraction of weights near threshold
- per-module quantizer scale / level drift
- relationship between bin crossing and validation / class-flip behavior where possible

优先 modules：

- `features.5.5`
- `features.7.1`
- attention q/k/v/proj
- MLP `fc1` and `fc2`
- high-drift modules previously identified by short-update diagnostics

诊断必须区分：

```text
有益探索：
  能保持或改善 validation/class-logit behavior 的 bin changes

有害漂移：
  与 class regressions、activation scale collapse、
  attention/logit degradation 或 validation loss increase 相关的 bin changes
```

输出保存到 `docs/` 下的 TSV/JSON，并在中文 progress doc 中总结解释。

## 必做工作流 B：clean Swin-T W4A4 No-QKR/No-StatsQ 分支

构建一个不依赖 OFQ-specific QKR/StatsQ machinery 的 clean strict W4A4 分支。

初始约束：

- disable QKR
- 避免把 StatsQ 当作 main weight quantizer
- 使用 LSQ-style 或 AOQ-compatible weight quantization
- 保持 strict W4A4
- 只做 single-checkpoint evaluation
- 只把 full ImageNet raw validation 作为有效指标

这个分支仍可使用：

- teacher KD
- teacher feature-output supervision
- fixed public pretrained/fixed checkpoint initialization
- existing data loader and distributed launcher machinery

但不能用 QKR/StatsQ 解释增益。

第一目标不是立刻到 81，而是证明更干净的 AOQ-native Swin-T strict W4A4 分支可以：

- 正确初始化
- 稳定训练
- 保存 checkpoint
- 跑通 `Samples=50000` 的 full validation
- 保持在 fixed start baseline 附近的合理范围内

## 必做工作流 C：AOQ-stage schedule gate

clean 分支跑通后，实现一个短程 AOQ-inspired schedule。

### 阶段 1：Exploration

对选定 weight quantizers 施加 threshold 或 scale narrowing。

候选机制：

- 缩小 selected weight quantizer scale 或 threshold interval
- 如果可行，解耦 threshold interval 和 quantization level interval
- 先只作用于 selected modules，而不是全模型

目的：

```text
增加 selected weight bin crossing
测试 controlled exploration 是否能逃离当前 local basin
```

### 阶段 2：Level / Scale Adaptation

允许 selected quantization level/scale parameters 和必要 weights 适应。

目的：

```text
让新的 bin assignments 变得有用，而不只是噪声
```

### 阶段 3：Delayed Stabilization

在 gate 后期施加较小的 delayed dampening 或 bin-center regularization。

候选机制：

- delayed weight bin-center loss
- delayed selective bin regularization
- delayed threshold-margin stabilization

目的：

```text
稳定探索得到的 solution
避免最后停在不稳定 threshold boundaries 附近
```

不要从一开始就启用强 damping，否则会违背 AOQ 假设。

## Gate 规则

使用短 gate；证据差时及时停止。

Gate 0：startup / smoke

- 必须在真实 GPU worker 中运行
- 必须写出 `args.yaml`
- 必须保存 checkpoint
- 必须用 `Samples=50000` 跑 full ImageNet raw validation

Gate 1：第一次 full-val

- 不能显著低于 fixed start `80.3640`
- 如果低于 `80.0` 且没有明确 recovery 机制，停止

Gate 2：前 1-2 个 full-val 点

- 如果低于 Phase 1F/2S 附近 `80.5220` 且没有上升趋势，停止
- 如果只在 `80.3-80.55` 附近震荡，不继续做标量扫描

Gate 3：current-best 对比

- 只有超过全历史本地 best `80.5540` 的 checkpoint 才进入更长扩展
- clean AOQ-native 分支内部也要同时比较当前 clean AOQ best `80.1660`

最终成功标准：

```text
Top-1 >= 81.0
Samples=50000
strict W4A4
single checkpoint
```

## 避免重复的实验

除非 AOQ 诊断给出实质新证据，不重复已经关闭的局部分支。

已关闭或低收益的实验族包括：

- pure quant-only polish from the same starting point
- longer continuation of the same Phase 2W/2Z recipe
- confidence-band KD variants
- local-ref KD variants
- class-protect KL variants
- broad hard damping
- move-v damping
- direct qkx direction anchor
- qkx ACT-MSE calibration
- fc2 ACT-MSE source calibration
- no-feature source
- attention-only source
- attention-output source
- simple update-count sweeps around Phase 2W/2Z

只有 AOQ bin-crossing 诊断显示新理由时，才重新访问这些轴。

## 建议的第一轮实现计划

### 第 1 步：编写 bin-crossing 诊断

在下面目录实现脚本：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/
```

建议名称：

```text
diagnose_resume10_aoq_bin_crossing_20260708.py
```

输入：

- checkpoint A
- checkpoint B
- model config / strict W4A4 settings
- selected module patterns

输出：

- module-level TSV
- parameter-level TSV/JSON
- summary JSON

最低指标：

- bin_changed_fraction
- near_threshold_fraction
- mean_abs_bin_delta
- scale_ratio
- high-risk module ranking

### 第 2 步：在历史 checkpoints 上跑诊断

比较：

```text
checkpoint-10 -> Phase 2S
Phase 2S -> Phase 2W
Phase 2W -> Phase 2Z
Phase 2Z -> Phase 2BR epoch3
```

用诊断结果选择 exploration modules。

### 第 3 步：构建 clean No-QKR/No-StatsQ gate

在下面目录创建短 run script：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/
```

建议名称：

```text
run_resume10_clean_lsq_noqkr_gate_20260708.sh
```

第一轮必须证明：

- no QKR
- no StatsQ
- strict W4A4
- full-val works
- checkpoint resume/init 正确

### 第 4 步：加入 AOQ-explore schedule

只有第 3 步跑通后，才加入：

- selected weight scale/threshold narrowing
- stage start/end controls
- delayed dampening controls
- logging for bin crossing and threshold proximity

建议脚本：

```text
run_resume10_aoq_native_stage_gate_20260708.sh
```

### 第 5 步：记录每个结果

每个实验必须追加到：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_to81_goal_progress_20260706.md
```

## 完成审计要求

宣布 goal 完成前，必须审计：

- 精确 checkpoint 路径
- `args.yaml`
- strict W4A4 证据
- no soup / no averaging / no ensemble
- full ImageNet raw validation 原始摘要行
- `Samples=50000`
- Top-1 >= 81.0
- 中文 progress doc 条目
- 是否为 single checkpoint

只有所有审计项都通过后，才能把 goal 标记为 complete。
