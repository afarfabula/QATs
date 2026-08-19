# Recipe1 ViT/Swin QAT literature survey and executable plan

Goal: design a 5-epoch, <=30 minute Swin-T W4A4 fake-quant/QAT recipe whose full ImageNet raw Top-1 reaches >=80.0. All candidate checkpoints default to `/tmp`; only best artifacts are copied to system disk.

## Baseline evidence from local run

Existing stable stage1 recipe: Swin-T W4A4 OFQ QAT from scratch, KD hard+soft, no augmentation, global batch 2048, lr=2e-4, min_lr=1e-5, weight_decay=0, bf16 AMP.

Observed full ImageNet validation:

| epoch | Top-1 | Top-5 |
|---:|---:|---:|
| 0 | 71.890 | 90.832 |
| 1 | 72.686 | 91.334 |
| 2 | 73.322 | 91.658 |
| 3 | 73.690 | 91.596 |
| 4 | 73.994 | 91.880 |
| 5 | 74.156 | 91.852 |
| 99 / ckpt100 | 78.476 | 94.342 |
| ~147-151 continued | ~78.65 | ~94.42 |

Baseline gap to the new gate: about +5.8 to +6.0 Top-1 at 5 epochs.

## Papers reviewed

| # | Work | Year | Model / target | Bit-width | Core method | Fit for W4A4 ViT QAT | Recipe1 transfer |
|---:|---|---:|---|---|---|---|---|
| 1 | GPLQ: A General, Practical, and Lightning QAT Method for Vision Transformers | 2025 | ViT/DeiT/Swin | W4A4 | Sequential quantization: activation-only QAT first, then weight PTQ. Shows direct W4A4 QAT is inferior to activation-first optimization. | Very high. Directly names Swin W4A4 and fast QAT. | Recipe1-A/B: activation/quantizer-first warmup, freeze most FP weights early, then full W4A4 fine-tune. |
| 2 | Q-ViT: Fully Differentiable Quantization for Vision Transformer | 2022 | ViT/DeiT | low-bit mixed | Learnable scales and bit-widths; differentiable quantization. | Medium-high. Learnable scale logic matches OFQ/LSQ-style trainable quantizers. | Emphasize quantizer parameter learning, higher quant LR, quant-only or head_norm_quant warmup. |
| 3 | Q-ViT: Accurate and Fully Quantized Low-bit Vision Transformer | 2022 | DeiT/Swin | W/A low-bit | Finds attention-map distortion as bottleneck; uses information rectification and distribution-guided distillation. | High for attention-heavy Swin QAT. | Recipe1-D: attention/QKV relation distillation or attention output/feature distillation from FP teacher. |
| 4 | PTQ4ViT | 2022 | ViT | PTQ low-bit | Twin uniform quantization for softmax/GELU activations; Hessian-guided scale selection. | Medium. PTQ method, but calibration insight is useful. | Increase setup-alpha/calibration batches; consider better activation calibration before short QAT. |
| 5 | FQ-ViT | 2022 | Fully quantized ViT | low-bit, incl. 4-bit softmax | Power-of-Two Factor for LayerNorm inputs and Log-Int-Softmax. Targets LN/softmax distribution issues. | Medium. OFQ does not implement PTF/LIS, but identifies LN/softmax activation sensitivity. | Train head/norm/quant parameters and add feature/output distillation around block outputs. |
| 6 | RepQ-ViT | 2023 | ViT/Swin | PTQ low-bit | Scale reparameterization for post-LN activation outliers; decouples calibration/inference quantizers. | High concept fit for Swin activation outliers. | Recipe1-A: PTQ-like strong calibration; Recipe1-B: train norm/quant and attention projections. |
| 7 | NoisyQuant | 2022/2023 | ViT | PTQ activation quant | Adds uniform noisy bias to reshape heavy-tailed activations for lower quantization error. | Medium. Implementation not present, but suggests activation distribution smoothing. | Potential code change later: activation noise/bias during setup-alpha or early QAT; not first candidate. |
| 8 | PSAQ-ViT / PSAQ-ViT V2 | 2022 | ViT data-free quant | PTQ/DFQ | Patch similarity aware calibration; V2 uses adaptive teacher-student data generation. | Medium. Uses ViT-specific patch/teacher signals. | Teacher-student losses and calibration data quality matter; use real ImageNet plus teacher KD/feature distill. |
| 9 | I&S-ViT | 2023/2026 journal | ViT | PTQ low-bit | Shift-uniform-log2 quantizer and three-stage smooth optimization. | Medium-high. Three-stage smooth optimization aligns with staged Recipe1. | Recipe1 staged optimization: calibration -> quant/head/norm warmup -> full W4A4 KD. |
| 10 | ADFQ-ViT | 2024 | ViT | PTQ low-bit | Per-patch outlier-aware quantizer for post-LN activations and non-uniform attention scores. | High concept fit, especially Swin windows/outliers. | Prioritize activation/LN outlier handling via norm/quant training and block-output distillation. |
| 11 | LRP-QViT | 2024 | ViT | W/A 4-bit and mixed | Layer-wise relevance to identify important layers; clipped channel-wise quant for post-LN activations. | Medium. We must stay W4A4, but layer importance motivates nonuniform loss/LR. | Weight early/mid/late layer feature losses differently; train head_norm_attn_quant if full update is too slow. |
| 12 | APHQ-ViT | 2025 | ViT | PTQ low-bit | Average perturbation Hessian reconstruction; focuses on post-GELU activation degradation. | Medium-high. Reconstruction objective is portable. | Recipe1-E: block-output reconstruction / teacher_feature_output norm_mse. |
| 13 | Mix-QViT | 2025 | ViT | PTQ/QAT mixed bit | Layer importance + quant sensitivity for mixed precision; clipped channel-wise post-LN quant. | Medium. Constraint here is W4A4, but sensitivity still informs where to focus. | Layer/head selective distillation and grouped LR for quant-sensitive modules. |
| 14 | MixA-Q | 2025 | window-based ViT/Swin | mixed activation precision, OFQ W4A4 baseline | Activation sparsity/mixed precision on Swin windows; starts from OFQ W4A4 baseline. | Conceptual only because goal requires W4A4. | Use window/activation sensitivity as motivation for activation-first and attention-output distillation. |

## Recipe1 design principles from literature

1. Direct W4A4 QAT is not enough for fast 5-epoch convergence. GPLQ and I&S-ViT argue for staged/sequential optimization.
2. Activation quantization, especially post-LN/GELU/softmax/attention, is the main ViT bottleneck. This supports setup-alpha calibration, activation-quant warmup, norm/quant trainable policy, and attention/block-output distillation.
3. Teacher guidance should be more structural than only logits: attention map, QKV relation, feature/block output reconstruction are recurring themes.
4. Quantizer/scales should adapt quickly. Short-horizon QAT should emphasize quantizer parameters early via quant-only or grouped LR.
5. Storage discipline: all candidates output to `/tmp/qat_recipe1_runs`; only raw best copied to system disk.

## Executable Recipe1 candidate matrix

All candidates are 5 epochs, 8xH100, global batch 2048, full validation every epoch, output under `/tmp/qat_recipe1_runs`, bf16 AMP, W4A4 fake quant enabled.

- Baseline-B0: current KD hard+soft no-aug recipe, 5 epochs, for a fresh wall-time/metric baseline.
- Recipe1-A GPLQ-inspired calibration/activation-first: larger setup-alpha calibration, quant/head/norm trainable warmup for epoch 0, then full train via update override if supported. If no update override works, use `head_norm_quant` for all 5 epochs as a safe first proxy.
- Recipe1-B Quantizer fast adaptation: full params, setup-alpha 16, quant_lr_multiplier 4 or 8, KD hard+soft. Prior C52 at late stage showed high quant LR can hurt, but early from-scratch may benefit.
- Recipe1-C Teacher block-output reconstruction: full params, KD hard+soft, normalized block-output loss on Swin stage ends (`features.1.1,features.3.1,features.5.5,features.7.1`) with weight sweep around 0.02.
- Recipe1-D Attention/QKV structural distillation: KD hard+soft qk/qkv or teacher_qk_rel/attention output loss; target attention distortion identified by Q-ViT.
- Recipe1-E Strong supervised augmentation/KD hybrid: introduce controlled label smoothing/mixup/cutmix/randaugment only if it improves 5-epoch raw Top-1; not the first candidate because current no-aug converges fast.

## Gate and audit

A candidate only satisfies the goal if raw full ImageNet Top-1 >=80.0 within 5 epochs and total wall time <=30 minutes. EMA or soup cannot substitute for raw checkpoint. If no candidate reaches 80, goal remains active and next iteration should implement missing mechanisms such as activation-only fake quant or outlier-aware post-LN quantizer.
