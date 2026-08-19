# Swin-T W4A4 public methods survey

Date: 2026-07-04

Scope: public paper/repo audit for ImageNet-1K ViT/Swin low-bit quantization methods relevant to the current QATs/OFQ Swin-T W4A4 goal. This document is not a local training result. It separates official reported metrics from local reproducibility.

## Current local anchor

The current local non-QKR Recipe1 line is healthy but too low:

| local run | raw full ImageNet Top-1 | note |
|---|---:|---|
| non-QKR epoch1 gate | 73.3100 | chain is functional |
| non-QKR epoch10 | 76.8180 | below target trajectory |
| W3/QSC-style epoch10 | 77.3420 | strongest local early result |
| W3/QSC-style epoch20 | 77.4120-77.5920 | plateau, not enough |
| QSS-v1 epoch100 | 78.4020 | long run failed to catch OFQ |
| strict stage1 epoch100 | 78.4760 | best no-QSS 100-epoch same-codepath baseline |

Conclusion: the current local path is not competitive with the official OFQ W4A4 Swin-T result. The next step should be public-baseline reproduction, not more ungrounded local tricks.

## Comparable public results

| priority | method | type | official source checked | model | bits | reported Top-1 | Top-5 | comparable to Swin-T W4A4 full-val? | runnable/checkpoint evidence | key mechanism |
|---:|---|---|---|---|---|---:|---:|---|---|---|
| 1 | VVTQ / Quantization Variation | QAT | `HuangOwen/Quantization-Variation` README, arXiv HTML, official log, code | Swin-T | W4A4 for transformer blocks; first/patch and head are 8-bit in code | 82.42 / 82.424 | 96.026 | mostly yes for paper-style W4A4, but not strict all-layer W4A4 | checkpoint links in README; log `log/Swin-T-W4A4.log`; entry `train_VVTQ.py`; model `quantization/Swin_quant.py` | multi-crop soft-label KD, module-dependent quantization, LSQ-style quantizers, q/k/v split, oscillation-aware bin regularization |
| 2 | OFQ | QAT + CGA fine-tune | `nbasyl/OFQ` README, local mirrored README, train/eval scripts | Swin-T | W4A4 | 81.88 | not in README table | yes | checkpoint link in README; eval script `eval_scripts/swin_t/w4a4.sh`; train script `train_scripts/swin_t/w4a4_swin_t.sh` | StatsQ weights, LSQ activations, KD, QK reparameterization, CGA |
| 2 | OFQ | QAT | `nbasyl/OFQ` README | DeiT-S | W4A4 | 81.10 | not in README table | model differs, but same QAT family | checkpoint link and eval script `eval_scripts/deit_s/w4a4.sh` | same OFQ stack |
| 2 | PTQ4ViT | PTQ | `hahnyuan/PTQ4ViT` README | Swin-T | W6A6 | 80.47 | not in README table | no, W6A6 not W4A4 | Google checkpoint links listed; `example/test_all.py` | twin uniform quantization for softmax/GELU and Hessian-guided calibration |
| 3 | FQ-ViT | PTQ / integer-oriented | `megvii-research/FQ-ViT` README | Swin-T | W8/A8/Attn4 | 80.04 | not in README table | no, weights/linear activations are 8-bit | `test_quant.py swin_tiny ... --quant --ptf --lis` | PTF LayerNorm quantization and Log-Int-Softmax |
| 4 | RepQ-ViT | PTQ | `zkkli/RepQ-ViT` README and `classification/README.md` | Swin-T | W4A4 | 72.31 | not in README table | yes setting, but accuracy is far below target | `classification/test_quant.py --model swin_tiny --w_bit 4 --a_bit 4` | scale reparameterization for post-LN/post-softmax activation distributions |
| 5 | APHQ-ViT | PTQ + reconstruction | `GoatWu/APHQ-ViT` README | Swin-S | W4A4 | 81.81 | not in README table | no, Swin-S not Swin-T; no Swin-T row in README | checkpoints on Google Drive/HuggingFace; `test_quant.py` | average perturbation Hessian reconstruction, MLP reconstruction |
| 6 | Q-ViT accurate fully quantized | QAT | `YanjingLi0202/Q-ViT` README, arXiv search result | DeiT-S / ViT-S | low-bit, repo table incomplete | DeiT-S 3-bit 79.1 in README; arXiv snippet claims ViT-S about 80.9 | README gives Top-5 94.3 for DeiT-S 3-bit | not direct Swin-T W4A4 | README only provides DeiT-T 4-bit and DeiT-S 2/3-bit checkpoints; no Swin-T W4A4 checkpoint table | information rectification / distribution-guided distillation |
| 7 | GPLQ | staged QAT + PTQ | `wujx2001/GPLQ` README and examples | Swin-T supported | W32A4 stage1 then W4A4 stage2 | not extracted from PDF in this environment | unknown | likely relevant but metric still needs table verification | official scripts `examples/run_stage1_qat.sh`, `examples/run_stage2_ptq.sh`; PCA assets for Swin-T | activation-first QAT with TCS/PCA feature mimic, then weight PTQ + compensation |
| 8 | I&S-ViT | PTQ | paper search result only | ViT/Swin | low-bit | not repo-verified | unknown | not yet usable | no official repo/config verified in this pass | staged smooth optimization and inclusive quantization |
| 9 | QDrop | PTQ | paper search result only | mostly CNN in accessible result | low-bit | not ViT/Swin repo-verified | unknown | not direct | no ViT/Swin official repo path verified | random quantization dropping for reconstruction |
| 10 | LSQ / LSQ+ and PACT family | generic QAT quantizers | VVTQ paper/code references; OFQ activation mode uses LSQ | generic CNN/ViT components | varies | no standalone Swin-T W4A4 public target verified here | unknown | no as standalone methods | LSQ appears as implementation component in OFQ/VVTQ; PACT not found as an official Swin-T W4A4 repo target in this pass | learned step-size / learned activation clipping baselines |

## Setting and migration matrix

| method | epoch/update detail found | batch detail found | augmentation / KD / teacher | external assets required | special quantization or operators | current QATs/OFQ migration assessment | risk |
|---|---|---|---|---|---|---|---|
| VVTQ / Quantization Variation | README command uses 150 epochs; log shows final epoch 148 validation and multiple final eval points | README command uses `--batch-size 512`; code divides by `num_crops` with default 4 | FKD multi-crop soft labels; soft-label CE; no one-hot target in main loss | full-precision Swin-T checkpoint; official quantized checkpoints; FKD soft-label files, recommended Marginal Smoothing Top-5 500-crop | LSQ-style quantizers; q/k/v split; first/last 8-bit; `BinReg` oscillation-aware regularizer | Do official-repo eval first. Porting into QATs/OFQ is medium-to-high effort because it needs FKD loader, VVTQ Swin module, first/last bit policy, and BinReg. | Best verified Top-1, but comparability depends on accepting first/last 8-bit and external soft labels. |
| OFQ | official script uses 300 epochs; local target can validate every <=10 epochs | per-GPU batch 64, world size 8, effective batch 512 | Swin-T teacher; hard+soft KD; OFQ augmentation/script policy | full-precision teacher/pretrained init; official quantized checkpoint links | StatsQ weights, LSQ activations, QKR, CGA | High. This is closest to QATs/OFQ; likely mainly launcher/script parity and QKR/CGA stability. | Local QKR chain previously failed one gate, so preflight/smoke is mandatory. |
| GPLQ | Stage 1 is 1 epoch W32A4 activation QAT; Stage 2 is eval/PTQ compensation | example defaults batch 16 for Stage 1, 64 for Stage 2 unless overridden | TCS/PCA feature mimicking with teacher model | FP32 Swin weights; PCA assets for Swin; Stage1 checkpoint | activation-first QAT then W4A4 weight PTQ + compensation | Medium. Reproduce in GPLQ repo first; porting into QATs/OFQ would require new staged training/eval flow. | Exact Swin-T W4A4 table value not yet extracted in this environment. |
| PTQ4ViT | PTQ calibration, minutes-scale; no W4A4 Swin-T target in README | calibration images 32/128 in README table | no QAT teacher path in README result table | saved quantized checkpoints listed on Google Drive | twin uniform quantization for softmax/GELU; Hessian-guided scale search | Low as a main path. Use only as activation calibration reference. | Public Swin-T 80+ number is W6A6, not W4A4. |
| FQ-ViT | PTQ/eval only in README | not a QAT update-budget match | no direct QAT teacher path | timm/pretrained models | PTF for LayerNorm, Log-Int-Softmax | Low as a main path. Could inspire LN/softmax handling. | Swin-T 80+ result is W8/A8/Attn4, not W4A4. |
| RepQ-ViT | PTQ/eval command only | not a QAT update-budget match | no QAT teacher path | timm/pretrained models | scale reparameterization for post-LN/post-softmax | Low as a main path because Swin-T W4A4 is 72.31. | Mechanism useful, result too low. |
| APHQ-ViT | PTQ reconstruction/calibration/optimization | val batch example 500 | reconstruction-oriented, no Swin-T result row | Google Drive/HuggingFace checkpoints; timm pretrained models | MLP reconstruction, average perturbation Hessian reconstruction | Low-to-medium as mechanism reference after VVTQ/OFQ. | No Swin-T W4A4 README row; Swin-S/B are not target model. |
| Q-ViT | README training commands use 300 epochs for DeiT variants | batch 512 for DeiT training commands | hard distillation, teacher models for DeiT | DeiT checkpoint links only for listed bits | information rectification / distribution-guided distillation in paper family | Low for Swin-T target until Swin-T W4A4 repo artifact is found. | README lacks Swin-T W4A4 checkpoint/eval row. |
| I&S-ViT | not repo-verified | not repo-verified | not repo-verified | repo path surfaced by search (`zysxmu/IaS-ViT`) but not audited here | SULQ and staged smooth optimization per search/paper snippet | Not ready. Needs official repo/config audit before any plan. | Insufficient official-source evidence in this pass. |
| QDrop | not ViT/Swin repo-verified | not repo-verified | PTQ reconstruction concept | not repo-verified for target | random activation quantization dropping | Not a target path. | No official Swin-T W4A4 path verified. |
| LSQ / PACT | generic quantizer families | varies | varies | no standalone target artifact verified | learned step-size / learned clipping | Already embedded as components in OFQ/VVTQ; do not pursue as standalone Swin-T goal. | No standalone official Swin-T W4A4 >=80 path verified. |

## Source notes

### VVTQ / Quantization Variation

Official paper/code:

- arXiv: `https://arxiv.org/pdf/2307.00331`
- repo: `https://github.com/HuangOwen/Quantization-Variation`
- older ar5iv page links `https://github.com/HuangOwen/VVTQ`; the current arXiv page and active repo use `HuangOwen/Quantization-Variation`.

The official README table reports:

| model | bits | Top-1 | weights | logs |
|---|---|---:|---|---|
| Swin-T | 32/32 | 81.0 | SharePoint link | none |
| Swin-T | 4/4 | 82.42 | SharePoint link | `log/Swin-T-W4A4.log` |
| Swin-T | 3/3 | 81.37 | SharePoint link | `log/Swin-T-W3A3.log` |
| Swin-T | 2/2 | 77.66 | SharePoint link | `log/Swin-T-W2A2.log` |

The official `log/Swin-T-W4A4.log` ends with:

```text
INFO:root: * Acc@1 82.424 Acc@5 96.026
```

Key code evidence:

- entry: `train_VVTQ.py`
- Swin model: `quantization/Swin_quant.py`
- quantizer base: `quantization/lsq_layer.py`, `quantization/_quan_base.py`
- regularizer: `util_loss.py`
- initialization helper: `engine.py`
- soft labels and multi-crop data: `utils_FKD.py`

The official README W4A4 command shown for DeiT-T uses:

- `--epochs 150`
- `--batch-size 512`
- `--lr 5e-4`
- `--warmup-epochs 0`
- `--min-lr 0`
- `--wbits 4 --abits 4`
- `--reg`
- `--softlabel_path ./FKD_soft_label_500_crops_marginal_smoothing_k_5`
- `--finetune [path to full precision baseline model]`

Implementation details that matter for our setting:

1. It uses precomputed FKD soft labels, not only online teacher KD. This is a large external dependency and is likely part of the speed/stability gain.
2. It uses multi-crop training. In `train_VVTQ.py`, `args.batch_size` is divided by `num_crops`, and the default `num_crops` is 4.
3. It initializes quantization scales with `initialize_quantization(..., sample_iters=1)`.
4. It trains with soft-label cross entropy, not the standard supervised CE objective.
5. The `--reg` flag enables oscillation-aware bin regularization. `BinReg` penalizes within-bin FP weight variance around detached quantized weights and is annealed with `CosineTempDecay(t_max=args.epochs, temp_range=(0, 0.01), rel_decay_start=0.25)`.
6. `Swin_quant.py` splits Swin attention into separate `proj_q`, `proj_k`, and `proj_v`, quantizes q/k/v activations, attention maps, and attention output.
7. The first patch embedding and final classification head are fixed to 8-bit in code (`PatchEmbed` quant/proj and `self.head` use `nbits=8`). Therefore, this is paper-style W4A4 for the main transformer blocks, not strict all-layer W4A4.
8. The paper explicitly says the first patch embedding and last classification layers are fixed to 8-bit because they are more sensitive. This must be called out in any comparison with a stricter local setting.

Interpretation: VVTQ is now the highest verified public Swin-T result in this audit. It is not just a metric source; it gives a concrete alternative explanation for our current plateau: our local Recipe1 lacks multi-crop soft-label KD, module-dependent bit allocation, and explicit oscillation-aware bin regularization. It also suggests that a strict all-layer W4A4 constraint may be harder than common paper settings.

### OFQ

Official repo: `https://github.com/nbasyl/OFQ`.

The official README table reports:

| model | bits | Top-1 | eval path |
|---|---|---:|---|
| OFQ Swin-T | 2-2 | 78.52 | `eval_scripts/swin_t/w2a2.sh` |
| OFQ Swin-T | 3-3 | 81.09 | `eval_scripts/swin_t/w3a3.sh` |
| OFQ Swin-T | 4-4 | 81.88 | `eval_scripts/swin_t/w4a4.sh` |
| OFQ DeiT-S | 4-4 | 81.10 | `eval_scripts/deit_s/w4a4.sh` |

The official Swin-T W4A4 train script uses:

- `--epochs 300`
- per-GPU `--batch-size 64` with `--world_size 8`, effective batch 512
- `--aq-mode lsq`, `--aq-bitw 4`, `--wq-mode statsq`, `--wq-bitw 4`
- `--use-kd --teacher swin_t --kd_hard_and_soft 1`
- `--qk_reparam --qk_reparam_type 0`
- CGA fine-tune path with `cga.py`, `--qk_reparam_type 1`, `--boundaryRange 0.005`, `--freeze_for_n_epochs 30`

This is the only fully comparable official source in this audit that directly reports Swin-T W4A4 above 80. It should be treated as the reproduction baseline.

### PTQ4ViT

Official repo: `https://github.com/hahnyuan/PTQ4ViT`.

The README reports Swin-T/224:

- original 81.39
- W8A8 81.246
- W6A6 80.47

It does not provide a Swin-T W4A4 target in the README. It is not a direct solution for this goal, but its softmax/GELU twin quantization and Hessian-guided calibration are relevant to activation calibration.

### FQ-ViT

Official repo: `https://github.com/megvii-research/FQ-ViT`.

The README reports Swin-T:

- full precision 81.35
- Ours W8/A8/Attn8 80.51
- Ours W8/A8/Attn4 80.04

This is not W4A4. It is useful as evidence that LayerNorm/Softmax handling matters, not as a target result.

### RepQ-ViT

Official repo: `https://github.com/zkkli/RepQ-ViT`.

The classification README reports:

- Swin-T W4/A4 72.31
- Swin-T W6/A6 80.69
- Swin-S W4/A4 79.45

This is a direct warning: PTQ scale reparameterization alone is not enough for Swin-T W4A4 80. It is a mechanism reference for activation outliers, not a main reproduction target.

### APHQ-ViT

Official repo: `https://github.com/GoatWu/APHQ-ViT`.

The README reports:

- Swin-S W4/A4 81.81
- Swin-B W4/A4 83.42
- no Swin-T row in the README result table

This is not directly comparable to the target. Its reconstruction machinery is worth studying if OFQ reproduction still falls short, especially for MLP/post-GELU reconstruction, but it should not be claimed as a Swin-T W4A4 result.

### Q-ViT

Official repo checked: `https://github.com/YanjingLi0202/Q-ViT`.

The README provides runnable training commands for DeiT-T/DeiT-S and checkpoint table entries for DeiT-T 4-bit and DeiT-S 2/3-bit. It does not provide a Swin-T W4A4 checkpoint/eval row. The arXiv search result mentions about 80.9 Top-1 for ViT-S, but that is not a Swin-T W4A4 reproduction path.

### GPLQ

Official repo: `https://github.com/wujx2001/GPLQ`.

The README and scripts show an implementation path that is highly relevant:

- Stage 1: `tools/train_stage1.py` via `examples/run_stage1_qat.sh`
- Stage 1 uses W32A4, one epoch, TCS/PCA feature mimicking, `--wbits -1 --abits 4`
- Stage 2: `tools/evaluate_stage2.py` via `examples/run_stage2_ptq.sh`
- Stage 2 applies W4A4 with weight quantization and compensation
- Swin-T is directly supported as `swin_tiny_patch4_window7_224`
- PCA parameters for Swin-T are referenced by script

The PDF was downloaded locally, but this environment lacks `pdftotext`, so exact GPLQ paper table values were not extracted in this pass. Do not quote a GPLQ Swin-T W4A4 number until the table is verified from a readable source.

## Why current local results are low

The strongest public Swin-T W4A4-family result verified here is VVTQ Swin-T W4A4 Top-1 82.424 / Top-5 96.026. The strongest previously audited same-OFQ-family baseline is OFQ Swin-T W4A4 81.88. The current local Recipe1 variants are around 77-78 because they miss or underuse the public baselines' main conditions:

1. VVTQ uses FKD multi-crop soft labels and soft-label CE as the main objective. Local Recipe1 uses online teacher/KD but not the same multi-crop precomputed soft-label supervision.
2. VVTQ uses module-dependent quantization and keeps patch embedding / final head at 8-bit. A strict local all-layer W4A4 setting is harder and should not be compared without this caveat.
3. VVTQ explicitly regularizes weight-bin oscillation; OFQ uses StatsQ/QKR/CGA for oscillation. QSS-v1 only regularizes quant/shift state drift and did not fix long-horizon accuracy.
4. The official OFQ Swin-T W4A4 path uses QK reparameterization and CGA. The current accepted local non-QKR path cannot address Q/K intertwined attention-logit oscillation.
5. The official OFQ script uses effective batch 512, giving about 4x more optimizer updates per data epoch than the local large-batch 2048 Recipe1 setting.
6. Public successful paths emphasize structural calibration or staged quantization: VVTQ uses multi-crop KD plus variation-aware regularization, OFQ uses StatsQ/QKR/CGA, GPLQ uses activation-first QAT then weight PTQ/compensation, and PTQ4ViT/FQ-ViT/RepQ all point at post-LN/softmax/GELU activation distributions.

## Priority recommendation

1. Reproduce/evaluate VVTQ Swin-T W4A4 first as the highest verified public result, but report the 8-bit first/last-layer caveat. This is the best public target for "how did others get above 80".
2. Reproduce OFQ Swin-T W4A4 as the closest method family to our current QATs/OFQ code. It remains the best direct OFQ-family baseline with QKR/CGA.
3. Audit GPLQ as a staged alternative after VVTQ/OFQ are understood. It is promising because it directly attacks short-horizon convergence by avoiding direct W4A4 full-model QAT at the beginning.
4. Use PTQ4ViT/FQ-ViT/RepQ/APHQ/LSQ/PACT only as mechanism references unless their code is ported into a clearly gated local experiment.
