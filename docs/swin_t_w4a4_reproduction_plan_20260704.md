# Swin-T W4A4 public-baseline reproduction goal

Date: 2026-07-04

Status: proposed goal for user review. Do not launch training until this plan is approved.

## Goal

Reproduce or tightly align the best public Swin-T W4A4-family ImageNet-1K result before attempting new local algorithm ideas.

The highest verified public target is:

| method | model | bits | source | reported Top-1 | reproduction evidence |
|---|---|---|---|---:|---|
| VVTQ / Quantization Variation | Swin-T | W4A4 for main transformer blocks; first/last layers are 8-bit in code | official README + official log | 82.424 | checkpoint link, `log/Swin-T-W4A4.log`, `train_VVTQ.py`, `quantization/Swin_quant.py` |
| OFQ | Swin-T | W4A4 | official OFQ README | 81.88 | checkpoint link, `eval_scripts/swin_t/w4a4.sh`, `train_scripts/swin_t/w4a4_swin_t.sh` |

Success for this goal means a local checkpoint evaluated on full ImageNet-1K raw validation reaches Top-1 >=80.0 by <=50 data epochs, with no EMA/soup/proxy substitution.

## Non-goals

- Do not claim public paper numbers as local success.
- Do not run more QSS-v1/refmodel/custom augmentation experiments before public-baseline reproduction is reviewed.
- Do not use calibration subset, partial validation, training loss, or checkpoint README numbers as success.
- Do not treat PTQ4ViT W6A6, FQ-ViT W8/A8/Attn4, APHQ Swin-S W4A4, or Q-ViT DeiT/ViT results as Swin-T W4A4 success.
- Do not hide VVTQ's first/last 8-bit caveat when comparing against strict all-layer W4A4 local results.

## Why this goal is necessary

The current local line is too far below the official public baseline:

| run | Top-1 | interpretation |
|---|---:|---|
| local non-QKR epoch10 | 76.8180 | healthy but weak |
| local W3/QSC epoch10 | 77.3420 | best early local result |
| local W3/QSC epoch20 | 77.4120-77.5920 | plateau |
| local QSS-v1 epoch100 | 78.4020 | failed long-run path |
| local strict stage1 epoch100 | 78.4760 | best same-codepath no-QSS baseline |
| official OFQ Swin-T W4A4 | 81.88 | closest OFQ-family public baseline |
| official VVTQ Swin-T W4A4-family | 82.424 | highest verified public result; first/last layers 8-bit |

The gap is not a small regularization issue. The public paths differ in structural components, supervision, bit allocation, and update budget:

- VVTQ uses FKD multi-crop soft labels, soft-label CE, module-dependent quantization, q/k/v split, LSQ-style quantizers, and oscillation-aware bin regularization.
- VVTQ keeps patch embedding and final classification head at 8-bit; this likely matters for its high score.
- OFQ enables QK reparameterization and CGA.
- OFQ effective batch is 512 in the official 8-GPU script, not the local large-batch 2048 recipe.
- OFQ uses StatsQ weights, LSQ activations, KD, QKR, and CGA together.

## Phase 0: preflight only

No long training in this phase.

Checklist:

- Confirm local data path maps to full ImageNet train/val.
- Confirm public-repo reproduction path chosen by the user: VVTQ first, OFQ first, or GPLQ first.
- Confirm full-val parser extracts `Acc@1`, `Acc@5`, loss, and samples from logs for the chosen repo.
- For VVTQ: confirm FKD soft-label availability or a plan to download/use the official soft labels.
- For VVTQ: confirm whether comparison target permits first/last 8-bit as in the paper.
- For OFQ: confirm Swin-T W4A4 train/eval flags are preserved in `qat_launch.py` or a direct OFQ script.
- For OFQ: confirm QKR smoke still starts and exits cleanly.
- Confirm outputs keep checkpoint every 10 epochs, not only `last.pth.tar`.

Exit criteria:

- For VVTQ: a no-train model/eval wiring check can import/build `swin_tiny_patch4_window7_224_quant` and load or identify required checkpoint/soft-label files.
- For OFQ: a one-update QKR construction/training smoke completes.
- No claim about accuracy is made from this phase.

## Candidate A: VVTQ public-top reproduction

Purpose: reproduce the highest verified public Swin-T W4A4-family result and understand which components explain the 82.424 Top-1.

Why this candidate:

- It is the best verified public result found so far: official log reports `Acc@1 82.424 Acc@5 96.026`.
- It directly targets variation sources that match our failure modes: module sensitivity, activation/weight outliers, and dynamic oscillation.
- It gives concrete code for q/k/v split quantization, multi-crop soft-label KD, and oscillation-aware bin regularization.

Minimum implementation path:

1. First reproduce in the official VVTQ repo or a local vendor copy, not inside QATs/OFQ.
2. Use official entry `train_VVTQ.py` and official model `quantization/Swin_quant.py`.
3. Use official Swin-T W4A4 checkpoint/eval first if checkpoint can be downloaded.
4. If training is approved later, use the official W4A4 recipe with FKD soft labels.

Files to inspect or vendor before any run:

- `train_VVTQ.py`
- `quantization/Swin_quant.py`
- `quantization/lsq_layer.py`
- `quantization/_quan_base.py`
- `util_loss.py`
- `engine.py`
- `utils_FKD.py`
- `log/Swin-T-W4A4.log`

Configuration:

- model: Swin-T
- dataset: ImageNet-1K full train/val
- bits: W4A4 for main transformer blocks; patch embedding and classification head use 8-bit in official code
- quantizer: LSQ-style learned step size
- attention: q/k/v split with quantized q/k/v activations, attention map, and attention output
- supervision: FKD multi-crop soft labels
- loss: soft-label CE plus optional `--reg`
- regularizer: oscillation-aware bin regularization from `BinReg`
- official epochs: 150
- official log evidence: final `Acc@1 82.424 Acc@5 96.026`

Gate policy:

| checkpoint | required action |
|---|---|
| eval official ckpt | must reproduce or explain discrepancy from 82.424 full-val |
| epoch1 if training | full-val Top-1 >72 before continuing |
| every 10 epochs | compare to local non-QKR 76.818 and W3/QSC 77.342 |
| epoch20 | if below 78.5 and trend is flat, stop and diagnose |
| epoch50 | success for our goal if raw Top-1 >=80.0 |

Diagnostics if low:

- verify FKD soft-label path and `soft_label_type`;
- verify `num_crops` effective batch logic;
- verify patch/head are 8-bit as in official code, not accidentally forced to 4-bit;
- verify `--reg` is active and `BinReg` contributes nonzero loss after annealing starts;
- verify quantization scale initialization ran;
- verify official checkpoint's first/last layer bit policy before comparing to strict W4A4.

Expected bottlenecks solved:

- quantization initialization: `initialize_quantization`;
- attention/logit drift: q/k/v split and attention quantization;
- activation outlier: module-dependent quantization and first/last 8-bit exception;
- quantizer state instability: LSQ-style scales and BinReg;
- insufficient update budget: 150-epoch official recipe, but we will only advance in <=10-epoch local chunks;
- augmentation/KD mismatch: FKD multi-crop soft-label supervision.

## Candidate B: official OFQ 50-epoch reproduction

Purpose: recover the strongest same-code-family public baseline before new OFQ-side algorithm work.

Why this candidate:

- It is closest to the current QATs/OFQ code and already reports Swin-T W4A4 81.88.
- It directly tests whether our local failure is mostly missing/unstable QKR+CGA/update-budget alignment.

Minimum implementation path:

1. Prefer a direct OFQ-script reproduction first, or `qat_launch.py` only after confirming flag parity.
2. Reuse official `train_scripts/swin_t/w4a4_swin_t.sh` and `eval_scripts/swin_t/w4a4.sh` as the source of truth.
3. If adapting to QATs, only change path/data/output plumbing, not algorithm flags.

Files to inspect or adapt:

- `third_party/OFQ/train_scripts/swin_t/w4a4_swin_t.sh`
- `third_party/OFQ/eval_scripts/swin_t/w4a4.sh`
- `third_party/OFQ/train.py`
- `third_party/OFQ/cga.py`
- `third_party/OFQ/src/quantization/modules/utils.py`
- `qat_launch.py` only if using the unified launcher

Configuration:

- model: Swin-T
- dataset: ImageNet-1K full train/val
- bits: W4A4
- weight quantizer: StatsQ, per-channel
- activation quantizer: LSQ, per-channel, learnable clip
- teacher: Swin-T FP teacher
- KD: hard + soft
- QKR: enabled, `qk_reparam_type=0`
- batch/update budget: match official OFQ as closely as possible, effective global batch 512
- validation: full raw ImageNet every 10 epochs
- checkpoints: every 10 epochs, keep all validation checkpoints

Gate policy:

| checkpoint | required action |
|---|---|
| epoch1 | must be full-val Top-1 >72 before continuing |
| epoch10 | compare to local non-QKR 76.818 and W3/QSC 77.342 |
| epoch20 | if below 78.5 and trend is flat, stop and diagnose |
| epoch30 | if below 79.2, do not spend more compute without root-cause review |
| epoch40 | continue only if trajectory can plausibly reach 80 |
| epoch50 | success if raw Top-1 >=80.0 |

Diagnostics if low:

- verify QKR flags reached OFQ model construction, not only CLI;
- compare update count to official script;
- verify teacher/KD path is active;
- verify augmentation and optimizer settings match the official script;
- verify checkpoint resume/load order does not overwrite quantizer state;
- inspect whether CGA is required earlier than planned.

Expected bottlenecks solved:

- attention/logit drift: QKR;
- dynamic oscillation: StatsQ and CGA;
- quantizer state instability: OFQ quantizer setup and CGA freeze;
- insufficient update budget: official effective batch 512 and 10-epoch checkpoint cadence;
- augmentation/KD mismatch: official KD/augmentation policy instead of local no-aug large batch.

## Candidate B2: OFQ CGA alignment

Purpose: reproduce the full official OFQ stack if Candidate B gets close but does not cross 80.

Configuration:

- resume from best Candidate B checkpoint;
- run OFQ CGA path with `qk_reparam_type=1`;
- use `boundaryRange=0.005`;
- use `freeze_for_n_epochs=30` as official script reference;
- evaluate every 5-10 epochs.

Gate:

- Run only if Candidate B reaches a plausible neighborhood, for example >=79 by epoch30/40 or has a clear positive trend.
- Stop if CGA causes two consecutive full-val drops or fails to improve within 10 epochs.

## Candidate C: GPLQ staged reproduction candidate

Purpose: test a public staged alternative, not a local trick.

Rationale:

GPLQ's official repo uses:

- Stage 1 W32A4 activation QAT for 1 epoch;
- TCS/PCA feature mimicking;
- Stage 2 W4A4 weight PTQ plus compensation;
- direct Swin-T support via `swin_tiny_patch4_window7_224`;
- scripts `examples/run_stage1_qat.sh` and `examples/run_stage2_ptq.sh`.

This is relevant because the local direct W4A4 QAT path is slow and plateaus. GPLQ attacks the likely bottleneck, activation quantization, before weight quantization.

Minimum implementation path:

1. Reproduce GPLQ in its official repo first.
2. Run Stage 1 W32A4 activation QAT using `examples/run_stage1_qat.sh`.
3. Run Stage 2 W4A4 weight PTQ/compensation using `examples/run_stage2_ptq.sh`.
4. Only after reproducing, decide whether to port the mechanism into QATs/OFQ.

Files to inspect or adapt:

- `examples/run_stage1_qat.sh`
- `examples/run_stage2_ptq.sh`
- `tools/train_stage1.py`
- `tools/evaluate_stage2.py`
- `gplq/`
- `models/`
- `teacher_thing/` or shipped PCA assets

Constraint:

- Exact GPLQ Swin-T W4A4 paper table value still needs readable PDF/table verification. Do not quote or target an unverified number.
- Treat GPLQ as second priority until OFQ same-setting reproduction is understood.

Gate:

- Stage1 output must evaluate cleanly.
- Stage2 W4A4 full-val must beat the local epoch10/20 plateau before any porting into QATs is considered.
- If reproduced in GPLQ repo first, only then decide whether to port the mechanism into QATs/OFQ.

Expected bottlenecks solved:

- quantization initialization: activation-first stage;
- activation outlier: W32A4 activation QAT and TCS/PCA feature mimic;
- insufficient update budget: fast stage1+stage2 instead of long direct W4A4 QAT;
- quantizer state instability: separates activation adaptation from weight PTQ/compensation.

## Deliverables

1. For every run:
   - exact command/script;
   - checkpoint path;
   - epoch/data_epoch and optimizer update count;
   - raw full ImageNet Top-1/Top-5/loss/samples;
   - whether QKR/CGA/GPLQ stages were active.

2. For every stop:
   - gate failed;
   - most likely root cause;
   - next diagnostic, not next blind experiment.

3. For success:
   - local checkpoint at <=50 epochs with raw Top-1 >=80.0;
   - independent eval log;
   - diff against official OFQ setting.

## Decision summary

Recommended first approved action: Phase 0 preflight for VVTQ, then try to reproduce the official VVTQ Swin-T W4A4 checkpoint/eval because it is the highest verified public result. If the first/last 8-bit caveat is unacceptable for the target, switch first action to OFQ-QKR reproduction with effective batch 512 and 10-epoch checkpoints.

Recommended hold: any custom QSS/refmodel/augmentation idea not derived from public VVTQ/OFQ/GPLQ reproduction evidence.
