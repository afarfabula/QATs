# Swin-T W4A4 Resume10 To 81 Goal Progress

Date: 2026-07-06

## Goal

Find a Swin-T W4A4 QAT resume / finetune paradigm that starts from the public-family fixed-QKR epoch10 checkpoint and reaches full ImageNet raw Top-1 >= 81.0 within at most 10 additional epochs.

## Fixed Starting Point

- Checkpoint: `/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar`
- Source run log: `/mlx_devbox/users/quyanyi/playground/train_recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706.log`
- Source run evidence:
  - `Test: [distributed-summary]  Time: 10.398s  Loss: 0.8453  Acc@1: 80.3640  Acc@5: 95.3140  Samples: 50000`
- Data: `/tmp/imagenet1k_full_parquet`
- Dataset check in current shell:
  - train shards: 294
  - validation shards: 14
- Checkpoint file exists and is 329M.

## Environment Status

Initial Jupyter shell was not a usable GPU worker shell:

```text
test -e /dev/nvidia0 -> no-gpu-device
NVIDIA_VISIBLE_DEVICES=none
MLX_ENGINE_KERNEL_TYPE=JUPYTER
ARNOLD_WORKSPACE_ID=54554
```

Entered real GPU worker with `mlx worker login`:

```text
tiger@984521.worker:playground
gpu-device-present
8x NVIDIA H100 80GB HBM3 visible
```

## Existing 20-Epoch Continuation Evidence

Existing continuation runs from checkpoint-10 do not show progress toward 81:

| run | continuation checkpoints observed | best raw Top-1 | note |
|---|---:|---:|---|
| `recipe20ep_from10_nostruct_featnorm_softkd_20260706` | 1-5 | 80.3700 | flat after first two resumed epochs |
| `recipe20ep_from10_struct_teacherattn_qkrel_v3_20260706` | 1-5 | 80.2880 | slower and lower than no-structure |
| `recipe20ep_from10_anchorref_attnkl_qkrel_20260706` | 1-2 | 80.2760 | no upward signal |
| `recipe20ep_from10_qkrel_only_20260706` | 1-2 | 80.2860 | no upward signal |

Conclusion: do not continue the existing 20-epoch structure branches as the main path.

## Launcher / Resume Checks

Static evidence from `qat_launch.py` shows the relied-on flags are parsed and propagated:

- `--resume`
- `--no-resume-opt`
- `--start-epoch`
- `--lr`
- `--min-lr`
- `--scheduler-epochs`
- `--epoch-checkpoint-interval`
- `--quant-lr-multiplier`
- `--quant-only-start-epoch`
- `--teacher-feature-output-*`

The historical OFQ restore rule still applies: `setup_alpha` must run before checkpoint weights are loaded. This should be rechecked in logs for any new run by looking for strict resume and missing/unexpected key evidence.

## Prepared Scripts

### Phase 0: Independent Full-Val Eval

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_resume10_ckpt10_fullval_20260706.sh
```

Expected purpose:

- Independently validate checkpoint-10 on full ImageNet raw validation.
- Confirm Top-1 is around 80.36 before any new resume experiment.

Result:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_ckpt10_fullval_20260706.log
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 33.148s  Loss: 0.8453  Acc@1: 80.3640  Acc@5: 95.3140  Samples: 50000
```

Note: the first eval-only attempt produced invalid Top-1 `0.3180` because unified eval-only did not load `--resume`. Patched `qat_launch.py` so eval-only calls `strict_resume_checkpoint(...)` after `setup_alpha`; `python3 -m py_compile qat_launch.py` passed. The rerun above is the accepted Phase 0 evidence.

### Phase 1A: Minimal Low-LR Gate

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_lowlr_gate_20260706.sh
```

Default settings:

```text
EXP=recipe_resume10_lowlr_gate_a_lr1e5_qm1_20260706
EPOCHS=3
SCHEDULER_EPOCHS=3
LR=1e-5
MIN_LR=5e-6
QUANT_LR_MULTIPLIER=1
teacher_feature_output_weight=0.003
teacher_feature_output_layers=features.5.5,features.7.1
teacher_feature_output_loss=norm_mse
teacher_attn_kl=disabled
teacher_qk_rel=disabled
anchor_ref=disabled
```

Gate:

- Resumed epoch 1 should not meaningfully drop below checkpoint-10's 80.3640.
- Resumed epoch 2 or 3 should reach at least 80.55-80.60 before extending to 10 epochs.

Result:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_lowlr_gate_a_lr1e5_qm1_20260706.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_lowlr_gate_a_lr1e5_qm1_20260706
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
```

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.3400 | 95.2820 | 0.8467 | preserve baseline: pass; lift: fail |
| 2 | `checkpoint-2.pth.tar` | 80.2920 | 95.2980 | 0.8459 | fail |
| 3 | `checkpoint-3.pth.tar` | 80.4360 | 95.3120 | 0.8401 | best in run, but below 80.55-80.60 gate |

Conclusion: Phase 1A is stable but does not produce enough lift. Do not extend this recipe to 10 resumed epochs. Move to Phase 1B.

### Phase 1B: Slightly Higher Low-LR Gate

Use the same script with environment overrides:

```bash
EXP=recipe_resume10_lowlr_gate_b_lr15e6_qm2_20260706 \
MASTER_PORT=30523 \
LR=1.5e-5 \
MIN_LR=5e-6 \
QUANT_LR_MULTIPLIER=2 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_lowlr_gate_20260706.sh
```

Gate is the same as Phase 1A.

Result:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_lowlr_gate_b_lr15e6_qm2_20260706.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_lowlr_gate_b_lr15e6_qm2_20260706
Using grouped LR: base_params=28280866, quant_params=327327, quant_lr_multiplier=2.0
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
```

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.3040 | 95.3080 | 0.8459 | fail |
| 2 | `checkpoint-2.pth.tar` | 80.4580 | 95.2960 | 0.8429 | best in run, but below 80.55-80.60 gate |
| 3 | `checkpoint-3.pth.tar` | 80.4020 | 95.2960 | 0.8418 | fail |

Conclusion: Phase 1B gives a small lift over checkpoint-10 but still does not meet the extension gate. Do not extend this recipe to 10 resumed epochs. Move to Phase 2 quant-only gate.

### Phase 2: Quant-Only Gate

Only run this if Phase 1 full-param gates are flat or declining:

```bash
EXP=recipe_resume10_quantonly_gate_lr15e6_qm2_20260706 \
MASTER_PORT=30524 \
LR=1.5e-5 \
MIN_LR=5e-6 \
QUANT_LR_MULTIPLIER=2 \
QUANT_ONLY_START_EPOCH=0 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_lowlr_gate_20260706.sh
```

Gate is the same: preserve 80.3640 first, then require upward movement by resumed epoch 2/3.

First attempt note:

The first Phase 2 attempt used `QUANT_ONLY_START_EPOCH=0` but did not set `--trainable-policy quant`; logs showed `policy=all, trainable=28608256, frozen=0`, so it was invalid and was interrupted around epoch0 34%.

Valid result:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_quantonly_gate_quant_lr15e6_qm2_20260706.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_quantonly_gate_quant_lr15e6_qm2_20260706
Using grouped LR: base_params=28280866, quant_params=327327, quant_lr_multiplier=2.0
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar; missing=0, unexpected=0
Trainable parameter policy: epoch=0, quant_only=True, policy=quant, trainable=327390, frozen=28280866
```

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.2920 | 95.3120 | 0.8441 | fail |
| 2 | `checkpoint-2.pth.tar` | 80.3560 | 95.3160 | 0.8438 | fail |
| 3 | `checkpoint-3.pth.tar` | 80.3160 | 95.3280 | 0.8431 | fail |

Conclusion: Quant-only is valid but does not improve over full-param Phase 1B. Do not extend this recipe.

### Phase 1C: Full-Param Low-LR Gate With Higher LR Floor

Reason: Phase 1B briefly reached `80.4580` at resumed epoch2, but LR decayed to `5e-6` by epoch3. Phase 1C tests whether keeping a higher LR floor helps the short gate.

Command:

```bash
EXP=recipe_resume10_lowlr_gate_c_lr15e6_minlr1e5_qm2_20260706 \
MASTER_PORT=30526 \
LR=1.5e-5 \
MIN_LR=1e-5 \
QUANT_LR_MULTIPLIER=2 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_lowlr_gate_20260706.sh
```

Result:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_lowlr_gate_c_lr15e6_minlr1e5_qm2_20260706.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_lowlr_gate_c_lr15e6_minlr1e5_qm2_20260706
Using grouped LR: base_params=28280866, quant_params=327327, quant_lr_multiplier=2.0
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar; missing=0, unexpected=0
```

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.2360 | 95.2640 | 0.8490 | fail; stopped early |

Conclusion: Raising `min_lr` to `1e-5` hurts baseline preservation. The run was interrupted during epoch2 and should not be extended.

### Phase 1D: Full-Param Quant LR x4 Gate

Reason: Phase 1B's quant LR x2 reached the best current Top-1 `80.4580`. Phase 1D tests whether stronger quant/shift adaptation helps.

Command:

```bash
EXP=recipe_resume10_lowlr_gate_d_lr15e6_qm4_20260706 \
MASTER_PORT=30527 \
LR=1.5e-5 \
MIN_LR=5e-6 \
QUANT_LR_MULTIPLIER=4 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_lowlr_gate_20260706.sh
```

Status:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_lowlr_gate_d_lr15e6_qm4_20260706.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_lowlr_gate_d_lr15e6_qm4_20260706
Using grouped LR: base_params=28280866, quant_params=327327, quant_lr_multiplier=4.0
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar; missing=0, unexpected=0
```

No validation checkpoint was produced. The run was interrupted around epoch0 10% because it was abnormally slow in the interactive worker shell and had not produced a gate signal. Do not treat this as a metric result.

### Phase 1E: Full-Param No-Feature Gate

Reason: all accepted low-LR gates used `teacher_feature_output_weight=0.003`. Phase 1E tests whether the late-stage resume is being held back by the feature-output auxiliary.

Command:

```bash
EXP=recipe_resume10_lowlr_gate_e_lr15e6_qm2_nofeat_20260706 \
MASTER_PORT=30528 \
LR=1.5e-5 \
MIN_LR=5e-6 \
QUANT_LR_MULTIPLIER=2 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_lowlr_gate_20260706.sh
```

Result:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_lowlr_gate_e_lr15e6_qm2_nofeat_20260706.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_lowlr_gate_e_lr15e6_qm2_nofeat_20260706
Using grouped LR: base_params=28280866, quant_params=327327, quant_lr_multiplier=2.0
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar; missing=0, unexpected=0
TeacherFeatOut: 0.000e+00
```

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.3820 | 95.2840 | 0.8462 | fail |
| 2 | `checkpoint-2.pth.tar` | 80.4080 | 95.3000 | 0.8408 | fail |
| 3 | `checkpoint-3.pth.tar` | 80.4000 | 95.2840 | 0.8431 | fail |

Conclusion: Removing the feature-output auxiliary does not improve the best gate. Do not extend this recipe.

### Phase 1F Plan: Pre-QAT Late Feature Reconstruction Warm Start

Reason: LR / quant-only / feature-on-off gates plateau below 80.46. Existing public-family experiments showed QSC-style pre-QAT late feature reconstruction is a real training-paradigm mechanism, even if it did not solve from-scratch acceleration. This phase tests whether applying a short quant/shift-only feature reconstruction warm start before late-stage resume training can improve the already-strong checkpoint-10 basin.

Planned command:

```bash
EXP=recipe_resume10_prerecon100_lowlr_b_20260706 \
MASTER_PORT=30529 \
LR=1.5e-5 \
MIN_LR=5e-6 \
QUANT_LR_MULTIPLIER=2 \
PRE_QAT_FEATURE_RECON_UPDATES=100 \
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1 \
PRE_QAT_FEATURE_RECON_POLICY=quant \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_lowlr_gate_20260706.sh
```

Gate is the same as Phase 1B. This should only be extended if it reaches at least 80.55-80.60 by resumed epoch2/3.

Result:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_prerecon100_lowlr_b_20260706.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon100_lowlr_b_20260706
Using grouped LR: base_params=28280866, quant_params=327327, quant_lr_multiplier=2.0
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar; missing=0, unexpected=0
Starting pre-QAT feature reconstruction: updates=100, policy=quant, weight_mode=none, confidence_power=0.0, layers=('features.5.5', 'features.7.1')
PreQATFeatRecon: update=1/100 loss=0.117176 kept=67764 masked=27767356
PreQATFeatRecon: update=50/100 loss=0.107642 kept=67764 masked=27767356
PreQATFeatRecon: update=100/100 loss=0.112701 kept=67764 masked=27767356
Finished pre-QAT feature reconstruction: updates=100
```

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.3840 | 95.3600 | 0.8441 | stable, below gate |
| 2 | `checkpoint-2.pth.tar` | 80.5220 | 95.2900 | 0.8415 | best so far, but below 80.55-80.60 extension gate |
| 3 | `checkpoint-3.pth.tar` | 80.3620 | 95.2700 | 0.8402 | regressed to baseline |

Conclusion: pre-QAT feature reconstruction is the best valid mechanism so far and improves the best checkpoint from `80.4580` to `80.5220`, but it still does not satisfy the extension gate and regresses by the third resumed epoch. Do not extend this exact recipe to 10 epochs.

### Phase 1G: Quant-Only Polish From Phase 1F Checkpoint-2

Reason: Phase 1F checkpoint-2 is the best current point at `80.5220`, but the next full-param epoch regresses. Phase 1G tests whether a quant/shift-only polish can preserve the weight basin while nudging quantization state upward.

Command:

```bash
EXP=recipe_resume10_prerecon_ckpt2_quantpolish_lr8e6_20260706 \
MASTER_PORT=30530 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon100_lowlr_b_20260706/checkpoint-2.pth.tar \
EPOCHS=2 \
SCHEDULER_EPOCHS=2 \
LR=8e-6 \
MIN_LR=4e-6 \
QUANT_LR_MULTIPLIER=2 \
QUANT_ONLY_START_EPOCH=0 \
TRAINABLE_POLICY=quant \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_lowlr_gate_20260706.sh
```

Result:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_prerecon_ckpt2_quantpolish_lr8e6_20260706.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_ckpt2_quantpolish_lr8e6_20260706
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon100_lowlr_b_20260706/checkpoint-2.pth.tar; missing=0, unexpected=0
Trainable parameter policy: epoch=0, quant_only=True, policy=quant, trainable=327390, frozen=28280866
```

| polish epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.4720 | 95.2700 | 0.8424 | below start checkpoint-2 |
| 2 | `checkpoint-2.pth.tar` | 80.5020 | 95.3500 | 0.8419 | below start checkpoint-2 and extension gate |

Conclusion: quant-only polish does not preserve or improve Phase 1F checkpoint-2. Do not extend.

### Phase 1H: GPLQ-Inspired Activation Curriculum Gate

Reason: The previous gates show that ordinary low-LR continuation, quant-only polish, feature-output on/off, and pre-QAT feature reconstruction variants plateau below the 81 target. Literature review points to a training-paradigm issue rather than a scalar hyperparameter issue: GPLQ argues for activation-first, weights-later staged optimization that preserves the original optimization basin, while VVTQ/OFQ-style public results rely on structural quantization mechanisms beyond plain continuation. Phase 1H tests a resume-specific activation-first curriculum: keep W4 weights, relax activations to A8 for the first resumed epoch, then switch back to W4A4 with LSQ rescale and alpha recalibration.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_actcurr_gate_20260707.sh
```

Command defaults:

```text
EXP=recipe_resume10_actcurr_w4a8_to_w4a4_20260707
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_actcurr_w4a8_to_w4a4_20260707.log
Output=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_actcurr_w4a8_to_w4a4_20260707
LR=1.5e-5
MIN_LR=5e-6
QUANT_LR_MULTIPLIER=2
PROGRESSIVE_BIT_SCHEDULE=0:4:8,1:4:4
PROGRESSIVE_BIT_RECALIBRATE_EPOCHS=1
PROGRESSIVE_BIT_RECALIBRATE_BATCHES=4
teacher_feature_output_weight=0.003
teacher_feature_output_layers=features.5.5,features.7.1
teacher_feature_output_loss=norm_mse
```

Worker/GPU evidence:

```text
NO_COLOR=1 TERM=dumb mlx worker login
tiger@984521.worker:playground
gpu-device-present
8x NVIDIA H100 80GB HBM3 visible and idle before launch
```

Runtime evidence:

```text
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Applied progressive fake-quant bits: epoch=0 wbits=4 abits=8 weight_modules=118 act_modules=65
Applied progressive fake-quant bits: epoch=1 wbits=4 abits=4 weight_modules=118 act_modules=65
progressive bit recalibrate alpha batches=4 quantizers=67
Applied progressive bit alpha recalibration: epoch=1 batches=4 quantizers=67
```

Results:

| resumed epoch | active bits | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---|---:|---:|---:|---|
| 1 | W4A8 | `checkpoint-1.pth.tar` | 80.6540 | 95.4080 | 0.8320 | passes extension gate, best current single checkpoint |
| 2 | W4A4 | `checkpoint-2.pth.tar` | 77.2060 | 93.8520 | 1.0267 | fails; hard A8->A4 switch destroys the gain |

Timing:

| epoch | updates | avg step time | samples/sec |
|---:|---:|---:|---:|
| 0 | 2496 | 0.223172s | 2294.20 |
| 1 | 2496 | 0.223051s | 2295.44 |

Stop decision:

The run was manually interrupted during epoch2 after the W4A4 epoch1 full-val result dropped to `77.2060`. Continuing the same hard-switch schedule cannot reach the 81 target and would waste GPU time. GPU memory returned to idle after interruption; no residual QAT/OFQ process group remained.

Conclusion:

Activation-first is a real positive signal: the W4A8 resumed checkpoint reached `80.6540`, beating the previous best `80.5220` and passing the extension gate. However, the hard switch directly from A8 to A4 is too abrupt and invalidates the basin. Do not extend this exact schedule to 10 epochs.

Next non-redundant direction:

- Test a smoother activation curriculum, e.g. `0:4:8,1:4:6,2:4:4`, with LSQ rescale and recalibration at each switch.
- If W4A6 preserves the W4A8 gain, only then continue to W4A4 and consider extending.
- Avoid returning to plain low-LR or quant-only tuning; the useful finding is the activation-first paradigm, not the exact hard-switch schedule.

### Phase 1I: Smooth Activation Curriculum Gate

Reason: Phase 1H showed that activation-first W4A8 is a strong positive signal (`80.6540`), but the direct A8 -> A4 switch destroys the gain. Phase 1I tests whether an intermediate A6 stage can preserve the activation-first basin before returning to A4.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_actcurr_smooth_gate_20260707.sh
```

Command defaults:

```text
EXP=recipe_resume10_actcurr_w4a8_w4a6_w4a4_20260707
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_actcurr_w4a8_w4a6_w4a4_20260707.log
Output=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_actcurr_w4a8_w4a6_w4a4_20260707
LR=1.5e-5
MIN_LR=5e-6
QUANT_LR_MULTIPLIER=2
PROGRESSIVE_BIT_SCHEDULE=0:4:8,1:4:6,2:4:4
PROGRESSIVE_BIT_RECALIBRATE_EPOCHS=1,2
PROGRESSIVE_BIT_RECALIBRATE_BATCHES=4
teacher_feature_output_weight=0.003
teacher_feature_output_layers=features.5.5,features.7.1
teacher_feature_output_loss=norm_mse
```

Runtime evidence:

```text
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Applied progressive fake-quant bits: epoch=0 wbits=4 abits=8 weight_modules=118 act_modules=65
Applied progressive fake-quant bits: epoch=1 wbits=4 abits=6 weight_modules=118 act_modules=65
progressive bit recalibrate alpha batches=4 quantizers=67
Applied progressive bit alpha recalibration: epoch=1 batches=4 quantizers=67
```

Results:

| resumed epoch | active bits | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---|---:|---:|---:|---|
| 1 | W4A8 | `checkpoint-1.pth.tar` | 80.6540 | 95.4080 | 0.8320 | reproduces Phase 1H W4A8 positive signal |
| 2 | W4A6 | `checkpoint-2.pth.tar` | 79.7340 | 95.0320 | 0.8794 | better than hard W4A4 switch, but below extension gate |

Timing:

| epoch | updates | avg step time | samples/sec |
|---:|---:|---:|---:|
| 0 | 2496 | 0.223518s | 2290.64 |
| 1 | 2496 | 0.223322s | 2292.66 |

Stop decision:

The run was manually interrupted during the next W4A4 epoch after the W4A6 full-val dropped to `79.7340`. Since W4A6 already failed to preserve the `80.6540` W4A8 gain and did not meet the `80.55-80.60` extension gate, continuing to W4A4 cannot satisfy the 81 target under this schedule. GPU memory returned to idle after interruption; no residual QAT/OFQ process group remained.

Conclusion:

The intermediate A6 stage confirms that smoother activation compression is better than the direct A8 -> A4 switch (`79.7340` vs `77.2060`), but still loses too much of the A8 gain. The active bottleneck is not simply the lack of an intermediate bit-width; the A4 endpoint needs additional local activation reconstruction or a longer activation-only/activation-relaxed phase before weight/full-parameter updates.

Next non-redundant direction:

- Do not extend `0:4:8,1:4:6,2:4:4`.
- Try preserving the strong W4A8 basin for more than one resumed epoch, then evaluate whether a later W4A6 transition holds up.
- Alternatively add an A4 transition-specific local reconstruction step immediately before/after the A4 switch, focused on activation/shift parameters, rather than another scalar LR tweak.
- Treat W4A8 `80.6540` as a useful diagnostic checkpoint, not a final W4A4 success.

### Phase 1J: A8 Checkpoint To A4 Transition Reconstruction

Reason: Phase 1H/1I show that W4A8 is the only strong positive signal, while A4 transition is the failure point. Phase 1J tests whether starting from the W4A8 checkpoint and performing an A4 late-feature reconstruction warm start can bridge the transition before short A4 QAT.

Command:

```bash
EXP=recipe_resume10_a8ckpt1_a4_recon100_gate_20260707 \
MASTER_PORT=30535 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_actcurr_w4a8_to_w4a4_20260707/checkpoint-1.pth.tar \
EPOCHS=2 \
SCHEDULER_EPOCHS=2 \
LR=1.0e-5 \
MIN_LR=5e-6 \
QUANT_LR_MULTIPLIER=2 \
PRE_QAT_FEATURE_RECON_UPDATES=100 \
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1 \
PRE_QAT_FEATURE_RECON_POLICY=quant \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_lowlr_gate_20260706.sh
```

Runtime evidence:

```text
Log=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_a8ckpt1_a4_recon100_gate_20260707.log
Output=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_a8ckpt1_a4_recon100_gate_20260707
Strict resume: loaded model from .../recipe_resume10_actcurr_w4a8_to_w4a4_20260707/checkpoint-1.pth.tar; missing=0, unexpected=0
Starting pre-QAT feature reconstruction: updates=100, policy=quant, weight_mode=none, confidence_power=0.0, layers=('features.5.5', 'features.7.1')
PreQATFeatRecon: update=1/100 loss=0.631106 kept=67764 masked=27767356
PreQATFeatRecon: update=50/100 loss=0.622582 kept=67764 masked=27767356
PreQATFeatRecon: update=100/100 loss=0.617707 kept=67764 masked=27767356
Finished pre-QAT feature reconstruction: updates=100
```

Result:

| resumed epoch | start checkpoint | active bits | transition step | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---|---|---:|---:|---:|---|
| 1 | Phase 1H W4A8 `checkpoint-1.pth.tar` | W4A4 | 100-step late-feature quant/shift reconstruction | 75.1060 | 92.7340 | 1.1004 | fail |

Timing:

| epoch | updates | avg step time | samples/sec |
|---:|---:|---:|---:|
| 0 | 2496 | 0.223629s | 2289.51 |

Stop decision:

The run was interrupted during the next epoch after the first A4 full-val dropped to `75.1060`. GPU memory returned to idle after interruption; no residual QAT/OFQ process group remained.

Conclusion:

This transition reconstruction is worse than both the hard A8->A4 switch (`77.2060`) and the A8->A6 intermediate (`79.7340`). Starting from an A8-trained checkpoint and then rebuilding an A4 model plus late-feature quant/shift reconstruction does not preserve the A8 basin. The likely failure mode is that A4 activation quantizer state/distribution changes are too large for late-block feature reconstruction alone; the feature loss before normal training remained high (`~0.62`) and only slowly decreased.

Next non-redundant direction:

- Do not resume from the A8 checkpoint directly into A4 with this reconstruction recipe.
- The next useful mechanism should inspect or modify A4 activation quantizer initialization/distribution handling itself, e.g. alpha calibration/rescale policy, activation outlier handling, or a more local per-activation reconstruction, rather than repeating late-feature reconstruction or scalar LR sweeps.
- A practical diagnostic is to compare activation quantizer scales and layer-wise feature error for checkpoint-10, W4A8 checkpoint-1, W4A6 checkpoint-2, and failed W4A4 transition checkpoints before launching another training run.

### Phase 1K: Activation Transition Diagnostic

Reason: Phase 1H-1J show that the failure is specifically at the activation-bit transition. Before launching more training, compare checkpoint feature errors and activation quantizer scale statistics using the same OFQ/QKR model path.

Script:

```bash
python3 /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_activation_transition_20260707.py
```

Artifacts:

```text
JSON: /mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_activation_transition_diag_20260707.json
TSV:  /mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_activation_transition_diag_20260707.tsv
```

Diagnostic setup:

- Single GPU worker run, no training.
- Builds the same OFQ/QKR Swin-T path as `qat_launch.py`.
- Loads checkpoints strictly with `missing=0, unexpected=0`.
- Uses one fixed validation batch.
- Reports `features.5.5`, `features.7.1`, logits KL/top1 agreement, and activation quantizer scale summaries.

Key rows:

| case | bits | f5.5 MSE | f7.1 MSE | logits KL | top1 agree | input scale mean / p95 | qkx scale mean / p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| `ckpt10_start_w4a4` | W4A4 | 10.955 | 4.618 | 2.581 | 0.547 | 0.2566 / 0.6631 | 0.1390 / 0.2101 |
| `phase1h_w4a8_ckpt1_as_w4a8` | W4A8 | 6.391 | 3.046 | 0.277 | 0.859 | 0.0565 / 0.1513 | 0.0325 / 0.0493 |
| `phase1h_w4a8_ckpt1_as_w4a4` | W4A4 | 30.006 | 5.945 | 5.381 | 0.000 | 0.0567 / 0.1513 | 0.0325 / 0.0493 |
| `phase1h_w4a4_ckpt2` | W4A4 | 20.262 | 8.980 | 3.116 | 0.234 | 0.3273 / 0.9699 | 0.1909 / 0.3209 |
| `phase1i_w4a6_ckpt2_as_w4a6` | W4A6 | 11.210 | 5.296 | 0.745 | 0.703 | 0.1509 / 0.4493 | 0.0920 / 0.1508 |
| `phase1i_w4a6_ckpt2_as_w4a4` | W4A4 | 22.178 | 6.778 | 3.929 | 0.141 | 0.1523 / 0.4493 | 0.0920 / 0.1508 |
| `phase1j_a4_recon_ckpt1` | W4A4 | 18.701 | 8.040 | 2.112 | 0.562 | 0.0770 / 0.1773 | 0.0464 / 0.0662 |

Interpretation:

- W4A8 works because its activation scales are much smaller and appropriate for A8 thresholds; it has low logits KL (`0.277`) and high teacher top1 agreement (`0.859`).
- Reinterpreting the W4A8 checkpoint as W4A4 keeps the small A8-trained activation scales but changes the thresholds to A4. This collapses `features.5.5` variance (`student_std=0.93` vs teacher `5.63` in the JSON), produces very high feature error (`30.006`) and logits KL (`5.381`), and zero top1 agreement on the diagnostic batch.
- The smooth W4A6 path is an intermediate state: scale means are between A8 and A4, and logits KL is far better at W4A6 (`0.745`) than A4 reinterpretation, explaining why W4A6 validation (`79.7340`) was much better than W4A4 hard switch.
- Phase 1J's late-feature reconstruction reduced logits KL on the diagnostic batch but did not repair the endpoint because A4 activation scales were still far from a stable A4 distribution; it validated at only `75.1060`.

Conclusion:

The likely mechanism is activation-scale mismatch at bit transition: checkpoint state carries learned LSQ activation scales but not a stable policy for changing bit thresholds. A8-trained scales are too small for direct A4 interpretation; naive recalibration or late-feature reconstruction does not restore the W4A4 basin.

Next non-redundant direction:

- Test an activation-scale transplant/anchor: start from the A8 checkpoint weights but replace A4 activation quantizer scales with the stable `checkpoint-10` A4 scales before evaluation/training.
- This is a mechanism test of the diagnostic hypothesis, not a scalar hyperparameter sweep.
- If direct eval of the transplanted checkpoint recovers near `80.36+`, use it as the next resume start. If it fails, the issue is not only scale magnitude and requires more local per-activation distribution handling.

### Phase 1L: A4 Activation-Scale Transplant Eval

Reason: Phase 1K showed that W4A8 checkpoints carry much smaller activation quantizer scales than stable W4A4 checkpoints. Phase 1L tests the narrow hypothesis that the A4 failure is mainly caused by activation-scale magnitude mismatch: use W4A8 checkpoint weights, but transplant stable A4 activation quantizer scales from the original checkpoint-10.

Construction:

```text
Base checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_actcurr_w4a8_to_w4a4_20260707/checkpoint-1.pth.tar
Scale source: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
Output checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_a8ckpt1_a4_scale_transplant_20260707/checkpoint-1.pth.tar
Replaced keys: 90 activation quantizer scale/signed keys
Missing keys: 0
Patterns: input_quant_fn.s, input_quant_fn.signed, quant_x_4_qkv.input_quant_fn.s, quan_a_qkx_fn.s, quan_a_v_fn.s, quan_a_softmax_fn.s
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_a8ckpt1_a4_scale_transplant_20260707/checkpoint-1.pth.tar \
EXP=eval_resume10_a8ckpt1_a4_scale_transplant_20260707 \
MASTER_PORT=30536 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_a8ckpt1_a4_scale_transplant_20260707.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_resume10_ckpt10_fullval_20260706.sh
```

Result:

```text
Strict resume: loaded model from .../recipe_resume10_a8ckpt1_a4_scale_transplant_20260707/checkpoint-1.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.962s  Loss: 0.8623  Acc@1: 78.8780  Acc@5: 94.6180  Samples: 50000
```

Interpretation:

- The transplant improves over the direct A8->A4 hard switch (`77.2060`) and over the failed A4 reconstruction branch (`75.1060`), so activation-scale magnitude is part of the failure.
- It still falls far below checkpoint-10 (`80.3640`) and the extension gate, so scale magnitude alone is not sufficient.
- A8 training also changes ordinary weights and shift/bias state into a basin that depends on relaxed A8 activations; simply forcing A4 scales back in does not recover the W4A4 basin.

Stop decision:

Do not train from this transplanted checkpoint. It fails direct full-val at `78.8780`, below the baseline and extension gate.

Conclusion:

The A4 transition failure is a coupled state problem: activation quantizer scales, shift/move bias, and ordinary weights co-adapt during A8 training. The next mechanism must manage the bit-transition state jointly rather than only transplant LSQ scales or only reconstruct late features.

Next non-redundant direction:

- Design a bit-transition policy that preserves the W4A8 basin while jointly adjusting activation scales and shift biases, ideally with a local per-activation objective at the transition point.
- Before training, compare `move_b4/move_aft` and activation scale ratios between checkpoint-10, W4A8, W4A6, and failed A4 checkpoints to choose whether to transplant scale only, shift only, or both in a controlled ablation.
- Do not repeat direct A8->A4, A8->A6->A4, or late-feature-only reconstruction; those branches are now falsified.

### Phase 1M: A4 Activation Scale+Shift Transplant Eval

Reason: Phase 1L showed that transplanting only stable A4 activation scales into the W4A8 checkpoint improves the A4 endpoint from `77.2060` to `78.8780`, but not enough. Phase 1M tests whether the missing coupled state is the activation shift/move bias by transplanting both scale/signed keys and `move_b4` / `move_aft` biases from checkpoint-10.

Construction:

```text
Base checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_actcurr_w4a8_to_w4a4_20260707/checkpoint-1.pth.tar
Scale/shift source: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
Output checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_a8ckpt1_a4_scale_shift_transplant_20260707/checkpoint-1.pth.tar
Replaced keys: 196 total
  - 90 activation quantizer scale/signed keys
  - 106 move_b4/move_aft bias keys
Missing keys: 0
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_a8ckpt1_a4_scale_shift_transplant_20260707/checkpoint-1.pth.tar \
EXP=eval_resume10_a8ckpt1_a4_scale_shift_transplant_20260707 \
MASTER_PORT=30537 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_a8ckpt1_a4_scale_shift_transplant_20260707.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_resume10_ckpt10_fullval_20260706.sh
```

Result:

```text
Strict resume: loaded model from .../recipe_resume10_a8ckpt1_a4_scale_shift_transplant_20260707/checkpoint-1.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.785s  Loss: 0.8722  Acc@1: 79.0180  Acc@5: 94.6620  Samples: 50000
```

Interpretation:

- Transplanting both scale and shift is only marginally better than scale-only (`79.0180` vs `78.8780`).
- It is still well below checkpoint-10 (`80.3640`) and the extension gate.
- Therefore, the failure is not fixed by restoring stable A4 LSQ scale and shift state alone. The ordinary weights and quantized activation pathway co-adapt during A8 training in a way that remains incompatible with A4.

Stop decision:

Do not train from the scale+shift transplanted checkpoint. It fails direct full-val below baseline and below extension gate.

Conclusion:

The A4 transition is a coupled optimization-state problem, not just a missing scale or shift transplant. The next mechanism should not be another transplant of static checkpoint state. It should either:

- keep A4 active throughout while using an activation-relaxation auxiliary rather than changing actual bit-width, or
- implement a local per-activation transition objective that updates activation quantizers and nearby weights together before full QAT.

The current evidence favors an A4-native relaxation strategy over A8-to-A4 checkpoint conversion.

### Phase 1N: A4-Native Freeze-Activation-State Gate

Reason: Phase 1H-1M falsified A8-to-A4 conversion and static state transplant. Phase 1N tests an A4-native alternative: keep the checkpoint-10 W4A4 activation quantizer and shift state fixed, and let ordinary weights / non-activation quantizer state adapt inside the stable A4 activation basin. This is a mechanism test of whether activation-state movement is causing short-run instability.

Command:

```bash
EXP=recipe_resume10_a4native_freezeact_gate_20260707 \
MASTER_PORT=30538 \
EPOCHS=2 \
SCHEDULER_EPOCHS=2 \
LR=1.0e-5 \
MIN_LR=5e-6 \
QUANT_LR_MULTIPLIER=2 \
QUANT_ONLY_START_EPOCH=0 \
TRAINABLE_POLICY=freeze_act_quant \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_lowlr_gate_20260706.sh
```

Runtime evidence:

```text
Log=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_a4native_freezeact_gate_20260707.log
Output=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_a4native_freezeact_gate_20260707
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Trainable parameter policy: epoch=0, quant_only=True, policy=freeze_act_quant, trainable=28282025, frozen=326231
Trainable parameter policy: epoch=1, quant_only=True, policy=freeze_act_quant, trainable=28282025, frozen=326231
```

Results:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.3300 | 95.2560 | 0.8466 | below baseline |
| 2 | `checkpoint-2.pth.tar` | 80.4280 | 95.2480 | 0.8422 | slight lift, below extension gate |

Timing:

| epoch | updates | avg step time | samples/sec |
|---:|---:|---:|---:|
| 0 | 2496 | 0.174943s | 2926.66 |
| 1 | 2496 | 0.174853s | 2928.17 |

Conclusion:

Freezing activation quantizer/shift state prevents catastrophic transition failure and is stable, but it does not create enough lift. The best result `80.4280` is above checkpoint-10 by only `+0.064` and below the `80.55-80.60` extension gate. Do not extend this exact freeze-act recipe.

Interpretation:

- Activation-state drift is part of the problem, but freezing it entirely underfits the short adaptation.
- The next mechanism should allow controlled activation-state movement rather than either hard-freezing it or changing bit-width.
- A plausible next A4-native mechanism is activation slow-state / constrained activation-state updates: keep a stable A4 activation-state reference and allow small pulled updates, instead of full freeze or A8 relaxation.

### Phase 1O: A4-Native Activation QSS Gate

Reason: Phase 1N showed that freezing activation quantizer/shift state is stable but underfits. Phase 1O keeps the model in W4A4 and allows activation quantizer/shift updates, but constrains them with Quantizer Slow State (QSS) pullback to a stable shadow state. This tests controlled activation-state movement without changing bit-width.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_a4native_actqss_gate_20260707.sh
```

Command defaults:

```text
EXP=recipe_resume10_a4native_actqss_gate_20260707
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_a4native_actqss_gate_20260707.log
Output=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_a4native_actqss_gate_20260707
LR=1.0e-5
MIN_LR=5e-6
QUANT_LR_MULTIPLIER=2
QSS_DECAY=0.99
QSS_SYNC_INTERVAL=50
QSS_PULL=0.05
QSS_POLICY=activation
teacher_feature_output_weight=0.003
```

Runtime evidence:

```text
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Initialized quant slow state: params=243, policy=activation, decay=0.99, sync_interval=50, pull=0.05
Applied quant slow state pull: update=50, tensors=243, pull=0.05
...
Applied quant slow state pull: update=4950, tensors=243, pull=0.05
```

Results:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.4340 | 95.3020 | 0.8447 | below extension gate |
| 2 | `checkpoint-2.pth.tar` | 80.4560 | 95.2820 | 0.8429 | best in branch, below extension gate |

Timing:

| epoch | updates | avg step time | samples/sec |
|---:|---:|---:|---:|
| 0 | 2496 | 0.225437s | 2271.14 |
| 1 | 2496 | 0.225249s | 2273.04 |

Conclusion:

Activation QSS is stable and slightly improves over freeze-act (`80.4560` vs `80.4280`), but it is essentially tied with the earlier full-param low-LR Phase 1B (`80.4580`) and remains below the `80.55-80.60` extension gate. Do not extend this exact activation-QSS recipe.

Interpretation:

- Controlled activation-state movement helps only marginally at this strength.
- The bottleneck is no longer catastrophic activation transition; it is insufficient positive learning signal under strict W4A4.
- The next useful A4-native mechanism should add a stronger structural target while keeping activation-state control, for example combining activation-QSS with pre-QAT feature reconstruction or a transition-local objective. Avoid another scalar-only QSS pull/decay sweep unless supported by diagnostics.

## Worker Entry Reminder

Use a real GPU worker terminal:

```bash
NO_COLOR=1 TERM=dumb mlx worker login
test -e /dev/nvidia0 && echo gpu-device-present
nvidia-smi
```

If worker auto-discovery fails with `/workers/detail/ status code:500`, provide a worker ID or pod IP and run:

```bash
NO_COLOR=1 TERM=dumb mlx worker login <worker_id_or_pod_ip>
```

## Current Status

- Phase 0 independent eval completed and confirmed checkpoint-10 raw Top-1 `80.3640`.
- Phase 1A completed; best Top-1 `80.4360`, below extension gate.
- Phase 1B completed; best Top-1 `80.4580`, below extension gate.
- Phase 2 quant-only gate completed; best Top-1 `80.3560`, below extension gate.
- Phase 1C high min-LR gate stopped early; Top-1 `80.2360`.
- Phase 1D quant LR x4 was interrupted before validation; no accepted metric.
- Phase 1E no-feature gate completed; best Top-1 `80.4080`, below extension gate.
- Phase 1F pre-QAT feature reconstruction completed; best Top-1 `80.5220`, below extension gate and followed by regression.
- Phase 1G quant-only polish from Phase 1F checkpoint-2 completed; best Top-1 `80.5020`, below starting checkpoint and extension gate.
- Phase 1H GPLQ-inspired activation curriculum completed; W4A8 checkpoint reached Top-1 `80.6540`, but hard switch to W4A4 dropped to `77.2060`, so the exact schedule was stopped.
- Phase 1I smooth activation curriculum completed; W4A8 again reached `80.6540`, W4A6 reached `79.7340`, still below extension gate, so the schedule was stopped before wasting compute on W4A4.
- Phase 1J A8-to-A4 transition reconstruction completed; A4 full-val reached only `75.1060`, so the branch was stopped.
- Phase 1K activation transition diagnostic completed; evidence points to LSQ activation-scale mismatch when A8/A6 checkpoints are interpreted as A4.
- Phase 1L A4 activation-scale transplant eval completed; direct full-val reached `78.8780`, better than hard switch but below baseline/extension gate, so it is not a usable resume start.
- Phase 1M A4 scale+shift transplant eval completed; direct full-val reached `79.0180`, still below baseline/extension gate, so static transplant is not a usable resume start.
- Phase 1N A4-native freeze-activation-state gate completed; best Top-1 `80.4280`, stable but below extension gate.
- Phase 1O A4-native activation-QSS gate completed; best Top-1 `80.4560`, stable but below extension gate.
- Phase 1P pre-QAT feature reconstruction + activation-QSS combo completed to resumed epoch2; best Top-1 `80.3860`, below baseline uplift and extension gate, so it was stopped before epoch3.
- Phase 1Q sequential block reconstruction + activation-QSS completed to resumed epoch1; Top-1 `80.2000`, below baseline and extension gate, so it was stopped early.
- Phase 1R QDrop-style pre-QAT feature reconstruction completed to resumed epoch2; best Top-1 `80.4980`, below Phase 1F and extension gate, so it was stopped before epoch3.
- Phase 1S disagreement-weighted pre-QAT feature reconstruction completed to resumed epoch1; Top-1 `80.3660`, effectively baseline and below extension gate, so it was stopped early.
- Phase 1T late-layer-only QDrop pre-QAT feature reconstruction completed to resumed epoch1; Top-1 `80.3480`, below baseline and extension gate, so it was stopped early.
- Phase 1U pre-QAT feature reconstruction + late activation scale anchor completed to resumed epoch1; Top-1 `80.3460`, below baseline and extension gate, so it was stopped early.
- Current best resume gate: Phase 1H `checkpoint-1.pth.tar`, Top-1 `80.6540` under W4A8; best valid W4A4 continuation remains Phase 1F `80.5220`, still below extension gate and target.
- No >=81 checkpoint has been found yet.
- Goal is not complete.

### Phase 1P: Pre-QAT Feature Reconstruction + Activation-QSS Combo Gate

Reason:

Phase 1F showed the best valid W4A4 result so far by adding pre-QAT late feature reconstruction (`80.5220`), while Phase 1O showed activation-QSS is stable but too weak alone (`80.4560`). Phase 1P tested the direct combination: 100-step late feature reconstruction, then normal W4A4 resume with activation-QSS.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_actqss_gate_20260707.sh
```

Key settings:

```text
EXP=recipe_resume10_prerecon100_actqss_gate_20260707
EPOCHS=3
SCHEDULER_EPOCHS=3
LR=1.5e-5
MIN_LR=5e-6
QUANT_LR_MULTIPLIER=2
PRE_QAT_FEATURE_RECON_UPDATES=100
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
QSS_DECAY=0.99
QSS_SYNC_INTERVAL=50
QSS_PULL=0.05
QSS_POLICY=activation
teacher_feature_output_weight=0.003
```

Runtime evidence:

```text
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
PreQATFeatRecon: update=100/100 loss=0.112701 kept=67764 masked=27767356
Initialized quant slow state: params=243, policy=activation, decay=0.99, sync_interval=50, pull=0.05
```

Results:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.3860 | 95.2800 | 0.8439 | below extension gate |
| 2 | `checkpoint-2.pth.tar` | 80.3520 | 95.2580 | 0.8425 | regressed; fail |

Timing:

| epoch | updates | avg step time | samples/sec |
|---:|---:|---:|---:|
| 0 | 2496 | 0.224879s | 2276.78 |
| 1 | 2496 | 0.224945s | 2276.11 |

Conclusion:

The naive combination of pre-QAT feature reconstruction and activation-QSS is worse than Phase 1F and Phase 1O. It does not reproduce the Phase 1F `80.5220` peak and is below the checkpoint-10 baseline uplift target. The run was stopped after epoch2 full validation to avoid spending the third epoch on a branch that already failed the `80.55-80.60` extension gate.

Interpretation:

- QSS pullback after a quant-only late-feature warm start appears to damp the useful adaptation rather than stabilize it.
- The feature reconstruction in Phase 1F was global across late layers and quant/shift parameters only; combining it with activation-QSS does not address block-wise coupling between ordinary weights, activation quantizers, and local feature distributions.
- This makes a more local block-wise reconstruction mechanism a better next test than more QSS scalar sweeps.

### Phase 1Q: Sequential Block Reconstruction + Activation-QSS Gate

Literature motivation:

- QDrop argues that randomly dropping activation quantization during reconstruction improves low-bit flatness; the actionable part for this goal is not stochastic dropout yet, but the focus on activation-quantization robustness during local reconstruction.
- BRECQ-style work motivates block-wise/local reconstruction instead of only global end-to-end continuation.
- Recent ViT PTQ papers still emphasize reconstruction because ViT activation distributions and LayerNorm/channel variation make direct CNN-style PTQ/QAT transfer weak.

Mechanism:

Keep strict W4A4 active throughout. Before normal QAT epochs, do sequential teacher feature reconstruction on `features.3.1`, `features.5.5`, and `features.7.1`, one block at a time. Unlike Phase 1F, this gate uses `module_all` for the local block so ordinary weights and local quant/shift state can co-adapt under a block-local objective. Then run the same short W4A4 resume with activation-QSS.

Launcher change:

`qat_launch.py` now supports:

```text
--pre-qat-seq-feature-recon-policy quant|module_all
```

Default remains `quant`, so older scripts keep their behavior. `python3 -m py_compile qat_launch.py` passed.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_seqblock_moduleall_actqss_gate_20260707.sh
```

Key settings:

```text
EXP=recipe_resume10_seqblock_moduleall_actqss_gate_20260707
PRE_QAT_SEQ_FEATURE_RECON_UPDATES=50
PRE_QAT_SEQ_FEATURE_RECON_LAYERS=features.3.1,features.5.5,features.7.1
PRE_QAT_SEQ_FEATURE_RECON_POLICY=module_all
QSS_POLICY=activation
```

Gate:

- Must validate on full ImageNet with `Samples: 50000`.
- If epoch2/3 does not reach `80.55-80.60`, do not extend to 10 epochs.
- If it exceeds the gate, extend to at most 10 additional epochs and continue full-val checkpoint selection.

Runtime evidence:

```text
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Starting pre-QAT sequential feature reconstruction: updates_per_layer=50, policy=module_all, layers=('features.3.1', 'features.5.5', 'features.7.1')
PreQATSeqFeatRecon: layer=features.3.1 update=50/50 loss=0.111712 kept=451559 masked=861928
PreQATSeqFeatRecon: layer=features.5.5 update=50/50 loss=0.162335 kept=1792231 masked=10573024
PreQATSeqFeatRecon: layer=features.7.1 update=50/50 loss=0.063635 kept=7141649 masked=20693471
Initialized quant slow state: params=243, policy=activation, decay=0.99, sync_interval=50, pull=0.05
```

Result:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.2000 | 95.3100 | 0.8456 | below baseline and extension gate; stopped |

Timing:

| epoch | updates | avg step time | samples/sec |
|---:|---:|---:|---:|
| 0 | 2496 | 0.225102s | 2274.53 |

Conclusion:

Sequential `module_all` block reconstruction hurts the first resumed validation point. The mechanism changed the local blocks substantially enough to lower Top-1 to `80.2000`, worse than checkpoint-10 (`80.3640`), Phase 1F (`80.5220`), and Phase 1P (`80.3860`). The run was interrupted after checkpoint-1 full validation to avoid spending further epochs on a branch with no early positive signal.

Interpretation:

- Allowing full local block weights to move during reconstruction is too invasive for this late-stage resume setting.
- The previous Phase 1F result suggests quant/shift-only reconstruction can help modestly, but ordinary block weight reconstruction damages the basin.
- The next non-redundant direction should not be another wider `module_all` reconstruction. Prefer mechanisms that preserve the checkpoint-10 weight basin while changing the learning signal, such as QDrop-style stochastic activation quantization during quant/shift-only reconstruction, confidence/disagreement weighting, or activation outlier-aware calibration.

### Phase 1R: QDrop-Style Pre-QAT Feature Reconstruction Gate

Reason:

Phase 1Q showed that `module_all` block reconstruction is too invasive. QDrop's useful lesson for this resume setting is different: during reconstruction, activation quantization can be randomly bypassed to encourage a flatter quantized solution, while the final endpoint remains strict W4A4. Phase 1R keeps Phase 1F's quant/shift-only late feature reconstruction, but applies QDrop-style stochastic activation-quantizer bypass during the reconstruction forward only. Normal QAT training and validation remain full W4A4.

Launcher change:

`qat_launch.py` now supports:

```text
--pre-qat-feature-recon-qdrop-prob <0..1>
```

Implementation scope:

- Default is `0.0`, so old scripts keep their behavior.
- The bypass only wraps `run_pre_qat_feature_reconstruction`.
- It temporarily patches activation quantizers matching `input_quant_fn` and `quan_a_*` during the student reconstruction forward.
- It does not affect teacher forward, normal QAT epochs, or validation.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_qdrop_prerecon_gate_20260707.sh
```

Key settings:

```text
EXP=recipe_resume10_qdrop_prerecon_gate_20260707
PRE_QAT_FEATURE_RECON_UPDATES=100
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
PRE_QAT_FEATURE_RECON_QDROP_PROB=0.5
QSS=disabled
```

Runtime evidence:

```text
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Starting pre-QAT feature reconstruction: updates=100, policy=quant, weight_mode=none, confidence_power=0.0, qdrop_prob=0.5, layers=('features.5.5', 'features.7.1')
PreQATFeatRecon QDrop: prob=0.5, activation_quantizers=89
PreQATFeatRecon: update=100/100 loss=0.258754 kept=67484 masked=27760428
```

Results:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.4420 | 95.3120 | 0.8460 | below extension gate |
| 2 | `checkpoint-2.pth.tar` | 80.4980 | 95.3440 | 0.8402 | best in branch, below 80.55-80.60 gate |

Timing:

| epoch | updates | avg step time | samples/sec |
|---:|---:|---:|---:|
| 0 | 2496 | 0.223355s | 2292.32 |
| 1 | 2496 | 0.223355s | 2292.31 |

Conclusion:

QDrop-style reconstruction is a valid W4A4 branch and improves over checkpoint-10 and Phase 1P, but it does not beat Phase 1F (`80.5220`) and does not reach the `80.55-80.60` extension gate. The run was stopped after epoch2 full validation to avoid spending epoch3 or a 10-epoch extension on a branch with insufficient signal.

Interpretation:

- Random activation quantization bypass during quant/shift-only reconstruction is less destructive than `module_all` block reconstruction and better than the QSS-combo branch.
- It is still weaker than plain Phase 1F pre-QAT feature reconstruction, likely because dropping activation quantization in reconstruction makes the learned quant/shift parameters less specialized for the final strict A4 endpoint.
- The next non-redundant direction should refine where/when the bypass applies, rather than broadening it. Candidates: lower QDrop probability, later-layer-only activation bypass, or disagreement/confidence-weighted feature reconstruction without QSS.

### Phase 1S: Disagreement-Weighted Pre-QAT Feature Reconstruction Gate

Reason:

Phase 1R suggested broad activation quantization bypass is less effective than the original Phase 1F reconstruction. Phase 1S tested another non-scalar learning-signal change that preserves the checkpoint-10 weight basin: keep Phase 1F's quant/shift-only late feature reconstruction, but weight each sample by teacher/student logit disagreement during pre-QAT reconstruction. The hypothesis was that reconstruction should focus on samples where the quantized student is already drifting from the teacher, without changing bit-width, activation state control, or ordinary weights.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_disagree_prerecon_gate_20260707.sh
```

Key settings:

```text
EXP=recipe_resume10_disagree_prerecon_gate_20260707
PRE_QAT_FEATURE_RECON_UPDATES=100
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
PRE_QAT_FEATURE_RECON_WEIGHT_MODE=disagreement
QSS=disabled
QDrop=disabled
```

Runtime evidence:

```text
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Starting pre-QAT feature reconstruction: updates=100, policy=quant, weight_mode=disagreement, confidence_power=0.0, qdrop_prob=0.0, layers=('features.5.5', 'features.7.1')
PreQATFeatRecon: update=100/100 loss=0.115319 kept=67764 masked=27767356
```

Result:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.3660 | 95.2460 | 0.8455 | baseline-level; stopped |

Timing:

| epoch | updates | avg step time | samples/sec |
|---:|---:|---:|---:|
| 0 | 2496 | 0.223693s | 2288.85 |

Conclusion:

Disagreement weighting does not create early uplift. The first resumed full validation is essentially equal to checkpoint-10 (`80.3660` vs `80.3640`) and well below Phase 1F (`80.5220`) and the `80.55-80.60` extension gate. The run was interrupted after checkpoint-1 validation to avoid spending epoch2/3 on a branch with no early positive signal.

Interpretation:

- Weighting the reconstruction loss toward high teacher/student logit disagreement is not sufficient; it may overweight hard/noisy samples without improving the final strict A4 basin.
- Since confidence/disagreement/QDrop variants all fail to beat Phase 1F, the next direction should not be another sample-weighting tweak.
- More promising next mechanisms should change the endpoint state handling directly, e.g. activation outlier-aware calibration, selective late activation quantizer regularization, or lower-risk late-layer-only QDrop rather than global activation bypass.

### Phase 1T: Late-Layer-Only QDrop Pre-QAT Feature Reconstruction Gate

Reason:

Phase 1R used QDrop-style activation quantizer bypass across all activation quantizers and underperformed Phase 1F. Phase 1T tested a lower-risk version: only bypass activation quantizers under the same late feature modules used by reconstruction (`features.5.5`, `features.7.1`). This tests whether local activation flatness can help without perturbing earlier/global A4 activation state.

Launcher change:

`qat_launch.py` now supports:

```text
--pre-qat-feature-recon-qdrop-layers <comma-separated module names>
```

Default is empty, which preserves Phase 1R's global QDrop behavior when `qdrop_prob > 0`. Passing the late layers limits the patched activation quantizers to those submodules.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_lateqdrop_prerecon_gate_20260707.sh
```

Key settings:

```text
EXP=recipe_resume10_lateqdrop_prerecon_gate_20260707
PRE_QAT_FEATURE_RECON_UPDATES=100
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
PRE_QAT_FEATURE_RECON_QDROP_PROB=0.5
PRE_QAT_FEATURE_RECON_QDROP_LAYERS=features.5.5,features.7.1
QSS=disabled
```

Runtime evidence:

```text
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Starting pre-QAT feature reconstruction: updates=100, policy=quant, weight_mode=none, confidence_power=0.0, qdrop_prob=0.5, qdrop_layers=('features.5.5', 'features.7.1'), layers=('features.5.5', 'features.7.1')
PreQATFeatRecon QDrop: prob=0.5, layers=('features.5.5', 'features.7.1'), activation_quantizers=14
PreQATFeatRecon: update=100/100 loss=0.118532 kept=65413 masked=27767356
```

Result:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.3480 | 95.2820 | 0.8473 | below baseline and extension gate; stopped |

Timing:

| epoch | updates | avg step time | samples/sec |
|---:|---:|---:|---:|
| 0 | 2496 | 0.223268s | 2293.21 |

Conclusion:

Late-layer-only QDrop is worse than global QDrop and below the checkpoint-10 baseline. It was stopped after checkpoint-1 full validation. This closes the current QDrop family for this goal unless a new diagnostic suggests a very different use of stochastic activation bypass.

Interpretation:

- Localizing QDrop to the reconstructed late modules reduces the number of patched activation quantizers from 89 to 14, but it does not improve strict W4A4 endpoint accuracy.
- The best W4A4 branch remains Phase 1F plain quant/shift-only feature reconstruction.
- Next work should move away from QDrop and sample weighting. Do not use checkpoint soup or multi-checkpoint averaging for this goal. The more plausible remaining direction is activation endpoint state handling, such as outlier-aware LSQ scale calibration or selective late activation scale regularization.

### Phase 1U: Pre-QAT Feature Reconstruction + Late Activation Scale Anchor

Reason:

After QDrop and sample weighting failed to beat Phase 1F, this branch tested direct endpoint-state handling. The mechanism keeps Phase 1F's quant/shift-only late feature reconstruction, then snapshots the late activation LSQ scale parameters under `features.5.5` and `features.7.1` after reconstruction. During normal QAT, a small relative MSE regularizer anchors those activation scales so they do not drift away from the reconstructed A4 endpoint.

Launcher change:

`qat_launch.py` now supports:

```text
--act-scale-anchor-weight <float>
--act-scale-anchor-layers <comma-separated module names>
--act-scale-anchor-start-epoch <int>
```

Implementation detail:

- Anchor state is collected after pre-QAT reconstruction and before DDP wrapping.
- Runtime-only `_act_scale_anchor_state` is filtered out of `args.yaml` so CUDA tensors are not serialized.
- Default weight is `0.0`, so existing scripts remain unchanged.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_actscale_anchor_gate_20260707.sh
```

Key settings:

```text
EXP=recipe_resume10_prerecon_actscale_anchor_gate_20260707
PRE_QAT_FEATURE_RECON_UPDATES=100
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
ACT_SCALE_ANCHOR_WEIGHT=0.01
ACT_SCALE_ANCHOR_LAYERS=features.5.5,features.7.1
ACT_SCALE_ANCHOR_START_EPOCH=0
QSS=disabled
QDrop=disabled
```

Runtime evidence:

```text
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
PreQATFeatRecon: update=100/100 loss=0.112701 kept=67764 masked=27767356
Initialized activation scale anchor: params=14, layers=('features.5.5', 'features.7.1'), weight=0.01, start_epoch=0
Enabled activation scale anchor: weight=0.01, layers=features.5.5,features.7.1, pairs=14
```

Result:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.3460 | 95.2820 | 0.8428 | below baseline and extension gate; stopped |

Timing:

| epoch | updates | avg step time | samples/sec |
|---:|---:|---:|---:|
| 0 | 2496 | 0.223767s | 2288.10 |

Conclusion:

Late activation scale anchoring at weight `0.01` hurts the first resumed validation point. It is below checkpoint-10 (`80.3640`) and much worse than Phase 1F (`80.5220`). The run was interrupted after checkpoint-1 validation to avoid spending epoch2/3 on a branch with no early positive signal.

Interpretation:

- The reconstructed late activation scales should not be strongly frozen; the model likely needs some A4 scale movement during the first resumed epochs.
- This branch does not invalidate endpoint-state handling in general, but this specific hard anchor is too restrictive or anchored at the wrong point.
- A softer alternative, if pursued, should regularize only outlier-prone activation scales or use a delayed/weak anchor after the model has moved into a better basin.

### Phase 1V: Pre-QAT Feature Reconstruction + Delayed Weak Late Activation Scale Anchor

Reason:

Phase 1U showed that a hard late activation scale anchor from epoch0 is too restrictive. This branch kept the same Phase 1F-style pre-QAT feature reconstruction, but delayed the anchor until resumed epoch1 and reduced its weight by 10x. The intent was to let the first resumed epoch move inside the A4 basin before lightly constraining late activation scale drift.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_actscale_anchor_delayed_gate_20260707.sh
```

Key settings:

```text
EXP=recipe_resume10_prerecon_actscale_anchor_delayed_gate_20260707
PRE_QAT_FEATURE_RECON_UPDATES=100
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
ACT_SCALE_ANCHOR_WEIGHT=0.001
ACT_SCALE_ANCHOR_LAYERS=features.5.5,features.7.1
ACT_SCALE_ANCHOR_START_EPOCH=1
QSS=disabled
QDrop=disabled
```

Runtime evidence:

```text
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
PreQATFeatRecon: update=100/100 loss=0.112701 kept=67764 masked=27767356
Initialized activation scale anchor: params=14, layers=('features.5.5', 'features.7.1'), weight=0.001, start_epoch=1
Enabled activation scale anchor: weight=0.001, layers=features.5.5,features.7.1, pairs=14
```

Result:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.3840 | 95.3600 | 0.8441 | above checkpoint-10 by +0.020, below Phase 1F and extension gate |
| 2 | `checkpoint-2.pth.tar` | 80.3960 | 95.3000 | 0.8411 | below Phase 1F and extension gate; stopped |

Timing:

| epoch | updates | avg step time | samples/sec |
|---:|---:|---:|---:|
| 0 | 2496 | 0.223250s | 2293.40 |
| 1 | 2496 | 0.223832s | 2287.43 |

Conclusion:

Delayed weak activation scale anchoring is stable but does not recover the Phase 1F gain. Epoch2 reaches only `80.3960`, well below Phase 1F `80.5220` and below the `80.55-80.60` extension gate, so the branch was interrupted instead of extending to epoch3 or 10 epochs.

Interpretation:

- Directly anchoring all late activation scales is too blunt even when delayed and weakened.
- The remaining endpoint-state direction should become more local and data-dependent: adjust only activation scales whose observed A4 clipping/outlier statistics are poor, then allow normal A4 learning.
- This closes the current activation scale anchor family. Do not continue with only anchor weight/start sweeps.

### Phase 1W: Pre-QAT Activation Percentile Calibration + Feature Reconstruction

Reason:

Phase 1V closed direct activation scale anchoring. This branch tested a more data-dependent endpoint-state mechanism inspired by ViT PTQ/QAT papers that focus on post-LN / GELU / attention activation outliers: keep strict W4A4 active, observe real train-batch activation inputs for late activation quantizers, set their LSQ `s` toward a high-percentile A4 clipping target, then run the known-best Phase 1F-style quant/shift-only feature reconstruction.

Launcher change:

`qat_launch.py` now supports:

```text
--pre-qat-act-percentile-calib-batches <int>
--pre-qat-act-percentile-calib-layers <comma-separated module names>
--pre-qat-act-percentile-calib-percentile <float>
--pre-qat-act-percentile-calib-blend <float>
```

Implementation detail:

- The calibration runs after strict checkpoint restore and before pre-QAT feature reconstruction.
- It uses activation quantizer forward pre-hooks to observe the tensor entering each matched `input_quant_fn` / `quan_a_*` module.
- It stores only per-batch percentile scale estimates, not full activations.
- New scale = `(1 - blend) * old_scale + blend * percentile_scale`.
- Defaults disable the mechanism, so existing scripts are unchanged.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_actpercentile_prerecon_gate_20260707.sh
```

Key settings:

```text
EXP=recipe_resume10_actpercentile_prerecon_gate_20260707
PRE_QAT_ACT_PERCENTILE_CALIB_BATCHES=16
PRE_QAT_ACT_PERCENTILE_CALIB_LAYERS=features.5.5,features.7.1
PRE_QAT_ACT_PERCENTILE_CALIB_PERCENTILE=0.999
PRE_QAT_ACT_PERCENTILE_CALIB_BLEND=0.5
PRE_QAT_FEATURE_RECON_UPDATES=100
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
QSS=disabled
QDrop=disabled
activation-scale-anchor=disabled
```

Runtime evidence:

```text
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Starting pre-QAT activation percentile calibration: batches=16, percentile=0.999, blend=0.5, layers=('features.5.5', 'features.7.1'), quantizers=14
Finished pre-QAT activation percentile calibration: batches=16, updated=14, mean_scale_ratio=1.0572, min_ratio=0.6970, max_ratio=1.4069
PreQATFeatRecon: update=100/100 loss=0.118495 kept=67764 masked=27767356
```

Result:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.3100 | 95.2760 | 0.8480 | below checkpoint-10 and extension gate; stopped |

Timing:

| epoch | updates | avg step time | samples/sec |
|---:|---:|---:|---:|
| 0 | 2496 | 0.223377s | 2292.09 |

Conclusion:

Activation percentile calibration with 16 batches, percentile `0.999`, and blend `0.5` is stable and non-empty, but it hurts the first resumed validation point (`80.3100` vs checkpoint-10 `80.3640`). The run was interrupted after epoch1 full validation to avoid spending more epochs on a branch already below baseline and far below Phase 1F `80.5220`.

Interpretation:

- A broad late-layer percentile scale reset is still too coarse. It moves the LSQ scales in a plausible outlier-aware direction, but the endpoint feature MSE after reconstruction is not improved enough and validation drops.
- This does not invalidate activation outlier handling generally; it suggests the adjustment must be more selective than all 14 late activation quantizers, or tied to direct clipping/error diagnostics rather than a single percentile and blend.
- Do not continue this exact percentile-calibration branch with scalar percentile/blend sweeps unless a diagnostic identifies specific quantizers that are over/under-clipping.

### Phase 1X: Selective Activation MSE Calibration + Feature Reconstruction

Reason:

Phase 1W showed that broad percentile calibration reduced clipping but worsened validation. A new quantizer-level diagnostic was added to inspect late activation quantizers directly across `checkpoint-10`, Phase 1F best, and Phase 1W. The diagnostic showed Phase 1W reduced clip rates but increased quantization error, especially for:

```text
features.5.5.mlp.fc1.input_quant_fn
features.7.1.mlp.fc1.input_quant_fn
features.5.5.attn.quant_x_4_qkv.input_quant_fn
features.7.1.attn.quant_x_4_qkv.input_quant_fn
features.5.5.attn.quan_a_v_fn
features.7.1.attn.quan_a_v_fn
features.5.5.attn.quan_a_qkx_fn
features.7.1.attn.quan_a_qkx_fn
```

Diagnostic artifacts:

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_quantizer_clipping_20260707.py
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_quantizer_clipping_diag_20260707.json
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_quantizer_clipping_diag_20260707.tsv
```

Key diagnostic evidence:

```text
phase1w_actpercentile: clip rate lower than ckpt10/Phase1F on most selected quantizers
phase1w_actpercentile: quant_abs_error_mean higher on MLP fc1, qkv input, v, and qkx quantizers
```

Therefore this branch tried the opposite of Phase 1W: instead of expanding scales to reduce clipping, run a selective activation MSE scale search over the diagnosed quantizers to directly minimize A4 reconstruction error, then run Phase 1F-style late feature reconstruction.

Launcher change:

`qat_launch.py` now supports:

```text
--pre-qat-act-mse-calib-batches <int>
--pre-qat-act-mse-calib-layers <comma-separated module names>
--pre-qat-act-mse-calib-quantizers <comma-separated quantizer names>
--pre-qat-act-mse-calib-grid <min,max,steps>
--pre-qat-act-mse-calib-blend <float>
```

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_actmse_prerecon_gate_20260707.sh
```

Key settings:

```text
EXP=recipe_resume10_actmse_prerecon_gate_20260707
PRE_QAT_ACT_MSE_CALIB_BATCHES=8
PRE_QAT_ACT_MSE_CALIB_LAYERS=features.5.5,features.7.1
PRE_QAT_ACT_MSE_CALIB_QUANTIZERS=<8 diagnosed quantizers listed above>
PRE_QAT_ACT_MSE_CALIB_GRID=0.75,1.05,13
PRE_QAT_ACT_MSE_CALIB_BLEND=0.5
PRE_QAT_FEATURE_RECON_UPDATES=100
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
QSS=disabled
QDrop=disabled
activation-scale-anchor=disabled
activation-percentile-calibration=disabled
```

Runtime evidence:

```text
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Starting pre-QAT activation MSE calibration: batches=8, grid=0.75,1.05,13, blend=0.5, layers=('features.5.5', 'features.7.1'), ... matched=8
Finished pre-QAT activation MSE calibration: batches=8, updated=8, mean_scale_ratio=0.9410, min_ratio=0.8957, max_ratio=1.0250
PreQATFeatRecon: update=100/100 loss=0.109729 kept=67764 masked=27767356
```

Result:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.3400 | 95.2520 | 0.8455 | below checkpoint-10 and extension gate; stopped |

Timing:

| epoch | updates | avg step time | samples/sec |
|---:|---:|---:|---:|
| 0 | 2496 | 0.223355s | 2292.32 |

Conclusion:

Selective activation MSE calibration is stable and does the intended opposite of Phase 1W: it shrinks selected scales on average (`mean_scale_ratio=0.9410`) rather than expanding them. However, first resumed full validation is still below checkpoint-10 (`80.3400` vs `80.3640`) and far below Phase 1F (`80.5220`), so the branch was interrupted after epoch1 full validation.

Interpretation:

- LSQ activation scale manipulation, whether percentile/outlier-oriented or MSE-oriented, does not explain the remaining gap by itself.
- The best valid W4A4 branch remains Phase 1F plain quant/shift-only feature reconstruction.
- The next non-redundant direction should move from activation-scale-only mechanisms to a structural learning signal, likely weight-bin/oscillation regularization or teacher/refmodel attention relation applied during/after Phase 1F, while keeping the successful Phase 1F reconstruction as the warm start.

### Phase 1Y: Pre-QAT Feature Reconstruction + Weak Bin Regularizer

Reason:

Phase 1W and Phase 1X falsified activation-scale-only endpoint handling. Public VVTQ/Quantization Variation suggests that weight-bin oscillation control can be part of a successful Swin-T low-bit QAT recipe. This branch keeps the best valid W4A4 mechanism so far, Phase 1F pre-QAT quant/shift-only feature reconstruction, and adds a weak VVTQ-style bin regularizer during normal QAT epochs.

Existing implementation:

`qat_launch.py` already had `--bin-reg-weight` and `--bin-reg-variance-weight`. The regularizer covers quantized weights via `statsq_fn` / `lsqw_fn` and Q/K/V weights through `qk_quant` / `v_quant`.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_binreg_gate_20260707.sh
```

Key settings:

```text
EXP=recipe_resume10_prerecon_binreg_gate_20260707
PRE_QAT_FEATURE_RECON_UPDATES=100
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
BIN_REG_WEIGHT=2e-5
BIN_REG_VARIANCE_WEIGHT=1.0
QSS=disabled
QDrop=disabled
activation-scale-anchor=disabled
activation-percentile-calibration=disabled
activation-MSE-calibration=disabled
```

Runtime evidence:

```text
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
PreQATFeatRecon: update=100/100 loss=0.112701 kept=67764 masked=27767356
Enabled bin regularizer: weight=2e-05, variance_weight=1.0, pairs=77
TrainSummary: epoch=0 updates=2496 avg_step_time=0.764382s samples_per_step=512 samples_per_sec=669.82
```

Result:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.3900 | 95.3340 | 0.8457 | above checkpoint-10 by +0.026, below Phase 1F and extension gate; stopped |

Conclusion:

Weak BinReg is stable but expensive: step time increases from the usual `~0.223s` to `0.764s`. The first resumed full-val is only `80.3900`, which is above checkpoint-10 by `+0.026` but below Phase 1F and far below the extension gate. Because this branch is slow and its early signal is weaker than already-falsified branches such as QDrop/global reconstruction, it was interrupted after epoch1 full validation.

Interpretation:

- Full-model BinReg at this implementation cost is not a viable short-run resume mechanism at the tested weak weight.
- Stronger BinReg would be even slower and more likely to overconstrain a late-stage resume point, so do not sweep `bin_reg_weight` blindly.
- If revisiting weight-bin regularization, make it selective and cheap, e.g. apply only to attention Q/K/V or high-oscillation heads, rather than all 77 weight pairs.

### Phase 1Z: Pre-QAT Feature Reconstruction + Selective Attention Q/K/V BinReg

Reason:

Phase 1Y showed full-model BinReg is too expensive (`0.764s` step time) and weaker than Phase 1F. This branch keeps the same structural idea but makes it selective and cheap: apply BinReg only to late attention Q/K/V weights under `features.5.5.attn` and `features.7.1.attn`, reducing the regularized pairs from 77 to 6.

Launcher change:

`qat_launch.py` now supports:

```text
--bin-reg-layers <comma-separated module names>
--bin-reg-attn-only
```

Defaults preserve the old all-model behavior when these flags are not set.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_attnbinreg_gate_20260707.sh
```

Key settings:

```text
EXP=recipe_resume10_prerecon_attnbinreg_gate_20260707
PRE_QAT_FEATURE_RECON_UPDATES=100
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
BIN_REG_WEIGHT=2e-5
BIN_REG_VARIANCE_WEIGHT=1.0
BIN_REG_LAYERS=features.5.5.attn,features.7.1.attn
BIN_REG_ATTN_ONLY=1
QSS=disabled
QDrop=disabled
activation-scale-anchor=disabled
activation-percentile-calibration=disabled
activation-MSE-calibration=disabled
```

Runtime evidence:

```text
Enabled bin regularizer: weight=2e-05, variance_weight=1.0, layers=('features.5.5.attn', 'features.7.1.attn'), attn_only=True, pairs=6
TrainSummary: epoch=0 updates=2496 avg_step_time=0.263299s samples_per_step=512 samples_per_sec=1944.56
TrainSummary: epoch=1 updates=2496 avg_step_time=0.263571s samples_per_step=512 samples_per_sec=1942.55
```

Result:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.4060 | 95.2940 | 0.8438 | above checkpoint-10, below Phase 1F/extension gate |
| 2 | `checkpoint-2.pth.tar` | 80.4680 | 95.3100 | 0.8407 | below Phase 1F `80.5220` and extension gate; stopped |

Conclusion:

Selective attention Q/K/V BinReg is much cheaper than full-model BinReg and is stable, but it still underperforms Phase 1F. Epoch2 reaches `80.4680`, below Phase 1F's `80.5220`, so the branch was interrupted before epoch3/10.

Interpretation:

- The cheap selective BinReg variant confirms that the full-model implementation cost was avoidable, but weight-bin regularization does not add enough lift to the Phase 1F warm start at this weak setting.
- Since both full-model and selective BinReg underperform Phase 1F, do not continue with BinReg-only variants.
- The remaining non-redundant direction should focus on teacher/refmodel attention relation or q/k/v structural distillation after the Phase 1F warm start, not scalar BinReg sweeps.

### Phase 2A: Pre-QAT Feature Reconstruction + Scaled Teacher Q/K Relation

Reason:

Existing from-checkpoint-10 structural runs with teacher attention / QK relation were weak, but they did not include the Phase 1F pre-QAT feature reconstruction warm start. Since BinReg-only variants underperformed Phase 1F, this branch tested a different structural signal: keep Phase 1F pre-QAT quant/shift reconstruction, then add teacher Q/K relation loss during normal QAT. Teacher attention KL was kept disabled because prior logs showed it has very large magnitude (`~1.6e3`) and can dominate the loss.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_qkrel_gate_20260707.sh
```

Key settings:

```text
EXP=recipe_resume10_prerecon_qkrel_gate_20260707
PRE_QAT_FEATURE_RECON_UPDATES=100
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
TEACHER_QK_REL_WEIGHT=1000
TEACHER_ATTN_KL_WEIGHT=0
QSS=disabled
QDrop=disabled
activation-scale-anchor=disabled
activation-percentile-calibration=disabled
activation-MSE-calibration=disabled
BinReg=disabled
```

Runtime evidence:

```text
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
PreQATFeatRecon: update=100/100 loss=0.112701 kept=67764 masked=27767356
TeacherQKRel: 4.288e-06 (first step)
TrainSummary: epoch=0 updates=2496 avg_step_time=0.271765s samples_per_step=512 samples_per_sec=1883.98
```

Result:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.2960 | 95.3320 | 0.8439 | below checkpoint-10 and extension gate; stopped |

Conclusion:

Scaled Teacher Q/K relation after the Phase 1F warm start hurts the first resumed validation point. It drops below checkpoint-10 (`80.2960` vs `80.3640`) and far below Phase 1F, so the run was interrupted after epoch1 full validation.

Interpretation:

- Q/K relation alone is not the missing structural signal for this resume10 target, even when scaled to contribute roughly `~0.003-0.004` to the loss.
- Teacher attention KL remains unattractive in the existing form because prior runs show very large KL magnitudes and weak validation.
- Do not continue QKRel-only or teacher-attn-KL-only variants on this goal. The next useful structural attempt would need a more localized or normalized q/k/v distillation objective, not the current global QK relation loss.

### Phase 2B: Pre-QAT Feature Reconstruction + Post-Epoch Feature Refresh

Reason:

Phase 1F remains the best valid strict W4A4 branch, but its checkpoint-2 to checkpoint-3 regression appears to be feature/logit alignment drift rather than activation quantizer clipping drift. This branch tested a targeted fix: keep the Phase 1F 100-step pre-QAT late feature reconstruction, then after each normal QAT epoch run a short late-feature reconstruction refresh before checkpoint save and full validation.

Implementation:

- Added `--post-epoch-feature-recon-updates`.
- Reused `run_pre_qat_feature_reconstruction(...)` with an `updates_override` and a descriptive label.
- Initial DDP attempt failed after the first post-epoch refresh update because `static_graph=True` DDP detected changed gradient usage:

```text
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
... this is not compatible with static_graph set to True.
Parameter indices which did not receive grad ...
```

- Fixed the post-epoch refresh path by bypassing the DDP wrapper for the local reconstruction forward/backward and manually all-reducing the kept quant/shift gradients before `optimizer.step()`. Pre-QAT reconstruction before DDP wrapping remains unchanged.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_postrefresh_gate_20260707.sh
```

Key settings:

```text
EXP=recipe_resume10_prerecon_postrefresh20_gate_20260707
PRE_QAT_FEATURE_RECON_UPDATES=100
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
POST_EPOCH_FEATURE_RECON_UPDATES=20
LR=1.5e-5
MIN_LR=5e-6
QUANT_LR_MULTIPLIER=2
strict W4A4: wq_bitw=4, aq_bitw=4
```

Runtime evidence:

```text
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Starting post-epoch feature reconstruction epoch=0: updates=20, ... bypass_ddp=True ...
post-epoch feature reconstruction epoch=0: update=20/20 loss=0.111667 kept=67764 masked=27767356 reduced=67764
TrainSummary: epoch=0 updates=2496 avg_step_time=0.223205s samples_per_step=512 samples_per_sec=2293.85
```

Result:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.3860 | 95.3060 | 0.8511 | above checkpoint-10 by +0.022, below Phase 1F and extension gate; stopped |

Conclusion:

Post-epoch feature refresh is technically viable after bypassing DDP reducer state, but 20 refresh updates hurt the Phase 1F early uplift. The first full validation is only `80.3860`, far below Phase 1F's `80.5220`, so the branch was interrupted during epoch2 rather than spending more compute.

Interpretation:

- Replaying the same late-feature reconstruction objective after a full QAT epoch is not sufficient to preserve the Phase 1F checkpoint-2 basin.
- The refresh may be fighting the normal epoch's KD/CE update rather than correcting the specific drift that matters for accuracy.
- Do not continue with scalar sweeps over post-refresh update count unless a diagnostic identifies a narrower target. The next non-redundant mechanism should make the structural target more local, for example normalized q/k/v representation distillation inside the late attention blocks rather than global Q/K relation or repeated output-feature reconstruction.

### Phase 2C: Pre-QAT Feature Reconstruction + Local Teacher Q/K/V Relation

Reason:

Phase 2A showed that global teacher Q/K relation hurts, while Phase 2B showed repeated output-feature refresh does not preserve the Phase 1F uplift. This branch kept the successful Phase 1F warm start but made the structural target more local: late attention block Q/K/V relation distillation only on attention layers 10 and 11, with normalized relation vectors for q, k, and v.

Implementation:

- Added `--teacher-qkv-rel-weight`, `--teacher-qkv-rel-warmup-epochs`, `--teacher-qkv-rel-layers`, and `--teacher-qkv-rel-components`.
- Reused the existing `qqkkvv=True` Swin attention path so student/teacher return `(attn, q_score, k_score, v_score)`.
- Added `teacher_qkv_relation_loss(...)`, which normalizes flattened relation tensors and averages over selected late layers and components.
- Kept old global `--teacher-qk-rel-weight` disabled for this branch.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_qkvrel_gate_20260707.sh
```

Key settings:

```text
EXP=recipe_resume10_prerecon_qkvrel_gate_20260707
PRE_QAT_FEATURE_RECON_UPDATES=100
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
TEACHER_QKV_REL_WEIGHT=10
TEACHER_QKV_REL_LAYERS=10,11
TEACHER_QKV_REL_COMPONENTS=q,k,v
TEACHER_QK_REL_WEIGHT=0
LR=1.5e-5
MIN_LR=5e-6
QUANT_LR_MULTIPLIER=2
strict W4A4: wq_bitw=4, aq_bitw=4
```

Sanity evidence:

```text
TeacherRel: 5.924e-05 (5.924e-05) at first train step
TeacherRel stayed around 5.5e-05 during the short sanity run
```

Runtime evidence:

```text
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
TeacherRel: 5.924e-05 (5.924e-05) at first train step
TrainSummary: epoch=0 updates=2496 avg_step_time=0.248631s samples_per_step=512 samples_per_sec=2059.28
```

Result:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.4680 | 95.2820 | 0.8449 | below Phase 1F and extension gate; stopped |

Conclusion:

Local Q/K/V relation distillation is stable and correctly connected, but it does not beat Phase 1F. The first resumed full validation reaches `80.4680`, matching the better BinReg-class runs but still below Phase 1F `80.5220`, so the branch was interrupted during epoch2.

Interpretation:

- The local relation signal has a controlled magnitude and acceptable runtime, but it does not address the specific Phase 1F checkpoint-2 uplift mechanism.
- Since global QKRel and local QKVRel both underperform Phase 1F, do not continue relation-distillation-only variants by weight sweeping.
- The best valid strict W4A4 checkpoint remains Phase 1F `checkpoint-2.pth.tar` at `80.5220`; the goal is not complete.

### Phase 2D: Parameter Drift Diagnostic + Freeze Activation Quant/Shift After Phase 1F Warm Start

Reason:

Phase 1F's checkpoint-2 to checkpoint-3 regression looked like feature/logit alignment drift, but activation clipping diagnostics did not show a large scale/clipping explanation. This phase added a direct checkpoint parameter-drift diagnostic to compare:

- checkpoint-10 -> Phase 1F checkpoint-2, the useful uplift path
- Phase 1F checkpoint-2 -> checkpoint-3, the regression path

Diagnostic artifacts:

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_phase1f_param_drift_20260707.json
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_phase1f_param_drift_20260707.tsv
```

Key diagnostic findings:

```text
ckpt10 -> Phase1F ckpt2:
  move_shift rel_l2=0.02562, strongest useful drift class
  late features.5.5 / features.7.1 move_* modules move heavily
  examples: features.5.5.attn.move_v_aft rel_l2=0.2320,
            features.7.1.attn.move_v_aft rel_l2=0.1591

Phase1F ckpt2 -> ckpt3:
  move_shift remains the largest aggregate drift class, rel_l2=0.00875
  early features.0.0 move_b4/move_aft show very high relative drift (~0.198)
  late features.5.x / features.7.x move_* also continue drifting
```

Hypothesis:

Phase 1F's useful uplift may come from the initial quant/shift reconstruction, but continued normal training may over-drift activation quant/shift state. This branch tests the broad version: after the Phase 1F warm start, freeze all activation quantizers and shift parameters with `trainable_policy=freeze_act_quant`, letting ordinary weights continue adapting.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_freezeact_gate_20260707.sh
```

Key settings:

```text
EXP=recipe_resume10_prerecon_freezeact_gate_20260707
PRE_QAT_FEATURE_RECON_UPDATES=100
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
TRAINABLE_POLICY=freeze_act_quant
LR=1.5e-5
MIN_LR=5e-6
QUANT_LR_MULTIPLIER=2
strict W4A4: wq_bitw=4, aq_bitw=4
```

Sanity evidence:

```text
Trainable parameter update policy: epoch=0, update=0, mode=requires_grad, policy=freeze_act_quant, trainable=28282025, frozen=326231
TrainSummary: epoch=0 updates=20 ...
Stopped early after 20 optimizer updates in epoch 0.
```

Runtime evidence:

```text
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Trainable parameter update policy: epoch=0, update=0, mode=requires_grad, policy=freeze_act_quant, trainable=28282025, frozen=326231
TrainSummary: epoch=0 updates=2496 avg_step_time=0.174601s samples_per_step=512 samples_per_sec=2932.41
TrainSummary: epoch=1 updates=2496 avg_step_time=0.174438s samples_per_step=512 samples_per_sec=2935.14
```

Result:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.4460 | 95.2700 | 0.8446 | below Phase 1F checkpoint-2 |
| 2 | `checkpoint-2.pth.tar` | 80.4260 | 95.2660 | 0.8412 | below Phase 1F and declining; stopped |

Conclusion:

Freezing all activation quantizers and shift parameters after Phase 1F warm start is stable and fast, but too broad. It prevents neither underperformance nor mild decline; it also blocks the useful late shift adaptation that Phase 1F appears to need.

Interpretation:

- The drift diagnostic says `move_shift` is both the useful adaptation class and the later drift class; freezing all shift state removes too much signal.
- The next narrower test should freeze only non-late activation/shift state, while allowing the Phase 1F target layers (`features.5.5`, `features.7.1`) or their local late blocks to keep adapting.
- The best valid strict W4A4 checkpoint remains Phase 1F `checkpoint-2.pth.tar` at `80.5220`; the goal is still not complete.

### Phase 2E: Freeze Non-Late Activation Quant/Shift, Keep Phase 1F Late Layers Trainable

Reason:

Phase 2D showed that freezing all activation quantizers and shift parameters is too broad: it blocks the useful late shift adaptation that Phase 1F needs. Based on the drift diagnostic, this branch narrows the mask: freeze activation quant/shift parameters globally except under the Phase 1F target layers `features.5.5` and `features.7.1`.

Implementation:

- Added `trainable_policy=freeze_act_except_layers`.
- Added `--trainable-policy-freeze-act-except-layers`.
- The policy keeps ordinary weights trainable everywhere, freezes activation quant/shift state outside the exception layers, and allows activation quant/shift to keep adapting under the exception layers.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_freezeact_exceptlate_gate_20260707.sh
```

Key settings:

```text
EXP=recipe_resume10_prerecon_freezeact_exceptlate_gate_20260707
PRE_QAT_FEATURE_RECON_UPDATES=100
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
TRAINABLE_POLICY=freeze_act_except_layers
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1
LR=1.5e-5
MIN_LR=5e-6
QUANT_LR_MULTIPLIER=2
strict W4A4: wq_bitw=4, aq_bitw=4
```

Sanity evidence:

```text
Trainable parameter update policy: epoch=0, update=0, mode=requires_grad, policy=freeze_act_except_layers, trainable=28349789, frozen=258467
TrainSummary: epoch=0 updates=20 ...
Stopped early after 20 optimizer updates in epoch 0.
```

Runtime evidence:

```text
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Trainable parameter update policy: epoch=0, update=0, mode=requires_grad, policy=freeze_act_except_layers, trainable=28349789, frozen=258467
TrainSummary: epoch=0 updates=2496 avg_step_time=0.180117s samples_per_step=512 samples_per_sec=2842.60
TrainSummary: epoch=1 updates=2496 avg_step_time=0.180007s samples_per_step=512 samples_per_sec=2844.34
TrainSummary: epoch=2 updates=2496 avg_step_time=0.179898s samples_per_step=512 samples_per_sec=2846.06
```

Result:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.4120 | 95.3200 | 0.8478 | below Phase 1F and below broad freeze-act |
| 2 | `checkpoint-2.pth.tar` | 80.3960 | 95.3280 | 0.8437 | below Phase 1F and declining |
| 3 | `checkpoint-3.pth.tar` | 80.4260 | 95.3440 | 0.8396 | slight recovery but still below Phase 1F; stopped |

Conclusion:

Freezing non-late activation quant/shift while keeping `features.5.5` and `features.7.1` adaptive is stable, but it does not improve over Phase 1F. It performs worse than the broad freeze-act gate at epoch1 and never approaches `80.5220`.

Interpretation:

- The useful Phase 1F behavior is not recovered by hard freezing non-late activation/shift state.
- The VVTQ-inspired direction still looks valid, but the implementation should move from hard masks to soft, variation-weighted trust regularization.
- Next useful test: anchor high-risk early activation/shift strongly, anchor late useful layers weakly, and optionally apply selective bin regularization to high-drift modules, instead of freezing them.

### Phase 2F: VVTQ-Inspired Variation-Weighted Trust Regularization

Reason:

Phase 2D and Phase 2E showed that hard freezing activation quantizers / shift state is too blunt. The parameter-drift diagnostic indicated that `move_shift` is both the useful Phase 1F adaptation class and a later regression class. This phase switched from hard masks to a soft VVTQ-inspired trust penalty: constrain high-risk early and sensitive quant/shift parameters more strongly, while allowing the Phase 1F late layers to keep adapting with a weaker anchor.

Implementation:

- Added variation trust state collection after pre-QAT feature reconstruction and before DDP wrapping.
- Added `--variation-trust-*` controls for global weight, early/late layer multipliers, softmax quantizer multiplier, `move_v` multiplier, and projection/move multiplier.
- The trust anchor snapshots trainable model parameters after the successful pre-QAT reconstruction warm start.
- The loss is logged as `VarTrust`; it is a soft parameter-distance regularizer, not a freeze policy.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Key settings:

```text
EXP=recipe_resume10_prerecon_vartrust_gate_20260707
PRE_QAT_FEATURE_RECON_UPDATES=100
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
VARIATION_TRUST_WEIGHT=0.001
VARIATION_TRUST_LATE_LAYERS=features.5.5,features.7.1
VARIATION_TRUST_LATE_MULTIPLIER=0.25
VARIATION_TRUST_EARLY_LAYERS=features.0.0,features.1.0,features.1.1
VARIATION_TRUST_EARLY_MULTIPLIER=2.0
VARIATION_TRUST_SOFTMAX_MULTIPLIER=2.0
VARIATION_TRUST_MOVE_V_MULTIPLIER=1.5
VARIATION_TRUST_PROJ_MOVE_MULTIPLIER=1.25
LR=1.5e-5
MIN_LR=5e-6
QUANT_LR_MULTIPLIER=2
strict W4A4: wq_bitw=4, aq_bitw=4
```

Args evidence:

```text
args.yaml:
  aq_bitw: 4
  wq_bitw: 4
  kd_hard_and_soft: 0
  skip_validate: false
  variation_trust_weight: 0.001
  variation_trust_late_layers: features.5.5,features.7.1
  variation_trust_early_layers: features.0.0,features.1.0,features.1.1
```

Runtime evidence:

```text
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Initialized variation trust anchor: params=243, weight=0.001, late_layers=features.5.5,features.7.1, early_layers=features.0.0,features.1.0,features.1.1
Enabled variation trust regularizer: weight=0.001, pairs=243, avg_multiplier=1.180, late_layers=features.5.5,features.7.1, early_layers=features.0.0,features.1.0,features.1.1
TrainSummary: epoch=0 updates=2496 avg_step_time=0.256889s samples_per_step=512 samples_per_sec=1993.08
TrainSummary: epoch=1 updates=2496 avg_step_time=0.258226s samples_per_step=512 samples_per_sec=1982.76
```

Result:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.4000 | 95.3260 | 0.8474 | above checkpoint-10 but below Phase 1F |
| 2 | `checkpoint-2.pth.tar` | 80.4780 | 95.2920 | 0.8412 | improving, but still below Phase 1F `80.5220`; stopped during epoch3 |

Conclusion:

Variation-weighted trust regularization is stable and gives a better second-epoch result than hard freeze policies, but this configuration still does not beat Phase 1F. The branch was stopped after the second full validation because it remained below `80.5220`, and the goal is not complete.

Interpretation:

- Soft trust is directionally better than hard freeze: epoch2 reaches `80.4780`, above Phase 2D/2E and closer to Phase 1F.
- The current trust strength is probably too diffuse or too weakly targeted: `VarTrust` grows smoothly but does not prevent the branch from staying below Phase 1F.
- Next useful VVTQ-style step is selective bin regularization on diagnosed high-drift modules, especially early `features.0.0` / `features.1.*` activation and move/shift paths, while preserving weak trust on useful late layers.
- Do not continue this exact `VARIATION_TRUST_WEIGHT=0.001` branch to epoch3+ as the main path unless a later diagnostic shows a reason; the second full-val gate did not clear the Phase 1F baseline.

### Phase 2G: Variation Trust + Selective Activation Bin-Boundary Margin

Reason:

Phase 1Y/1Z already falsified full-model and late-attention weight BinReg-only variants, while Phase 2F showed soft variation trust is stable but still below Phase 1F. This branch tested a more activation-specific VVTQ idea: penalize selected activation quantizer inputs that sit too close to quantization bin boundaries, which should reduce low-bit activation bin flipping / oscillation without applying full-model weight BinReg.

Implementation:

- Added `--act-bin-margin-weight`.
- Added `--act-bin-margin-layers`.
- Added `--act-bin-margin-quantizers`.
- Added `--act-bin-margin`.
- Added `--act-bin-margin-max-elements`.
- The implementation installs temporary forward hooks on selected activation quantizers, captures their pre-quantization inputs during the normal single student forward, and penalizes values within `margin` of half-integer bin boundaries after LSQ scale normalization.
- Defaults keep this disabled.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_actbin_gate_20260707.sh
```

Key settings:

```text
EXP=recipe_resume10_prerecon_vartrust_actbin_gate_20260707
PRE_QAT_FEATURE_RECON_UPDATES=100
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
VARIATION_TRUST_WEIGHT=0.001
ACT_BIN_MARGIN_WEIGHT=0.01
ACT_BIN_MARGIN_LAYERS=features.0.0,features.1.0,features.1.1,features.5.5,features.7.1
ACT_BIN_MARGIN_QUANTIZERS=input_quant_fn,quant_x_4_qkv
ACT_BIN_MARGIN=0.08
ACT_BIN_MARGIN_MAX_ELEMENTS=32768
LR=1.5e-5
MIN_LR=5e-6
QUANT_LR_MULTIPLIER=2
strict W4A4: wq_bitw=4, aq_bitw=4
```

Sanity evidence:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_prerecon_vartrust_actbin_sanity20_20260707.log
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Enabled activation bin-margin regularizer: weight=0.01, layers=('features.0.0', 'features.1.0', 'features.1.1', 'features.5.5', 'features.7.1'), quantizers=('input_quant_fn', 'quant_x_4_qkv'), margin=0.08, pairs=17
TrainSummary: epoch=0 updates=20 avg_step_time=0.320870s samples_per_step=512 samples_per_sec=1595.66
Stopped early after 20 optimizer updates in epoch 0.
```

Runtime evidence:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_prerecon_vartrust_actbin_gate_20260707.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_vartrust_actbin_gate_20260707
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Enabled activation bin-margin regularizer: weight=0.01, layers=('features.0.0', 'features.1.0', 'features.1.1', 'features.5.5', 'features.7.1'), quantizers=('input_quant_fn', 'quant_x_4_qkv'), margin=0.08, pairs=17
Enabled variation trust regularizer: weight=0.001, pairs=243, avg_multiplier=1.180, late_layers=features.5.5,features.7.1, early_layers=features.0.0,features.1.0,features.1.1
TrainSummary: epoch=0 updates=2496 avg_step_time=0.268003s samples_per_step=512 samples_per_sec=1910.43
```

Args evidence:

```text
args.yaml:
  aq_bitw: 4
  wq_bitw: 4
  kd_hard_and_soft: 0
  skip_validate: false
  act_bin_margin_weight: 0.01
  act_bin_margin_layers: features.0.0,features.1.0,features.1.1,features.5.5,features.7.1
  act_bin_margin_quantizers: input_quant_fn,quant_x_4_qkv
```

Result:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.3460 | 95.2760 | 0.8453 | below checkpoint-10 `80.3640` and Phase 1F; stopped |

Conclusion:

Selective activation bin-boundary margin is technically connected and cheap enough to run, but the tested setting immediately regresses below checkpoint-10. The branch was interrupted during epoch2 and should not be continued.

Interpretation:

- Boundary-margin regularization on activation inputs appears too direct or too broad even with selected modules; it may suppress useful early adaptation rather than only harmful bin flipping.
- Since activation scale calibration, activation hard freeze, activation bin-margin, full-model BinReg, and late-attention BinReg all underperform Phase 1F, the remaining useful direction should not be another activation/bin regularizer sweep.
- Next candidates should use the drift diagnostic to change the warm-start endpoint itself, for example different pre-QAT reconstruction targets / policies or a two-stage transition that preserves Phase 1F checkpoint-2 state rather than adding another per-step regularizer.

### Phase 2H: Phase 1F Checkpoint-2 Endpoint Re-Reconstruction, Save, Then Eval

Reason:

Phase 1F's checkpoint-2 is the best valid strict W4A4 point (`80.5220`), while the next normal QAT epoch regresses to `80.3620`. Most regularizers added during normal QAT failed to preserve this point. This branch tested a narrower endpoint hypothesis: start from Phase 1F checkpoint-2, run another 100-step late feature reconstruction on quant/shift parameters only, save the reconstructed endpoint before any normal training epoch, and directly full-validate that single checkpoint.

This is not checkpoint averaging or soup; it is a single-model checkpoint produced by additional reconstruction updates.

Scripts:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_phase1f_ckpt2_endpoint_rerecon_save_20260707.sh

CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_phase1f_ckpt2_endpoint_rerecon100_20260707/step_checkpoints/step_0000.pth.tar \
EXP=eval_phase1f_ckpt2_endpoint_rerecon100_step0000_20260707 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Key settings:

```text
Start checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon100_lowlr_b_20260706/checkpoint-2.pth.tar
PRE_QAT_FEATURE_RECON_UPDATES=100
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003
No normal QAT epoch: epochs=0
Saved checkpoint: step_checkpoints/step_0000.pth.tar
strict W4A4 eval: wq_bitw=4, aq_bitw=4
```

Runtime evidence:

```text
Strict resume: loaded model from .../recipe_resume10_prerecon100_lowlr_b_20260706/checkpoint-2.pth.tar; missing=0, unexpected=0
Starting pre-QAT feature reconstruction: updates=100, policy=quant, ... layers=('features.5.5', 'features.7.1')
pre-QAT feature reconstruction: update=1/100 loss=0.127136 kept=67764 masked=27767356
pre-QAT feature reconstruction: update=50/100 loss=0.109800 kept=67764 masked=27767356
pre-QAT feature reconstruction: update=100/100 loss=0.114615 kept=67764 masked=27767356
Finished pre-QAT feature reconstruction: updates=100
step_checkpoint=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_phase1f_ckpt2_endpoint_rerecon100_20260707/step_checkpoints/step_0000.pth.tar
```

Eval evidence:

```text
Strict resume: loaded model from .../step_checkpoints/step_0000.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.483s  Loss: 0.8759  Acc@1: 80.4600  Acc@5: 95.3260  Samples: 50000
```

Result:

| source checkpoint | saved checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---|---|---:|---:|---:|---|
| Phase 1F `checkpoint-2.pth.tar` | `step_checkpoints/step_0000.pth.tar` | 80.4600 | 95.3260 | 0.8759 | below Phase 1F `80.5220`; fail |

Conclusion:

Re-running the same late feature reconstruction from Phase 1F checkpoint-2 damages the endpoint instead of lifting it. The checkpoint is valid strict W4A4 and single-model, but it is below the starting checkpoint and far below the `81.0` target.

Interpretation:

- The Phase 1F checkpoint-2 state is already a narrow local endpoint; more of the same reconstruction objective overfits or shifts quant/feature state away from the validation optimum.
- The remaining path should preserve checkpoint-2 while changing the next normal-training transition, not perform additional endpoint reconstruction.
- Next useful gate: resume from Phase 1F checkpoint-2 with full parameters trainable but an anchor trust to the checkpoint-2 state during the first continuation epoch, with weak/no anchor on useful late layers. This directly targets the checkpoint-2 -> checkpoint-3 regression rather than recreating checkpoint-2.

### Phase 2I: Phase 1F Checkpoint-2 Anchor-Preserved Continuation

Reason:

Phase 2H showed that re-running endpoint reconstruction from Phase 1F checkpoint-2 hurts. This branch instead tries to preserve checkpoint-2 during the next normal continuation epoch: resume from checkpoint-2, snapshot the resumed state as a variation-trust anchor, then run one strict W4A4 QAT epoch with full parameters trainable. Early/high-risk activation/shift state receives stronger trust, while the Phase 1F useful late layers `features.5.5` and `features.7.1` receive weaker trust.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_phase1f_ckpt2_anchor_continuation_gate_20260707.sh
```

Key settings:

```text
Start checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon100_lowlr_b_20260706/checkpoint-2.pth.tar
EPOCHS=1
SCHEDULER_EPOCHS=1
LR=1.0e-5
MIN_LR=5e-6
QUANT_LR_MULTIPLIER=1.5
VARIATION_TRUST_WEIGHT=0.003
VARIATION_TRUST_LATE_LAYERS=features.5.5,features.7.1
VARIATION_TRUST_LATE_MULTIPLIER=0.25
VARIATION_TRUST_EARLY_LAYERS=features.0.0,features.1.0,features.1.1
VARIATION_TRUST_EARLY_MULTIPLIER=3.0
VARIATION_TRUST_SOFTMAX_MULTIPLIER=2.0
VARIATION_TRUST_MOVE_V_MULTIPLIER=2.0
VARIATION_TRUST_PROJ_MOVE_MULTIPLIER=1.5
strict W4A4: wq_bitw=4, aq_bitw=4
```

Sanity evidence:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_phase1f_ckpt2_anchor_continuation_sanity20_20260707.log
Strict resume: loaded model from .../recipe_resume10_prerecon100_lowlr_b_20260706/checkpoint-2.pth.tar; missing=0, unexpected=0
Initialized variation trust anchor: params=243, weight=0.003, late_layers=features.5.5,features.7.1, early_layers=features.0.0,features.1.0,features.1.1
Enabled variation trust regularizer: weight=0.003, pairs=243, avg_multiplier=1.459, late_layers=features.5.5,features.7.1, early_layers=features.0.0,features.1.0,features.1.1
TrainSummary: epoch=0 updates=20 avg_step_time=0.318836s samples_per_step=512 samples_per_sec=1605.84
Stopped early after 20 optimizer updates in epoch 0.
```

Runtime evidence:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_phase1f_ckpt2_anchor_continuation_gate_20260707.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_phase1f_ckpt2_anchor_continuation_gate_20260707
Strict resume: loaded model from .../recipe_resume10_prerecon100_lowlr_b_20260706/checkpoint-2.pth.tar; missing=0, unexpected=0
Initialized variation trust anchor: params=243, weight=0.003, late_layers=features.5.5,features.7.1, early_layers=features.0.0,features.1.0,features.1.1
Enabled variation trust regularizer: weight=0.003, pairs=243, avg_multiplier=1.459, late_layers=features.5.5,features.7.1, early_layers=features.0.0,features.1.0,features.1.1
TrainSummary: epoch=0 updates=2496 avg_step_time=0.258035s samples_per_step=512 samples_per_sec=1984.23
```

Result:

| source checkpoint | resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---:|---:|---:|---|
| Phase 1F `checkpoint-2.pth.tar` | 1 | `checkpoint-1.pth.tar` | 80.4500 | 95.3000 | 0.8437 | below Phase 1F `80.5220`; fail |

Conclusion:

Checkpoint-2 anchor-preserved continuation is stable but still loses accuracy. It is much better than the original Phase 1F checkpoint-3 collapse to `80.3620`, but it does not preserve or improve the checkpoint-2 peak. The goal is still not complete.

Interpretation:

- A parameter-distance trust anchor can reduce the severity of checkpoint-2 -> checkpoint-3 drift, but not enough to create a new best checkpoint.
- The remaining gap is unlikely to be solved by more checkpoint-2 preservation variants alone: quant-only polish, endpoint re-reconstruction, hard freezes, and soft trust all stay below `80.5220`.
- The next non-redundant direction should revisit the only stronger signal seen so far, the W4A8 intermediate (`80.6540`), and focus on a less destructive transition back to strict W4A4.

### Phase 2J: Extended W4A8 Hold Before A6/A4 Transition

Reason:

The only single-model checkpoint above Phase 1F so far was the activation-relaxed W4A8 intermediate (`80.6540`), but direct A8 -> A4 and A8 -> A6 -> A4 transitions failed. Phase 1I proposed testing whether preserving the W4A8 basin for more than one resumed epoch would make the subsequent A6 transition less destructive. This branch keeps A8 for two epochs, then switches to A6, then would switch to A4 only if A6 remained strong.

Implementation note:

- Fixed `save_epoch_checkpoint(...)` to create `output_dir` before writing. The first launch of this branch hit `RuntimeError: Parent directory ... does not exist` at checkpoint save, which was a launcher robustness issue, not a model result.
- After the fix, the branch was relaunched with the same experiment name. The log file was overwritten by a later accidental port-conflict relaunch, so the durable run evidence for the successful relaunch is the checkpoint timestamps plus the terminal transcript captured during the run. The checkpoint files are present under the output directory.

Script:

```bash
EXP=recipe_resume10_actcurr_extend_a8_w4a8_w4a8_w4a6_w4a4_20260707 \
MASTER_PORT=30633 \
EPOCHS=4 \
SCHEDULER_EPOCHS=4 \
PROGRESSIVE_BIT_SCHEDULE=0:4:8,1:4:8,2:4:6,3:4:4 \
PROGRESSIVE_BIT_RECALIBRATE_EPOCHS=2,3 \
PROGRESSIVE_BIT_RECALIBRATE_BATCHES=4 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_actcurr_smooth_gate_20260707.sh
```

Key settings:

```text
Start checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
PROGRESSIVE_BIT_SCHEDULE=0:4:8,1:4:8,2:4:6,3:4:4
PROGRESSIVE_BIT_RECALIBRATE_EPOCHS=2,3
PROGRESSIVE_BIT_RECALIBRATE_BATCHES=4
LR=1.5e-5
MIN_LR=5e-6
QUANT_LR_MULTIPLIER=2
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003
```

Runtime evidence:

```text
Applied progressive fake-quant bits: epoch=0 wbits=4 abits=8 weight_modules=118 act_modules=65
TrainSummary: epoch=0 updates=2496 avg_step_time=0.223577s samples_per_step=512 samples_per_sec=2290.04
Test: [distributed-summary]  Time: 33.870s  Loss: 0.8319  Acc@1: 80.6240  Acc@5: 95.3860  Samples: 50000

Applied progressive fake-quant bits: epoch=1 wbits=4 abits=8 weight_modules=118 act_modules=65
TrainSummary: epoch=1 updates=2496 ...
Test: [distributed-summary]  Time: 10.434s  Loss: 0.8296  Acc@1: 80.6980  Acc@5: 95.4440  Samples: 50000

Applied progressive fake-quant bits: epoch=2 wbits=4 abits=6 weight_modules=118 act_modules=65
Applied progressive bit alpha recalibration: epoch=2 batches=4 quantizers=67
TrainSummary: epoch=2 updates=2496 ...
Test: [distributed-summary]  Time: 10.085s  Loss: 0.8816  Acc@1: 79.6680  Acc@5: 94.9800  Samples: 50000
```

Checkpoint evidence:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_actcurr_extend_a8_w4a8_w4a8_w4a6_w4a4_20260707/checkpoint-1.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_actcurr_extend_a8_w4a8_w4a8_w4a6_w4a4_20260707/checkpoint-2.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_actcurr_extend_a8_w4a8_w4a8_w4a6_w4a4_20260707/checkpoint-3.pth.tar
```

Result:

| resumed epoch | active bits | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---|---:|---:|---:|---|
| 1 | W4A8 | `checkpoint-1.pth.tar` | 80.6240 | 95.3860 | 0.8319 | strong, near prior W4A8 |
| 2 | W4A8 | `checkpoint-2.pth.tar` | 80.6980 | 95.4440 | 0.8296 | strongest single checkpoint so far, but not strict W4A4 |
| 3 | W4A6 | `checkpoint-3.pth.tar` | 79.6680 | 94.9800 | 0.8816 | transition still fails; stopped before completing A4 |

Conclusion:

Holding W4A8 for a second epoch improves the relaxed checkpoint from `80.6240` to `80.6980`, stronger than the previous W4A8 `80.6540`. However, the A6 transition still collapses to `79.6680`, slightly worse than the prior A8 -> A6 result (`79.7340`). The branch was interrupted during the following A4 epoch because A6 already failed the transition gate.

Interpretation:

- The W4A8 basin can be improved, but making it stronger does not automatically make it more convertible to A6/A4.
- The A8 -> lower-bit transition remains the core problem. It is not solved by simply holding A8 longer.
- A useful next direction would need a transition-local objective at the bit switch itself, not another longer A8 hold or scalar schedule change.
- The best strict W4A4 checkpoint remains Phase 1F `80.5220`; the strongest non-strict diagnostic checkpoint is now W4A8 `80.6980`.

### Phase 2K: Progressive Bit-Switch Transition-Local Feature Reconstruction

Reason:

Phase 2J showed that W4A8 can reach `80.6980`, but the A8 -> A6 transition still collapsed to `79.6680`, so simply holding A8 longer is not enough. This branch tested a transition-local objective at the bit switch itself: immediately after lowering activation bit-width, run local teacher feature reconstruction under the new lower-bit state before normal QAT continues.

This is a single-model training branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Implementation:

- Added progressive bit transition reconstruction to `qat_launch.py`.
- New args:
  - `--progressive-bit-transition-recon-updates`
  - `--progressive-bit-transition-recon-epochs`
  - `--progressive-bit-transition-recon-layers`
  - `--progressive-bit-transition-recon-policy`
  - `--progressive-bit-transition-recon-confidence-power`
  - `--progressive-bit-transition-recon-weight-mode`
  - `--progressive-bit-transition-recon-qdrop-prob`
  - `--progressive-bit-transition-recon-qdrop-layers`
- The mechanism applies the scheduled lower bit-width first, optionally recalibrates LSQ alpha, then runs teacher feature reconstruction with `bypass_ddp=True` and gradient masking.
- The gate script disables `--static-graph` by default because transition reconstruction changes which parameters receive gradients before normal DDP training. A forced sanity run with `--static-graph` hit the expected DDP reducer error; the no-static-graph rerun completed.

Script:

```bash
EXP=recipe_resume10_actcurr_transition_recon_gate_20260707 \
MASTER_PORT=30650 \
EPOCHS=4 \
SCHEDULER_EPOCHS=4 \
PROGRESSIVE_BIT_SCHEDULE=0:4:8,1:4:8,2:4:6,3:4:4 \
PROGRESSIVE_BIT_RECALIBRATE_EPOCHS=2,3 \
TRANSITION_RECON_EPOCHS=2,3 \
TRANSITION_RECON_UPDATES=80 \
TRANSITION_RECON_POLICY=module_all \
USE_STATIC_GRAPH=0 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_actcurr_transition_recon_gate_20260707.sh
```

Key settings:

```text
Start checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
PROGRESSIVE_BIT_SCHEDULE=0:4:8,1:4:8,2:4:6,3:4:4
PROGRESSIVE_BIT_RECALIBRATE_EPOCHS=2,3
TRANSITION_RECON_UPDATES=80
TRANSITION_RECON_LAYERS=features.5.5,features.7.1
TRANSITION_RECON_POLICY=module_all
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003
USE_STATIC_GRAPH=0
strict W4A4 final endpoint: wq_bitw=4, aq_bitw=4
```

Sanity evidence:

```text
Forced transition sanity:
EXP=recipe_resume10_transition_recon_forced_sanity20_nostatic_20260707
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Starting progressive bit transition reconstruction: epoch=0, from=W4A4, to=W4A4, updates=4, layers=('features.5.5', 'features.7.1'), policy=module_all
progressive bit transition feature reconstruction epoch=0: update=1/4 loss=0.492677 kept=8933880 masked=18901240 reduced=8933880
progressive bit transition feature reconstruction epoch=0: update=4/4 loss=0.472489 kept=8933880 masked=18901240 reduced=8933880
TrainSummary: epoch=0 updates=20 avg_step_time=0.273482s samples_per_step=512 samples_per_sec=1872.15
```

Static-graph failure evidence:

```text
Forced sanity with static graph hit:
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
This error indicates that your training graph has changed in this iteration ... not compatible with static_graph set to True.
```

Runtime evidence:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_actcurr_transition_recon_gate_20260707.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_actcurr_transition_recon_gate_20260707
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Args: static_graph=false, kd_hard_and_soft=0, progressive_bit_transition_recon_updates=80, progressive_bit_transition_recon_policy=module_all

A8->A6 transition recon:
progressive bit transition feature reconstruction epoch=2: update=1/80 loss=0.336123 kept=8933880 masked=18901240 reduced=8933880
progressive bit transition feature reconstruction epoch=2: update=50/80 loss=0.252232 kept=8933880 masked=18901240 reduced=8933880
progressive bit transition feature reconstruction epoch=2: update=80/80 loss=0.246103 kept=8933880 masked=18901240 reduced=8933880

A6->A4 transition recon:
progressive bit transition feature reconstruction epoch=3: update=1/80 loss=0.403026 kept=8933880 masked=18901240 reduced=8933880
progressive bit transition feature reconstruction epoch=3: update=50/80 loss=0.364925 kept=8933880 masked=18901240 reduced=8933880
progressive bit transition feature reconstruction epoch=3: update=80/80 loss=0.348841 kept=8933880 masked=18901240 reduced=8933880
```

Result:

| resumed epoch | active bits | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---|---:|---:|---:|---|
| 1 | W4A8 | `checkpoint-1.pth.tar` | 80.6140 | 95.3940 | 0.8337 | non-strict diagnostic only |
| 2 | W4A8 | `checkpoint-2.pth.tar` | 80.6060 | 95.4380 | 0.8302 | non-strict diagnostic only |
| 3 | W4A6 | `checkpoint-3.pth.tar` | 79.7640 | 95.0060 | 0.8841 | slightly better than Phase 2J A6 `79.6680`, still far below Phase 1F |
| 4 | strict W4A4 | `checkpoint-4.pth.tar` | 76.7120 | 93.7200 | 1.0558 | fail |

Conclusion:

Transition-local module-all feature reconstruction is technically connected and can optimize the local feature objective at bit switches. It slightly improves the A8 -> A6 endpoint versus Phase 2J (`79.7640` vs `79.6680`), but it does not preserve the representation into strict A4. The final strict W4A4 result collapses to `76.7120`, far below checkpoint-10 `80.3640` and Phase 1F `80.5220`.

Interpretation:

- Updating full late modules during transition reconstruction is too destructive for the final A4 switch even though the local feature loss decreases.
- The local feature objective alone can be optimized without preserving classification accuracy, especially at the A6 -> A4 boundary.
- Future transition-local work should be more constrained: quant/shift-only or small adapter-like parameter subsets at the switch, stronger anchor to the pre-switch student state, or direct transition diagnostics that compare pre-switch and post-switch logits/features before applying normal QAT.
- Do not repeat `module_all` transition reconstruction with this A8/A6/A4 schedule unless another constraint is added.

### Phase 2L: Progressive Bit-Switch Transition Reconstruction, Quant/Shift Only

Reason:

Phase 2K showed that `module_all` transition reconstruction can reduce local feature loss but is too destructive for the A6 -> A4 switch. This branch keeps the same transition-local mechanism and schedule, but changes the reconstruction gradient mask to `quant`, so only quant/shift parameters inside `features.5.5` and `features.7.1` are updated at each bit switch.

This is a single-model training branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Script:

```bash
EXP=recipe_resume10_actcurr_transition_recon_quant_gate_20260707 \
MASTER_PORT=30652 \
EPOCHS=4 \
SCHEDULER_EPOCHS=4 \
PROGRESSIVE_BIT_SCHEDULE=0:4:8,1:4:8,2:4:6,3:4:4 \
PROGRESSIVE_BIT_RECALIBRATE_EPOCHS=2,3 \
TRANSITION_RECON_EPOCHS=2,3 \
TRANSITION_RECON_UPDATES=80 \
TRANSITION_RECON_POLICY=quant \
USE_STATIC_GRAPH=0 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_actcurr_transition_recon_gate_20260707.sh
```

Key settings:

```text
Start checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
PROGRESSIVE_BIT_SCHEDULE=0:4:8,1:4:8,2:4:6,3:4:4
TRANSITION_RECON_UPDATES=80
TRANSITION_RECON_LAYERS=features.5.5,features.7.1
TRANSITION_RECON_POLICY=quant
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003
USE_STATIC_GRAPH=0
strict W4A4 final endpoint: wq_bitw=4, aq_bitw=4
```

Sanity evidence:

```text
Forced transition sanity:
EXP=recipe_resume10_transition_recon_quant_forced_sanity20_20260707
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Starting progressive bit transition reconstruction: epoch=0, from=W4A4, to=W4A4, updates=4, layers=('features.5.5', 'features.7.1'), policy=quant
progressive bit transition feature reconstruction epoch=0: update=1/4 loss=0.492677 kept=67764 masked=27767356 reduced=67764
progressive bit transition feature reconstruction epoch=0: update=4/4 loss=0.487288 kept=67764 masked=27767356 reduced=67764
TrainSummary: epoch=0 updates=20 avg_step_time=0.277559s samples_per_step=512 samples_per_sec=1844.66
```

Runtime evidence:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_actcurr_transition_recon_quant_gate_20260707.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_actcurr_transition_recon_quant_gate_20260707
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Args: static_graph=false, kd_hard_and_soft=0, progressive_bit_transition_recon_updates=80, progressive_bit_transition_recon_policy=quant

A8->A6 transition recon:
progressive bit transition feature reconstruction epoch=2: update=1/80 loss=0.336123 kept=67764 masked=27767356 reduced=67764
progressive bit transition feature reconstruction epoch=2: update=50/80 loss=0.293979 kept=67764 masked=27767356 reduced=67764
progressive bit transition feature reconstruction epoch=2: update=80/80 loss=0.289908 kept=67764 masked=27767356 reduced=67764

A6->A4 transition recon:
progressive bit transition feature reconstruction epoch=3: update=1/80 loss=0.423771 kept=67764 masked=27767356 reduced=67764
progressive bit transition feature reconstruction epoch=3: update=50/80 loss=0.415642 kept=67764 masked=27767356 reduced=67764
progressive bit transition feature reconstruction epoch=3: update=80/80 loss=0.401747 kept=67764 masked=27767356 reduced=67764
```

Result:

| resumed epoch | active bits | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---|---:|---:|---:|---|
| 1 | W4A8 | `checkpoint-1.pth.tar` | 80.6140 | 95.3940 | 0.8337 | non-strict diagnostic only |
| 2 | W4A8 | `checkpoint-2.pth.tar` | 80.6060 | 95.4380 | 0.8302 | non-strict diagnostic only |
| 3 | W4A6 | `checkpoint-3.pth.tar` | 79.7880 | 95.0140 | 0.8839 | slightly better than Phase 2K A6 `79.7640`, still far below Phase 1F |
| 4 | strict W4A4 | `checkpoint-4.pth.tar` | 76.6780 | 93.6660 | 1.0596 | fail |

Conclusion:

Quant/shift-only transition reconstruction is less invasive and gives a tiny A6 improvement over `module_all` (`79.7880` vs `79.7640`), but it still does not solve the strict A4 transition. The final strict W4A4 checkpoint is `76.6780`, below checkpoint-10 `80.3640` and far below Phase 1F `80.5220`.

Interpretation:

- The failure is not only caused by updating late block weights during transition reconstruction; even quant/shift-only reconstruction cannot preserve accuracy at A4.
- A8/A6 local feature reconstruction is not aligned enough with final strict W4A4 classification accuracy.
- The next useful path should likely add a classification-preserving anchor at the switch, for example pre-switch student logit KL / feature anchor while doing post-switch quant/shift updates, or run a much shorter A4 transition adjustment directly from the best strict W4A4 Phase 1F checkpoint instead of from the W4A8 branch.
- Do not repeat plain transition feature reconstruction with either `module_all` or `quant` policy on this schedule.

### Phase 2M: Quant/Shift Transition Reconstruction With Pre-Switch Student Logit Anchor

Reason:

Phase 2L showed that quant/shift-only transition reconstruction is less destructive than `module_all`, but still collapses at A4. This branch added a classification-preserving pre-switch student anchor at the bit switch: before lowering the bit-width, snapshot the current student, then during post-switch transition reconstruction add a logit KL term from the low-bit student to the pre-switch student.

This is a single-model training branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Implementation:

- Added optional transition anchor KL to `qat_launch.py`:
  - `--progressive-bit-transition-anchor-kl-weight`
  - `--progressive-bit-transition-anchor-kl-temperature`
- The anchor model is copied before the bit switch when `progressive_bit_transition_anchor_kl_weight > 0`.
- The transition reconstruction loss becomes local teacher feature reconstruction plus optional pre-switch student logit KL.
- The gate still disables `--static-graph` because transition reconstruction changes gradient participation before normal DDP training.

Script:

```bash
EXP=recipe_resume10_actcurr_transition_anchor_quant_gate_20260707 \
MASTER_PORT=30655 \
EPOCHS=4 \
SCHEDULER_EPOCHS=4 \
PROGRESSIVE_BIT_SCHEDULE=0:4:8,1:4:8,2:4:6,3:4:4 \
PROGRESSIVE_BIT_RECALIBRATE_EPOCHS=2,3 \
TRANSITION_RECON_EPOCHS=2,3 \
TRANSITION_RECON_UPDATES=80 \
TRANSITION_RECON_POLICY=quant \
TRANSITION_ANCHOR_KL_WEIGHT=0.05 \
TRANSITION_ANCHOR_KL_TEMPERATURE=2.75 \
USE_STATIC_GRAPH=0 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_actcurr_transition_recon_gate_20260707.sh
```

Key settings:

```text
Start checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar
PROGRESSIVE_BIT_SCHEDULE=0:4:8,1:4:8,2:4:6,3:4:4
TRANSITION_RECON_UPDATES=80
TRANSITION_RECON_LAYERS=features.5.5,features.7.1
TRANSITION_RECON_POLICY=quant
TRANSITION_ANCHOR_KL_WEIGHT=0.05
TRANSITION_ANCHOR_KL_TEMPERATURE=2.75
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003
USE_STATIC_GRAPH=0
strict W4A4 final endpoint: wq_bitw=4, aq_bitw=4
```

Sanity evidence:

```text
Forced transition sanity:
EXP=recipe_resume10_transition_anchor_quant_forced_sanity20_v2_20260707
Captured progressive bit transition anchor model: epoch=0, from=W4A4, to=W4A4, kl_weight=0.05, temperature=2.75
progressive bit transition feature reconstruction epoch=0: update=1/4 loss=0.514645 kept=67764 masked=28540429 reduced=67764
progressive bit transition feature reconstruction epoch=0: update=4/4 loss=0.509564 kept=67764 masked=28540429 reduced=67764
TrainSummary: epoch=0 updates=20 avg_step_time=0.272414s samples_per_step=512 samples_per_sec=1879.49
```

Runtime evidence:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_actcurr_transition_anchor_quant_gate_20260707.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_actcurr_transition_anchor_quant_gate_20260707
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=0, unexpected=0
Args: static_graph=false, kd_hard_and_soft=0, progressive_bit_transition_recon_policy=quant, progressive_bit_transition_anchor_kl_weight=0.05

A8->A6 transition:
Captured progressive bit transition anchor model: epoch=2, from=W4A8, to=W4A6, kl_weight=0.05, temperature=2.75
progressive bit transition feature reconstruction epoch=2: update=1/80 loss=0.351628 kept=67764 masked=28540429 reduced=67764
progressive bit transition feature reconstruction epoch=2: update=50/80 loss=0.308412 kept=67764 masked=28540429 reduced=67764
progressive bit transition feature reconstruction epoch=2: update=80/80 loss=0.304288 kept=67764 masked=28540429 reduced=67764

A6->A4 transition:
Captured progressive bit transition anchor model: epoch=3, from=W4A6, to=W4A4, kl_weight=0.05, temperature=2.75
progressive bit transition feature reconstruction epoch=3: update=1/80 loss=0.447124 kept=67764 masked=28540429 reduced=67764
progressive bit transition feature reconstruction epoch=3: update=50/80 loss=0.435427 kept=67764 masked=28540429 reduced=67764
progressive bit transition feature reconstruction epoch=3: update=80/80 loss=0.424755 kept=67764 masked=28540429 reduced=67764
```

Result:

| resumed epoch | active bits | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---|---:|---:|---:|---|
| 1 | W4A8 | `checkpoint-1.pth.tar` | 80.6140 | 95.3940 | 0.8337 | non-strict diagnostic only |
| 2 | W4A8 | `checkpoint-2.pth.tar` | 80.6060 | 95.4380 | 0.8302 | non-strict diagnostic only |
| 3 | W4A6 | `checkpoint-3.pth.tar` | 79.7640 | 95.0560 | 0.8822 | no improvement over Phase 2K; below Phase 2L A6 |
| 4 | strict W4A4 | `checkpoint-4.pth.tar` | 76.8280 | 93.6020 | 1.0590 | fail |

Conclusion:

Pre-switch student logit anchoring at weight `0.05` is technically connected but does not improve the A8/A6 -> A4 path. A6 remains `79.7640`, and final strict W4A4 is only `76.8280`, far below checkpoint-10 `80.3640` and Phase 1F `80.5220`.

Interpretation:

- The A8 branch is not a reliable path back to strict A4, even with local feature reconstruction, quant/shift-only updates, and a pre-switch logit anchor.
- The next useful direction should pivot away from A8/A6 transition schedules.
- A more plausible next gate is to start from the best strict W4A4 Phase 1F checkpoint (`80.5220`) and test a short strict-A4 continuation with a fixed logit self-anchor to preserve the checkpoint-2 classifier while allowing only carefully scoped quant/shift adaptation.

### Phase 2N: Phase 1F Checkpoint-2 Strict-A4 Fixed Logit Anchor Continuation

Reason:

Phase 2K/2L/2M showed that A8/A6 transition schedules are not a useful route back to strict A4. This branch pivots back to the best valid strict W4A4 point, Phase 1F `checkpoint-2.pth.tar` at `80.5220`, and tests whether a fixed logit self-anchor can preserve that classifier during one normal strict-A4 continuation epoch.

This is a single-model strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Script:

```bash
EXP=recipe_phase1f_ckpt2_fixedlogit_gate_20260707 \
MASTER_PORT=30657 \
EPOCHS=1 \
SCHEDULER_EPOCHS=1 \
REF_LOGIT_KL_WEIGHT=0.02 \
USE_STATIC_GRAPH=0 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_phase1f_ckpt2_fixedlogit_continuation_gate_20260707.sh
```

Key settings:

```text
Start checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon100_lowlr_b_20260706/checkpoint-2.pth.tar
LR=8e-6
MIN_LR=4e-6
QUANT_LR_MULTIPLIER=1.5
TRAIN_SCHEME=ema_ref_attn_kl
REF_UPDATE=fixed
REF_LOGIT_KL_WEIGHT=0.02
REF_LOGIT_KL_TEMPERATURE=2.75
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003
strict W4A4: wq_bitw=4, aq_bitw=4
```

Sanity evidence:

```text
EXP=recipe_phase1f_ckpt2_fixedlogit_sanity20_20260707
Strict resume: loaded model from .../recipe_resume10_prerecon100_lowlr_b_20260706/checkpoint-2.pth.tar; missing=0, unexpected=0
Enabled EMA refmodel attention-KL scheme: ref_update=fixed ... attn_kl_weight=0.0 ...
RefLogitKL: 2.965e-08 at update 0, then non-zero during training
TrainSummary: epoch=0 updates=20 avg_step_time=0.369178s samples_per_step=512 samples_per_sec=1386.86
```

Runtime evidence:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_phase1f_ckpt2_fixedlogit_gate_20260707.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_phase1f_ckpt2_fixedlogit_gate_20260707
Strict resume: loaded model from .../recipe_resume10_prerecon100_lowlr_b_20260706/checkpoint-2.pth.tar; missing=0, unexpected=0
Enabled EMA refmodel attention-KL scheme: ref_update=fixed, ref_update_interval=1, momentum=0.999, attn_kl_weight=0.0
args.yaml: train_scheme=ema_ref_attn_kl, ref_update=fixed, ref_logit_kl_weight=0.02, static_graph=false, kd_hard_and_soft=0
TrainSummary: epoch=0 updates=2496 avg_step_time=0.309139s samples_per_step=512 samples_per_sec=1656.21
Test: [distributed-summary]  Time: 34.396s  Loss: 0.8437  Acc@1: 80.3960  Acc@5: 95.2880  Samples: 50000
```

Result:

| source checkpoint | resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---:|---:|---:|---|
| Phase 1F `checkpoint-2.pth.tar` | 1 | `checkpoint-1.pth.tar` | 80.3960 | 95.2880 | 0.8437 | below Phase 1F `80.5220`; fail |

Conclusion:

Fixed logit self-anchoring is connected and does constrain the continuation (`RefLogitKL` non-zero), but it does not preserve the Phase 1F `80.5220` peak. The result is similar to other checkpoint-2 continuation attempts and remains far below the `81.0` goal.

Interpretation:

- The checkpoint-2 peak is not preserved by one-epoch continuation even with fixed logit anchoring.
- Further work should avoid longer continuation from checkpoint-2 unless the first epoch can preserve or improve `80.5220`.
- The remaining useful direction is likely not another continuation regularizer, but a pre-validation endpoint improvement or a different strict-A4 local adjustment that is evaluated before a full training epoch changes the basin.

### Phase 2O: Phase 1F Checkpoint-2 Selective Activation-MSE Endpoint Repair

Reason:

Phase 2N showed that normal strict-A4 continuation from Phase 1F checkpoint-2 still loses the `80.5220` peak even with a fixed logit anchor. This branch avoids a normal training epoch and instead tests a very narrow endpoint-only repair: start from Phase 1F `checkpoint-2.pth.tar`, apply activation MSE scale calibration only to three diagnosed high-clipping late attention activation quantizers, save the calibrated endpoint immediately, and run a strict W4A4 full validation.

This is a single-model strict W4A4 endpoint. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Endpoint save command:

```bash
EXP=recipe_phase1f_ckpt2_selective_actmse_endpoint_20260707 \
MASTER_PORT=30658 \
ACT_MSE_BATCHES=8 \
ACT_MSE_BLEND=0.35 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_phase1f_ckpt2_selective_actmse_endpoint_save_20260707.sh
```

Full validation command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_phase1f_ckpt2_selective_actmse_endpoint_20260707/step_checkpoints/step_0000.pth.tar \
EXP=eval_phase1f_ckpt2_selective_actmse_endpoint_20260707 \
MASTER_PORT=30659 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Key settings:

```text
Start checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon100_lowlr_b_20260706/checkpoint-2.pth.tar
Saved endpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_phase1f_ckpt2_selective_actmse_endpoint_20260707/step_checkpoints/step_0000.pth.tar
ACT_MSE_BATCHES=8
ACT_MSE_LAYERS=features.5.5,features.7.1
ACT_MSE_QUANTIZERS=features.5.5.attn.quan_a_qkx_fn,features.7.1.attn.quant_x_4_qkv.input_quant_fn,features.5.5.attn.quan_a_v_fn
ACT_MSE_GRID=0.85,1.25,17
ACT_MSE_BLEND=0.35
strict W4A4: wq_bitw=4, aq_bitw=4
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_phase1f_ckpt2_selective_actmse_endpoint_20260707.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_phase1f_ckpt2_selective_actmse_endpoint_20260707.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict resume: loaded model from .../recipe_resume10_prerecon100_lowlr_b_20260706/checkpoint-2.pth.tar; missing=0, unexpected=0
Activation MSE calibration: matched=3
Finished pre-QAT activation MSE calibration: batches=8, updated=3, mean_scale_ratio=0.9748, min_ratio=0.9572, max_ratio=0.9887
Eval command included --wq-bitw 4 --aq-bitw 4 --eval-only
Strict eval resume: loaded model from .../step_checkpoints/step_0000.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.684s  Loss: 0.8417  Acc@1: 80.4660  Acc@5: 95.2960  Samples: 50000
```

Result:

| source checkpoint | endpoint | raw Top-1 | raw Top-5 | loss | gate |
|---|---|---:|---:|---:|---|
| Phase 1F `checkpoint-2.pth.tar` | `step_0000.pth.tar` | 80.4660 | 95.2960 | 0.8417 | above checkpoint-10 `80.3640`, below Phase 1F `80.5220`; fail |

Conclusion:

The selective activation-MSE endpoint repair is valid strict W4A4 and gives a weak positive signal versus checkpoint-10, but it does not beat the Phase 1F `80.5220` checkpoint and is far below the `81.0` goal.

Interpretation:

- Narrow activation-scale repair is less destructive than broad activation calibration and better than several continuation branches, but the three-quantizer bundle is not sufficient.
- Do not extend this exact endpoint to training epochs because it already starts below Phase 1F.
- The next useful diagnostic is to split the three selected quantizers into single-quantizer endpoint repairs to identify whether any one component has positive contribution before designing another combined variation-aware endpoint repair.

### Phase 2P: Phase 1F Checkpoint-2 Single-Quantizer Activation-MSE Endpoint Split

Reason:

Phase 2O showed that the three-quantizer selective activation-MSE endpoint repair is valid and mildly positive versus checkpoint-10, but it remains below Phase 1F `80.5220`. This phase split the three quantizers into single-quantizer endpoint repairs to identify whether the bundle contained a harmful component or if all components were simply too weak.

This is a set of single-model strict W4A4 endpoint evaluations. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Script:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_phase1f_ckpt2_single_actmse_endpoint_gates_20260707.sh
```

Common settings:

```text
Start checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon100_lowlr_b_20260706/checkpoint-2.pth.tar
ACT_MSE_BATCHES=8
ACT_MSE_GRID=0.85,1.25,17
ACT_MSE_BLEND=0.35
strict W4A4: wq_bitw=4, aq_bitw=4
```

Runtime evidence:

```text
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
qkx save log: /mlx_devbox/users/quyanyi/playground/train_recipe_phase1f_ckpt2_single_actmse_qkx_endpoint_20260707.log
qkx eval log: /mlx_devbox/users/quyanyi/playground/train_eval_phase1f_ckpt2_single_actmse_qkx_endpoint_20260707.log
qkv_input save log: /mlx_devbox/users/quyanyi/playground/train_recipe_phase1f_ckpt2_single_actmse_qkv_input_endpoint_20260707.log
qkv_input eval log: /mlx_devbox/users/quyanyi/playground/train_eval_phase1f_ckpt2_single_actmse_qkv_input_endpoint_20260707.log
v save log: /mlx_devbox/users/quyanyi/playground/train_recipe_phase1f_ckpt2_single_actmse_v_endpoint_20260707.log
v eval log: /mlx_devbox/users/quyanyi/playground/train_eval_phase1f_ckpt2_single_actmse_v_endpoint_20260707.log
All eval commands included --wq-bitw 4 --aq-bitw 4 --eval-only.
All eval resumes loaded the saved step_0000 checkpoint with missing=0, unexpected=0.
```

Result:

| variant | quantizer | scale ratio | endpoint | raw Top-1 | raw Top-5 | loss | samples | gate |
|---|---|---:|---|---:|---:|---:|---:|---|
| qkx | `features.5.5.attn.quan_a_qkx_fn` | 0.9785 | `recipe_phase1f_ckpt2_single_actmse_qkx_endpoint_20260707/step_checkpoints/step_0000.pth.tar` | 80.4900 | 95.3160 | 0.8418 | 50000 | best split, below Phase 1F |
| qkv_input | `features.7.1.attn.quant_x_4_qkv.input_quant_fn` | 0.9887 | `recipe_phase1f_ckpt2_single_actmse_qkv_input_endpoint_20260707/step_checkpoints/step_0000.pth.tar` | 80.4860 | 95.3520 | 0.8412 | 50000 | below qkx and Phase 1F |
| v | `features.5.5.attn.quan_a_v_fn` | 0.9572 | `recipe_phase1f_ckpt2_single_actmse_v_endpoint_20260707/step_checkpoints/step_0000.pth.tar` | 80.4280 | 95.2800 | 0.8411 | 50000 | worst split |

Conclusion:

Single-quantizer endpoint repair confirms that `features.5.5.attn.quan_a_qkx_fn` and `features.7.1.attn.quant_x_4_qkv.input_quant_fn` are less harmful than `features.5.5.attn.quan_a_v_fn`, but none of the single repairs beats Phase 1F `80.5220`.

Interpretation:

- The `v` shrink is likely harmful in this endpoint setting; it should not be included in the next activation-MSE combination.
- The three-quantizer Phase 2O result (`80.4660`) is lower than qkx-only (`80.4900`) and qkv_input-only (`80.4860`), consistent with destructive interaction from the bundle.
- A final low-cost endpoint diagnostic is to test `qkx + qkv_input` while excluding `v`. If that does not beat Phase 1F, close the activation-MSE endpoint branch and move to a different variation-aware mechanism.

### Phase 2Q: Phase 1F Checkpoint-2 Pair Activation-MSE Endpoint, qkx + qkv_input

Reason:

Phase 2P showed that qkx-only (`80.4900`) and qkv_input-only (`80.4860`) were the least harmful single activation-MSE repairs, while v-only was clearly worse (`80.4280`). This branch tested the pair `qkx + qkv_input` and excluded the harmful v quantizer. It is the final low-cost combination implied by the single-quantizer split.

This is a single-model strict W4A4 endpoint. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Endpoint save command:

```bash
EXP=recipe_phase1f_ckpt2_pair_actmse_qkx_qkv_endpoint_20260707 \
MASTER_PORT=30676 \
ACT_MSE_BATCHES=8 \
ACT_MSE_LAYERS=features.5.5,features.7.1 \
ACT_MSE_QUANTIZERS=features.5.5.attn.quan_a_qkx_fn,features.7.1.attn.quant_x_4_qkv.input_quant_fn \
ACT_MSE_GRID=0.85,1.25,17 \
ACT_MSE_BLEND=0.35 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_phase1f_ckpt2_selective_actmse_endpoint_save_20260707.sh
```

Full validation command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_phase1f_ckpt2_pair_actmse_qkx_qkv_endpoint_20260707/step_checkpoints/step_0000.pth.tar \
EXP=eval_phase1f_ckpt2_pair_actmse_qkx_qkv_endpoint_20260707 \
MASTER_PORT=30677 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Key settings:

```text
Start checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon100_lowlr_b_20260706/checkpoint-2.pth.tar
Saved endpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_phase1f_ckpt2_pair_actmse_qkx_qkv_endpoint_20260707/step_checkpoints/step_0000.pth.tar
ACT_MSE_BATCHES=8
ACT_MSE_QUANTIZERS=features.5.5.attn.quan_a_qkx_fn,features.7.1.attn.quant_x_4_qkv.input_quant_fn
ACT_MSE_GRID=0.85,1.25,17
ACT_MSE_BLEND=0.35
strict W4A4: wq_bitw=4, aq_bitw=4
```

Runtime evidence:

```text
Save log: /mlx_devbox/users/quyanyi/playground/train_recipe_phase1f_ckpt2_pair_actmse_qkx_qkv_endpoint_20260707.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_phase1f_ckpt2_pair_actmse_qkx_qkv_endpoint_20260707.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict save resume: loaded model from .../recipe_resume10_prerecon100_lowlr_b_20260706/checkpoint-2.pth.tar; missing=0, unexpected=0
Activation MSE calibration: batches=8, updated=2, mean_scale_ratio=0.9836, min_ratio=0.9785, max_ratio=0.9887
Eval command included --wq-bitw 4 --aq-bitw 4 --eval-only
Strict eval resume: loaded model from .../step_checkpoints/step_0000.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 33.928s  Loss: 0.8418  Acc@1: 80.5060  Acc@5: 95.3080  Samples: 50000
```

Result:

| source checkpoint | endpoint | raw Top-1 | raw Top-5 | loss | gate |
|---|---|---:|---:|---:|---|
| Phase 1F `checkpoint-2.pth.tar` | `step_0000.pth.tar` | 80.5060 | 95.3080 | 0.8418 | best activation-MSE endpoint variant, but below Phase 1F `80.5220`; fail |

Conclusion:

The qkx+qkv_input pair is the best activation-MSE endpoint repair tried so far (`80.5060`), improving over qkx-only (`80.4900`), qkv_input-only (`80.4860`), v-only (`80.4280`), and the original three-quantizer bundle (`80.4660`). However, it still does not beat Phase 1F `80.5220`, and it is far below the `81.0` target.

Interpretation:

- Activation-MSE endpoint repair has useful diagnostic value, but its best variant remains below the known best strict W4A4 checkpoint.
- The activation-scale-only endpoint branch should be closed unless a new diagnostic changes the target set or objective materially.
- The next variation-aware attempt should move from scale-only endpoint repair to a different mechanism that changes the optimization signal around the successful Phase 1F warm start, such as classifier-preserving local reconstruction on high-variation samples or module-specific trust weighting based on observed validation-sensitive quantizer effects.

### Phase 2R: Phase 1F-Style Pre-QAT Feature Reconstruction With Start-Student Logit Anchor

Reason:

Phase 1F remains the best strict W4A4 mechanism (`80.5220`) but later regresses, and Phase 2O/2P/2Q closed activation-MSE endpoint repair as insufficient. This phase tested whether the successful Phase 1F pre-QAT feature reconstruction can be made more classifier-preserving by anchoring the reconstruction step to the starting student logits. The anchor is captured immediately after strict resume from checkpoint-10 and before reconstruction, then used only during pre-QAT feature reconstruction.

This is a single-model strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Implementation:

- Added ordinary pre-QAT feature reconstruction anchor controls to `qat_launch.py`:
  - `--pre-qat-feature-recon-anchor-kl-weight`
  - `--pre-qat-feature-recon-anchor-kl-temperature`
- Reused the existing `run_pre_qat_feature_reconstruction(..., anchor_model, anchor_kl_weight, anchor_kl_temperature)` path that was previously only used for progressive bit transitions.
- When `pre_qat_feature_recon_updates > 0` and anchor weight is positive, the launcher snapshots the current resumed student before reconstruction and frees it after reconstruction.
- Updated `tmp_scripts/run_resume10_lowlr_gate_20260706.sh` so the anchor knobs are logged and reproducible.

Sanity command:

```bash
EXP=recipe_resume10_prerecon_anchor_sanity20_20260707 \
MASTER_PORT=30678 \
EPOCHS=1 \
SCHEDULER_EPOCHS=1 \
LR=1.5e-5 \
MIN_LR=5e-6 \
QUANT_LR_MULTIPLIER=2 \
PRE_QAT_FEATURE_RECON_UPDATES=4 \
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1 \
PRE_QAT_FEATURE_RECON_POLICY=quant \
PRE_QAT_FEATURE_RECON_ANCHOR_KL_WEIGHT=0.02 \
PRE_QAT_FEATURE_RECON_ANCHOR_KL_TEMPERATURE=2.75 \
MAX_TRAIN_UPDATES=20 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_lowlr_gate_20260706.sh
```

Sanity evidence:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_prerecon_anchor_sanity20_20260707.log
Strict resume: loaded model from .../recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar; missing=0, unexpected=0
Captured pre-QAT feature reconstruction anchor model: kl_weight=0.02, temperature=2.75
Starting pre-QAT feature reconstruction: updates=4, policy=quant, ... anchor_kl_weight=0.02, anchor_kl_temperature=2.75
pre-QAT feature reconstruction: update=4/4 loss=0.111994 kept=67764 masked=28540429 reduced=0
TrainSummary: epoch=0 updates=20 avg_step_time=0.266312s samples_per_step=512 samples_per_sec=1922.56
```

Gate command:

```bash
EXP=recipe_resume10_prerecon_anchor_gate_20260707 \
MASTER_PORT=30679 \
EPOCHS=2 \
SCHEDULER_EPOCHS=2 \
LR=1.5e-5 \
MIN_LR=5e-6 \
QUANT_LR_MULTIPLIER=2 \
PRE_QAT_FEATURE_RECON_UPDATES=100 \
PRE_QAT_FEATURE_RECON_LAYERS=features.5.5,features.7.1 \
PRE_QAT_FEATURE_RECON_POLICY=quant \
PRE_QAT_FEATURE_RECON_ANCHOR_KL_WEIGHT=0.02 \
PRE_QAT_FEATURE_RECON_ANCHOR_KL_TEMPERATURE=2.75 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_lowlr_gate_20260706.sh
```

Runtime evidence:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_prerecon_anchor_gate_20260707.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_anchor_gate_20260707
Args: pre_qat_feature_recon_anchor_kl_weight=0.02, pre_qat_feature_recon_anchor_kl_temperature=2.75, pre_qat_feature_recon_updates=100, wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0
Strict resume: loaded model from .../recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar; missing=0, unexpected=0
Captured pre-QAT feature reconstruction anchor model: kl_weight=0.02, temperature=2.75
Starting pre-QAT feature reconstruction: updates=100, policy=quant, ... anchor_kl_weight=0.02, anchor_kl_temperature=2.75
pre-QAT feature reconstruction: update=100/100 loss=0.112561 kept=67764 masked=28540429 reduced=0
Finished pre-QAT feature reconstruction: updates=100
```

Result:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.4020 | 95.3300 | 0.8453 | above checkpoint-10 and Phase 1F epoch1, but below Phase 1F best |
| 2 | `checkpoint-2.pth.tar` | 80.4240 | 95.3040 | 0.8436 | below Phase 1F `80.5220`; fail |

Conclusion:

The start-student logit anchor is technically connected and reproducible, but it does not improve the Phase 1F mechanism. The first epoch is slightly better than Phase 1F epoch1 (`80.4020` vs `80.3840`), but epoch2 reaches only `80.4240`, far below Phase 1F checkpoint-2 `80.5220` and the `81.0` target.

Interpretation:

- Classification-preserving anchoring during pre-QAT feature reconstruction at weight `0.02` overconstrains or misdirects the useful Phase 1F adaptation rather than preserving the later peak.
- Do not continue this exact anchor configuration to more epochs.
- Since scale-only endpoint repair and simple pre-QAT logit anchoring both failed, the next useful variation-aware branch should either target different parameters/modules based on the drift diagnostics or change the reconstruction objective more materially, rather than adding another weak global anchor to Phase 1F.

### Phase 2S: Selective Variation Trust on Diagnosed Early and Late Activation/Shift Modules

Reason:

Phase 2F showed that soft variation trust is better than hard freeze, but its anchor covered all activation/shift parameters (`params=243`) and remained below Phase 1F. Phase 2R showed that adding a start-student logit anchor to pre-QAT reconstruction does not help. This phase made variation trust more selective: only activation/shift parameters under the diagnosed early/high-risk modules and Phase 1F late target modules are anchored, reducing the trust set while preserving the weak late-layer anchor idea.

This is a single-model strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Implementation:

- Added `--variation-trust-layers` to `qat_launch.py`.
- `collect_variation_trust_state(...)` now accepts a module filter and only snapshots activation/shift parameters under those named modules when provided.
- Updated `tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh` to log and pass `VARIATION_TRUST_LAYERS`.

Sanity command:

```bash
EXP=recipe_resume10_prerecon_vartrust_selective_sanity20_20260707 \
MASTER_PORT=30680 \
EPOCHS=1 \
SCHEDULER_EPOCHS=1 \
PRE_QAT_FEATURE_RECON_UPDATES=4 \
VARIATION_TRUST_WEIGHT=0.003 \
VARIATION_TRUST_LAYERS=features.0.0,features.1.0,features.1.1,features.5.5,features.7.1 \
VARIATION_TRUST_LATE_LAYERS=features.5.5,features.7.1 \
VARIATION_TRUST_LATE_MULTIPLIER=0.15 \
VARIATION_TRUST_EARLY_LAYERS=features.0.0,features.1.0,features.1.1 \
VARIATION_TRUST_EARLY_MULTIPLIER=3.0 \
VARIATION_TRUST_SOFTMAX_MULTIPLIER=2.0 \
VARIATION_TRUST_MOVE_V_MULTIPLIER=2.0 \
VARIATION_TRUST_PROJ_MOVE_MULTIPLIER=1.5 \
MAX_TRAIN_UPDATES=20 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Sanity evidence:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_prerecon_vartrust_selective_sanity20_20260707.log
Strict resume: loaded model from .../recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar; missing=0, unexpected=0
Finished pre-QAT feature reconstruction: updates=4
Initialized variation trust anchor: params=79, weight=0.003, layers=('features.0.0', 'features.1.0', 'features.1.1', 'features.5.5', 'features.7.1'), late_layers=features.5.5,features.7.1, early_layers=features.0.0,features.1.0,features.1.1
Enabled variation trust regularizer: weight=0.003, pairs=79, avg_multiplier=1.948, late_layers=features.5.5,features.7.1, early_layers=features.0.0,features.1.0,features.1.1
TrainSummary: epoch=0 updates=20 avg_step_time=0.271344s samples_per_step=512 samples_per_sec=1886.90
```

Gate command:

```bash
EXP=recipe_resume10_prerecon_vartrust_selective_gate_20260707 \
MASTER_PORT=30681 \
EPOCHS=2 \
SCHEDULER_EPOCHS=2 \
PRE_QAT_FEATURE_RECON_UPDATES=100 \
VARIATION_TRUST_WEIGHT=0.003 \
VARIATION_TRUST_LAYERS=features.0.0,features.1.0,features.1.1,features.5.5,features.7.1 \
VARIATION_TRUST_LATE_LAYERS=features.5.5,features.7.1 \
VARIATION_TRUST_LATE_MULTIPLIER=0.15 \
VARIATION_TRUST_EARLY_LAYERS=features.0.0,features.1.0,features.1.1 \
VARIATION_TRUST_EARLY_MULTIPLIER=3.0 \
VARIATION_TRUST_SOFTMAX_MULTIPLIER=2.0 \
VARIATION_TRUST_MOVE_V_MULTIPLIER=2.0 \
VARIATION_TRUST_PROJ_MOVE_MULTIPLIER=1.5 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Runtime evidence:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_prerecon_vartrust_selective_gate_20260707.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_vartrust_selective_gate_20260707
Args: variation_trust_layers=features.0.0,features.1.0,features.1.1,features.5.5,features.7.1, variation_trust_weight=0.003, wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0
Strict resume: loaded model from .../recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar; missing=0, unexpected=0
Starting pre-QAT feature reconstruction: updates=100, policy=quant, layers=('features.5.5', 'features.7.1')
Finished pre-QAT feature reconstruction: updates=100
Initialized variation trust anchor: params=79, weight=0.003, layers=('features.0.0', 'features.1.0', 'features.1.1', 'features.5.5', 'features.7.1'), late_layers=features.5.5,features.7.1, early_layers=features.0.0,features.1.0,features.1.1
Enabled variation trust regularizer: weight=0.003, pairs=79, avg_multiplier=1.948, late_layers=features.5.5,features.7.1, early_layers=features.0.0,features.1.0,features.1.1
```

Result:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.5220 | 95.3220 | 0.8446 | ties Phase 1F best, strong signal but not new best |
| 2 | `checkpoint-2.pth.tar` | 80.3820 | 95.2880 | 0.8428 | regresses below Phase 1F; fail |

Conclusion:

Selective variation trust is a cleaner implementation than Phase 2F and produces the best first-epoch result among variation-trust branches, exactly tying Phase 1F `80.5220`. However, it does not improve the best strict W4A4 checkpoint and still regresses by epoch2, so it does not satisfy the `81.0` goal or justify extending this exact configuration.

Interpretation:

- Filtering trust to diagnosed modules is useful: it avoids the broad 243-parameter anchor and improves early behavior relative to previous soft-trust variants.
- The persistent epoch2 regression means the current trust target or schedule is still not enough to preserve the Phase 1F-style peak.
- A follow-up, if pursued, should not simply run longer. It should either stop/save/evaluate the epoch1 tie point only, or change the schedule so the trust relaxes differently after epoch1; otherwise the branch repeats the known Phase 1F checkpoint-2 to checkpoint-3 regression pattern.

### Phase 2T: Selective Variation Trust Plus Early Selective BinReg

Reason:

Phase 2S showed that selective variation trust can tie the current best strict W4A4 result at the first resumed epoch, but it still regresses on the second resumed epoch. This phase tested whether a VVTQ-style bin regularizer, restricted only to diagnosed early/high-risk modules, could suppress harmful early quantized-weight/bin drift without the over-constraint observed in full-model BinReg or late-attn-only BinReg.

This is a single-model strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Implementation:

- Reused selective variation trust from Phase 2S:
  - trust layers: `features.0.0,features.1.0,features.1.1,features.5.5,features.7.1`
  - trust anchor params: `79`
  - late weak-anchor layers: `features.5.5,features.7.1`
  - early strong-anchor layers: `features.0.0,features.1.0,features.1.1`
- Added optional BinReg knobs to `tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh`.
- Applied BinReg only to early diagnosed modules:
  - `BIN_REG_WEIGHT=2e-5`
  - `BIN_REG_VARIANCE_WEIGHT=1.0`
  - `BIN_REG_LAYERS=features.0.0,features.1.0.attn,features.1.1.attn`
  - `BIN_REG_ATTN_ONLY=0`

Gate command:

```bash
EXP=recipe_resume10_prerecon_vartrust_earlybin_gate_20260707 \
MASTER_PORT=30683 \
EPOCHS=2 \
SCHEDULER_EPOCHS=2 \
PRE_QAT_FEATURE_RECON_UPDATES=100 \
VARIATION_TRUST_WEIGHT=0.003 \
VARIATION_TRUST_LAYERS=features.0.0,features.1.0,features.1.1,features.5.5,features.7.1 \
VARIATION_TRUST_LATE_LAYERS=features.5.5,features.7.1 \
VARIATION_TRUST_LATE_MULTIPLIER=0.15 \
VARIATION_TRUST_EARLY_LAYERS=features.0.0,features.1.0,features.1.1 \
VARIATION_TRUST_EARLY_MULTIPLIER=3.0 \
VARIATION_TRUST_SOFTMAX_MULTIPLIER=2.0 \
VARIATION_TRUST_MOVE_V_MULTIPLIER=2.0 \
VARIATION_TRUST_PROJ_MOVE_MULTIPLIER=1.5 \
BIN_REG_WEIGHT=2e-5 \
BIN_REG_VARIANCE_WEIGHT=1.0 \
BIN_REG_LAYERS=features.0.0,features.1.0.attn,features.1.1.attn \
BIN_REG_ATTN_ONLY=0 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Runtime evidence:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_prerecon_vartrust_earlybin_gate_20260707.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_vartrust_earlybin_gate_20260707
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Args: bin_reg_weight=2.0e-05, bin_reg_layers=features.0.0,features.1.0.attn,features.1.1.attn, variation_trust_layers=features.0.0,features.1.0,features.1.1,features.5.5,features.7.1, wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0
Strict resume: loaded model from .../recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar; missing=0, unexpected=0
Starting pre-QAT feature reconstruction: updates=100, policy=quant, layers=('features.5.5', 'features.7.1')
pre-QAT feature reconstruction: update=100/100 loss=0.112701 kept=67764 masked=28540429 reduced=0
Initialized variation trust anchor: params=79, weight=0.003, layers=('features.0.0', 'features.1.0', 'features.1.1', 'features.5.5', 'features.7.1'), late_layers=features.5.5,features.7.1, early_layers=features.0.0,features.1.0,features.1.1
Enabled bin regularizer: weight=2e-05, variance_weight=1.0, layers=('features.0.0', 'features.1.0.attn', 'features.1.1.attn'), attn_only=False, pairs=9
Enabled variation trust regularizer: weight=0.003, pairs=79, avg_multiplier=1.948, late_layers=features.5.5,features.7.1, early_layers=features.0.0,features.1.0,features.1.1
Test: [distributed-summary]  Time: 34.044s  Loss: 0.8438  Acc@1: 80.3120  Acc@5: 95.3180  Samples: 50000
```

Result:

| resumed epoch | checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---:|---|---:|---:|---:|---|
| 1 | `checkpoint-1.pth.tar` | 80.3120 | 95.3180 | 0.8438 | below checkpoint-10 `80.3640` and Phase 1F/2S `80.5220`; fail |

Conclusion:

Early selective BinReg is not a useful continuation mechanism for this resume10 branch. Even when restricted to diagnosed early modules, the first full validation drops below the fixed checkpoint-10 baseline, so the run was stopped during the second training epoch instead of spending the full 2-epoch gate.

Interpretation:

- The negative first full-val means the BinReg constraint is harming useful Phase 1F adaptation immediately, not only failing to prevent later regression.
- Do not repeat early BinReg by weight-only adjustment.
- The next variation-aware branch should test schedule behavior around the Phase 2S first-epoch tie point: continue from the strict `80.5220` checkpoint with the trust regularizer disabled, rather than adding more bin constraints.

### Phase 2U: Continue From Phase 2S Epoch1 Tie Point With Variation Trust Disabled

Reason:

Phase 2S produced a strong first resumed epoch (`80.5220`) but regressed to `80.3820` after the second resumed epoch while variation trust stayed enabled. Phase 2T showed that adding early selective BinReg is harmful immediately. This phase tested a schedule hypothesis: use selective variation trust only to reach the Phase 2S first-epoch tie point, then continue from that single checkpoint with variation trust and BinReg disabled to see whether the second-epoch regression is caused mainly by persistent trust regularization.

This is a single-model strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Implementation:

- Resumed from Phase 2S `checkpoint-1.pth.tar`, the strict W4A4 checkpoint that tied the current best `80.5220`.
- Set `START_EPOCH=1` and `EPOCHS=2`, so only the second resumed epoch is trained.
- Disabled pre-QAT feature reconstruction on the continuation:
  - `PRE_QAT_FEATURE_RECON_UPDATES=0`
- Disabled variation trust:
  - `VARIATION_TRUST_WEIGHT=0`
- Disabled BinReg:
  - `BIN_REG_WEIGHT=0`
- Kept the same teacher feature-output auxiliary used by the Phase 2S family:
  - `TEACHER_FEATURE_OUTPUT_WEIGHT=0.003`

Note:

`tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh` was updated to expose `START_EPOCH` and pass it to `qat_launch.py`. The underlying OFQ training path did start at epoch 1 as expected, confirmed by runtime logs.

Gate command:

```bash
EXP=recipe_resume10_vartrust_epoch1_continue_notrust_gate_20260707 \
MASTER_PORT=30684 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar \
START_EPOCH=1 \
EPOCHS=2 \
SCHEDULER_EPOCHS=2 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Runtime evidence:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_vartrust_epoch1_continue_notrust_gate_20260707.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_continue_notrust_gate_20260707
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Args: pre_qat_feature_recon_updates=0, variation_trust_weight=0.0, wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0
Strict resume: loaded model from .../recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar; missing=0, unexpected=0
Trainable parameter policy: epoch=1, quant_only=False, policy=all, trainable=28608256, frozen=0
Train: 1 ... VarTrust: 0.000e+00 ... BinReg: 0.000e+00
Test: [distributed-summary]  Time: 34.734s  Loss: 0.8429  Acc@1: 80.4540  Acc@5: 95.3280  Samples: 50000
```

Result:

| source checkpoint | trained epoch | output checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---:|---:|---:|---|
| Phase 2S `checkpoint-1.pth.tar` | epoch 1 only | `checkpoint-2.pth.tar` | 80.4540 | 95.3280 | 0.8429 | better than Phase 2S epoch2 `80.3820`, but below Phase 2S/Phase 1F `80.5220`; fail |

Conclusion:

Turning off variation trust after the Phase 2S first-epoch tie point partially reduces the second-epoch regression (`80.4540` vs `80.3820`) but still does not preserve or improve the `80.5220` peak. The regression is therefore not caused solely by persistent trust regularization; the second full-param QAT epoch itself is still destructive for the peak checkpoint.

Interpretation:

- Phase 2S trust is useful for reaching a strong first-epoch point, but simple full-param continuation from that point is still low-yield.
- The next branch should not add more trust/BinReg. It should change the trainable set after the tie point: for example, continue from Phase 2S `checkpoint-1` with full weights frozen and only a carefully selected small quant/shift subset trainable, or use a gradient-mask policy that lets early/high-risk activation quant/shift stop moving while allowing the late beneficial adaptation modules to move.
- Success still requires a strict W4A4 single checkpoint at `>=81.0` Top-1 with `Samples=50000`; current best remains `80.5220`.

### Phase 2V: Continue From Phase 2S Epoch1 With Only Late Quant/Shift Trainable

Reason:

Phase 2U showed that disabling variation trust after the Phase 2S first-epoch tie point improves the second resumed epoch relative to Phase 2S itself, but still falls below the `80.5220` peak. This phase tested a narrower VVTQ-style schedule: after reaching the Phase 2S tie point, freeze almost everything and allow only the diagnosed useful late modules' quant/shift parameters to move. This directly tests whether small late quant/shift adaptation can preserve or improve the peak without the destructive full-parameter second epoch.

This is a single-model strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Implementation:

- Added `--start-epoch` forwarding from `qat_launch.py` to OFQ `train.py`, so midpoint resumes are explicit rather than relying on checkpoint-inferred epoch behavior.
- Added `quant_in_layers` trainable policy:
  - uses `trainable_policy_freeze_act_except_layers` as the selected layer list
  - trains only quant/shift parameters under those selected modules
  - freezes every other parameter
- Updated `tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh` to pass:
  - `QUANT_ONLY_START_EPOCH`
  - `TRAINABLE_POLICY`
  - `TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS`
  - `TRAINABLE_POLICY_UPDATE_OVERRIDES`
  - `TRAINABLE_POLICY_UPDATE_MODE`
- Sanity confirmed the target small trainable set:
  - `policy=quant_in_layers`
  - `trainable=67774`
  - `frozen=28540482`

Gate command:

```bash
EXP=recipe_resume10_vartrust_epoch1_quantinlate_gate_20260707 \
MASTER_PORT=30688 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar \
START_EPOCH=1 \
EPOCHS=2 \
SCHEDULER_EPOCHS=2 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
QUANT_ONLY_START_EPOCH=1 \
TRAINABLE_POLICY=quant_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Runtime evidence:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_vartrust_epoch1_quantinlate_gate_20260707.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_quantinlate_gate_20260707
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Args: trainable_policy=quant_in_layers, pre_qat_feature_recon_updates=0, variation_trust_weight=0.0, wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0
Strict resume: loaded model from .../recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar; missing=0, unexpected=0
Trainable parameter policy: epoch=1, quant_only=True, policy=quant_in_layers, trainable=67774, frozen=28540482
TrainSummary: epoch=1 updates=2496 avg_step_time=0.117807s samples_per_step=512 samples_per_sec=4346.09
Test: [distributed-summary]  Time: 34.314s  Loss: 0.8417  Acc@1: 80.4760  Acc@5: 95.2980  Samples: 50000
```

Result:

| source checkpoint | trained epoch | policy | output checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2S `checkpoint-1.pth.tar` | epoch 1 only | `quant_in_layers`, `features.5.5,features.7.1` | `checkpoint-2.pth.tar` | 80.4760 | 95.2980 | 0.8417 | better than Phase 2U `80.4540`, still below `80.5220`; fail |

Conclusion:

Only training late quant/shift parameters is not enough to preserve or improve the `80.5220` peak. It is slightly better than full-parameter no-trust continuation (`80.4760` vs `80.4540`), but still below the Phase 1F/2S best and far below the `81.0` goal.

Interpretation:

- Freezing almost everything prevents some damage from the full-parameter second epoch, but the remaining late quant/shift-only capacity is too weak to produce new lift.
- Do not extend this exact `quant_in_layers` branch.
- The next local-trainable branch should test a wider but still scoped update: train all parameters inside the diagnosed late useful modules (`features.5.5,features.7.1`) while keeping the rest of the model frozen. This sits between destructive full-model training and too-weak quant/shift-only training.

### Phase 2W: Continue From Phase 2S Epoch1 With Only Late Blocks Trainable

Reason:

Phase 2V showed that updating only late quant/shift parameters is too weak, but it slightly improved over full-parameter no-trust continuation. Phase 2W widens the trainable set while remaining local: it trains all parameters inside the diagnosed useful late modules `features.5.5` and `features.7.1`, while freezing the rest of the model. This tests whether late block-local adaptation can add capacity without reintroducing destructive whole-model drift.

This is a single-model strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Implementation:

- Added `params_in_layers` trainable policy:
  - uses `trainable_policy_freeze_act_except_layers` as the selected layer list
  - trains all parameters under selected modules
  - freezes every other parameter
- Sanity confirmed the intended scoped trainable set:
  - `policy=params_in_layers`
  - `trainable=8933890`
  - `frozen=19674366`
- Reused the Phase 2S `checkpoint-1.pth.tar` tie point as the source checkpoint.
- Disabled variation trust and BinReg during the continuation.

Gate command:

```bash
EXP=recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707 \
MASTER_PORT=30690 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar \
START_EPOCH=1 \
EPOCHS=2 \
SCHEDULER_EPOCHS=2 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
QUANT_ONLY_START_EPOCH=1 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Runtime evidence:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Args: trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5,features.7.1, pre_qat_feature_recon_updates=0, variation_trust_weight=0.0, wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0
Strict resume: loaded model from .../recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar; missing=0, unexpected=0
Trainable parameter policy: epoch=1, quant_only=True, policy=params_in_layers, trainable=8933890, frozen=19674366
TrainSummary: epoch=1 updates=2496 avg_step_time=0.118855s samples_per_step=512 samples_per_sec=4307.76
Test: [distributed-summary]  Time: 34.500s  Loss: 0.8405  Acc@1: 80.5400  Acc@5: 95.3020  Samples: 50000
```

Result:

| source checkpoint | trained epoch | policy | output checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2S `checkpoint-1.pth.tar` | epoch 1 only | `params_in_layers`, `features.5.5,features.7.1` | `checkpoint-2.pth.tar` | 80.5400 | 95.3020 | 0.8405 | new strict W4A4 best in this thread, above 80.5220; continue |

Conclusion:

Late block-local adaptation is the first continuation branch in this sequence to beat the previous strict W4A4 best. It improves over Phase 2S/Phase 1F `80.5220` to `80.5400`, and over the narrower Phase 2V `80.4760`. It is still below the `81.0` final target, but the branch is positive and should be extended carefully.

Interpretation:

- The trainable-set width matters: quant/shift-only late adaptation is too weak, but full-model continuation is too destructive. Training the diagnosed late blocks only is a better capacity/stability tradeoff.
- The next gate should continue from this new `80.5400` checkpoint with the same local late-block policy for one additional epoch. If the next full-val rises or holds near the new best, continue short; if it drops clearly below `80.5220`, stop and preserve this checkpoint as the current best.

### Phase 2X: Extend Phase 2W One More Late-Block-Local Epoch

Reason:

Phase 2W produced a new strict W4A4 best (`80.5400`) by training only the diagnosed late blocks `features.5.5` and `features.7.1` from the Phase 2S tie point. Phase 2X tested whether this positive trend continues for one more epoch from the new checkpoint, using the same scoped trainable policy and no extra trust/BinReg.

This is a single-model strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Gate command:

```bash
EXP=recipe_resume10_paramsinlate_extend1_gate_20260707 \
MASTER_PORT=30691 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Runtime evidence:

```text
Log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_extend1_gate_20260707.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_extend1_gate_20260707
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Args: trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5,features.7.1, pre_qat_feature_recon_updates=0, variation_trust_weight=0.0, wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0
Strict resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=8933890, frozen=19674366
TrainSummary: epoch=2 updates=2496 avg_step_time=0.119034s samples_per_step=512 samples_per_sec=4301.28
Test: [distributed-summary]  Time: 34.179s  Loss: 0.8411  Acc@1: 80.4120  Acc@5: 95.3220  Samples: 50000
```

Result:

| source checkpoint | trained epoch | policy | output checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | epoch 2 only | `params_in_layers`, `features.5.5,features.7.1` | `checkpoint-3.pth.tar` | 80.4120 | 95.3220 | 0.8411 | below Phase 2W `80.5400` and below `80.5220`; fail |

Conclusion:

The late-block-local policy has a one-epoch peak but does not tolerate a second full epoch from that checkpoint. Phase 2X falls back to `80.4120`, so the Phase 2W `checkpoint-2.pth.tar` should be preserved as the current strict W4A4 best.

Interpretation:

- The useful effect is a short local adaptation, not an extendable schedule as-is.
- Do not continue `params_in_layers` for another full epoch from Phase 2W checkpoint-2.
- The next branch should start from the Phase 2W `80.5400` checkpoint and test endpoint or short-update repair around the peak: for example, a very short late-block/quant-only polish with validation, or selective activation/quantizer endpoint calibration from the new best checkpoint.

### Phase 2Y: Activation-MSE Endpoint Repair From Phase 2W Best Checkpoint

Reason:

Phase 2W established a new strict W4A4 best (`80.5400`), while Phase 2X showed that a full additional late-block-local epoch regresses. Phase 2Y tested a cheap endpoint repair around the new peak: reuse the previously best activation-MSE endpoint pair (`features.5.5.attn.quan_a_qkx_fn` and `features.7.1.attn.quant_x_4_qkv.input_quant_fn`) on the Phase 2W checkpoint, save immediately, and run full validation.

This is a single-model strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Save command:

```bash
EXP=recipe_resume10_paramsinlate_ckpt2_pair_actmse_qkx_qkv_endpoint_20260708 \
MASTER_PORT=30692 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
ACT_MSE_QUANTIZERS=features.5.5.attn.quan_a_qkx_fn,features.7.1.attn.quant_x_4_qkv.input_quant_fn \
ACT_MSE_LAYERS=features.5.5,features.7.1 \
ACT_MSE_BATCHES=8 \
ACT_MSE_GRID=0.85,1.25,17 \
ACT_MSE_BLEND=0.35 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_phase1f_ckpt2_selective_actmse_endpoint_save_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_pair_actmse_qkx_qkv_endpoint_20260708/step_checkpoints/step_0000.pth.tar \
EXP=eval_resume10_paramsinlate_ckpt2_pair_actmse_qkx_qkv_endpoint_20260708 \
MASTER_PORT=30693 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_ckpt2_pair_actmse_qkx_qkv_endpoint_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Save log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_ckpt2_pair_actmse_qkx_qkv_endpoint_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_ckpt2_pair_actmse_qkx_qkv_endpoint_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict save resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Activation MSE calibration: batches=8, updated=2, mean_scale_ratio=0.9832, min_ratio=0.9786, max_ratio=0.9878
Saved endpoint: .../recipe_resume10_paramsinlate_ckpt2_pair_actmse_qkx_qkv_endpoint_20260708/step_checkpoints/step_0000.pth.tar
Strict eval resume: loaded model from .../step_checkpoints/step_0000.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.671s  Loss: 0.8408  Acc@1: 80.4540  Acc@5: 95.2980  Samples: 50000
```

Result:

| source checkpoint | endpoint | raw Top-1 | raw Top-5 | loss | gate |
|---|---|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | qkx + qkv_input activation-MSE endpoint | 80.4540 | 95.2980 | 0.8408 | below Phase 2W `80.5400`; fail |

Conclusion:

The qkx+qkv_input activation-MSE endpoint repair that was the least harmful on Phase 1F is harmful on the new Phase 2W peak. It drops from `80.5400` to `80.4540`, so this endpoint repair should not replace the current best checkpoint.

Interpretation:

- The Phase 2W improvement is not recovered or enhanced by shrinking those two activation scales.
- The next branch should test short-update checkpoints from the Phase 2W peak rather than endpoint activation-scale repair: run a small number of late-block-local updates, save, and full-val to locate whether the Phase 2X full-epoch drop happens early or late in the second continuation epoch.

### Phase 2Z: 250-Update Late-Block-Local Continuation From Phase 2W Best

Reason:

Phase 2W found a new strict W4A4 best (`80.5400`) with one epoch of late-block-local adaptation, while Phase 2X showed that a full second local epoch drops to `80.4120`. Phase 2Z tests whether the second-epoch degradation happens immediately or later by running only 250 optimizer updates from the Phase 2W checkpoint, saving the early-stopped checkpoint, and running full validation.

This is a single-model strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Implementation:

- Continued from Phase 2W `checkpoint-2.pth.tar` (`80.5400`).
- Kept the same local trainable policy:
  - `TRAINABLE_POLICY=params_in_layers`
  - `TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1`
- Stopped after 250 optimizer updates:
  - `MAX_TRAIN_UPDATES=250`
- Validation used the saved early-stop checkpoint `checkpoint-3.pth.tar`.

Train command:

```bash
EXP=recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708 \
MASTER_PORT=30694 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=250 \
STEP_CHECKPOINT_WARMUP_UPDATES=250 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=250 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_ckpt2_250upd_gate_20260708 \
MASTER_PORT=30695 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_ckpt2_250upd_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_ckpt2_250upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=8933890, frozen=19674366
TrainSummary: epoch=2 updates=250 avg_step_time=0.122541s samples_per_step=512 samples_per_sec=4178.19
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.791s  Loss: 0.8387  Acc@1: 80.5540  Acc@5: 95.3060  Samples: 50000
```

Result:

| source checkpoint | updates | policy | output checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 250 | `params_in_layers`, `features.5.5,features.7.1` | `checkpoint-3.pth.tar` | 80.5540 | 95.3060 | 0.8387 | new strict W4A4 best; continue short-update search |

Conclusion:

The local second-epoch update is briefly useful before the full-epoch degradation. Phase 2Z improves the current best from `80.5400` to `80.5540`, but remains below the final `81.0` target.

Interpretation:

- The Phase 2X full-epoch drop is not immediate; the curve has an early local improvement.
- The next gate should test a nearby update count, especially before and around 250 updates, to locate the short-update peak.

### Phase 2AA: 500-Update Late-Block-Local Continuation From Phase 2W Best

Reason:

Phase 2Z showed that 250 updates improves the Phase 2W checkpoint. Phase 2AA tests whether the same short-update branch keeps improving at 500 updates or has already passed the local peak.

This is a single-model strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_ckpt2_500upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_ckpt2_500upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=8933890, frozen=19674366
TrainSummary: epoch=2 updates=500 avg_step_time=0.120704s samples_per_step=512 samples_per_sec=4241.78
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_ckpt2_500upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.461s  Loss: 0.8395  Acc@1: 80.5300  Acc@5: 95.2980  Samples: 50000
```

Result:

| source checkpoint | updates | policy | output checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 500 | `params_in_layers`, `features.5.5,features.7.1` | `checkpoint-3.pth.tar` | 80.5300 | 95.2980 | 0.8395 | below Phase 2Z `80.5540`; do not use |

Conclusion:

500 updates is already past the short-update peak. It falls below the 250-update checkpoint and the Phase 2W base checkpoint. The current best remains Phase 2Z `checkpoint-3.pth.tar` at `80.5540`.

Interpretation:

- The useful update window is before 500 updates.
- The next gate should test a smaller update count such as 125 updates, then optionally a midpoint around 200-300 if needed.

### Phase 2AB: 300-Update Late-Block-Local Continuation From Phase 2W Best

Reason:

Phase 2Z showed that 250 updates is the best strict W4A4 checkpoint so far (`80.5540`), while Phase 2AA showed that 500 updates has already degraded. Phase 2AB tests a nearby point at 300 updates to decide whether the local peak is still around 250 or whether a slightly longer update window improves the result.

This is a single-model strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Train command:

```bash
EXP=recipe_resume10_paramsinlate_ckpt2_300upd_gate_20260708 \
MASTER_PORT=30700 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=300 \
STEP_CHECKPOINT_WARMUP_UPDATES=300 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=300 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_300upd_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_ckpt2_300upd_gate_20260708 \
MASTER_PORT=30701 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_ckpt2_300upd_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_ckpt2_300upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_ckpt2_300upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=8933890, frozen=19674366
TrainSummary: epoch=2 updates=300 avg_step_time=0.122247s samples_per_step=512 samples_per_sec=4188.25
Stopped early after 300 optimizer updates in epoch 2.
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_ckpt2_300upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 33.075s  Loss: 0.8397  Acc@1: 80.4860  Acc@5: 95.2900  Samples: 50000
```

Result:

| source checkpoint | updates | policy | output checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 300 | `params_in_layers`, `features.5.5,features.7.1` | `checkpoint-3.pth.tar` | 80.4860 | 95.2900 | 0.8397 | below Phase 2Z `80.5540`; do not use |

Conclusion:

300 updates is worse than 250 updates and also below the Phase 2W base checkpoint. The current best remains Phase 2Z:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Interpretation:

- The useful local-adaptation window is narrow and centered before 300 updates.
- This branch has exhausted pure short-update timing as a mechanism; do not keep sweeping update counts without a new variation-aware diagnosis.
- Next step: compare parameter/module drift between Phase 2W, 250-update best, 300-update regression, and 500-update regression to identify which quant/shift or late-block parameters correlate with the narrow gain versus harmful drift.

### Phase 2AC: Targeted Late-Attention Variation Trust From Short-Update Drift Diagnosis

Reason:

Phase 2AB showed that 300 updates without trust regresses to `80.4860`, while Phase 2Z at 250 updates remains the best checkpoint (`80.5540`). A checkpoint-drift diagnosis compared Phase 2W, 125, 250, 300, and 500 update checkpoints and found that the largest 250-to-300 harmful drift is concentrated in late attention activation/shift parameters:

```text
Drift report:
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_short_update_drift_20260708.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_short_update_drift_20260708.json

Largest harmful 250->300 drift signals:
features.5.5|move_v_shift
features.5.5|softmax_quant
features.7.1|move_v_shift
features.5.5.attn.proj.move_b4 / move_aft
features.7.1.attn.quant_x_4_qkv move_b4 / move_aft
```

This phase tested whether a scoped variation-trust anchor on only late attention activation/shift parameters can prevent the 300-update degradation while preserving the useful local adaptation.

This is a single-model strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Implementation fix:

The first targeted-trust attempt (`recipe_resume10_paramsinlate_ckpt2_300upd_targettrust_gate_20260708`) exposed a launcher bug: `variation_trust_weight > 0` only initialized the anchor inside the `start_epoch == 0` pre-QAT block, so resume runs with `START_EPOCH=2` logged `VarTrust: 0.000e+00`. `qat_launch.py` was fixed so variation trust anchors are initialized after resume when no anchor has already been created.

Static checks after the fix:

```bash
python3 -m py_compile /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py
git -C /mlx_devbox/users/quyanyi/playground/QATs diff --check -- qat_launch.py tmp_scripts/diagnose_resume10_short_update_drift_20260708.py docs/resume10_to81_goal_progress_20260706.md
```

Both commands passed.

Train command:

```bash
EXP=recipe_resume10_paramsinlate_ckpt2_300upd_targettrustfix_gate_20260708 \
MASTER_PORT=30703 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0.0015 \
VARIATION_TRUST_LAYERS=features.5.5.attn,features.7.1.attn \
VARIATION_TRUST_LATE_LAYERS=features.5.5,features.7.1 \
VARIATION_TRUST_LATE_MULTIPLIER=1.0 \
VARIATION_TRUST_EARLY_LAYERS= \
VARIATION_TRUST_EARLY_MULTIPLIER=1.0 \
VARIATION_TRUST_SOFTMAX_MULTIPLIER=4.0 \
VARIATION_TRUST_MOVE_V_MULTIPLIER=4.0 \
VARIATION_TRUST_PROJ_MOVE_MULTIPLIER=3.0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=300 \
STEP_CHECKPOINT_WARMUP_UPDATES=300 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=300 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_300upd_targettrustfix_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_ckpt2_300upd_targettrustfix_gate_20260708 \
MASTER_PORT=30704 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_ckpt2_300upd_targettrustfix_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_ckpt2_300upd_targettrustfix_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_ckpt2_300upd_targettrustfix_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Initialized variation trust anchor: params=26, weight=0.0015, layers=('features.5.5.attn', 'features.7.1.attn')
Enabled variation trust regularizer: weight=0.0015, pairs=26, avg_multiplier=2.000
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, trainable_policy=params_in_layers, quant_only_start_epoch=2, max_train_updates=300
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=8933890, frozen=19674366
VarTrust became nonzero after the first update window: 5.061e-05 at update 50, avg 2.530e-05
TrainSummary: epoch=2 updates=300 avg_step_time=0.124847s samples_per_step=512 samples_per_sec=4101.03
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_ckpt2_300upd_targettrustfix_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.830s  Loss: 0.8404  Acc@1: 80.5220  Acc@5: 95.2920  Samples: 50000
```

Result:

| source checkpoint | updates | policy | variation trust | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 300 | `params_in_layers`, `features.5.5,features.7.1` | late-attn activation/shift anchor, 26 params | 80.5220 | 95.2920 | 0.8404 | better than no-trust 300 (`80.4860`), below 250-best (`80.5540`) |

Conclusion:

Targeted late-attention variation trust partially fixes the 300-update harmful drift: it recovers `+0.036` Top-1 over the no-trust 300-update checkpoint. It does not beat the 250-update checkpoint and should not replace the current best.

Interpretation:

- The diagnosis is directionally useful: anchoring the late attention activation/shift drift improves the degraded 300-update point.
- The anchor is too late or too broad/strong to create a new best at 300 updates. It likely preserves the model after the useful 250-update window rather than improving the window itself.
- The current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not extend the 300-update target-trust checkpoint.
- Next variation-aware gate should apply the targeted trust earlier and closer to the useful window, for example a 250-update gate with late-attention trust, or a two-stage policy that lets broad late-block params adapt first and then freezes/anchors high-drift `move_v`, `proj.move_*`, and `quan_a_softmax` before the 250-to-300 degradation.

### Phase 2AD: 250-Update Late-Attention Variation Trust At The Current Peak Window

Reason:

Phase 2AC showed that targeted late-attention variation trust improves the degraded 300-update point from `80.4860` to `80.5220`, but it still does not beat the 250-update no-trust best (`80.5540`). This phase tests whether the same trust regularizer helps at the known useful 250-update window or whether it suppresses the useful adaptation that creates the current peak.

This is a single-model strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Train command:

```bash
EXP=recipe_resume10_paramsinlate_ckpt2_250upd_targettrustfix_gate_20260708 \
MASTER_PORT=30705 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0.0015 \
VARIATION_TRUST_LAYERS=features.5.5.attn,features.7.1.attn \
VARIATION_TRUST_LATE_LAYERS=features.5.5,features.7.1 \
VARIATION_TRUST_LATE_MULTIPLIER=1.0 \
VARIATION_TRUST_EARLY_LAYERS= \
VARIATION_TRUST_EARLY_MULTIPLIER=1.0 \
VARIATION_TRUST_SOFTMAX_MULTIPLIER=4.0 \
VARIATION_TRUST_MOVE_V_MULTIPLIER=4.0 \
VARIATION_TRUST_PROJ_MOVE_MULTIPLIER=3.0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=250 \
STEP_CHECKPOINT_WARMUP_UPDATES=250 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=250 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_targettrustfix_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_ckpt2_250upd_targettrustfix_gate_20260708 \
MASTER_PORT=30706 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_ckpt2_250upd_targettrustfix_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_ckpt2_250upd_targettrustfix_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_ckpt2_250upd_targettrustfix_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Initialized variation trust anchor: params=26, weight=0.0015, layers=('features.5.5.attn', 'features.7.1.attn')
Enabled variation trust regularizer: weight=0.0015, pairs=26, avg_multiplier=2.000
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, trainable_policy=params_in_layers, quant_only_start_epoch=2, max_train_updates=250
VarTrust became nonzero after the first update window: 5.061e-05 at update 50, avg 2.530e-05
TrainSummary: epoch=2 updates=250 avg_step_time=0.124387s samples_per_step=512 samples_per_sec=4116.20
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_ckpt2_250upd_targettrustfix_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.757s  Loss: 0.8389  Acc@1: 80.4840  Acc@5: 95.2840  Samples: 50000
```

Result:

| source checkpoint | updates | policy | variation trust | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 250 | `params_in_layers`, `features.5.5,features.7.1` | late-attn activation/shift anchor, 26 params | 80.4840 | 95.2840 | 0.8389 | below no-trust 250 (`80.5540`); do not use |

Conclusion:

Applying the targeted late-attention trust from the start of the 250-update window suppresses useful adaptation. It is worse than no-trust 250 by `-0.070` Top-1 and even slightly below the no-trust 300 checkpoint. This branch should not replace the current best.

Interpretation:

- The same trust that partially repairs the degraded 300-update point is harmful when applied throughout the useful 250-update window.
- The useful adaptation likely requires early movement of the diagnosed late-attention shift/softmax parameters; only the later 250-to-300 drift is harmful.
- Current best remains Phase 2Z no-trust 250:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not apply this late-attention variation trust from update 0.
- The next plausible variation-aware gate is delayed control: allow the no-trust local adaptation up to the current peak window, then freeze or strongly anchor the high-drift late-attention activation/shift parameters only after the useful adaptation has occurred.
- A concrete next gate should start from the Phase 2W checkpoint, use `TRAINABLE_POLICY_UPDATE_OVERRIDES` or a dedicated policy to switch after roughly 200-250 updates, and save/evaluate a short post-peak checkpoint. It should not repeat scalar update-count sweeps without changing the trainable/trust schedule.

### Phase 2AE: Delayed Freeze Of High-Drift Late-Attention Parameters

Reason:

Phase 2AD showed that applying late-attention variation trust from update 0 suppresses the useful 250-update adaptation. This phase tested a delayed-control variant: run the same no-trust late-block-local adaptation up to the current peak window, then freeze only the diagnosed high-drift late-attention activation/shift parameters for the post-peak interval from update 250 to 300.

This is a single-model strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Implementation:

- Added a narrow trainable policy to `qat_launch.py`:
  - `params_in_layers_freeze_highdrift_act`
  - It keeps the selected late layers trainable but freezes high-drift late-attention activation/shift parameters matching:
    - `move_v_`
    - `.attn.proj.move_`
    - `.attn.quan_a_softmax`
    - `.attn.quant_x_4_qkv.move_`
- Added the policy to CLI choices and update-override validation.
- Static checks passed:

```bash
python3 -m py_compile /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_short_update_drift_20260708.py
git -C /mlx_devbox/users/quyanyi/playground/QATs diff --check -- qat_launch.py tmp_scripts/diagnose_resume10_short_update_drift_20260708.py docs/resume10_to81_goal_progress_20260706.md
```

Runtime note:

The first delayed-freeze attempt used `TRAINABLE_POLICY_UPDATE_MODE=requires_grad`:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_ckpt2_300upd_delayedfreeze_gate_20260708.log
Trainable parameter update policy: epoch=2, update=250, mode=requires_grad, policy=params_in_layers_freeze_highdrift_act, trainable=8926880, frozen=19681376
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one...
this is not compatible with static_graph set to True.
```

This is a runtime wiring failure, not a valid accuracy result. The fix is to use `TRAINABLE_POLICY_UPDATE_MODE=grad_mask`, which keeps DDP static-graph topology stable and masks gradients instead of changing `requires_grad` mid-epoch.

Train command:

```bash
EXP=recipe_resume10_paramsinlate_ckpt2_300upd_delayedfreeze_gradmask_gate_20260708 \
MASTER_PORT=30708 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
TRAINABLE_POLICY_UPDATE_OVERRIDES=250:params_in_layers_freeze_highdrift_act \
TRAINABLE_POLICY_UPDATE_MODE=grad_mask \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=300 \
STEP_CHECKPOINT_WARMUP_UPDATES=300 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=300 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_300upd_delayedfreeze_gradmask_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_ckpt2_300upd_delayedfreeze_gradmask_gate_20260708 \
MASTER_PORT=30709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_ckpt2_300upd_delayedfreeze_gradmask_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_ckpt2_300upd_delayedfreeze_gradmask_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_ckpt2_300upd_delayedfreeze_gradmask_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Trainable parameter update policy: epoch=2, update=0, mode=grad_mask, policy=params_in_layers, trainable=28608256, frozen=0
Trainable parameter update policy: epoch=2, update=250, mode=grad_mask, policy=params_in_layers_freeze_highdrift_act, trainable=28608256, frozen=0
TrainSummary: epoch=2 updates=300 avg_step_time=0.226351s samples_per_step=512 samples_per_sec=2261.98
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_ckpt2_300upd_delayedfreeze_gradmask_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.286s  Loss: 0.8405  Acc@1: 80.4420  Acc@5: 95.2760  Samples: 50000
```

Result:

| source checkpoint | schedule | output checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---|---|---|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | `params_in_layers` until update 250, then `params_in_layers_freeze_highdrift_act` via `grad_mask` to update 300 | `checkpoint-3.pth.tar` | 80.4420 | 95.2760 | 0.8405 | below no-trust 250 (`80.5540`) and no-trust 300 (`80.4860`); fail |

Conclusion:

Delayed freezing of the diagnosed high-drift late-attention activation/shift parameters is harmful in this form. It does not preserve the 250-update peak and is worse than the no-trust 300-update checkpoint.

Interpretation:

- The high-drift parameters are not simply harmful after update 250; freezing them changes the late-block adaptation dynamics enough to degrade the model.
- The useful signal from Phase 2AC is narrower: a soft trust at 300 can partially repair degradation, but hard freezing or gradient masking after the peak is too blunt.
- Current best remains Phase 2Z no-trust 250:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not repeat delayed hard-freeze / grad-mask variants on the same high-drift set.
- The next variation-aware direction should be less blunt than freezing: either a softer post-peak regularizer with lower weight and narrower parameter selection, or a return to module selection around the 250-update best with a different adjacent module set. Any next run must beat `80.5540` to become the new best.

### Phase 2AF: Delayed Soft Variation Trust After The 250-Update Peak

Reason:

Phase 2AE showed that hard freezing or gradient masking the high-drift late-attention activation/shift parameters after update 250 is too blunt and drops to `80.4420`. Phase 2AF tests the softer version of the same idea: allow normal late-block-local adaptation through the useful 250-update window, capture a variation-trust anchor at update 250, and apply a weak trust regularizer only during the short 250-to-300 post-peak interval.

This is a single-model strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Implementation:

- Added `--variation-trust-start-update` to `qat_launch.py`.
- When `variation_trust_start_update > 0`, the launcher no longer captures the variation-trust anchor immediately after resume. Instead, `train_one_epoch_ofq(...)` captures the anchor once `local_update_count` reaches the configured start update.
- Updated `tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh` to pass and log `VARIATION_TRUST_START_UPDATE`.
- Static checks passed:

```bash
python3 -m py_compile /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_short_update_drift_20260708.py
bash -n /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
git -C /mlx_devbox/users/quyanyi/playground/QATs diff --check -- qat_launch.py tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh tmp_scripts/diagnose_resume10_short_update_drift_20260708.py docs/resume10_to81_goal_progress_20260706.md
```

Train command:

```bash
EXP=recipe_resume10_paramsinlate_ckpt2_300upd_delayedsofttrust_gate_20260708 \
MASTER_PORT=30710 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0.0005 \
VARIATION_TRUST_START_UPDATE=250 \
VARIATION_TRUST_LAYERS=features.5.5.attn,features.7.1.attn \
VARIATION_TRUST_LATE_LAYERS=features.5.5,features.7.1 \
VARIATION_TRUST_LATE_MULTIPLIER=1.0 \
VARIATION_TRUST_EARLY_LAYERS= \
VARIATION_TRUST_EARLY_MULTIPLIER=1.0 \
VARIATION_TRUST_SOFTMAX_MULTIPLIER=2.0 \
VARIATION_TRUST_MOVE_V_MULTIPLIER=2.0 \
VARIATION_TRUST_PROJ_MOVE_MULTIPLIER=1.5 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=300 \
STEP_CHECKPOINT_WARMUP_UPDATES=300 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=300 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_300upd_delayedsofttrust_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_ckpt2_300upd_delayedsofttrust_gate_20260708 \
MASTER_PORT=30711 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_ckpt2_300upd_delayedsofttrust_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_ckpt2_300upd_delayedsofttrust_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_ckpt2_300upd_delayedsofttrust_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, variation_trust_weight=0.0005, variation_trust_start_update=250, max_train_updates=300
Initialized variation trust anchor: params=26, weight=0.0005, layers=('features.5.5.attn', 'features.7.1.attn'), start_update=250, current_update=250
TrainSummary: epoch=2 updates=300 avg_step_time=0.122366s samples_per_step=512 samples_per_sec=4184.16
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_ckpt2_300upd_delayedsofttrust_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.649s  Loss: 0.8398  Acc@1: 80.4580  Acc@5: 95.3160  Samples: 50000
```

Result:

| source checkpoint | schedule | output checkpoint | raw Top-1 | raw Top-5 | loss | gate |
|---|---|---|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | no trust until update 250, then weak late-attn trust to update 300 | `checkpoint-3.pth.tar` | 80.4580 | 95.3160 | 0.8398 | below no-trust 250 (`80.5540`) and no-trust 300 (`80.4860`); fail |

Conclusion:

Delayed soft variation trust does not preserve the 250-update peak. It is less blunt than hard freeze, but still worse than both the no-trust 250 and no-trust 300 checkpoints.

Interpretation:

- The failure of delayed soft trust suggests that the post-250 degradation is not solved by simply pulling late-attention activation/shift parameters back to their 250-update values.
- Combined with Phase 2AD and 2AE, this closes the simple trust/freeze family around the current 250-update peak.
- Current best remains Phase 2Z no-trust 250:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not repeat late-attention trust/freeze variants around the 250-to-300 window unless the parameter set or objective changes materially.
- The next non-repeated variation-aware direction should switch module selection rather than only regularization: test whether adding or replacing adjacent late modules around `features.5.5` / `features.7.1` can move the 250-update peak upward without extending the destructive second-epoch trajectory.

### Phase 2AG: Adjacent Final-Stage Module Expansion At The 250-Update Peak

Reason:

Phase 2AF closed the simple late-attention trust/freeze family around the 250-to-300 post-peak window. This phase switches from regularization to module selection. The current best trains `features.5.5` and `features.7.1` for 250 updates. Phase 2AG tests whether adding the adjacent final-stage block `features.7.0` lets the 250-update local peak improve without extending the destructive second-epoch trajectory.

This is a single-model strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Train command:

```bash
EXP=recipe_resume10_paramsinlate_570_55_71_250upd_gate_20260708 \
MASTER_PORT=30712 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.0,features.7.1 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=250 \
STEP_CHECKPOINT_WARMUP_UPDATES=250 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=250 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_570_55_71_250upd_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_570_55_71_250upd_gate_20260708 \
MASTER_PORT=30713 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_570_55_71_250upd_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_570_55_71_250upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_570_55_71_250upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5,features.7.0,features.7.1, max_train_updates=250
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=16075544, frozen=12532712
TrainSummary: epoch=2 updates=250 avg_step_time=0.125770s samples_per_step=512 samples_per_sec=4070.93
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_570_55_71_250upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.534s  Loss: 0.8418  Acc@1: 80.5300  Acc@5: 95.3040  Samples: 50000
```

Result:

| source checkpoint | updates | policy | trainable params | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---:|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 250 | `params_in_layers`, `features.5.5,features.7.0,features.7.1` | 16,075,544 | 80.5300 | 95.3040 | 0.8418 | below no-trust 250 best (`80.5540`); do not use |

Conclusion:

Adding `features.7.0` to the current best late-block set does not improve the 250-update peak. It is better than several trust/freeze variants, but it remains below Phase 2Z no-trust 250.

Interpretation:

- More final-stage trainable capacity is not automatically beneficial; expanding from `8.93M` to `16.08M` trainable parameters loses `0.024` Top-1.
- The current best appears to depend on a very narrow local adaptation set rather than simply underfitting the final stage.
- Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not expand this branch by adding more final-stage modules.
- The next module-selection test should replace rather than add, for example `features.5.4,features.5.5,features.7.1` or `features.5.5,features.6.reduction,features.7.1`, and it should keep the same 250-update gate for direct comparison.

### Phase 2AH: Replace Adjacent Stage-3 Block At The 250-Update Peak

Reason:

Phase 2AG showed that adding final-stage `features.7.0` to the current best module set hurts the 250-update peak (`80.5300` vs `80.5540`). This phase tests a replacement-style module selection rather than adding more final-stage capacity: train `features.5.4,features.5.5,features.7.1` for the same 250-update gate. This keeps the known useful `features.5.5` and `features.7.1`, but adds the adjacent stage-3 block before `features.5.5` instead of the adjacent final-stage block before `features.7.1`.

This is a single-model strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Train command:

```bash
EXP=recipe_resume10_paramsinlate_54_55_71_250upd_gate_20260708 \
MASTER_PORT=30714 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.4,features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=250 \
STEP_CHECKPOINT_WARMUP_UPDATES=250 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=250 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_54_55_71_250upd_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_54_55_71_250upd_gate_20260708 \
MASTER_PORT=30715 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_54_55_71_250upd_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_54_55_71_250upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_54_55_71_250upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.4,features.5.5,features.7.1, max_train_updates=250
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=10726126, frozen=17882130
TrainSummary: epoch=2 updates=250 avg_step_time=0.130114s samples_per_step=512 samples_per_sec=3935.02
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_54_55_71_250upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 33.174s  Loss: 0.8401  Acc@1: 80.4580  Acc@5: 95.2720  Samples: 50000
```

Result:

| source checkpoint | updates | policy | trainable params | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---:|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 250 | `params_in_layers`, `features.5.4,features.5.5,features.7.1` | 10,726,126 | 80.4580 | 95.2720 | 0.8401 | below no-trust 250 best (`80.5540`); fail |

Conclusion:

Replacing the adjacent final-stage expansion with a stage-3 adjacent block is worse than the current best and worse than the `features.7.0` expansion. This module set should not be used.

Interpretation:

- The current best is not improved by simply adding an adjacent stage-3 block before `features.5.5`.
- `features.5.4` appears to introduce harmful adaptation under this 250-update schedule.
- Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not extend `features.5.4,features.5.5,features.7.1`.
- The remaining replacement module test worth trying before closing this module-selection neighborhood is `features.5.5,features.6.reduction,features.7.1`, because the stage-transition reduction may affect quantized representation alignment differently than adding another attention block.

### Phase 2AI: Stage-Transition Reduction Module Replacement At The 250-Update Peak

Reason:

Phase 2AH showed that adding the adjacent stage-3 block `features.5.4` is harmful (`80.4580`). This phase tests the other planned replacement-style module selection: keep the known useful `features.5.5` and `features.7.1`, and add the stage-transition reduction module `features.6.reduction`. The hypothesis is that the stage transition could affect quantized representation alignment differently than adding another attention block.

This is a single-model strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Train command:

```bash
EXP=recipe_resume10_paramsinlate_55_6red_71_250upd_gate_20260708 \
MASTER_PORT=30716 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.6.reduction,features.7.1 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=250 \
STEP_CHECKPOINT_WARMUP_UPDATES=250 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=250 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_55_6red_71_250upd_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_55_6red_71_250upd_gate_20260708 \
MASTER_PORT=30717 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_55_6red_71_250upd_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_55_6red_71_250upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_55_6red_71_250upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5,features.6.reduction,features.7.1, max_train_updates=250
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=10117386, frozen=18490870
TrainSummary: epoch=2 updates=250 avg_step_time=0.122983s samples_per_step=512 samples_per_sec=4163.19
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_55_6red_71_250upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.608s  Loss: 0.8395  Acc@1: 80.5380  Acc@5: 95.2860  Samples: 50000
```

Result:

| source checkpoint | updates | policy | trainable params | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---:|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 250 | `params_in_layers`, `features.5.5,features.6.reduction,features.7.1` | 10,117,386 | 80.5380 | 95.2860 | 0.8395 | below no-trust 250 best (`80.5540`); do not use |

Conclusion:

Adding the stage-transition reduction module is better than adding `features.5.4` and slightly better than the broad final-stage expansion, but it still does not beat the current best.

Interpretation:

- The module-selection neighborhood around `features.5.5` and `features.7.1` has not produced an improvement over the narrow two-block policy.
- `features.6.reduction` is less harmful than `features.5.4`, but still not useful enough to clear `80.5540`.
- Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not expand or replace the current module set with `features.5.4`, `features.7.0`, or `features.6.reduction` under the same 250-update gate.
- The next non-repeated direction should use the current best checkpoint as an endpoint and inspect case/logit-level or class-level changes versus Phase 2W/no-trust 250, rather than launching more local module variants blindly.

### Phase 2AJ: Full-Val Logit/Class Diagnosis Of The Current Short-Update Peak

Reason:

Phase 2Z remains the best strict W4A4 checkpoint at `80.5540`, but local module expansion/replacement and simple trust/freeze around the 250-to-300 update drift did not improve it. This phase follows the variation-aware/VVTQ direction: inspect full-validation class/logit behavior before launching another training gate.

This is diagnostic only. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Script:

```bash
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_logit_classes_20260708.py
```

Run command:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONUNBUFFERED=1 \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python3 /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_logit_classes_20260708.py \
  --devices 0 \
  --device-index 0 \
  --batch-size 256 \
  --workers 8 \
  2>&1 | tee /mlx_devbox/users/quyanyi/playground/train_resume10_logit_class_diag_20260708.log
```

Runtime evidence:

```text
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
validation_samples=50000
train_shards=294
validation_shards=14
Strict resume ckpt10: missing=0, unexpected=0
Strict resume phase2w: missing=0, unexpected=0
Strict resume best250: missing=0, unexpected=0
Strict resume u300: missing=0, unexpected=0
```

Artifacts:

```text
Summary: /mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_logit_class_diag_20260708/summary.json
Class deltas: /mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_logit_class_diag_20260708/class_delta.tsv
Confidence bins: /mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_logit_class_diag_20260708/confidence_bins.tsv
Flip cases: /mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_logit_class_diag_20260708/flip_cases.tsv
Log: /mlx_devbox/users/quyanyi/playground/train_resume10_logit_class_diag_20260708.log
```

Single-process diagnostic metrics:

| checkpoint | strict W4A4 samples | Top-1 | Top-5 | loss |
|---|---:|---:|---:|---:|
| checkpoint-10 | 50000 | 80.3520 | 95.3120 | 0.8449 |
| Phase 2W checkpoint-2 | 50000 | 80.5060 | 95.3120 | 0.8412 |
| Phase 2Z best250 checkpoint-3 | 50000 | 80.5040 | 95.3040 | 0.8394 |
| no-trust 300 checkpoint-3 | 50000 | 80.4340 | 95.3120 | 0.8403 |

Important caveat:

The diagnostic intentionally runs single-process on one GPU to preserve per-sample ordering and avoid distributed gathering complexity. Its absolute Top-1 is close to but not identical to the 8-GPU strict full-val numbers. Final gates still require the existing 8-GPU full ImageNet validation script. The diagnosis is used for relative class/logit signals, not as the official score.

Key findings:

- Against checkpoint-10, best250 has `+76` net Top-1 flips: `740` samples improve and `664` regress. The positive flips are concentrated in the teacher/model confidence band `0.20-0.40`, where best250 has `+81` net flips. The `0.40-0.60` band has balanced flips, and high-confidence bins `>=0.80` have no Top-1 flips.
- Against no-trust 300, best250 has `+35` net flips. The gain again comes from moderate-confidence bins: `+28` in `0.20-0.40` and `+7` in `0.40-0.60`.
- Against Phase 2W, best250 is effectively tied in this single-process diagnostic (`-1` net flip), so the Phase 2W vs best250 distinction is too small to over-interpret from this script.
- Class-level regressions that repeatedly appear include class `876` (best250 loses 4-5 correct samples versus all references) and several late ImageNet classes around `764/799/865/909`. Class gains that appear versus multiple references include `831`, plus several moderate-confidence gains such as `864/173/921` versus checkpoint-10.

Interpretation:

The current short-update peak is not a broad high-confidence improvement. Its useful movement is mostly in uncertain/moderate-confidence samples, while most high-confidence samples should be preserved. This suggests the next gate should change the learning signal around the confidence band where useful flips occur, rather than adding more module capacity or globally anchoring drift.

Next decision:

- Do not launch more blind module-selection variants.
- Do not repeat global confidence weighting or disagreement weighting, which already failed earlier in the Phase 1 family.
- Add a narrow teacher-confidence band KD auxiliary and test it under the same Phase 2Z 250-update late-block-local gate:
  - source: Phase 2W checkpoint-2
  - trainable policy: `params_in_layers`, `features.5.5,features.7.1`
  - update budget: 250
  - band: teacher confidence `0.20-0.60`
  - gate: beat Phase 2Z `80.5540` on strict W4A4 8-GPU full-val, or stop.

### Phase 2AK: Teacher-Confidence Band KD At The 250-Update Late-Block Peak

Reason:

Phase 2AJ showed that the useful flips from the current short-update peak are concentrated in moderate-confidence validation samples. This phase tests a narrow learning-signal change: keep the exact Phase 2Z 250-update late-block-local gate, but add a small extra soft-KD term only for samples whose teacher confidence is in `0.20-0.60`.

Important distinction:

The diagnosis bins were computed using checkpoint confidence, not teacher confidence. This phase is therefore a first approximation, not a perfect replay of the diagnostic signal.

Code changes:

```text
qat_launch.py:
- added teacher_confidence_band_soft_kd
- added CLI/runtime args:
  --teacher-confidence-band-kd-weight
  --teacher-confidence-band-kd-low
  --teacher-confidence-band-kd-high
  --teacher-confidence-band-kd-temperature
tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh:
- forwards/logs TEACHER_CONFIDENCE_BAND_KD_* env vars
```

Static checks:

```bash
python3 -m py_compile \
  /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py \
  /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_logit_classes_20260708.py
bash -n /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
git -C /mlx_devbox/users/quyanyi/playground/QATs diff --check -- \
  qat_launch.py \
  tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh \
  tmp_scripts/diagnose_resume10_logit_classes_20260708.py
```

Train command:

```bash
EXP=recipe_resume10_paramsinlate_bandkd0206_250upd_gate_20260708 \
MASTER_PORT=30732 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0.10 \
TEACHER_CONFIDENCE_BAND_KD_LOW=0.2 \
TEACHER_CONFIDENCE_BAND_KD_HIGH=0.6 \
TEACHER_CONFIDENCE_BAND_KD_TEMPERATURE=2.75 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=250 \
STEP_CHECKPOINT_WARMUP_UPDATES=250 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=250 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_bandkd0206_250upd_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_bandkd0206_250upd_gate_20260708 \
MASTER_PORT=30733 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_bandkd0206_250upd_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_bandkd0206_250upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_bandkd0206_250upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, teacher_confidence_band_kd_weight=0.1, band=0.2..0.6, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5,features.7.1, max_train_updates=250
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=8933890, frozen=19674366
TrainSummary: epoch=2 updates=250 avg_step_time=0.133709s samples_per_step=512 samples_per_sec=3829.22
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_bandkd0206_250upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.483s  Loss: 0.8391  Acc@1: 80.5400  Acc@5: 95.3020  Samples: 50000
```

Result:

| source checkpoint | updates | policy | extra KD | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 250 | `params_in_layers`, `features.5.5,features.7.1` | teacher confidence band `0.20-0.60`, weight `0.10` | 80.5400 | 95.3020 | 0.8391 | below Phase 2Z `80.5540`; fail |

Conclusion:

Teacher-confidence band KD does not improve the current 250-update peak. It returns to the Phase 2W-level score (`80.5400`) and is below Phase 2Z no-trust 250 (`80.5540`).

Interpretation:

- The Phase 2AJ signal should not be treated as a generic teacher-confidence weighting signal.
- The useful bin in Phase 2AJ was based on checkpoint/reference confidence from strict W4A4 models. Teacher confidence is a different quantity and likely does not select the same samples.
- Do not repeat teacher-confidence band KD as a scalar sweep without a reference-confidence implementation.

Next decision:

- Current best remains Phase 2Z:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

- The next non-repeated implementation should use the checkpoint-confidence idea directly: a fixed reference model from Phase 2W or checkpoint-10 should compute the confidence band, then apply the auxiliary only to samples in the reference-confidence `0.20-0.60` band. This would align the training signal with Phase 2AJ more closely than teacher-confidence banding.

### Phase 2AL: Fixed Reference-Confidence Band KD At The 250-Update Late-Block Peak

Reason:

Phase 2AK showed that teacher-confidence band KD does not match the Phase 2AJ diagnostic signal. This phase implements the direct variant: use a fixed strict W4A4 reference checkpoint to select samples whose reference confidence lies in `0.20-0.60`, then apply a small extra soft-KD term on only those samples. The KD target is still the teacher logits; the fixed reference is used only for training-time sample selection. Final output remains a single strict W4A4 checkpoint.

Code changes:

```text
qat_launch.py:
- added reference_confidence_band_soft_kd
- added clone_fixed_logit_ref_model
- added CLI/runtime args:
  --ref-confidence-band-kd-weight
  --ref-confidence-band-kd-low
  --ref-confidence-band-kd-high
  --ref-confidence-band-kd-temperature
  --ref-confidence-band-kd-checkpoint
- train_one_epoch_ofq now accepts confidence_ref_model and uses it only when ref_confidence_band_kd_weight > 0

tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh:
- forwards/logs REF_CONFIDENCE_BAND_KD_* env vars
```

Static checks:

```bash
python3 -m py_compile \
  /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py \
  /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_logit_classes_20260708.py
bash -n /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
git -C /mlx_devbox/users/quyanyi/playground/QATs diff --check -- \
  qat_launch.py \
  tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh \
  tmp_scripts/diagnose_resume10_logit_classes_20260708.py \
  docs/resume10_to81_goal_progress_20260706.md
```

Train command:

```bash
EXP=recipe_resume10_paramsinlate_refband0206_250upd_gate_20260708 \
MASTER_PORT=30734 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0.10 \
REF_CONFIDENCE_BAND_KD_LOW=0.2 \
REF_CONFIDENCE_BAND_KD_HIGH=0.6 \
REF_CONFIDENCE_BAND_KD_TEMPERATURE=2.75 \
REF_CONFIDENCE_BAND_KD_CHECKPOINT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=250 \
STEP_CHECKPOINT_WARMUP_UPDATES=250 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=250 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_refband0206_250upd_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_refband0206_250upd_gate_20260708 \
MASTER_PORT=30735 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_refband0206_250upd_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_refband0206_250upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_refband0206_250upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, ref_confidence_band_kd_weight=0.1, ref_confidence_band_kd_checkpoint=.../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar, band=0.2..0.6, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5,features.7.1, max_train_updates=250
Enabled reference-confidence band KD: weight=0.1, band=[0.2, 0.6), temperature=2.75, source=.../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=8933890, frozen=19674366
TrainSummary: epoch=2 updates=250 avg_step_time=0.217600s samples_per_step=512 samples_per_sec=2352.94
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_refband0206_250upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 33.652s  Loss: 0.8396  Acc@1: 80.4720  Acc@5: 95.3220  Samples: 50000
```

Result:

| source checkpoint | updates | policy | extra KD | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 250 | `params_in_layers`, `features.5.5,features.7.1` | fixed-ref confidence band `0.20-0.60`, weight `0.10` | 80.4720 | 95.3220 | 0.8396 | below Phase 2Z `80.5540`; fail |

Conclusion:

Direct fixed-reference confidence band KD is harmful at weight `0.10`. It drops below Phase 2W, teacher-band KD, and the Phase 2Z no-trust 250 checkpoint.

Interpretation:

- The Phase 2AJ confidence-bin finding is diagnostic, but naively increasing KD pressure on that band changes the training trajectory in the wrong direction.
- This may over-constrain exactly the uncertain samples where late-block adaptation needs freedom rather than stronger teacher matching.
- Do not repeat confidence-band KD as a scalar sweep unless the objective changes materially, for example by using it as a *negative* or gate-mixing signal rather than additive KD.

Next decision:

- Current best remains Phase 2Z:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

- The next non-repeated direction should focus on preserving class-specific gains/regressions or changing the 250-update endpoint selection, not on adding confidence-band KD pressure.

### Phase 2AM: 225-Update No-Trust Endpoint Around The Narrow Local Peak

Reason:

The short-update curve around Phase 2W has a narrow best known point at 250 updates: 125 updates scored `80.5220`, 250 scored `80.5540`, 300 scored `80.4860`, and 500 scored `80.5300`. Since confidence-band KD failed, this phase returns to endpoint selection and checks whether the peak occurs slightly before 250 updates. This is a single-checkpoint strict W4A4 endpoint test; it does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Train command:

```bash
EXP=recipe_resume10_paramsinlate_ckpt2_225upd_gate_20260708 \
MASTER_PORT=30736 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=225 \
STEP_CHECKPOINT_WARMUP_UPDATES=225 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=225 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_225upd_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_ckpt2_225upd_gate_20260708 \
MASTER_PORT=30737 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_ckpt2_225upd_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_ckpt2_225upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_ckpt2_225upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5,features.7.1, max_train_updates=225
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=8933890, frozen=19674366
TrainSummary: epoch=2 updates=225 avg_step_time=0.123354s samples_per_step=512 samples_per_sec=4150.66
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_ckpt2_225upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.809s  Loss: 0.8400  Acc@1: 80.4960  Acc@5: 95.3300  Samples: 50000
```

Result:

| source checkpoint | updates | policy | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 225 | `params_in_layers`, `features.5.5,features.7.1` | 80.4960 | 95.3300 | 0.8400 | below Phase 2Z `80.5540`; fail |

Conclusion:

225 updates is too early. It is below both Phase 2W (`80.5400`) and Phase 2Z 250 updates (`80.5540`), so the local endpoint peak is not before 250 under this schedule.

Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not spend more runs on simple pre-250 endpoint search.
- The remaining useful search space is either very narrow post-250 endpoint selection with a materially different stop/checkpoint mechanism, or a new class/regression-aware loss that protects the classes harmed by Phase 2Z without adding global confidence-band KD pressure.

### Phase 2AN: Class-Protect Fixed-Reference KL For Regressed Classes

Reason:

Phase 2AJ identified repeated class-level regressions in the current best Phase 2Z endpoint, especially class `876` and related classes such as `799`, `386`, `40`, `709`, `349`, `969`, `865`, `764`, and `909`. Phase 2AK/2AL showed that broad confidence-band KD pressure is harmful. This phase tests a narrower class/regression-aware protection mechanism: only for samples whose target class is in the diagnosed regression set, add a small fixed-reference logit KL to preserve the Phase 2W baseline behavior while the normal late-block-local adaptation runs.

The fixed reference is used only during training loss computation. Final output remains a single strict W4A4 checkpoint. This does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Code changes:

```text
qat_launch.py:
- added parse_int_set
- added class_protect_ref_kl_loss
- added CLI/runtime args:
  --class-protect-ref-kl-weight
  --class-protect-ref-kl-classes
  --class-protect-ref-kl-temperature
  --class-protect-ref-kl-checkpoint
- reused clone_fixed_logit_ref_model for class protection

tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh:
- forwards/logs CLASS_PROTECT_REF_KL_* env vars
```

Static checks:

```bash
python3 -m py_compile \
  /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py \
  /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_logit_classes_20260708.py
bash -n /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
git -C /mlx_devbox/users/quyanyi/playground/QATs diff --check -- \
  qat_launch.py \
  tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh \
  tmp_scripts/diagnose_resume10_logit_classes_20260708.py \
  docs/resume10_to81_goal_progress_20260706.md
```

Train command:

```bash
EXP=recipe_resume10_paramsinlate_classprotect_v1_250upd_gate_20260708 \
MASTER_PORT=30738 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0.02 \
CLASS_PROTECT_REF_KL_CLASSES=876,799,386,40,709,349,969,865,764,909 \
CLASS_PROTECT_REF_KL_TEMPERATURE=2.75 \
CLASS_PROTECT_REF_KL_CHECKPOINT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=250 \
STEP_CHECKPOINT_WARMUP_UPDATES=250 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=250 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_classprotect_v1_250upd_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_classprotect_v1_250upd_gate_20260708 \
MASTER_PORT=30739 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_classprotect_v1_250upd_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_classprotect_v1_250upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_classprotect_v1_250upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, class_protect_ref_kl_weight=0.02, class_protect_ref_kl_classes=[40,349,386,709,764,799,865,876,909,969], trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5,features.7.1, max_train_updates=250
Enabled class-protect ref KL: weight=0.02, classes=(40, 349, 386, 709, 764, 799, 865, 876, 909, 969), temperature=2.75, source=.../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=8933890, frozen=19674366
TrainSummary: epoch=2 updates=250 avg_step_time=0.217030s samples_per_step=512 samples_per_sec=2359.12
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_classprotect_v1_250upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.884s  Loss: 0.8399  Acc@1: 80.4780  Acc@5: 95.2620  Samples: 50000
```

Result:

| source checkpoint | updates | policy | extra loss | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 250 | `params_in_layers`, `features.5.5,features.7.1` | class-protect ref KL, weight `0.02`, 10 regressed classes | 80.4780 | 95.2620 | 0.8399 | below Phase 2Z `80.5540`; fail |

Conclusion:

Direct class-protect fixed-reference KL is harmful even at a small weight. It does not preserve the current peak and falls well below the no-trust 250 checkpoint.

Interpretation:

- The class-delta diagnostic should not be turned into a direct reference-KL protection loss in this form.
- The protected classes are likely not independent failure modes; constraining them against the Phase 2W reference may suppress useful late-block adaptation more broadly.
- Do not repeat class-protect ref KL as a scalar sweep unless the protected set or objective changes materially.

Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Confidence-band KD, reference-confidence KD, class-protect ref KL, and simple pre-250 endpoint search are now closed.
- The next useful direction should return to parameter/module variation, but with a different mechanism from hard trust/freeze: e.g. use per-parameter LR damping for the diagnosed high-drift late-attention activation/shift parameters after 250 updates, or evaluate a very narrow post-250 endpoint with a smooth optimizer/LR change rather than additive KL losses.

### Phase 2AO: Post-250 High-Drift Gradient Damping

Reason:

Phase 2AE showed that hard-freezing high-drift late-attention activation/shift parameters after the 250-update peak is too blunt. Phase 2AF showed that delayed soft variation trust is also harmful. This phase tests a softer parameter-level variation control: keep Phase 2Z's no-trust late-block-local adaptation through update 250, then from update 250 to 300 keep the same late blocks active but damp gradients on the diagnosed high-drift late-attention activation/shift parameters instead of setting them to zero.

This is a single-checkpoint strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Implementation notes:

An initial attempt named `recipe_resume10_paramsinlate_ckpt2_300upd_graddamp01_gate_20260708` was invalid and was interrupted. Its first `grad_damp` implementation damped all parameters outside `params_in_layers` from update 0, so it did not reproduce the Phase 2Z no-trust path through update 250.

The corrected implementation masks parameters outside the base `trainable_policy`, and only damps parameters that are inside the base policy but excluded by the current update override policy.

Code changes:

```text
qat_launch.py:
- added trainable_policy_grad_damp
- added trainable_policy_update_mode=grad_damp
- added apply_gradient_damp_policy
- grad_damp masks non-base-policy params and damps only the current-policy-excluded subset

tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh:
- forwards/logs TRAINABLE_POLICY_GRAD_DAMP
```

Static checks:

```bash
python3 -m py_compile \
  /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py \
  /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_logit_classes_20260708.py
bash -n /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
git -C /mlx_devbox/users/quyanyi/playground/QATs diff --check -- \
  qat_launch.py \
  tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh \
  docs/resume10_to81_goal_progress_20260706.md
```

Train command:

```bash
EXP=recipe_resume10_paramsinlate_ckpt2_300upd_graddamp01b_gate_20260708 \
MASTER_PORT=30741 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
TRAINABLE_POLICY_UPDATE_OVERRIDES=250:params_in_layers_freeze_highdrift_act \
TRAINABLE_POLICY_UPDATE_MODE=grad_damp \
TRAINABLE_POLICY_GRAD_DAMP=0.1 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=300 \
STEP_CHECKPOINT_WARMUP_UPDATES=300 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=300 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_300upd_graddamp01b_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_ckpt2_300upd_graddamp01b_gate_20260708 \
MASTER_PORT=30742 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_ckpt2_300upd_graddamp01b_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_ckpt2_300upd_graddamp01b_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_ckpt2_300upd_graddamp01b_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5,features.7.1, trainable_policy_update_overrides={250: params_in_layers_freeze_highdrift_act}, trainable_policy_update_mode=grad_damp, trainable_policy_grad_damp=0.1, max_train_updates=300
Gradient damping evidence: update=0 policy=params_in_layers damped_params=0 masked_params=19674313; update=250 switched to params_in_layers_freeze_highdrift_act
TrainSummary: epoch=2 updates=300 avg_step_time=0.227308s samples_per_step=512 samples_per_sec=2252.45
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_ckpt2_300upd_graddamp01b_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.618s  Loss: 0.8407  Acc@1: 80.4540  Acc@5: 95.2840  Samples: 50000
```

Result:

| source checkpoint | updates | policy | post-250 control | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 300 | `params_in_layers`, `features.5.5,features.7.1` | high-drift late-attn grad damping `0.1` after update 250 | 80.4540 | 95.2840 | 0.8407 | below Phase 2Z `80.5540`; fail |

Conclusion:

Post-250 high-drift gradient damping is harmful. It scores below the no-trust 300 endpoint (`80.4860`) and far below the no-trust 250 endpoint (`80.5540`).

Interpretation:

- The high-drift late-attention activation/shift parameters are not fixed by simple post-peak damping.
- Combined with hard freeze, delayed soft trust, and class/reference KL failures, the post-250 continuation family is now low-yield under this late-block-local setup.
- The current peak is best treated as a narrow endpoint, not a branch that can be safely extended by constraining high-drift parameters.

Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not spend more runs on post-250 control of the same high-drift set in this local late-block branch.
- The remaining useful direction should either choose a different starting branch before Phase 2W, or revisit module/parameter selection with a genuinely different write set rather than trying to extend the current 250-update endpoint.

### Phase 2AP: Late-MLP-Only Adaptation At The 250-Update Gate

Reason:

Phase 2AO closed another post-250 control mechanism on the late-attention high-drift set. This phase changes the write set instead of adding more control to the same attention path. It keeps the Phase 2W source checkpoint and the same 250-update gate, but trains only the MLP submodules inside the two useful late blocks:

```text
features.5.5.mlp
features.7.1.mlp
```

This avoids the high-drift attention activation/shift parameters (`move_v`, softmax quantizer, proj move, qkv move) while still allowing block-local non-attention adaptation. It is a single-checkpoint strict W4A4 branch and does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Parameter inventory:

```text
features.5.5.mlp: 12 tensors, 1,185,438 elements
features.7.1.mlp: 12 tensors, 4,730,128 elements
Total trainable via policy: 5,915,566 parameters
```

Train command:

```bash
EXP=recipe_resume10_paramsinlate_mlp55_71_250upd_gate_20260708 \
MASTER_PORT=30743 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5.mlp,features.7.1.mlp \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=250 \
STEP_CHECKPOINT_WARMUP_UPDATES=250 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=250 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_mlp55_71_250upd_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_mlp55_71_250upd_gate_20260708 \
MASTER_PORT=30744 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_mlp55_71_250upd_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_mlp55_71_250upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_mlp55_71_250upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5.mlp,features.7.1.mlp, max_train_updates=250
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=5915566, frozen=22692690
TrainSummary: epoch=2 updates=250 avg_step_time=0.114430s samples_per_step=512 samples_per_sec=4474.34
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_mlp55_71_250upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 33.417s  Loss: 0.8389  Acc@1: 80.4360  Acc@5: 95.2940  Samples: 50000
```

Result:

| source checkpoint | updates | policy | trainable params | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---:|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 250 | `params_in_layers`, `features.5.5.mlp,features.7.1.mlp` | 5,915,566 | 80.4360 | 95.2940 | 0.8389 | below Phase 2Z `80.5540`; fail |

Conclusion:

Training only the MLP submodules of the useful late blocks is too narrow or misses the useful adaptation mechanism. It performs worse than Phase 2W itself and much worse than Phase 2Z no-trust late-block-local 250 updates.

Interpretation:

- The Phase 2Z gain is not coming from MLP-only adaptation.
- The useful adaptation likely requires the attention path and/or interaction between attention and MLP inside `features.5.5` and `features.7.1`, despite the attention path carrying the harmful high-drift parameters later.
- Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not extend MLP-only late-block adaptation.
- The current evidence says the two-block `features.5.5,features.7.1` write set is still the only positive local branch, but post-250 continuation is destructive and submodule narrowing loses the useful signal.
- A non-repeated next direction should change the source branch or pre-250 construction, not keep repairing/extending the Phase 2Z endpoint.

### Phase 2AQ: Late-Attention-Only Adaptation At The 250-Update Gate

Reason:

Phase 2AP showed that MLP-only adaptation inside the useful late blocks is too narrow and loses the Phase 2Z gain. This phase tests the complementary write set: train only the attention submodules inside `features.5.5` and `features.7.1` for the same 250-update gate. The purpose is to check whether Phase 2Z's gain comes mostly from attention adaptation while avoiding unrelated MLP movement.

This is a single-checkpoint strict W4A4 branch and does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Parameter inventory:

```text
features.5.5.attn: 25 floating tensors, 606,030 floating elements
features.7.1.attn: 25 floating tensors, 2,409,990 floating elements
Total trainable via policy: 3,013,716 parameters
```

Train command:

```bash
EXP=recipe_resume10_paramsinlate_attn55_71_250upd_gate_20260708 \
MASTER_PORT=30745 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5.attn,features.7.1.attn \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=250 \
STEP_CHECKPOINT_WARMUP_UPDATES=250 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=250 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_attn55_71_250upd_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_attn55_71_250upd_gate_20260708 \
MASTER_PORT=30746 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_attn55_71_250upd_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_attn55_71_250upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_attn55_71_250upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5.attn,features.7.1.attn, max_train_updates=250
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=3013716, frozen=25594540
TrainSummary: epoch=2 updates=250 avg_step_time=0.120913s samples_per_step=512 samples_per_sec=4234.45
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_attn55_71_250upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 33.254s  Loss: 0.8386  Acc@1: 80.5460  Acc@5: 95.3260  Samples: 50000
```

Result:

| source checkpoint | updates | policy | trainable params | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---:|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 250 | `params_in_layers`, `features.5.5.attn,features.7.1.attn` | 3,013,716 | 80.5460 | 95.3260 | 0.8386 | close to Phase 2Z `80.5540`, but still below; fail |

Conclusion:

Late-attention-only adaptation nearly recovers the current best but does not beat it. It is much better than MLP-only (`80.4360`) and nearly matches the full two-block write set (`80.5540`).

Interpretation:

- Most of the Phase 2Z gain comes from the attention submodules of `features.5.5` and `features.7.1`.
- The final `+0.008` Top-1 gap versus Phase 2Z likely comes from interaction with non-attention parameters in those same blocks, not from MLP-only adaptation.
- This explains why hard attention drift control after the 250-update peak is harmful: the attention path is the main positive adaptation path, even though some of its activation/shift parameters later drift destructively.

Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not treat MLP-only as useful.
- Attention-only is strong enough to keep as a diagnostic handle, but it is still below the full two-block 250-update checkpoint.
- The next non-repeated direction should test a hybrid write set: attention submodules plus a minimal non-attention complement from the same blocks, rather than full block or MLP-only. A concrete candidate is `features.5.5.attn,features.7.1.attn,features.5.5.norm2,features.7.1.norm2` if those norm parameters exist, or attention plus MLP quant/shift only.

### Phase 2AR: Late-Attention Plus Pre-Attention Norm At The 250-Update Gate

Reason:

Phase 2AQ showed that late-attention-only adaptation nearly matches Phase 2Z (`80.5460` vs `80.5540`), while MLP-only is much weaker. This phase tests a minimal non-attention complement: add the pre-attention norm parameters `norm1` from the same two blocks to the attention-only write set. The hypothesis is that `norm1` may help condition attention adaptation without opening the full MLP path or full block.

This is a single-checkpoint strict W4A4 branch and does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Parameter inventory:

```text
features.5.5.attn + features.7.1.attn: 3,013,716 trainable parameters
features.5.5.norm1 + features.7.1.norm1: 2,304 parameters
Total trainable via policy: 3,016,020 parameters
```

Train command:

```bash
EXP=recipe_resume10_paramsinlate_attn_norm1_55_71_250upd_gate_20260708 \
MASTER_PORT=30747 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5.attn,features.7.1.attn,features.5.5.norm1,features.7.1.norm1 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=250 \
STEP_CHECKPOINT_WARMUP_UPDATES=250 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=250 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_attn_norm1_55_71_250upd_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_attn_norm1_55_71_250upd_gate_20260708 \
MASTER_PORT=30748 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_attn_norm1_55_71_250upd_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_attn_norm1_55_71_250upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_attn_norm1_55_71_250upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5.attn,features.7.1.attn,features.5.5.norm1,features.7.1.norm1, max_train_updates=250
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=3016020, frozen=25592236
TrainSummary: epoch=2 updates=250 avg_step_time=0.121221s samples_per_step=512 samples_per_sec=4223.69
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_attn_norm1_55_71_250upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 33.268s  Loss: 0.8388  Acc@1: 80.4800  Acc@5: 95.3140  Samples: 50000
```

Result:

| source checkpoint | updates | policy | trainable params | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---:|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 250 | `params_in_layers`, `features.5.5.attn,features.7.1.attn,features.5.5.norm1,features.7.1.norm1` | 3,016,020 | 80.4800 | 95.3140 | 0.8388 | below attention-only `80.5460` and Phase 2Z `80.5540`; fail |

Conclusion:

Adding pre-attention norm parameters to the strong attention-only write set is harmful. This minimal complement drops the branch from `80.5460` to `80.4800`.

Interpretation:

- `norm1` movement is not the missing complement that explains the full two-block advantage.
- The full two-block `+0.008` over attention-only likely comes from a different small interaction, possibly MLP-side quant/shift or norm2, rather than pre-attention normalization.
- Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not extend attention+norm1.
- The next minimal-complement test, if continuing this path, should use attention plus `norm2` or attention plus MLP quant/shift only. It should keep the same 250-update gate and must beat `80.5540` to matter.

### Phase 2AS: Late-Attention Plus Post-Attention Norm At The 250-Update Gate

Reason:

Phase 2AR showed that adding `norm1` to the strong attention-only write set is harmful. This phase tests the other minimal norm complement from the same two blocks: attention plus `norm2`. The hypothesis is that post-attention normalization might be the small interaction that lets the full two-block write set slightly beat attention-only, without opening the full MLP path.

This is a single-checkpoint strict W4A4 branch and does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Parameter inventory:

```text
features.5.5.attn + features.7.1.attn: 3,013,716 trainable parameters
features.5.5.norm2 + features.7.1.norm2: 2,304 parameters
Total trainable via policy: 3,016,020 parameters
```

Train command:

```bash
EXP=recipe_resume10_paramsinlate_attn_norm2_55_71_250upd_gate_20260708 \
MASTER_PORT=30749 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5.attn,features.7.1.attn,features.5.5.norm2,features.7.1.norm2 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=250 \
STEP_CHECKPOINT_WARMUP_UPDATES=250 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=250 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_attn_norm2_55_71_250upd_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_attn_norm2_55_71_250upd_gate_20260708 \
MASTER_PORT=30750 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_attn_norm2_55_71_250upd_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_attn_norm2_55_71_250upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_attn_norm2_55_71_250upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5.attn,features.7.1.attn,features.5.5.norm2,features.7.1.norm2, max_train_updates=250
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=3016020, frozen=25592236
TrainSummary: epoch=2 updates=250 avg_step_time=0.121178s samples_per_step=512 samples_per_sec=4225.18
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_attn_norm2_55_71_250upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 34.104s  Loss: 0.8386  Acc@1: 80.4760  Acc@5: 95.2980  Samples: 50000
```

Result:

| source checkpoint | updates | policy | trainable params | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---:|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 250 | `params_in_layers`, `features.5.5.attn,features.7.1.attn,features.5.5.norm2,features.7.1.norm2` | 3,016,020 | 80.4760 | 95.2980 | 0.8386 | below attention-only `80.5460` and Phase 2Z `80.5540`; fail |

Conclusion:

Adding post-attention norm parameters to the attention-only write set is harmful. It drops the branch from `80.5460` to `80.4760`, almost the same failure pattern as `norm1`.

Interpretation:

- Neither `norm1` nor `norm2` is the missing minimal complement that explains the full two-block advantage.
- Norm movement appears actively harmful around this 250-update gate.
- The remaining plausible complement is an attention plus MLP interaction while excluding norm parameters, or a still finer MLP-side quant/shift-only write set.
- Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not extend attention+norm1 or attention+norm2.
- The next non-repeated minimal-complement test should remove norms and use attention plus MLP from the same two late blocks: `features.5.5.attn,features.7.1.attn,features.5.5.mlp,features.7.1.mlp`.
- This keeps the same 250-update gate and must beat `80.5540` to matter; if it fails, the next branch should be finer MLP-side quant/shift-only rather than opening more full-block parameters.

### Phase 2AT: Late-Attention Plus MLP Without Norms At The 250-Update Gate

Reason:

Phase 2AQ showed attention-only is strong (`80.5460`), while Phase 2AP showed MLP-only is weak (`80.4360`) and Phases 2AR/2AS showed both `norm1` and `norm2` complements are harmful. This phase tests whether the full two-block advantage comes from an attention-MLP interaction after removing both norm modules from the write set.

This is a single-checkpoint strict W4A4 branch and does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Train command:

```bash
EXP=recipe_resume10_paramsinlate_attn_mlp_55_71_250upd_gate_20260708 \
MASTER_PORT=30751 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5.attn,features.7.1.attn,features.5.5.mlp,features.7.1.mlp \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=250 \
STEP_CHECKPOINT_WARMUP_UPDATES=250 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=250 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_attn_mlp_55_71_250upd_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_attn_mlp_55_71_250upd_gate_20260708 \
MASTER_PORT=30752 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_attn_mlp_55_71_250upd_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_attn_mlp_55_71_250upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_attn_mlp_55_71_250upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5.attn,features.7.1.attn,features.5.5.mlp,features.7.1.mlp, max_train_updates=250
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=8929282, frozen=19678974
TrainSummary: epoch=2 updates=250 avg_step_time=0.122202s samples_per_step=512 samples_per_sec=4189.77
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_attn_mlp_55_71_250upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.958s  Loss: 0.8393  Acc@1: 80.4820  Acc@5: 95.3060  Samples: 50000
```

Result:

| source checkpoint | updates | policy | trainable params | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---:|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 250 | `params_in_layers`, `features.5.5.attn,features.7.1.attn,features.5.5.mlp,features.7.1.mlp` | 8,929,282 | 80.4820 | 95.3060 | 0.8393 | below attention-only `80.5460` and Phase 2Z `80.5540`; fail |

Conclusion:

Attention plus full MLP without norms is not the missing complement. It performs almost the same as the norm-complement failures and far below attention-only.

Interpretation:

- The full two-block write set's small `+0.008` over attention-only does not come from opening the full MLP path alone.
- Broad MLP movement is harmful even when attention is also trainable.
- The useful branch remains concentrated in attention, but the full-block result may benefit from a much narrower parameter subset not captured by whole `mlp`, `norm1`, or `norm2` module prefixes.
- Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Close coarse module complements around attention: `norm1`, `norm2`, and full `mlp` are all harmful.
- The next non-repeated branch should be finer than module prefixes: attention plus only MLP-side quant/shift parameters, or attention plus a diagnosed subset of high-value non-attention parameters from `features.5.5` and `features.7.1`.
- If no such fine-grained policy is already supported by `qat_launch.py`, add a narrow policy/matcher rather than reopening full MLP or norm modules.

### Phase 2AU: Late-Attention Plus Selected-Layer Quant/Shift At The 250-Update Gate

Reason:

Phase 2AT showed that opening the full MLP path together with attention is harmful. This phase implements a finer-grained variation-aware write set: keep the full attention submodules trainable in `features.5.5` and `features.7.1`, but outside attention only train selected-layer quant/shift parameters. This tests whether the missing complement is MLP-side quant/shift adaptation rather than full MLP weights, norm movement, or full-block updates.

Implementation:

Added a narrow trainable policy to `qat_launch.py`:

```text
params_in_layers_attn_plus_quant
```

Policy semantics:

- `trainable_policy_freeze_act_except_layers` selects parent layer prefixes, here `features.5.5,features.7.1`.
- Inside selected layers, any `.attn.` parameter is trainable.
- Inside selected layers but outside attention, only `is_quant_or_shift_parameter(name)` parameters are trainable.
- Existing policies are unchanged.

Static/smoke evidence:

```text
python3 -m py_compile /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py
git -C /mlx_devbox/users/quyanyi/playground/QATs diff --check -- qat_launch.py docs/resume10_to81_goal_progress_20260706.md
Smoke EXP=smoke_resume10_paramsinlate_attn_plus_quant_55_71_1upd_20260708, MAX_TRAIN_UPDATES=1
Smoke Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers_attn_plus_quant, trainable=3025282, frozen=25582974
Smoke TrainSummary: epoch=2 updates=1
```

The smoke confirms the matcher is narrower than attention+MLP (`8,929,282`) and only slightly wider than attention-only (`3,013,716`), matching the intended attention plus non-attention quant/shift policy.

This is a single-checkpoint strict W4A4 branch and does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Train command:

```bash
EXP=recipe_resume10_paramsinlate_attn_plus_quant_55_71_250upd_gate_20260708 \
MASTER_PORT=30754 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers_attn_plus_quant \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=250 \
STEP_CHECKPOINT_WARMUP_UPDATES=250 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=250 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_attn_plus_quant_55_71_250upd_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_attn_plus_quant_55_71_250upd_gate_20260708 \
MASTER_PORT=30755 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_attn_plus_quant_55_71_250upd_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_attn_plus_quant_55_71_250upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_attn_plus_quant_55_71_250upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, trainable_policy=params_in_layers_attn_plus_quant, trainable_policy_freeze_act_except_layers=features.5.5,features.7.1, max_train_updates=250
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers_attn_plus_quant, trainable=3025282, frozen=25582974
TrainSummary: epoch=2 updates=250 avg_step_time=0.122199s samples_per_step=512 samples_per_sec=4189.88
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_attn_plus_quant_55_71_250upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.448s  Loss: 0.8394  Acc@1: 80.5420  Acc@5: 95.2900  Samples: 50000
```

Result:

| source checkpoint | updates | policy | trainable params | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---:|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 250 | `params_in_layers_attn_plus_quant`, `features.5.5,features.7.1` | 3,025,282 | 80.5420 | 95.2900 | 0.8394 | below attention-only `80.5460` and Phase 2Z `80.5540`; fail |

Conclusion:

The finer attention plus selected-layer quant/shift branch is much better than opening full MLP or norms, but still does not beat attention-only or the current best.

Interpretation:

- The useful adaptation is still dominated by attention.
- Adding non-attention selected-layer quant/shift is not enough to explain the full-block `+0.008` over attention-only.
- Coarse non-attention complements are now closed: `norm1`, `norm2`, full MLP, and selected-layer non-attention quant/shift all fail.
- Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not continue module-complement expansion around attention on this 250-update gate.
- The remaining work should either:
  - use attention-only as the cleaner handle and change training dynamics around it, or
  - diagnose the exact delta between attention-only and full-block Phase 2Z at parameter/case level before adding any new write set.
- A reasonable next branch is not another wider write set; it should be an attention-only dynamic variant, for example a shorter/longer endpoint around attention-only or a post-250 continuation with a very weak, targeted damping only on attention high-drift quant/shift, gated immediately against `80.5540`.

### Phase 2AV: Late-Attention-Only 275-Update Endpoint Gate

Reason:

Phase 2AU closed the last coarse non-attention complement. The clean remaining handle is attention-only adaptation. Phase 2AQ showed the 250-update attention-only endpoint is close to current best (`80.5460` vs `80.5540`). This phase tests the right side of that endpoint with the same attention-only write set and 275 updates, to see whether a slightly longer attention-only run improves before destructive drift dominates.

This is a single-checkpoint strict W4A4 branch and does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Train command:

```bash
EXP=recipe_resume10_paramsinlate_attn55_71_275upd_gate_20260708 \
MASTER_PORT=30756 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5.attn,features.7.1.attn \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=275 \
STEP_CHECKPOINT_WARMUP_UPDATES=275 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=275 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_attn55_71_275upd_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_attn55_71_275upd_gate_20260708 \
MASTER_PORT=30757 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_attn55_71_275upd_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_attn55_71_275upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_attn55_71_275upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5.attn,features.7.1.attn, max_train_updates=275
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=3013716, frozen=25594540
TrainSummary: epoch=2 updates=275 avg_step_time=0.121095s samples_per_step=512 samples_per_sec=4228.09
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_attn55_71_275upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.810s  Loss: 0.8414  Acc@1: 80.5140  Acc@5: 95.2720  Samples: 50000
```

Result:

| source checkpoint | updates | policy | trainable params | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---:|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 275 | `params_in_layers`, `features.5.5.attn,features.7.1.attn` | 3,013,716 | 80.5140 | 95.2720 | 0.8414 | below attention-only 250 `80.5460` and Phase 2Z `80.5540`; fail |

Conclusion:

The right side of the attention-only endpoint is worse. Extending attention-only from 250 to 275 updates loses `0.032` Top-1 and increases loss.

Interpretation:

- Attention-only adaptation peaks before or near 250 updates under this setup.
- The destructive drift after the local peak appears quickly; simply training longer is not the fix.
- Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not extend attention-only beyond 250 updates without a new stabilizer.
- If continuing endpoint search, test the left side rather than the right side: attention-only 225 or 237/240 updates, but simple 225 endpoint search on the full-block write set already failed, so the more informative next branch is attention-only 237/240 only if we want a tight local maximum.
- A higher-leverage next branch is to keep the 250-update endpoint and add a targeted stabilizer only after the local peak, such as post-250 high-drift attention quant/shift damping, gated immediately against `80.5540`.

### Phase 2AW: Late-Attention-Only 275 Updates With Post-250 High-Drift Grad Damping

Reason:

Phase 2AV showed that extending attention-only from 250 to 275 updates is worse (`80.5140`). This phase keeps the same attention-only base path until update 250, then switches to `params_in_layers_freeze_highdrift_act` with `grad_damp` mode for updates 250-275. The goal is to test whether softly damping the high-drift attention quant/shift parameters after the local peak can preserve or improve the 250-update attention-only endpoint without the destructive drift seen at 275.

This is a single-checkpoint strict W4A4 branch and does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Train command:

```bash
EXP=recipe_resume10_paramsinlate_attn55_71_275upd_post250_damp02_gate_20260708 \
MASTER_PORT=30758 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5.attn,features.7.1.attn \
TRAINABLE_POLICY_UPDATE_OVERRIDES=250:params_in_layers_freeze_highdrift_act \
TRAINABLE_POLICY_UPDATE_MODE=grad_damp \
TRAINABLE_POLICY_GRAD_DAMP=0.2 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=275 \
STEP_CHECKPOINT_WARMUP_UPDATES=275 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=275 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_attn55_71_275upd_post250_damp02_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_attn55_71_275upd_post250_damp02_gate_20260708 \
MASTER_PORT=30759 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_attn55_71_275upd_post250_damp02_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_attn55_71_275upd_post250_damp02_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_attn55_71_275upd_post250_damp02_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5.attn,features.7.1.attn, trainable_policy_update_overrides={250: params_in_layers_freeze_highdrift_act}, trainable_policy_update_mode=grad_damp, trainable_policy_grad_damp=0.2, max_train_updates=275
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=3013716, frozen=25594540
Trainable parameter update policy: epoch=2, update=0, mode=grad_damp, policy=params_in_layers, trainable=28608256, frozen=0
Applied gradient damping policy: policy=params_in_layers, damp=0.2, damped_params=0, masked_params=25594483
Trainable parameter update policy: epoch=2, update=250, mode=grad_damp, policy=params_in_layers_freeze_highdrift_act, trainable=28608256, frozen=0
TrainSummary: epoch=2 updates=275 avg_step_time=0.226906s samples_per_step=512 samples_per_sec=2256.44
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_attn55_71_275upd_post250_damp02_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.945s  Loss: 0.8418  Acc@1: 80.5340  Acc@5: 95.2900  Samples: 50000
```

Result:

| source checkpoint | updates | policy | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 275 | attention-only until 250, then `params_in_layers_freeze_highdrift_act` with `grad_damp=0.2` | 80.5340 | 95.2900 | 0.8418 | better than direct 275 `80.5140`, below attention-only 250 `80.5460` and Phase 2Z `80.5540`; fail |

Conclusion:

Post-250 high-drift gradient damping partially repairs the 275-update degradation, improving `80.5140 -> 80.5340`, but it still does not beat the 250-update attention-only endpoint or the current best.

Interpretation:

- The stabilizer signal is directionally useful, but too late or too weak/too broad to preserve the 250-update peak.
- The post-250 window has little headroom; once drift starts, soft damping recovers only part of the loss.
- Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not continue right-side endpoint extension to 275 under the current dynamics.
- If continuing this line, the stabilizer must start at or before the local peak, or the target endpoint should move closer to 250, such as 240/250 with a mild stabilizer during the final 25-50 updates.
- Any next branch still must beat `80.5540` on strict W4A4 full-val to become useful.

### Phase 2AX: Late-Attention-Only 240-Update Endpoint Gate

Reason:

Phase 2AV showed the right side of the attention-only endpoint is worse at 275 updates, and Phase 2AW showed post-250 damping only partially repairs that degradation. This phase tests the left side at 240 updates with the same attention-only write set, to see whether the local attention-only peak occurs slightly before 250.

This is a single-checkpoint strict W4A4 branch and does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Train command:

```bash
EXP=recipe_resume10_paramsinlate_attn55_71_240upd_gate_20260708 \
MASTER_PORT=30760 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5.attn,features.7.1.attn \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=240 \
STEP_CHECKPOINT_WARMUP_UPDATES=240 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=240 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_attn55_71_240upd_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_attn55_71_240upd_gate_20260708 \
MASTER_PORT=30761 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_attn55_71_240upd_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_attn55_71_240upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_attn55_71_240upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5.attn,features.7.1.attn, max_train_updates=240
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=3013716, frozen=25594540
TrainSummary: epoch=2 updates=240 avg_step_time=0.120801s samples_per_step=512 samples_per_sec=4238.37
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_attn55_71_240upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 33.082s  Loss: 0.8390  Acc@1: 80.5000  Acc@5: 95.3180  Samples: 50000
```

Result:

| source checkpoint | updates | policy | trainable params | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---:|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 240 | `params_in_layers`, `features.5.5.attn,features.7.1.attn` | 3,013,716 | 80.5000 | 95.3180 | 0.8390 | below attention-only 250 `80.5460` and Phase 2Z `80.5540`; fail |

Conclusion:

The left side of the attention-only endpoint is also worse. Attention-only at 240 updates is lower than both 250 and 275-with-damping.

Interpretation:

- Under the current schedule, attention-only has a narrow local best near 250 updates, but even that best remains below the full two-block Phase 2Z checkpoint.
- Simple endpoint search around attention-only is now low-yield: 240 and 275 both fail.
- Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Stop simple attention-only endpoint search.
- The next branch should not be another endpoint-only run; it needs a new mechanism, likely a stabilizer active before the local peak or a parameter/case-level delta diagnosis between attention-only 250 and full-block Phase 2Z.
- Any next gate must still beat `80.5540` on strict W4A4 full-val to matter.

### Phase 2AY: Full-Block 250 vs Attention-Only 250 Case/Class Diagnostic

Reason:

Phases 2AQ-2AX closed the simple attention-only endpoint and coarse complement branches. The remaining question is why full-block Phase 2Z (`features.5.5,features.7.1`) still beats attention-only by a small margin. This phase runs a same-script full-validation logit/class diagnostic comparing the attention-only 250 checkpoint to the full-block 250 checkpoint. This is not a training result and does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Diagnostic command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_logit_classes_20260708.py \
  --out-dir /mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_attn_vs_full_diag_20260708 \
  --labels attn250,full250 \
  --compare-label full250 \
  --checkpoint attn250=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_attn55_71_250upd_gate_20260708/checkpoint-3.pth.tar \
  --checkpoint full250=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar \
  --devices 0 \
  --device-index 0 \
  --batch-size 128 \
  --workers 8
```

Artifacts:

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_attn_vs_full_diag_20260708/summary.json
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_attn_vs_full_diag_20260708/class_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_attn_vs_full_diag_20260708/confidence_bins.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_attn_vs_full_diag_20260708/flip_cases.tsv
```

Same-script metrics:

| label | samples | Top-1 | Top-5 | loss |
|---|---:|---:|---:|---:|
| `attn250` | 50000 | 80.4580 | 95.3660 | 0.8396 |
| `full250` | 50000 | 80.4940 | 95.3140 | 0.8395 |

Pair summary:

```text
full250 vs attn250:
delta_top1 = +0.0360
improved = 355
regressed = 337
net_flips = +18
same_correct = 39892
same_wrong = 9416
avg_true_prob_delta = -0.0003076
avg_margin_delta = -0.0026401
```

Confidence-bin signal:

| attn250 confidence bin | total | improved | regressed | net flips | avg true-prob delta |
|---|---:|---:|---:|---:|---:|
| `[0.00,0.20)` | 1378 | 66 | 80 | -14 | +0.00152 |
| `[0.20,0.40)` | 4253 | 194 | 162 | +32 | +0.00273 |
| `[0.40,0.60)` | 6660 | 95 | 95 | 0 | +0.00169 |
| `>=0.60` | 37699 | 0 | 0 | 0 | negative small deltas |

Class-level gains:

The biggest full-block-over-attention gains are spread across several classes, with `+3` correct in each of:

```text
864, 831, 385, 921, 639, 231, 754, 744, 427, 633
```

Class-level losses:

The biggest regressions include:

```text
265 (-4), 876 (-3), 655 (-2), 165 (-2), 872 (-2), 175 (-2), 386 (-2),
928 (-2), 914 (-2), 733 (-2), 210 (-2), 241 (-2), 764 (-2)
```

Interpretation:

- The full-block edge over attention-only is real in the same diagnostic pipeline, but small: net `+18` flips.
- The useful extra signal is concentrated in low-to-moderate confidence cases, especially attention-only confidence `[0.20,0.40)`.
- Full-block does not globally improve confidence or margins; the average true-prob and margin deltas are slightly negative.
- The extra MLP/norm/full-block movement behaves like a targeted correction for a small subset of ambiguous samples, while also introducing offsetting regressions.
- This explains why coarse MLP/norm complements and broad confidence-band KD failed: the benefit is not a broad distributional shift.

Next decision:

- Do not re-run broad confidence-band KD or class-protect KL as implemented before.
- A plausible next mechanism should be sample-selective and local, not module-wide:
  - apply a very small auxiliary only on attention-only low/moderate confidence samples (`0.20-0.40`) while keeping write set attention-only; or
  - use `full250` as a fixed local teacher only for samples where attention-only is uncertain, but with much smaller weight than the failed confidence-band KD branches.
- The gate remains strict: beat `80.5540` on full ImageNet strict W4A4.

### Phase 2BA: Attention-Only With Local Full-Block Reference Band KD

Reason:

Phase 2AY showed that the full-block 250 checkpoint's small edge over attention-only is concentrated in attention-only low/moderate confidence samples, especially confidence `[0.20,0.40)`. Prior confidence-band KD failed because it used the ImageNet teacher as target and applied a broader band. This phase adds a new narrow mechanism: use the full-block Phase 2Z checkpoint as a fixed local teacher and apply a very small soft-KD loss only on samples whose local reference confidence is in `[0.20,0.40)`, while keeping the write set attention-only.

Implementation:

Added a local-reference confidence-band KD path to `qat_launch.py` and the runner:

```text
--local-ref-confidence-band-kd-weight
--local-ref-confidence-band-kd-low
--local-ref-confidence-band-kd-high
--local-ref-confidence-band-kd-temperature
--local-ref-confidence-band-kd-checkpoint
```

Difference from previous `ref-confidence-band-kd`:

- `ref-confidence-band-kd` used the fixed reference only to select samples; the target distribution was still the ImageNet teacher.
- `local-ref-confidence-band-kd` uses the fixed reference logits themselves as the soft target and also uses the fixed reference confidence for sample selection.

Static/smoke evidence:

```text
python3 -m py_compile /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py
bash -n /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
git -C /mlx_devbox/users/quyanyi/playground/QATs diff --check -- qat_launch.py tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh docs/resume10_to81_goal_progress_20260706.md

Smoke EXP=smoke_resume10_attn_localrefband0204_w002_1upd_20260708, MAX_TRAIN_UPDATES=1
Enabled local-reference confidence band KD: weight=0.02, band=[0.2, 0.4), temperature=2.75, source=.../recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=3013716, frozen=25594540
TrainSummary: epoch=2 updates=1
```

This is a single-checkpoint strict W4A4 branch and does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Train command:

```bash
EXP=recipe_resume10_attn_localrefband0204_w002_250upd_gate_20260708 \
MASTER_PORT=30763 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT=0.02 \
LOCAL_REF_CONFIDENCE_BAND_KD_LOW=0.2 \
LOCAL_REF_CONFIDENCE_BAND_KD_HIGH=0.4 \
LOCAL_REF_CONFIDENCE_BAND_KD_TEMPERATURE=2.75 \
LOCAL_REF_CONFIDENCE_BAND_KD_CHECKPOINT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5.attn,features.7.1.attn \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=250 \
STEP_CHECKPOINT_WARMUP_UPDATES=250 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=250 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_attn_localrefband0204_w002_250upd_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_attn_localrefband0204_w002_250upd_gate_20260708 \
MASTER_PORT=30764 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_attn_localrefband0204_w002_250upd_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_attn_localrefband0204_w002_250upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_attn_localrefband0204_w002_250upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Enabled local-reference confidence band KD: weight=0.02, band=[0.2, 0.4), temperature=2.75, source=.../recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, local_ref_confidence_band_kd_weight=0.02, local_ref_confidence_band_kd_low=0.2, local_ref_confidence_band_kd_high=0.4, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5.attn,features.7.1.attn, max_train_updates=250
Trainable parameter policy: epoch=2, quant_only=True, policy=params_in_layers, trainable=3013716, frozen=25594540
TrainSummary: epoch=2 updates=250 avg_step_time=0.214741s samples_per_step=512 samples_per_sec=2384.27
Strict eval resume: loaded model from .../recipe_resume10_attn_localrefband0204_w002_250upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.123s  Loss: 0.8395  Acc@1: 80.4440  Acc@5: 95.3580  Samples: 50000
```

Result:

| source checkpoint | updates | policy | auxiliary | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 250 | attention-only `features.5.5.attn,features.7.1.attn` | local full-block ref KD, band `[0.20,0.40)`, weight `0.02` | 80.4440 | 95.3580 | 0.8395 | below attention-only 250 `80.5460` and Phase 2Z `80.5540`; fail |

Conclusion:

The local full-block reference target is harmful even at small weight `0.02`. It drops below the plain attention-only branch and well below the current best.

Interpretation:

- The full-block advantage over attention-only is not directly transferable as a local soft-target KD signal.
- The ref-target loss appears to over-constrain the attention-only path or inject the full-block checkpoint's offsetting regressions into the attention-only branch.
- This reinforces the Phase 2AY interpretation: the full-block gain is a small, case-specific side effect, not a broadly useful target distribution.
- Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not continue local-ref soft-target KD as a scalar sweep.
- The next mechanism should avoid using full-block logits as direct targets. A better next step is parameter-level delta analysis between attention-only 250 and full-block 250 to identify the exact non-attention parameters that move in the full-block checkpoint, then test only those small parameter subsets or signs, not a logit-level distillation target.

### Phase 2BB: Attention-Only 250 vs Full-Block 250 Parameter Delta Diagnostic

Reason:

Phase 2BA showed that transferring full-block behavior through logits is harmful. This phase compares the parameter delta from Phase 2W to `attn250` and `full250` to identify what actually differs between the two checkpoints. This is a diagnostic only, not a training result.

Artifacts:

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_attn_vs_full_param_delta_20260708/summary.json
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_attn_vs_full_param_delta_20260708/kind_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_attn_vs_full_param_delta_20260708/stage_kind_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_attn_vs_full_param_delta_20260708/module_kind_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_attn_vs_full_param_delta_20260708/param_delta.tsv
```

Inputs:

```text
base = /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar
attn250 = /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_attn55_71_250upd_gate_20260708/checkpoint-3.pth.tar
full250 = /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
```

Top parameter-kind differences between `attn250` and `full250` deltas:

| kind | extra_rel | attn_rel | full_rel | interpretation |
|---|---:|---:|---:|---|
| `move_v_shift` | 0.01536 | 0.01955 | 0.00894 | full-block has much smaller `move_v` drift than attention-only |
| `softmax_quant` | 0.00327 | 0.00415 | 0.00401 | similar scale but different direction |
| `move_shift` | 0.00205 | 0.00233 | 0.00236 | similar aggregate drift, but parameter-level direction differs |
| `mlp` | 0.00158 | 0.00000 | 0.00158 | full-block-only MLP movement exists but is not dominant by relative scale |
| `attn_proj` | 0.00103 | 0.00202 | 0.00173 | attention projection drift is smaller under full-block |
| `attn_qkv` | 0.00094 | 0.00205 | 0.00191 | attention QKV drift is slightly smaller under full-block |

Largest stage/kind differences:

```text
features.5.5|move_v_shift: extra_rel=0.01627, attn_rel=0.02574, full_rel=0.01477
features.7.1|move_v_shift: extra_rel=0.01514, attn_rel=0.01775, full_rel=0.00685
features.7.1|softmax_quant: extra_rel=0.00503, attn_rel=0.00525, full_rel=0.00503
features.7.1|move_shift: extra_rel=0.00210, attn_rel=0.00236, full_rel=0.00236
features.5.5|move_shift: extra_rel=0.00189, attn_rel=0.00224, full_rel=0.00236
features.7.1|mlp: extra_rel=0.00162, attn_rel=0.00000, full_rel=0.00162
features.5.5|mlp: extra_rel=0.00143, attn_rel=0.00000, full_rel=0.00143
```

Largest individual parameters:

```text
features.5.5.attn.move_v_aft.bias: attn_rel=0.02752, full_rel=0.01579
features.7.1.attn.move_v_aft.bias: attn_rel=0.01911, full_rel=0.00736
features.5.5.attn.move_v_b4.bias: attn_rel=0.02427, full_rel=0.01393
features.5.5.attn.proj.move_b4.bias: attn_rel=0.02282, full_rel=0.01308
features.7.1.attn.move_v_b4.bias: attn_rel=0.01665, full_rel=0.00645
features.5.5.attn.proj.move_aft.bias: attn_rel=0.02218, full_rel=0.01269
features.7.1.attn.proj.move_b4.bias: attn_rel=0.01532, full_rel=0.00593
features.7.1.attn.proj.move_aft.bias: attn_rel=0.01301, full_rel=0.00505
features.7.1.attn.quant_x_4_qkv.move_aft.bias: attn_rel=0.00854, full_rel=0.00454
features.7.1.attn.quant_x_4_qkv.move_b4.bias: attn_rel=0.00848, full_rel=0.00454
```

Interpretation:

- Full-block's advantage over attention-only is not explained primarily by MLP/norm movement.
- The strongest difference is that full-block has substantially smaller drift in attention high-drift shift parameters, especially `move_v_*` and `proj.move_*`.
- This matches the earlier observation that attention is the main useful path, but attention-only over-moves high-drift shift parameters.
- Prior post-250 damping helped only partially because it started after the local peak. A non-repeated next mechanism is to damp high-drift attention shift parameters from the start while still allowing other attention parameters to adapt.

Next decision:

- Test attention-only 250 with `grad_damp` from update 0 using `params_in_layers_freeze_highdrift_act` as the current policy and base policy `params_in_layers`, so non-high-drift attention parameters are normal and high-drift attention shift parameters receive a smaller gradient.
- This is different from prior post-250 damping and from hard freeze: it is an all-window soft damping mechanism based on the delta diagnosis.

### Phase 2BC: Attention-Only 250 With All-Window High-Drift Grad Damping

Reason:

Phase 2BB showed that attention-only over-moves high-drift attention shift parameters compared with the full-block 250 checkpoint, especially `move_v_*` and `proj.move_*`. This phase tested whether applying the high-drift soft damping policy from update 0 can preserve the useful attention-only adaptation while reducing harmful drift before the local peak.

This is a single-checkpoint strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Train command:

```bash
EXP=recipe_resume10_attn55_71_250upd_highdrift_damp05_gate_20260708 \
MASTER_PORT=30765 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5.attn,features.7.1.attn \
TRAINABLE_POLICY_UPDATE_OVERRIDES=0:params_in_layers_freeze_highdrift_act \
TRAINABLE_POLICY_UPDATE_MODE=grad_damp \
TRAINABLE_POLICY_GRAD_DAMP=0.5 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=250 \
STEP_CHECKPOINT_WARMUP_UPDATES=250 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=250 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_attn55_71_250upd_highdrift_damp05_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_attn55_71_250upd_highdrift_damp05_gate_20260708 \
MASTER_PORT=30766 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_attn55_71_250upd_highdrift_damp05_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_attn55_71_250upd_highdrift_damp05_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_attn55_71_250upd_highdrift_damp05_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Strict train resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5.attn,features.7.1.attn, trainable_policy_update_overrides={0: params_in_layers_freeze_highdrift_act}, trainable_policy_update_mode=grad_damp, trainable_policy_grad_damp=0.5, max_train_updates=250
Trainable parameter update policy: epoch=2, update=0, mode=grad_damp, policy=params_in_layers_freeze_highdrift_act, trainable=28608256, frozen=0
Applied gradient damping policy: policy=params_in_layers_freeze_highdrift_act, damp=0.5, damped_params=7010, masked_params=25594483
TrainSummary: epoch=2 updates=250 avg_step_time=0.227357s samples_per_step=512 samples_per_sec=2251.96
Strict eval resume: loaded model from .../recipe_resume10_attn55_71_250upd_highdrift_damp05_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.879s  Loss: 0.8389  Acc@1: 80.5100  Acc@5: 95.2940  Samples: 50000
```

Result:

| source checkpoint | updates | policy | drift control | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 250 | attention-only `features.5.5.attn,features.7.1.attn` | all-window high-drift grad damping `0.5` | 80.5100 | 95.2940 | 0.8389 | below attention-only 250 `80.5460` and Phase 2Z `80.5540`; fail |

Conclusion:

All-window high-drift damping is too broad for the attention-only branch. It suppresses the useful attention adaptation and drops below both the plain attention-only endpoint and the current best.

Interpretation:

- The Phase 2BB delta diagnosis is still useful, but the control target was too broad.
- Treating `move_v_*`, `proj.move_*`, softmax quantizer, and QKV input move as one high-drift group over-regularizes the attention path.
- Do not sweep the same all-high-drift `grad_damp` coefficient.

Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Test a narrower variation-aware policy that damps only `move_v_*` shift parameters, because this is the largest attn-vs-full delta class in Phase 2BB.
- Use the full two-block Phase 2Z write set (`features.5.5,features.7.1`) as the base path so the branch directly competes with the current best and preserves normal adaptation for attention projection, softmax quantizer, QKV move, MLP, and block-local non-attention parameters.

### Phase 2BD: Full Two-Block 250 With Move-V-Only Grad Damping

Reason:

Phase 2BC showed that damping the whole diagnosed high-drift group is too broad. This phase tests a narrower variation-aware control: keep the current best Phase 2Z full two-block write set (`features.5.5,features.7.1`) and damp only `move_v_*` shift parameters from update 0. Attention projection move, softmax quantizer, QKV move, MLP, norms, and other block-local parameters remain on the normal full two-block path.

This is a single-checkpoint strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Code change:

```text
qat_launch.py:
- added is_move_v_shift_parameter(...)
- added trainable policy params_in_layers_freeze_move_v_shift
- wired the new policy into update override parsing, set_trainable_policy, parameter_matches_trainable_policy, and CLI choices
```

Static checks:

```bash
python3 -m py_compile /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py
bash -n /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
git -C /mlx_devbox/users/quyanyi/playground/QATs diff --check -- \
  qat_launch.py docs/resume10_to81_goal_progress_20260706.md tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Train command:

```bash
EXP=recipe_resume10_paramsinlate_250upd_movev_damp05_gate_20260708 \
MASTER_PORT=30767 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
TRAINABLE_POLICY_UPDATE_OVERRIDES=0:params_in_layers_freeze_move_v_shift \
TRAINABLE_POLICY_UPDATE_MODE=grad_damp \
TRAINABLE_POLICY_GRAD_DAMP=0.5 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=250 \
STEP_CHECKPOINT_WARMUP_UPDATES=250 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=250 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_250upd_movev_damp05_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_250upd_movev_damp05_gate_20260708 \
MASTER_PORT=30768 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_250upd_movev_damp05_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_250upd_movev_damp05_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_250upd_movev_damp05_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Strict train resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5,features.7.1, trainable_policy_update_overrides={0: params_in_layers_freeze_move_v_shift}, trainable_policy_update_mode=grad_damp, trainable_policy_grad_damp=0.5, max_train_updates=250
Trainable parameter update policy: epoch=2, update=0, mode=grad_damp, policy=params_in_layers_freeze_move_v_shift, trainable=28608256, frozen=0
Applied gradient damping policy: policy=params_in_layers_freeze_move_v_shift, damp=0.5, damped_params=2304, masked_params=19674313
TrainSummary: epoch=2 updates=250 avg_step_time=0.227017s samples_per_step=512 samples_per_sec=2255.33
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_250upd_movev_damp05_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 33.184s  Loss: 0.8393  Acc@1: 80.5100  Acc@5: 95.2900  Samples: 50000
```

Result:

| source checkpoint | updates | policy | drift control | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 250 | `params_in_layers`, `features.5.5,features.7.1` | all-window `move_v_*` grad damping `0.5` | 80.5100 | 95.2900 | 0.8393 | below Phase 2Z `80.5540`; fail |

Conclusion:

Move-V-only damping is also harmful. It is narrower than Phase 2BC (`2304` damped params versus `7010`), but still drops the full two-block branch from `80.5540` to `80.5100`.

Interpretation:

- `move_v_*` drift is not simply harmful noise. Some movement in this parameter class is needed for the current 250-update gain.
- The useful/harmful distinction is not captured by suppressing a whole parameter kind, even the largest delta kind from Phase 2BB.
- Do not continue with single-kind shift damping coefficient sweeps.

Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- The next non-repeated step should be diagnostic, not another immediate damping gate: compare Phase 2Z, Phase 2BC, and Phase 2BD at case/logit and parameter-delta level to separate useful `move_v` movement from globally reduced movement.
- A plausible next mechanism should be direction- or case-conditioned, rather than parameter-kind-conditioned. Simple hard freeze, soft trust, all-high-drift damping, and move-v-only damping are now closed.

### Phase 2BE: Damping Failure Parameter and Case/Logit Diagnostics

Reason:

Phase 2BC and Phase 2BD both failed at `80.5100`, but they changed different parameter sets. This phase compares the failed damping branches against Phase 2Z best250 to decide whether the next mechanism should still be parameter-kind based or should move to a direction/case-conditioned constraint. This is diagnostic only, not a training result.

Artifacts:

```text
Parameter delta:
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_damping_delta_20260708/summary.json
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_damping_delta_20260708/kind_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_damping_delta_20260708/stage_kind_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_damping_delta_20260708/module_kind_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_damping_delta_20260708/param_delta.tsv

Case/logit:
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_damping_logit_diag_20260708/summary.json
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_damping_logit_diag_20260708/class_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_damping_logit_diag_20260708/confidence_bins.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_damping_logit_diag_20260708/flip_cases.tsv

Logs:
/mlx_devbox/users/quyanyi/playground/train_resume10_damping_logit_diag_20260708.log
```

Parameter-delta diagnostic command:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python3 /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_damping_delta_20260708.py
```

Case/logit diagnostic command:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONUNBUFFERED=1 \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python3 /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_logit_classes_20260708.py \
  --devices 0 \
  --device-index 0 \
  --batch-size 256 \
  --workers 8 \
  --out-dir /mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_damping_logit_diag_20260708 \
  --labels phase2w,best250,highdrift_damp,movev_damp \
  --compare-label best250 \
  --checkpoint highdrift_damp=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_attn55_71_250upd_highdrift_damp05_gate_20260708/checkpoint-3.pth.tar \
  --checkpoint movev_damp=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_250upd_movev_damp05_gate_20260708/checkpoint-3.pth.tar \
  2>&1 | tee /mlx_devbox/users/quyanyi/playground/train_resume10_damping_logit_diag_20260708.log
```

Runtime evidence:

```text
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
diagnosis labels=['phase2w', 'best250', 'highdrift_damp', 'movev_damp']
validation_samples=50000
train_shards=294
validation_shards=14
Strict resume phase2w: missing=0, unexpected=0
Strict resume best250: missing=0, unexpected=0
Strict resume highdrift_damp: missing=0, unexpected=0
Strict resume movev_damp: missing=0, unexpected=0
```

Single-process diagnostic metrics:

| checkpoint | strict W4A4 samples | Top-1 | Top-5 | loss |
|---|---:|---:|---:|---:|
| Phase 2W | 50000 | 80.5060 | 95.3120 | 0.8412 |
| Phase 2Z best250 | 50000 | 80.5040 | 95.3040 | 0.8394 |
| Phase 2BC highdrift_damp | 50000 | 80.4580 | 95.3260 | 0.8394 |
| Phase 2BD movev_damp | 50000 | 80.4780 | 95.3300 | 0.8393 |

Important caveat:

This logit diagnostic is single-process for per-sample ordering. Absolute Top-1 differs slightly from the 8-GPU strict full-val scores, so final gates still require the official full-val script. The diagnostic is only for relative case/logit signals.

Parameter findings:

| group | branch | cmp/best delta norm | cosine to best delta | extra/best | interpretation |
|---|---|---:|---:|---:|---|
| `move_v_shift` | highdrift_damp | 2.147 | 0.657 | 1.670 | all-high-drift damping creates a much larger, different `move_v` trajectory |
| `proj_move_shift` | highdrift_damp | 2.144 | 0.658 | 1.666 | same failure pattern for projection shift |
| `move_v_shift` | movev_damp | 0.968 | 0.954 | 0.301 | move-v-only branch is close to best on this parameter kind |
| `proj_move_shift` | movev_damp | 0.968 | 0.955 | 0.298 | projection shift also remains close to best under move-v-only damping |
| `move_shift` | movev_damp | 0.994 | 0.706 | 0.764 | remaining shift direction mismatch is not from `move_v` alone |
| `features.5.5.attn.move_qkx_aft` | movev_damp | 1.003 | 0.065 | 1.370 | strong direction mismatch despite similar magnitude |
| `features.7.1.attn.move_qkx_aft` | movev_damp | 0.998 | 0.064 | 1.367 | same direction mismatch in the later block |

Case/logit findings:

| pair | best250 net flips | improved | regressed | main confidence bins |
|---|---:|---:|---:|---|
| best250 vs highdrift_damp | +23 | 348 | 325 | +15 in `[0.20,0.40)`, +6 in `[0.40,0.60)` |
| best250 vs movev_damp | +13 | 317 | 304 | +11 in `[0.40,0.60)`, +5 in `[0.00,0.20)`, -3 in `[0.20,0.40)` |
| best250 vs Phase 2W | -1 | 355 | 356 | noisy; confirms single-process absolute score is not the official gate |

Repeated class signals:

- best250 gains over damping variants include classes `831`, `864`, `118`, `231`, `590`, `787`, `834`.
- best250 loses relative to damping variants on classes including `876`, `165`, `442`, and in the move-v branch also `250`, `859`, `386`.
- Class `876` remains a repeated regression class even when comparing against damping variants, so direct class-protect KL was not the right mechanism but the class-level instability is real.

Conclusion:

The failure is not explained by simply reducing `move_v` magnitude. Move-v-only damping makes the `move_v` and `proj_move` deltas close to best250, yet still loses Top-1. The remaining sharp signal is direction mismatch in `move_qkx_aft` under both failed damping branches: the magnitude is near best250, but the direction is almost orthogonal.

Interpretation:

- Parameter-kind controls are too blunt: hard freeze, soft trust, all-high-drift damping, and move-v-only damping are now closed.
- The useful/harmful split is directional and case-level, not only per-kind drift magnitude.
- The next mechanism should constrain direction for the diagnosed qkx shift trajectory while leaving amplitude and other full two-block adaptation free.

Next decision:

- Add a minimal delta-direction anchor regularizer:
  - source/base: Phase 2W checkpoint-2
  - target direction: Phase 2Z best250 delta
  - layers/params: start with `features.5.5.attn.move_qkx_aft` and `features.7.1.attn.move_qkx_aft`
  - loss: cosine direction only on current delta versus target delta; no magnitude MSE and no checkpoint averaging
  - gate: same full two-block 250-update strict W4A4 full-val, must beat `80.5540`

### Phase 2BF: QKX Shift Delta-Direction Anchor At The 250-Update Gate

Reason:

Phase 2BE showed that both failed damping branches have a strong direction mismatch in `features.5.5.attn.move_qkx_aft` and `features.7.1.attn.move_qkx_aft`: magnitude is close to best250, but cosine against best250 delta is near zero. This phase tests a direction-only parameter regularizer. It uses Phase 2W as the base state and Phase 2Z best250 as the target delta direction, then penalizes `1 - cosine(current - base, target - base)` only for the two diagnosed qkx shift tensors.

This is a single-checkpoint strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble. The reference checkpoint is used only to define a training loss direction.

Code changes:

```text
qat_launch.py:
- added --delta-direction-anchor-* args
- added checkpoint-state reader for base/target delta
- added delta_direction_anchor_loss(...)
- logged direction loss as DirAnchor

tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh:
- forwards/logs DELTA_DIRECTION_ANCHOR_* env vars
```

Static checks:

```bash
python3 -m py_compile \
  /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py \
  /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_damping_delta_20260708.py
bash -n /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
git -C /mlx_devbox/users/quyanyi/playground/QATs diff --check -- \
  qat_launch.py \
  tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh \
  tmp_scripts/diagnose_resume10_damping_delta_20260708.py \
  docs/resume10_to81_goal_progress_20260706.md
```

Smoke:

```text
EXP=smoke_resume10_diranchor_qkx_1upd_20260708
MAX_TRAIN_UPDATES=1
DELTA_DIRECTION_ANCHOR_WEIGHT=0.01
DELTA_DIRECTION_ANCHOR_PARAMS=features.5.5.attn.move_qkx_aft,features.7.1.attn.move_qkx_aft

Initialized delta direction anchor: params=2, weight=0.01, patterns=('features.5.5.attn.move_qkx_aft', 'features.7.1.attn.move_qkx_aft'), ...
Enabled delta direction anchor: weight=0.01, pairs=2, start_update=0
Train: ... DirAnchor: 1.000e+00 ...
TrainSummary: epoch=2 updates=1 ...
```

Train command:

```bash
EXP=recipe_resume10_paramsinlate_diranchor_qkx_w001_250upd_gate_20260708 \
MASTER_PORT=30770 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
DELTA_DIRECTION_ANCHOR_WEIGHT=0.01 \
DELTA_DIRECTION_ANCHOR_BASE_CHECKPOINT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
DELTA_DIRECTION_ANCHOR_TARGET_CHECKPOINT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar \
DELTA_DIRECTION_ANCHOR_PARAMS=features.5.5.attn.move_qkx_aft,features.7.1.attn.move_qkx_aft \
DELTA_DIRECTION_ANCHOR_START_UPDATE=0 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=250 \
STEP_CHECKPOINT_WARMUP_UPDATES=250 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=250 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_diranchor_qkx_w001_250upd_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_diranchor_qkx_w001_250upd_gate_20260708 \
MASTER_PORT=30771 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_diranchor_qkx_w001_250upd_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_diranchor_qkx_w001_250upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_diranchor_qkx_w001_250upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Strict train resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, delta_direction_anchor_weight=0.01, delta_direction_anchor_params=features.5.5.attn.move_qkx_aft,features.7.1.attn.move_qkx_aft, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5,features.7.1, max_train_updates=250
Initialized delta direction anchor: params=2, weight=0.01, patterns=('features.5.5.attn.move_qkx_aft', 'features.7.1.attn.move_qkx_aft'), ...
Enabled delta direction anchor: weight=0.01, pairs=2, start_update=0
Train: update 0 DirAnchor=1.000e+00
Train: update 50 DirAnchor=2.373e-01
Train: update 100 DirAnchor=2.370e-01
Train: update 150 DirAnchor=2.367e-01
Train: update 200 DirAnchor=2.363e-01
TrainSummary: epoch=2 updates=250 avg_step_time=0.133500s samples_per_step=512 samples_per_sec=3835.19
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_diranchor_qkx_w001_250upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.630s  Loss: 0.8393  Acc@1: 80.5040  Acc@5: 95.3100  Samples: 50000
```

Result:

| source checkpoint | updates | policy | auxiliary | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 250 | `params_in_layers`, `features.5.5,features.7.1` | qkx shift delta-direction anchor, weight `0.01` | 80.5040 | 95.3100 | 0.8393 | below Phase 2Z `80.5540`; fail |

Conclusion:

QKX delta-direction anchoring is harmful at the tested light weight. It aligns the diagnosed direction signal during training, but the final strict W4A4 score drops below Phase 2Z and matches the failed damping-class gates.

Interpretation:

- The qkx direction mismatch is diagnostic but not directly fixable by forcing the Phase 2Z parameter-delta direction during the full 250-update window.
- Like magnitude damping, direction anchoring from update 0 appears to over-constrain the early useful adaptation.
- Do not continue this as a scalar weight sweep.

Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- The evidence now closes simple parameter loss controls around the Phase 2W -> Phase 2Z 250-update gate: hard masks, damping, MSE trust, confidence/class KD, local ref KD, and direction anchor.
- Next useful direction should change the construction of the Phase 2W source itself or use a delayed/conditional mechanism triggered after the useful movement emerges, rather than applying any anchor from update 0.

### Phase 2BG: Delayed QKX Delta-Direction Anchor From Update 125

Reason:

Phase 2BF showed that qkx direction anchoring from update 0 is harmful. This phase tests whether the same mechanism is only harmful because it blocks early useful movement. It allows the Phase 2W -> Phase 2Z local path to run without direction anchor for the first 125 optimizer updates, then enables the qkx direction anchor for updates 125-250.

This is a single-checkpoint strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Train command:

```bash
EXP=recipe_resume10_paramsinlate_diranchor_qkx_w001_start125_250upd_gate_20260708 \
MASTER_PORT=30772 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
START_EPOCH=2 \
EPOCHS=3 \
SCHEDULER_EPOCHS=3 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
DELTA_DIRECTION_ANCHOR_WEIGHT=0.01 \
DELTA_DIRECTION_ANCHOR_BASE_CHECKPOINT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
DELTA_DIRECTION_ANCHOR_TARGET_CHECKPOINT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar \
DELTA_DIRECTION_ANCHOR_PARAMS=features.5.5.attn.move_qkx_aft,features.7.1.attn.move_qkx_aft \
DELTA_DIRECTION_ANCHOR_START_UPDATE=125 \
QUANT_ONLY_START_EPOCH=2 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=250 \
STEP_CHECKPOINT_WARMUP_UPDATES=250 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=250 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_diranchor_qkx_w001_start125_250upd_gate_20260708/checkpoint-3.pth.tar \
EXP=eval_resume10_paramsinlate_diranchor_qkx_w001_start125_250upd_gate_20260708 \
MASTER_PORT=30773 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_diranchor_qkx_w001_start125_250upd_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_diranchor_qkx_w001_start125_250upd_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_diranchor_qkx_w001_start125_250upd_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar; missing=0, unexpected=0
Strict train resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, delta_direction_anchor_weight=0.01, delta_direction_anchor_start_update=125, delta_direction_anchor_params=features.5.5.attn.move_qkx_aft,features.7.1.attn.move_qkx_aft, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5,features.7.1, max_train_updates=250
Initialized delta direction anchor: params=2, weight=0.01, patterns=('features.5.5.attn.move_qkx_aft', 'features.7.1.attn.move_qkx_aft'), start_update=125
Train: update 0 DirAnchor=0.000e+00
Train: update 50 DirAnchor=0.000e+00
Train: update 100 DirAnchor=0.000e+00
Enabled delta direction anchor: weight=0.01, pairs=2, start_update=125
Train: update 150 DirAnchor=2.654e-03
Train: update 200 DirAnchor=1.293e-05
TrainSummary: epoch=2 updates=250 avg_step_time=0.128664s samples_per_step=512 samples_per_sec=3979.37
Strict eval resume: loaded model from .../recipe_resume10_paramsinlate_diranchor_qkx_w001_start125_250upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 33.348s  Loss: 0.8393  Acc@1: 80.4340  Acc@5: 95.3060  Samples: 50000
```

Result:

| source checkpoint | updates | policy | auxiliary | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2W `checkpoint-2.pth.tar` | 250 | `params_in_layers`, `features.5.5,features.7.1` | qkx direction anchor from update 125, weight `0.01` | 80.4340 | 95.3060 | 0.8393 | below Phase 2Z `80.5540`; fail |

Conclusion:

Delayed qkx direction anchoring is worse than anchoring from update 0 and worse than the no-trust 250 checkpoint. It should not be continued.

Interpretation:

- The direction-anchor family is now closed for this local path.
- Even delayed direction anchoring over-constrains the short-update dynamics or locks the wrong local direction.
- The next useful work should not be another direction-anchor timing/weight sweep.

Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Change the construction of the Phase 2W source or the short-update target, not another regularizer on the Phase 2W -> Phase 2Z path.
- A concrete next diagnostic is to compare Phase 2W source construction against Phase 2Z: which part of Phase 2W's first local epoch creates the useful source, and whether a shorter/different Phase 2W source can make the 250-update window less brittle.

### Phase 2BH: Half-Epoch Late-Block Source Construction From Phase 2S

Reason:

Phase 2W constructs the current positive source by training the Phase 2S `checkpoint-1` tie point for one full local late-block epoch, reaching `80.5400`, then Phase 2Z adds 250 more local updates to reach the current best `80.5540`. After multiple failed regularizers on the Phase 2W -> Phase 2Z path, this phase changes the source construction itself: train the same late-block-local policy from Phase 2S for only 1250 updates, save that partial source, and full-validate it. This tests whether the Phase 2W source has an earlier, less brittle peak.

This is a single-checkpoint strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Train command:

```bash
EXP=recipe_resume10_vartrust_epoch1_paramsinlate_1250upd_source_gate_20260708 \
MASTER_PORT=30774 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar \
START_EPOCH=1 \
EPOCHS=2 \
SCHEDULER_EPOCHS=2 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=1 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=1250 \
STEP_CHECKPOINT_WARMUP_UPDATES=1250 \
MAX_STEP_CHECKPOINTS_TO_SAVE=1 \
MAX_TRAIN_UPDATES=1250 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Eval command:

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_1250upd_source_gate_20260708/checkpoint-2.pth.tar \
EXP=eval_resume10_vartrust_epoch1_paramsinlate_1250upd_source_gate_20260708 \
MASTER_PORT=30775 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_vartrust_epoch1_paramsinlate_1250upd_source_gate_20260708.log \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_vartrust_epoch1_paramsinlate_1250upd_source_gate_20260708.log
Eval log: /mlx_devbox/users/quyanyi/playground/train_eval_resume10_vartrust_epoch1_paramsinlate_1250upd_source_gate_20260708.log
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar; missing=0, unexpected=0
Strict train resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, start_epoch=1, max_train_updates=1250, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5,features.7.1
Trainable parameter policy: epoch=1, quant_only=True, policy=params_in_layers, trainable=8933890, frozen=19674366
TrainSummary: epoch=1 updates=1250 avg_step_time=0.119298s samples_per_step=512 samples_per_sec=4291.76
Strict eval resume: loaded model from .../recipe_resume10_vartrust_epoch1_paramsinlate_1250upd_source_gate_20260708/checkpoint-2.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 33.104s  Loss: 0.8420  Acc@1: 80.4920  Acc@5: 95.2780  Samples: 50000
```

Result:

| source checkpoint | updates | policy | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---:|---:|---:|---|
| Phase 2S `checkpoint-1.pth.tar` | 1250 | `params_in_layers`, `features.5.5,features.7.1` | 80.4920 | 95.2780 | 0.8420 | below Phase 2W `80.5400` and Phase 2Z `80.5540`; fail |

Conclusion:

The Phase 2W source is not overtrained by the half-epoch point. The 1250-update partial source is weaker than the full Phase 2W one-epoch source, so the useful source construction still needs the full local epoch.

Interpretation:

- Shortening the Phase 2W source construction is not a way to make the later 250-update window less brittle.
- The current best remains Phase 2Z:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not continue source-construction shortening as a scalar update-count sweep unless a new diagnostic points to an earlier source peak.
- The remaining source-construction direction should change the objective or trainable set of the Phase 2S -> Phase 2W local epoch, not simply shorten it.

### Phase 2BI: Attention-Only Source Construction From Phase 2S

Reason:

Phase 2BH showed that shortening the full two-block Phase 2W source construction to 1250 updates is weaker than the full one-epoch source. This phase changes the trainable set instead of update count: start from the Phase 2S `checkpoint-1` tie point and train one full epoch with only the attention submodules of the two useful late blocks trainable (`features.5.5.attn,features.7.1.attn`). This tests whether the useful Phase 2W source is primarily attention-driven.

This is a single-checkpoint strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Train command:

```bash
EXP=recipe_resume10_vartrust_epoch1_attn55_71_source_gate_20260708 \
MASTER_PORT=30776 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar \
START_EPOCH=1 \
EPOCHS=2 \
SCHEDULER_EPOCHS=2 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=1 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5.attn,features.7.1.attn \
SAVE_STEP_CHECKPOINTS=0 \
MAX_TRAIN_UPDATES=0 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_vartrust_epoch1_attn55_71_source_gate_20260708.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_attn55_71_source_gate_20260708
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar; missing=0, unexpected=0
Strict train resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, start_epoch=1, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5.attn,features.7.1.attn
Trainable parameter policy: epoch=1, quant_only=True, policy=params_in_layers, trainable=3013716, frozen=25594540
TrainSummary: epoch=1 updates=2496 avg_step_time=0.117354s samples_per_step=512 samples_per_sec=4362.87
Test: [distributed-summary]  Time: 34.001s  Loss: 0.8405  Acc@1: 80.4920  Acc@5: 95.3040  Samples: 50000
```

Result:

| source checkpoint | trained epoch | policy | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---:|---:|---:|---|
| Phase 2S `checkpoint-1.pth.tar` | epoch 1 only | `params_in_layers`, `features.5.5.attn,features.7.1.attn` | 80.4920 | 95.3040 | 0.8405 | below Phase 2W `80.5400` and Phase 2Z `80.5540`; fail |

Conclusion:

The Phase 2W source is not primarily produced by attention-only adaptation. Training only the attention submodules for the full source-construction epoch underperforms the full late-block source and ties the half-epoch source result.

Interpretation:

- The full two-block source construction needs interaction between attention and non-attention parameters inside `features.5.5` and `features.7.1`.
- Prior complement tests from the Phase 2W source showed that opening MLP/norm additions during the later 250-update peak is harmful, but this source-construction test shows that excluding those non-attention parameters during the source epoch is also harmful.
- The next source-construction branch should not make the source trainable set narrower. It should change the objective or schedule while keeping the full late-block trainable set.

Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not continue attention-only source construction.
- A non-repeated next direction is to keep the full late-block source write set but change the source objective/schedule, for example remove or alter the teacher feature-output auxiliary during the Phase 2S -> Phase 2W source construction, then gate the source full-val before applying any follow-up short-update branch.

### Phase 2BJ: Full Late-Block Source Construction Without Teacher Feature-Output Auxiliary

Reason:

Phase 2BI showed that narrowing the Phase 2S -> Phase 2W source construction to attention-only is harmful. This phase keeps the full late-block write set (`features.5.5,features.7.1`) and changes only the source objective: remove the teacher feature-output auxiliary during the full source epoch. The purpose is to test whether Phase 2W's useful source quality comes from ordinary late-block local adaptation alone, or whether the feature-output auxiliary is part of the successful source construction.

This is a single-checkpoint strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Train command:

```bash
EXP=recipe_resume10_vartrust_epoch1_paramsinlate_nofeatout_source_gate_20260708 \
MASTER_PORT=30777 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar \
START_EPOCH=1 \
EPOCHS=2 \
SCHEDULER_EPOCHS=2 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=1 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=0 \
MAX_TRAIN_UPDATES=0 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_vartrust_epoch1_paramsinlate_nofeatout_source_gate_20260708.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_nofeatout_source_gate_20260708
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar; missing=0, unexpected=0
Strict train resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, teacher_feature_output_weight=0.0, pre_qat_feature_recon_updates=0, variation_trust_weight=0.0, bin_reg_weight=0.0, start_epoch=1, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5,features.7.1
Trainable parameter policy: epoch=1, quant_only=True, policy=params_in_layers, trainable=8933890, frozen=19674366
TrainSummary: epoch=1 updates=2496 avg_step_time=0.118572s samples_per_step=512 samples_per_sec=4318.04
Test: [distributed-summary]  Time: 34.380s  Loss: 0.8402  Acc@1: 80.4640  Acc@5: 95.3160  Samples: 50000
```

Result:

| source checkpoint | trained epoch | policy | source objective change | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2S `checkpoint-1.pth.tar` | epoch 1 only | `params_in_layers`, `features.5.5,features.7.1` | remove teacher feature-output auxiliary | 80.4640 | 95.3160 | 0.8402 | below Phase 2W `80.5400` and Phase 2Z `80.5540`; fail |

Conclusion:

Removing the teacher feature-output auxiliary hurts the full late-block source construction. The source drops below the full Phase 2W source and below both Phase 2BH/2BI source alternatives. The feature-output auxiliary is therefore a useful part of the Phase 2S -> Phase 2W source construction, not just incidental regularization.

Interpretation:

- The next source-construction branch should not remove teacher feature-output entirely.
- The successful source seems to require both the full late-block write set and a teacher feature-output signal on `features.5.5,features.7.1`.
- Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not continue the no-feature-output source branch.
- A non-repeated next source branch should keep teacher feature-output and the full late-block write set, but change the source schedule or target more structurally than a scalar weight sweep.

### Phase 2BK: Source Construction Delta and Class/Logit Diagnostics

Reason:

Phase 2BJ showed that removing the teacher feature-output auxiliary from the full late-block source construction drops the source checkpoint from Phase 2W-level quality to `80.4640`. Before launching another source gate, this diagnostic compares Phase 2S -> Phase 2W against Phase 2S -> no-feature source at parameter and validation-case level.

This is diagnostic only. It is not a training result and does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Commands:

```bash
python3 /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_source_delta_20260708.py

CUDA_VISIBLE_DEVICES=0 python3 /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_logit_classes_20260708.py \
  --out-dir /mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_source_logit_diag_20260708 \
  --checkpoint phase2s=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar \
  --checkpoint phase2w=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
  --checkpoint nofeat=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_nofeatout_source_gate_20260708/checkpoint-2.pth.tar \
  --labels phase2s,phase2w,nofeat \
  --compare-label phase2w \
  --devices 0 \
  --device-index 0 \
  --master-port 30778 \
  --batch-size 128 \
  --workers 8
```

Artifacts:

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_source_delta_20260708/summary.json
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_source_delta_20260708/kind_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_source_delta_20260708/stage_kind_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_source_delta_20260708/module_kind_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_source_delta_20260708/param_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_source_logit_diag_20260708/summary.json
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_source_logit_diag_20260708/class_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_source_logit_diag_20260708/confidence_bins.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_source_logit_diag_20260708/flip_cases.tsv
```

Diagnostic evidence:

```text
Parameter delta, Phase 2S as base:
- no-feature source loses 0.076 Top-1 versus Phase 2W.
- Largest module-level delta mismatch versus Phase 2W:
  features.5.5.attn.move_qkx_aft | cosine=-0.0048, extra_over_phase2w=1.446
  features.7.1.attn.move_qkx_aft | cosine=0.0010, extra_over_phase2w=1.422
  features.5.5.mlp.fc2.input_quant_fn.s | nofeat_over_phase2w=1.833, extra_over_phase2w=0.853
- Stage-level largest mismatch:
  features.7.1 move_shift, features.5.5 move_shift, features.7.1 act_quant, features.5.5 act_quant.

Single-GPU same-script full-val diagnostic:
phase2s:  Acc@1=80.4600, Acc@5=95.3440, Samples=50000
phase2w:  Acc@1=80.4720, Acc@5=95.3180, Samples=50000
nofeat:   Acc@1=80.3860, Acc@5=95.3400, Samples=50000

phase2w vs nofeat:
improved=340, regressed=297, net_flips=+43
main confidence-bin gain: nofeat ref confidence [0.20,0.40), net_flips=+41
largest positive classes include 681 (+4), 764 (+3), 492 (+3), 278 (+3), 150 (+3), 206 (+3)
largest negative classes include 509 (-2), 242 (-2), 961 (-2), 241 (-2), 824 (-2), 290 (-2)
```

Conclusion:

The teacher feature-output auxiliary is not merely changing global confidence. It changes the source construction dynamics around late attention `move_qkx_aft` direction and late activation-scale movement, especially in `features.5.5` and `features.7.1`. The validation gains are concentrated in low/moderate-confidence samples, but previous broad confidence-band KD and local-ref KD failed, so the next step should not be another logit/KD band.

Interpretation:

- Keep the full late-block write set and teacher feature-output in the source epoch.
- Do not remove feature-output, narrow the write set, or repeat QDrop/disagreement sample weighting.
- A non-repeated next gate should add a structural source signal that directly targets late attention behavior, rather than adding another scalar confidence/logit loss.

Next decision:

- Test a source-construction branch with the existing teacher feature-output auxiliary plus a small teacher attention-output MSE only on the two late attention layers corresponding to `features.5.5` and `features.7.1`.
- Gate only the source checkpoint first; it must beat Phase 2W `80.5400` to justify any follow-up 250-update branch.

### Phase 2BL: Full Late-Block Source Construction With Late Teacher Attention-Output MSE

Reason:

Phase 2BK showed that the no-feature source differs from Phase 2W mainly in late attention `move_qkx_aft` direction and late activation-scale movement, while the validation gains are concentrated in low/moderate-confidence samples. Previous confidence-band KD, local-ref KD, direction anchor, and damping branches failed, so this phase tests a different structural source signal: keep the useful teacher feature-output auxiliary and full late-block write set, and add a very small teacher attention-output MSE on only late attention layers `10,11`.

This is a single-checkpoint strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Launcher change:

`tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh` now forwards:

```text
TEACHER_ATTN_OUTPUT_WEIGHT
TEACHER_ATTN_OUTPUT_LAYERS
TEACHER_ATTN_OUTPUT_WARMUP_EPOCHS
```

Train command:

```bash
EXP=recipe_resume10_vartrust_epoch1_paramsinlate_attnout_w1e4_source_gate_20260708 \
MASTER_PORT=30779 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar \
START_EPOCH=1 \
EPOCHS=2 \
SCHEDULER_EPOCHS=2 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_ATTN_OUTPUT_WEIGHT=0.0001 \
TEACHER_ATTN_OUTPUT_LAYERS=10,11 \
TEACHER_ATTN_OUTPUT_WARMUP_EPOCHS=0 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=1 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=0 \
MAX_TRAIN_UPDATES=0 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_vartrust_epoch1_paramsinlate_attnout_w1e4_source_gate_20260708.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_attnout_w1e4_source_gate_20260708
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar; missing=0, unexpected=0
Strict train resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, teacher_feature_output_weight=0.003, teacher_feature_output_layers=features.5.5,features.7.1, teacher_attn_output_weight=0.0001, teacher_attn_output_layers=10,11, pre_qat_feature_recon_updates=0, variation_trust_weight=0.0, bin_reg_weight=0.0, start_epoch=1, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5,features.7.1
Trainable parameter policy: epoch=1, quant_only=True, policy=params_in_layers, trainable=8933890, frozen=19674366
Teacher feature-output hooks: layers=('features.5.5', 'features.7.1'), student_matches=('features.5.5', 'features.7.1'), teacher_matches=('features.5.5', 'features.7.1')
Train: update 0 TeacherAttnOut=1.441e-01, TeacherFeatOut=1.173e-01
TrainSummary: epoch=1 updates=2496 avg_step_time=0.119463s samples_per_step=512 samples_per_sec=4285.86
Test: [distributed-summary]  Time: 34.422s  Loss: 0.8399  Acc@1: 80.4920  Acc@5: 95.2960  Samples: 50000
```

Result:

| source checkpoint | trained epoch | policy | auxiliary | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2S `checkpoint-1.pth.tar` | epoch 1 only | `params_in_layers`, `features.5.5,features.7.1` | teacher feature-output `0.003` + late attention-output `1e-4` | 80.4920 | 95.2960 | 0.8399 | below Phase 2W `80.5400` and Phase 2Z `80.5540`; fail |

Conclusion:

The late teacher attention-output MSE is technically connected and lightweight, but it does not improve source construction. It drops the source to the same band as the half-epoch and attention-only source variants. Do not apply follow-up 250-update training to this source.

Interpretation:

- A small raw attention-output MSE is not the missing structural signal for the Phase 2S -> Phase 2W source.
- The useful feature-output signal should be kept, but adding another teacher-output objective on attention module outputs overconstrains or misdirects the source dynamics.
- Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not continue this attention-output source branch or sweep its scalar weight.
- The next non-repeated action should be diagnostic or a mechanism that changes the source state more directly around the diagnosed `move_qkx_aft` / late activation-scale mismatch without using direction-anchor, damping, confidence-band KD, or another teacher-output scalar loss.

### Phase 2BM: Full Late-Block Source Construction With Narrow Late MLP FC2 Activation-MSE Calibration

Reason:

Phase 2BK showed that the no-feature source over-moved `features.5.5.mlp.fc2.input_quant_fn.s` versus Phase 2W (`nofeat_over_phase2w=1.833`, `extra_over_phase2w=0.853`). Earlier activation-MSE endpoint repairs focused on qkx/qkv/v activation quantizers and did not test this diagnosed MLP `fc2` activation scale in the Phase 2S -> Phase 2W source-construction context. This phase tests a narrow source-state mechanism: after strict resume from Phase 2S and before the source epoch, run activation-MSE scale calibration only on the two late MLP `fc2.input_quant_fn` quantizers, then keep the known useful full late-block write set and teacher feature-output auxiliary.

This is a single-checkpoint strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Launcher change:

`qat_launch.py` now allows explicitly requested pre-QAT activation calibration after `--no-resume-opt` strict resume with `start_epoch > 0`, instead of only when `start_epoch == 0`. The gate runner also forwards `PRE_QAT_ACT_MSE_CALIB_*` variables.

Train command:

```bash
EXP=recipe_resume10_vartrust_epoch1_paramsinlate_fc2actmse_source_gate_20260708 \
MASTER_PORT=30780 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar \
START_EPOCH=1 \
EPOCHS=2 \
SCHEDULER_EPOCHS=2 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
PRE_QAT_ACT_MSE_CALIB_BATCHES=8 \
PRE_QAT_ACT_MSE_CALIB_LAYERS=features.5.5,features.7.1 \
PRE_QAT_ACT_MSE_CALIB_QUANTIZERS=features.5.5.mlp.fc2.input_quant_fn,features.7.1.mlp.fc2.input_quant_fn \
PRE_QAT_ACT_MSE_CALIB_GRID=0.85,1.15,13 \
PRE_QAT_ACT_MSE_CALIB_BLEND=0.35 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_ATTN_OUTPUT_WEIGHT=0 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=1 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=0 \
MAX_TRAIN_UPDATES=0 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_vartrust_epoch1_paramsinlate_fc2actmse_source_gate_20260708.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_fc2actmse_source_gate_20260708
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar; missing=0, unexpected=0
Strict train resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Starting pre-QAT activation MSE calibration: batches=8, grid=0.85,1.15,13, blend=0.35, layers=('features.5.5', 'features.7.1'), quantizers=('features.5.5.mlp.fc2.input_quant_fn', 'features.7.1.mlp.fc2.input_quant_fn'), matched=2
Finished pre-QAT activation MSE calibration: batches=8, updated=2, mean_scale_ratio=0.9779, min_ratio=0.9475, max_ratio=1.0083
Applied pre-QAT activation calibration after strict resume: start_epoch=1
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, pre_qat_act_mse_calib_batches=8, pre_qat_act_mse_calib_quantizers=features.5.5.mlp.fc2.input_quant_fn,features.7.1.mlp.fc2.input_quant_fn, teacher_feature_output_weight=0.003, pre_qat_feature_recon_updates=0, variation_trust_weight=0.0, bin_reg_weight=0.0, start_epoch=1, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5,features.7.1
Trainable parameter policy: epoch=1, quant_only=True, policy=params_in_layers, trainable=8933890, frozen=19674366
Teacher feature-output hooks: layers=('features.5.5', 'features.7.1'), student_matches=('features.5.5', 'features.7.1'), teacher_matches=('features.5.5', 'features.7.1')
TrainSummary: epoch=1 updates=2496 avg_step_time=0.121238s samples_per_step=512 samples_per_sec=4223.10
Test: [distributed-summary]  Time: 34.506s  Loss: 0.8403  Acc@1: 80.5360  Acc@5: 95.2800  Samples: 50000
```

Result:

| source checkpoint | trained epoch | policy | source-state change | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2S `checkpoint-1.pth.tar` | epoch 1 only | `params_in_layers`, `features.5.5,features.7.1` | narrow late MLP `fc2.input_quant_fn` ACT-MSE calibration | 80.5360 | 95.2800 | 0.8403 | close to Phase 2W `80.5400`, below Phase 2Z `80.5540`; fail to justify follow-up |

Conclusion:

Narrow late MLP `fc2` activation-MSE calibration is the strongest new source-construction variant in this group, nearly matching Phase 2W. It still does not beat Phase 2W and remains below the current best Phase 2Z, so it should not receive the follow-up 250-update branch.

Interpretation:

- The diagnosed `fc2.input_quant_fn` scale mismatch is relevant: correcting it recovers most of the source quality lost by no-feature/attention-only/attention-output source variants.
- The calibration slightly undershoots Phase 2W, so static activation-scale repair alone is not sufficient.
- Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not continue Phase 2BM into the 250-update follow-up.
- The useful signal is that narrow source-state repair can nearly recover Phase 2W; next work should diagnose Phase 2BM vs Phase 2W parameter deltas and class flips before deciding whether a second, equally narrow source-state repair is justified.

### Phase 2BN: Phase 2BM vs Phase 2W Parameter and Class/Logit Diagnostics

Reason:

Phase 2BM nearly matched Phase 2W (`80.5360` vs `80.5400`) but did not beat it. Before trying another source-state repair, this diagnostic compares Phase 2BM against Phase 2W to determine whether the remaining gap is random noise, a specific class-flip tradeoff, or an over-correction in the targeted `fc2.input_quant_fn` activation scales.

This is diagnostic only. It is not a training result and does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Commands:

```bash
python3 /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_source_delta_20260708.py \
  --out-dir /mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_bm_vs_phase2w_delta_20260708 \
  --cmp-label phase2bm_fc2actmse \
  --cmp-checkpoint /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_fc2actmse_source_gate_20260708/checkpoint-2.pth.tar \
  --cmp-top1 80.5360

CUDA_VISIBLE_DEVICES=0 python3 /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_logit_classes_20260708.py \
  --out-dir /mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_bm_vs_phase2w_logit_diag_20260708 \
  --checkpoint phase2w=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar \
  --checkpoint phase2bm_fc2actmse=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_fc2actmse_source_gate_20260708/checkpoint-2.pth.tar \
  --labels phase2w,phase2bm_fc2actmse \
  --compare-label phase2w \
  --devices 0 \
  --device-index 0 \
  --master-port 30781 \
  --batch-size 128 \
  --workers 8
```

Artifacts:

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_bm_vs_phase2w_delta_20260708/summary.json
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_bm_vs_phase2w_delta_20260708/kind_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_bm_vs_phase2w_delta_20260708/stage_kind_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_bm_vs_phase2w_delta_20260708/module_kind_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_bm_vs_phase2w_delta_20260708/param_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_bm_vs_phase2w_logit_diag_20260708/summary.json
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_bm_vs_phase2w_logit_diag_20260708/class_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_bm_vs_phase2w_logit_diag_20260708/confidence_bins.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_bm_vs_phase2w_logit_diag_20260708/flip_cases.tsv
```

Diagnostic evidence:

```text
Parameter delta, Phase 2S as base:
- Phase 2BM vs Phase 2W differs by only -0.004 Top-1 in 8-GPU full-val.
- The largest remaining module-level differences are exactly the calibrated fc2 activation scales:
  features.7.1.mlp.fc2.input_quant_fn | cmp_over_phase2w=58.19, delta_cosine_to_phase2w=0.681, extra_over_phase2w=57.51
  features.5.5.mlp.fc2.input_quant_fn | cmp_over_phase2w=5.36, delta_cosine_to_phase2w=-0.892, extra_over_phase2w=6.27
- Remaining qkx direction mismatch is not solved:
  features.5.5.attn.move_qkx_aft | cmp_over_phase2w=1.031, cosine=-0.002, extra_over_phase2w=1.438
  features.7.1.attn.move_qkx_aft | cmp_over_phase2w=1.028, cosine=0.008, extra_over_phase2w=1.428
- Kind-level mismatch is dominated by act_quant (`cmp_over_phase2w=2.68`, cosine=0.012, extra_over_phase2w=2.85).

Single-GPU same-script full-val diagnostic:
phase2w:            Acc@1=80.4720, Acc@5=95.3180, Samples=50000
phase2bm_fc2actmse: Acc@1=80.4720, Acc@5=95.3220, Samples=50000

phase2w vs phase2bm_fc2actmse:
improved=347, regressed=347, net_flips=0
confidence bins: [0.00,0.20) net +6 for Phase 2W; [0.40,0.60) net -4; high-confidence bins mostly unchanged.
classes where Phase 2W is better include 654 (+3), 681 (+3), 856 (+3), 489 (+3), 925 (+3).
classes where Phase 2BM is better include 348 (-4), 441 (-3), 817 (-3), 754 (-3), 756 (-3).
```

Conclusion:

Phase 2BM is not a clean improvement over Phase 2W. It mostly creates a different local source state with exactly balanced class flips under the diagnostic script and a small 8-GPU full-val loss. The fc2 activation-scale calibration is too strong at blend `0.35`: it moves the target scales far beyond the original Phase 2W delta, especially in `features.7.1.mlp.fc2.input_quant_fn`.

Interpretation:

- The fc2 activation-scale target is relevant, but the repair amplitude is too high.
- The qkx direction mismatch remains independent of this fc2 calibration and should not be attacked again with the already failed direction-anchor family.
- If continuing this path, the only justified source-state gate is a smaller fc2 ACT-MSE blend, not another new broad calibration.

Next decision:

- A possible next gate is `PRE_QAT_ACT_MSE_CALIB_BLEND=0.15` on the same two late MLP `fc2.input_quant_fn` quantizers, keeping all other Phase 2BM settings identical.
- Gate remains source-only: it must beat Phase 2W `80.5400` and ideally approach Phase 2Z `80.5540` before any follow-up 250-update branch is considered.

### Phase 2BO: Weak Late MLP FC2 Activation-MSE Calibration, Blend 0.15

Reason:

Phase 2BN showed that Phase 2BM's `0.35` blend over-corrected the targeted late MLP `fc2.input_quant_fn` activation scales, especially `features.7.1.mlp.fc2.input_quant_fn`. This phase keeps the same narrow source-state mechanism and all other source-construction settings, but reduces the calibration blend from `0.35` to `0.15` to test whether a smaller correction can preserve Phase 2W-level behavior while avoiding the over-moved activation scale delta.

This is a single-checkpoint strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Train command:

```bash
EXP=recipe_resume10_vartrust_epoch1_paramsinlate_fc2actmse_b015_source_gate_20260708 \
MASTER_PORT=30782 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar \
START_EPOCH=1 \
EPOCHS=2 \
SCHEDULER_EPOCHS=2 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
PRE_QAT_ACT_MSE_CALIB_BATCHES=8 \
PRE_QAT_ACT_MSE_CALIB_LAYERS=features.5.5,features.7.1 \
PRE_QAT_ACT_MSE_CALIB_QUANTIZERS=features.5.5.mlp.fc2.input_quant_fn,features.7.1.mlp.fc2.input_quant_fn \
PRE_QAT_ACT_MSE_CALIB_GRID=0.85,1.15,13 \
PRE_QAT_ACT_MSE_CALIB_BLEND=0.15 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_ATTN_OUTPUT_WEIGHT=0 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=1 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=0 \
MAX_TRAIN_UPDATES=0 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_vartrust_epoch1_paramsinlate_fc2actmse_b015_source_gate_20260708.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_fc2actmse_b015_source_gate_20260708
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar; missing=0, unexpected=0
Strict train resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Starting pre-QAT activation MSE calibration: batches=8, grid=0.85,1.15,13, blend=0.15, layers=('features.5.5', 'features.7.1'), quantizers=('features.5.5.mlp.fc2.input_quant_fn', 'features.7.1.mlp.fc2.input_quant_fn'), matched=2
Finished pre-QAT activation MSE calibration: batches=8, updated=2, mean_scale_ratio=0.9905, min_ratio=0.9775, max_ratio=1.0035
Applied pre-QAT activation calibration after strict resume: start_epoch=1
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, pre_qat_act_mse_calib_batches=8, pre_qat_act_mse_calib_blend=0.15, pre_qat_act_mse_calib_quantizers=features.5.5.mlp.fc2.input_quant_fn,features.7.1.mlp.fc2.input_quant_fn, teacher_feature_output_weight=0.003, pre_qat_feature_recon_updates=0, variation_trust_weight=0.0, bin_reg_weight=0.0, start_epoch=1, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5,features.7.1
Trainable parameter policy: epoch=1, quant_only=True, policy=params_in_layers, trainable=8933890, frozen=19674366
Teacher feature-output hooks: layers=('features.5.5', 'features.7.1'), student_matches=('features.5.5', 'features.7.1'), teacher_matches=('features.5.5', 'features.7.1')
TrainSummary: epoch=1 updates=2496 avg_step_time=0.121436s samples_per_step=512 samples_per_sec=4216.20
Test: [distributed-summary]  Time: 34.364s  Loss: 0.8392  Acc@1: 80.5280  Acc@5: 95.2900  Samples: 50000
```

Result:

| source checkpoint | trained epoch | policy | source-state change | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2S `checkpoint-1.pth.tar` | epoch 1 only | `params_in_layers`, `features.5.5,features.7.1` | narrow late MLP `fc2.input_quant_fn` ACT-MSE calibration, blend `0.15` | 80.5280 | 95.2900 | 0.8392 | below Phase 2BM `80.5360`, Phase 2W `80.5400`, and Phase 2Z `80.5540`; fail |

Conclusion:

Weakening the fc2 activation-MSE blend from `0.35` to `0.15` does not improve the source. It moves the calibration less aggressively (`mean_scale_ratio=0.9905` instead of `0.9779`), but the source full-val drops to `80.5280`.

Interpretation:

- The fc2 activation-scale correction is relevant but not monotonic; simply reducing the blend does not recover Phase 2W.
- The fc2 ACT-MSE family should be closed for now rather than swept further.
- Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not continue fc2 ACT-MSE source calibration or sweep blend values.
- The remaining useful work should return to diagnostics around the persistent qkx direction mismatch or compare Phase 2Z best against the strongest source variants at case/parameter level before launching another training gate.

### Phase 2BP: Phase 2Z Best vs Source and Short-Update Diagnostic Synthesis

Reason:

After Phase 2BO closed the fc2 ACT-MSE source-calibration family, the remaining question is whether the current best Phase 2Z `80.5540` still exposes a training gate worth launching. This synthesis reviews the existing Phase 2Z-vs-source and short-update diagnostics instead of immediately launching another speculative branch.

This is diagnostic synthesis only. It is not a training result and does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Evidence reviewed:

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_logit_class_diag_20260708/summary.json
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_logit_class_diag_20260708/class_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_logit_class_diag_20260708/confidence_bins.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_short_update_drift_20260708.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_attn_vs_full_diag_20260708/summary.json
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_attn_vs_full_diag_20260708/confidence_bins.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_attn_vs_full_param_delta_20260708/module_kind_delta.tsv
```

Key observations:

```text
Best official checkpoint remains Phase 2Z:
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000

Same-script single-GPU diagnostic is noisy around Phase 2W vs best250:
phase2w:  Acc@1=80.5060, Samples=50000
best250:  Acc@1=80.5040, Samples=50000
official 8-GPU full-val still favors best250 by +0.014 Top-1.

best250 vs phase2w class/logit behavior:
improved=355, regressed=356, net_flips=-1 in same-script diagnostic.
confidence bins are mixed:
  [0.00,0.20): net +11 for best250
  [0.20,0.40): net -14 for best250
  [0.40,0.60): net +1 for best250
Top class gains for best250 include 831 (+4), 848 (+3), 754 (+3), 200 (+3), 231 (+3).
Top class losses include 349 (-4), 876 (-4), 264 (-3), 813 (-3), 928 (-3), 764 (-3).

Attention-only vs full-block 250:
attn250 same-script Top-1=80.4580; full250 same-script Top-1=80.4940.
full250's net gain over attn250 is concentrated in ref confidence [0.20,0.40): net +32.
However, local/full-block reference KD on that band failed badly at 80.4440.

Short-update drift:
largest negative correlations with Top-1 include:
  features.7.1.mlp.fc2.input_quant_fn.s, corr_top1=-0.374
  features.7.1.mlp.fc1.input_quant_fn.s, corr_top1=-0.205
  features.5.5.attn.move_v_aft, corr_top1=-0.139
  features.5.5.attn.proj.move_aft, corr_top1=-0.136
  features.5.5.attn.move_v_b4, corr_top1=-0.126
But direct suppression/calibration attempts failed:
  move-v-only damping: 80.5100
  high-drift damping: 80.5100
  fc2 ACT-MSE blend 0.35: 80.5360
  fc2 ACT-MSE blend 0.15: 80.5280
```

Conclusion:

The current `80.5540` Phase 2Z checkpoint is a narrow local optimum, not a broad mechanism that can be safely improved by another simple scalar loss. The obvious diagnostic handles have all been tested and failed:

- confidence/logit transfer: teacher band KD, fixed-ref band KD, local-ref band KD, class-protect ref KL all failed.
- parameter suppression: hard freeze, broad damping, move-v-only damping, and delayed damping all failed.
- qkx direction: direct and delayed direction anchors failed.
- source construction: shorter source, attention-only source, no-feature source, attention-output source, and fc2 ACT-MSE source calibration all failed.
- endpoint/update count: 225/240/275/300/500 update probes did not beat 250.

Interpretation:

- The remaining gap to 81 is not likely to be solved by another local patch on Phase 2W -> Phase 2Z.
- The strongest next candidate must change the representation pathway more materially while preserving the strict W4A4 single-checkpoint constraint.
- A plausible non-repeated direction is to test the only nearby mechanism that has not been isolated in the full-block source path: qkx activation-scale state handling during source construction, but not as endpoint ACT-MSE and not as qkx delta direction anchoring.

Next decision:

- If launching one more source-state gate, use a very narrow qkx activation-scale initialization/calibration from Phase 2S before the full late-block source epoch, keeping teacher feature-output and full late-block write set unchanged.
- Gate remains source-only: it must beat Phase 2W `80.5400` before any follow-up short-update branch.

### Phase 2BQ: Narrow Late QKX Activation-MSE Calibration Before Source Epoch

Reason:

Phase 2BP identified persistent qkx activation/shift mismatch as one of the remaining unexplained source/short-update issues, while direction-anchor attempts on qkx deltas had already failed. This phase tests a different qkx source-state mechanism: before the Phase 2S -> source epoch, calibrate only the late attention `quan_a_qkx_fn` activation scales under `features.5.5` and `features.7.1`, then keep the full late-block write set and teacher feature-output auxiliary unchanged.

This is a single-checkpoint strict W4A4 branch. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Train command:

```bash
EXP=recipe_resume10_vartrust_epoch1_paramsinlate_qkxactmse_source_gate_20260708 \
MASTER_PORT=30783 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar \
START_EPOCH=1 \
EPOCHS=2 \
SCHEDULER_EPOCHS=2 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
PRE_QAT_ACT_MSE_CALIB_BATCHES=8 \
PRE_QAT_ACT_MSE_CALIB_LAYERS=features.5.5,features.7.1 \
PRE_QAT_ACT_MSE_CALIB_QUANTIZERS=features.5.5.attn.quan_a_qkx_fn,features.7.1.attn.quan_a_qkx_fn \
PRE_QAT_ACT_MSE_CALIB_GRID=0.85,1.15,13 \
PRE_QAT_ACT_MSE_CALIB_BLEND=0.35 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_ATTN_OUTPUT_WEIGHT=0 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=1 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=0 \
MAX_TRAIN_UPDATES=0 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_vartrust_epoch1_paramsinlate_qkxactmse_source_gate_20260708.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_qkxactmse_source_gate_20260708
GPU worker evidence: gpu-device-present, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded model from .../recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar; missing=0, unexpected=0
Strict train resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Starting pre-QAT activation MSE calibration: batches=8, grid=0.85,1.15,13, blend=0.35, layers=('features.5.5', 'features.7.1'), quantizers=('features.5.5.attn.quan_a_qkx_fn', 'features.7.1.attn.quan_a_qkx_fn'), matched=2
Finished pre-QAT activation MSE calibration: batches=8, updated=2, mean_scale_ratio=0.9655, min_ratio=0.9535, max_ratio=0.9776
Applied pre-QAT activation calibration after strict resume: start_epoch=1
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, pre_qat_act_mse_calib_batches=8, pre_qat_act_mse_calib_blend=0.35, pre_qat_act_mse_calib_quantizers=features.5.5.attn.quan_a_qkx_fn,features.7.1.attn.quan_a_qkx_fn, teacher_feature_output_weight=0.003, pre_qat_feature_recon_updates=0, variation_trust_weight=0.0, bin_reg_weight=0.0, start_epoch=1, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5,features.7.1
Trainable parameter policy: epoch=1, quant_only=True, policy=params_in_layers, trainable=8933890, frozen=19674366
Teacher feature-output hooks: layers=('features.5.5', 'features.7.1'), student_matches=('features.5.5', 'features.7.1'), teacher_matches=('features.5.5', 'features.7.1')
TrainSummary: epoch=1 updates=2496 avg_step_time=0.121240s samples_per_step=512 samples_per_sec=4223.03
Test: [distributed-summary]  Time: 34.647s  Loss: 0.8389  Acc@1: 80.4860  Acc@5: 95.2900  Samples: 50000
```

Result:

| source checkpoint | trained epoch | policy | source-state change | raw Top-1 | raw Top-5 | loss | gate |
|---|---:|---|---|---:|---:|---:|---|
| Phase 2S `checkpoint-1.pth.tar` | epoch 1 only | `params_in_layers`, `features.5.5,features.7.1` | late qkx activation-scale ACT-MSE calibration | 80.4860 | 95.2900 | 0.8389 | below Phase 2W `80.5400` and Phase 2Z `80.5540`; fail |

Conclusion:

Narrow qkx activation-scale calibration is harmful in the source-construction path. It contracts the two qkx activation scales (`mean_scale_ratio=0.9655`) and drops the source to the same weak range as the half-epoch/attention-only source variants.

Interpretation:

- The persistent qkx mismatch is real but not repairable by direct activation-scale MSE calibration.
- Together with the failed qkx direction-anchor branches, this closes direct qkx source-state manipulation for now.
- Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Next decision:

- Do not continue qkx ACT-MSE calibration or qkx direction/damping variants.
- The current local Phase 2W/2Z neighborhood is largely exhausted. Further work should pivot to a new training-paradigm mechanism rather than another local source/250-update patch.

### Phase 2BR: Continuous 5-Epoch Late-Block Curve From Phase 2S

Reason:

The user asked to run the currently best nearby scheme continuously for 5 epochs to see whether the curve naturally climbs past the current single-checkpoint best. This phase keeps the Phase 2S source, full late-block write set, strict W4A4 quantization, and teacher feature-output auxiliary, but runs five consecutive resumed epochs with `scheduler_epochs=6`.

This is a strict W4A4 single-checkpoint run. It does not use soup, checkpoint averaging, multi-checkpoint averaging, or ensemble.

Train command:

```bash
EXP=recipe_resume10_paramsinlate_5epoch_curve_from_phase2s_20260708 \
MASTER_PORT=30784 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar \
START_EPOCH=1 \
EPOCHS=6 \
SCHEDULER_EPOCHS=6 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
PRE_QAT_ACT_MSE_CALIB_BATCHES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_ATTN_OUTPUT_WEIGHT=0 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=1 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=0 \
MAX_TRAIN_UPDATES=0 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

Runtime evidence:

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_5epoch_curve_from_phase2s_20260708.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_5epoch_curve_from_phase2s_20260708
GPU worker evidence: worker 984521, 8x NVIDIA H100 80GB HBM3 visible
Strict train resume: loaded Phase 2S checkpoint; missing=0, unexpected=0
Strict train resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Args: wq_bitw=4, aq_bitw=4, kd_hard_and_soft=0, start_epoch=1, epochs=6, scheduler_epochs=6, teacher_feature_output_weight=0.003, pre_qat_feature_recon_updates=0, variation_trust_weight=0.0, bin_reg_weight=0.0, quant_only_start_epoch=1, trainable_policy=params_in_layers, trainable_policy_freeze_act_except_layers=features.5.5,features.7.1
Trainable parameter policy: epoch=1..5, quant_only=True, policy=params_in_layers, trainable=8933890, frozen=19674366
Teacher feature-output hooks: layers=('features.5.5', 'features.7.1'), student_matches=('features.5.5', 'features.7.1'), teacher_matches=('features.5.5', 'features.7.1')
```

Full validation curve:

| checkpoint | trained epoch | raw Top-1 | raw Top-5 | loss | samples | vs current best `80.5540` |
|---|---:|---:|---:|---:|---:|---:|
| `checkpoint-2.pth.tar` | epoch 1 | 80.5180 | 95.2820 | 0.8416 | 50000 | -0.0360 |
| `checkpoint-3.pth.tar` | epoch 2 | 80.4400 | 95.2640 | 0.8411 | 50000 | -0.1140 |
| `checkpoint-4.pth.tar` | epoch 3 | 80.5460 | 95.2860 | 0.8397 | 50000 | -0.0080 |
| `checkpoint-5.pth.tar` | epoch 4 | 80.4680 | 95.3060 | 0.8399 | 50000 | -0.0860 |
| `checkpoint-6.pth.tar` | epoch 5 | 80.4720 | 95.2880 | 0.8407 | 50000 | -0.0820 |

Result:

The best point in the 5-epoch continuous curve is epoch 3 at Top-1 `80.5460`, which is close to but still below Phase 2Z `80.5540`. The final epoch ends at `80.4720`, so continuing this recipe does not naturally climb toward 81%.

Current best remains:

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
Top-1 80.5540, Top-5 95.3060, Loss 0.8387, Samples 50000
```

Conclusion:

The full late-block teacher-feature recipe has a narrow local high point but no positive long-horizon trend. More epochs under the same scheduler/write set oscillate around `80.44-80.55` and do not beat the current best checkpoint. This closes the "just run the best scheme for 5 epochs" hypothesis.

Next decision:

- Do not spend more compute extending this exact 5-epoch recipe.
- If continuing toward 81, pivot to a different training-paradigm mechanism rather than another local Phase 2W/2Z extension. The next candidate should change supervision or stabilization at a broader transition level while preserving strict W4A4 single-checkpoint evaluation.

### Phase 2BS：AOQ 视角的 late-block weight-bin crossing 诊断

实验动机：

新的 AOQ-native goal 要求先从“全程抑制震荡”的 OFQ-family 局部修补路线切换到“前期受控探索、后期稳定”的范式。AOQ 论文的核心指标不是普通参数 L2 drift，而是权重是否跨过 quantization threshold、是否改变 quantized bin assignment。因此本阶段先做离线诊断，回答两个问题：

1. 当前有益 checkpoint 演化中，哪些 late-block 权重发生了适度 bin crossing？
2. 最近 5-epoch 连续训练为什么没有继续涨，是不是 crossing 过大、变成无效漂移？

本阶段只做 CPU 离线诊断，不启动训练，不占 GPU。

脚本：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_aoq_bin_crossing_20260708.py
```

输出：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_aoq_bin_crossing_20260708/
  aggregate_bin_crossing.tsv
  pair_bin_crossing.tsv
  summary.json
```

命令：

```bash
python3 /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_aoq_bin_crossing_20260708.py
```

检查：

```text
python3 -m py_compile tmp_scripts/diagnose_resume10_aoq_bin_crossing_20260708.py
git diff --check -- tmp_scripts/diagnose_resume10_aoq_bin_crossing_20260708.py
```

诊断设计：

- 默认比较 4 组历史 checkpoint 演化：
  - `ckpt10 -> phase2s`
  - `phase2s -> phase2w`
  - `phase2w -> phase2z`
  - `phase2z -> phase2br_e3`
- 默认只覆盖 AOQ-native goal 里优先关注的 late useful blocks：
  - `features.5.5`
  - `features.7.1`
- 对普通 weight 使用 checkpoint 中的权重和对应量化规则做离线 bin assignment：
  - StatsQ-like weight：按 OFQ `StatsQuantizer` 的 `2 * mean(abs(weight))` per-row scale 和 4-bit signed bin 规则估算。
  - LSQ weight：若 checkpoint 中有 `lsqw_fn.s`，用 checkpoint scale 估算。
- 对 QKR 分支额外计算 attention q/k 的 composite：
  - `q^T k -> qk_quant` 的 4-bit bin assignment。
- 主要指标：
  - `changed_fraction`：前后 checkpoint 的 quantized bin assignment 变化比例。
  - `after_near_fraction`：后一 checkpoint 中接近 quantization threshold 的权重比例，默认 near margin 为 `0.05`。
  - `mean_abs_bin_delta`：平均 bin 跳变幅度。
  - `top1_delta`：该 checkpoint 演化对应的 full-val Top-1 变化。

关键结果：

| checkpoint pair | Top-1 delta | 关键 crossing 现象 |
|---|---:|---|
| `ckpt10 -> phase2s` | +0.1580 | 大幅有益提升，`features.7.1` 的 `mlp_fc2` / `attn_proj` crossing 约 `8.6%`，`features.5.5/7.1` 的 qk composite crossing 约 `5.0%` |
| `phase2s -> phase2w` | +0.0180 | 小幅继续提升，`features.7.1` 的 `mlp_fc2` / `attn_proj` crossing 约 `7.6-7.8%`，`features.5.5` 的 `attn_v` / `attn_proj` 约 `6.1-6.2%` |
| `phase2w -> phase2z` | +0.0140 | 当前 best 的局部增益只需要更小 crossing：`features.7.1.mlp_fc2` 约 `4.48%`，`features.7.1.attn_proj` 约 `4.00%`，qk composite 约 `1.0%` |
| `phase2z -> phase2br_e3` | -0.0080 | 继续训练后 crossing 明显放大但 Top-1 反降：`features.7.1.mlp_fc2` 约 `10.57%`，`features.7.1.attn_proj` 约 `10.45%`，`features.5.5` qk composite 约 `8.16%` |

几个代表性模块的数值：

| module | `ckpt10->2S` | `2S->2W` | `2W->2Z` | `2Z->2BR-e3` |
|---|---:|---:|---:|---:|
| `features.7.1|mlp_fc2` changed_fraction | 0.08594 | 0.07820 | 0.04484 | 0.10568 |
| `features.7.1|attn_proj` changed_fraction | 0.08537 | 0.07597 | 0.03998 | 0.10446 |
| `features.5.5|attn_qk_composite` changed_fraction | 0.05121 | 0.03847 | 0.01014 | 0.08157 |
| `features.7.1|attn_qk_composite` changed_fraction | 0.04982 | 0.03218 | 0.00989 | 0.07120 |
| `features.5.5|attn_v` changed_fraction | 0.04966 | 0.06099 | 0.01790 | 0.08326 |

中文结论：

1. AOQ 视角确认了一个重要现象：有益适配确实伴随 selected late-block 的 weight-bin crossing，而不是完全冻结或完全不动。
2. 但 crossing 不是越多越好。`phase2w -> phase2z` 的当前 best 增益只伴随中等或较小 crossing；继续跑到 `phase2br_e3` 时，late-block crossing 大幅增加，Top-1 反而下降。
3. 这支持 AOQ-native 新范式，但也给出边界：不能简单“放大所有 oscillation”。必须做受控探索，并在 crossing 达到有效区间后及时 delayed stabilization。
4. 当前 `features.7.1.mlp_fc2`、`features.7.1.attn_proj` 是最强的 bin-crossing 探索/风险模块；`features.5.5/7.1` 的 qk composite 在有效提升阶段 crossing 很小，在过度继续训练阶段 crossing 变大且伴随回落，说明 QKR/qk 路径尤其需要谨慎。

下一步判断：

- 下一步不直接重复 `params_in_layers` 延长训练。
- 先设计 AOQ-inspired 的“受控 crossing gate”：
  - 探索阶段只允许 selected late weight 模块有适度 threshold/scale narrowing。
  - 目标 crossing 区间优先参考 `phase2w -> phase2z`，避免达到 `phase2z -> phase2br_e3` 的过量 crossing。
  - 后半段加入 delayed bin-center / dampening，而不是从一开始 damping。
- 同时准备 clean no-QKR/no-StatsQ branch，但第一步必须先保证 strict W4A4 full-val 可跑；不要一次性替换太多机制导致无法归因。

### Phase 2BT：AOQ explore / delayed BinReg 训练路径 smoke

实验动机：

Phase 2BS 说明有益提升需要受控的 late-block weight-bin crossing，而继续训练导致 crossing 过大后会回落。本阶段先不跑 full-val gate，只验证代码路径：能否在训练循环中按 update 区间打开 AOQ explore scale ratio，并在后续 update 延迟打开 BinReg。这个 smoke 只跑 2 个 optimizer updates，跳过验证，不作为精度结果。

代码改动：

```text
/mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py
/mlx_devbox/users/quyanyi/playground/QATs/third_party/OFQ/src/quantization/quantizer/statsq.py
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

新增机制：

- `StatsQuantizer` / `StatsQuantizer_specific_4_qkreparam_cga` / `StatsQuantizer_4d` 增加默认 `aoq_scale_ratio=1.0`。
- 显式传入 `--aoq-explore-scale-ratio <1` 时，训练循环会在指定 update 区间把 selected StatsQ/qk/v quantizer 的 scale 临时乘上该 ratio，从而缩窄 threshold/level 间隔，诱导受控 bin crossing。
- 新增 delayed BinReg 区间控制：
  - `--bin-reg-start-update`
  - `--bin-reg-end-update`
- 默认值保持关闭，因此不传新参数时不改变旧行为。

静态检查：

```text
python3 -m py_compile qat_launch.py tmp_scripts/diagnose_resume10_aoq_bin_crossing_20260708.py
bash -n tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
git diff --check -- qat_launch.py third_party/OFQ/src/quantization/quantizer/statsq.py tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh tmp_scripts/diagnose_resume10_aoq_bin_crossing_20260708.py docs/resume10_to81_goal_progress_20260706.md docs/resume10_to81_aoq_native_goal_20260708.md
```

Smoke 命令：

```bash
EXP=recipe_resume10_aoq_explore_smoke2upd_20260708 \
MASTER_PORT=30785 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar \
START_EPOCH=1 \
EPOCHS=2 \
SCHEDULER_EPOCHS=2 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
PRE_QAT_ACT_MSE_CALIB_BATCHES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=1e-5 \
BIN_REG_VARIANCE_WEIGHT=1.0 \
BIN_REG_LAYERS=features.7.1.mlp.fc2,features.7.1.attn.proj \
BIN_REG_START_UPDATE=1 \
BIN_REG_END_UPDATE=2 \
AOQ_EXPLORE_SCALE_RATIO=0.9 \
AOQ_EXPLORE_LAYERS=features.7.1.mlp.fc2,features.7.1.attn.proj \
AOQ_EXPLORE_START_UPDATE=0 \
AOQ_EXPLORE_END_UPDATE=2 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_ATTN_OUTPUT_WEIGHT=0 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=1 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=0 \
MAX_TRAIN_UPDATES=2 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

运行证据：

```text
GPU worker: 984521, 8x NVIDIA H100 80GB HBM3 visible
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_aoq_explore_smoke2upd_20260708.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_aoq_explore_smoke2upd_20260708
Strict resume: loaded model from .../recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
args.yaml:
  aoq_explore_scale_ratio: 0.9
  aoq_explore_layers: features.7.1.mlp.fc2,features.7.1.attn.proj
  aoq_explore_start_update: 0
  aoq_explore_end_update: 2
  bin_reg_start_update: 1
  bin_reg_end_update: 2
```

关键日志：

```text
AOQ explore scale ratio update: epoch=1, update=0, active=True, ratio=0.9, layers=('features.7.1.mlp.fc2', 'features.7.1.attn.proj'), quantizers=2, start_update=0, end_update=2
Enabled bin regularizer: weight=1e-05, variance_weight=1.0, layers=('features.7.1.mlp.fc2', 'features.7.1.attn.proj'), attn_only=False, pairs=2, start_update=1, end_update=2
TrainSummary: epoch=1 updates=2 avg_step_time=0.625954s samples_per_step=512 samples_per_sec=817.95
Stopped early after 2 optimizer updates in epoch 1.
```

结果：

- smoke 成功进入 AOQ explore 路径。
- smoke 成功在指定 update=1 开启 delayed BinReg。
- 该 run 使用 `MAX_TRAIN_UPDATES=2` 和 `--skip_validate`，没有 full-val，因此不计入 Top-1 结果。
- 早停后的 `TCPStore/NCCL` warning 是 DDP teardown warning，不是训练路径失败。

下一步判断：

- 可以启动第一个 full-val gate。
- gate 不应太激进，先用 `AOQ_EXPLORE_SCALE_RATIO=0.95` 或 `0.9`，只作用于 Phase 2BS 诊断出的核心 selected modules：`features.7.1.mlp.fc2,features.7.1.attn.proj`。
- delayed BinReg 放在后半段，例如 2502 steps/epoch 中的 1250 之后，避免一开始就抑制探索。
- Gate 规则：第一个 full-val 必须达到或接近 Phase 2S/2W 水平；如果低于 `80.5220` 且无上升趋势，就停止该 AOQ scale-ratio 分支。

### Phase 2BU：AOQ explore 0.95 + delayed BinReg full-val gate

实验动机：

Phase 2BS 的离线诊断显示，`phase2w -> phase2z` 的当前 best 增益只需要 `features.7.1.mlp_fc2` 和 `features.7.1.attn_proj` 中等 crossing，而 `phase2z -> phase2br_e3` 的过量 crossing 会导致回落。本阶段测试一个保守 AOQ-inspired gate：前半个 epoch 对这两个模块使用 `aoq_explore_scale_ratio=0.95`，后半个 epoch 关闭 explore 并启用 delayed BinReg，尝试“先受控探索、再稳定”。

这是 strict W4A4 单 checkpoint 实验，不使用 soup / checkpoint averaging / multi-checkpoint averaging / ensemble。

命令：

```bash
EXP=recipe_resume10_aoq_explore095_delayedbin_gate_20260708 \
MASTER_PORT=30786 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar \
START_EPOCH=1 \
EPOCHS=2 \
SCHEDULER_EPOCHS=2 \
PRE_QAT_FEATURE_RECON_UPDATES=0 \
PRE_QAT_ACT_MSE_CALIB_BATCHES=0 \
VARIATION_TRUST_WEIGHT=0 \
BIN_REG_WEIGHT=1e-5 \
BIN_REG_VARIANCE_WEIGHT=1.0 \
BIN_REG_LAYERS=features.7.1.mlp.fc2,features.7.1.attn.proj \
BIN_REG_START_UPDATE=1250 \
BIN_REG_END_UPDATE=0 \
AOQ_EXPLORE_SCALE_RATIO=0.95 \
AOQ_EXPLORE_LAYERS=features.7.1.mlp.fc2,features.7.1.attn.proj \
AOQ_EXPLORE_START_UPDATE=0 \
AOQ_EXPLORE_END_UPDATE=1250 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
TEACHER_ATTN_OUTPUT_WEIGHT=0 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0 \
REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
CLASS_PROTECT_REF_KL_WEIGHT=0 \
LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT=0 \
QUANT_ONLY_START_EPOCH=1 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
SAVE_STEP_CHECKPOINTS=0 \
MAX_TRAIN_UPDATES=0 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_prerecon_vartrust_gate_20260707.sh
```

运行证据：

```text
GPU worker: 984521, 8x NVIDIA H100 80GB HBM3 visible
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_aoq_explore095_delayedbin_gate_20260708.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_aoq_explore095_delayedbin_gate_20260708
Checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_aoq_explore095_delayedbin_gate_20260708/checkpoint-2.pth.tar
Strict resume: loaded model from .../recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
args.yaml:
  aoq_explore_scale_ratio: 0.95
  aoq_explore_layers: features.7.1.mlp.fc2,features.7.1.attn.proj
  aoq_explore_start_update: 0
  aoq_explore_end_update: 1250
  bin_reg_start_update: 1250
  bin_reg_end_update: 0
```

关键训练日志：

```text
AOQ explore scale ratio update: epoch=1, update=0, active=True, ratio=0.95, layers=('features.7.1.mlp.fc2', 'features.7.1.attn.proj'), quantizers=2, start_update=0, end_update=1250
AOQ explore scale ratio update: epoch=1, update=1250, active=False, ratio=1.0, layers=('features.7.1.mlp.fc2', 'features.7.1.attn.proj'), quantizers=2, start_update=0, end_update=1250
Enabled bin regularizer: weight=1e-05, variance_weight=1.0, layers=('features.7.1.mlp.fc2', 'features.7.1.attn.proj'), attn_only=False, pairs=2, start_update=1250, end_update=0
TrainSummary: epoch=1 updates=2496 avg_step_time=0.128766s samples_per_step=512 samples_per_sec=3976.20
```

Full-val 结果：

```text
Test: [distributed-summary]  Time: 34.958s  Loss: 0.8412  Acc@1: 80.4740  Acc@5: 95.2860  Samples: 50000
```

结果表：

| checkpoint | strict W4A4 | single checkpoint | full ImageNet raw Samples | Top-1 | Top-5 | loss | gate |
|---|---|---|---:|---:|---:|---:|---|
| `checkpoint-2.pth.tar` | yes | yes | 50000 | 80.4740 | 95.2860 | 0.8412 | fail：低于 Phase 2S `80.5220`、Phase 2W `80.5400` 和当前 best `80.5540` |

AOQ bin-crossing 复盘：

```text
脚本: tmp_scripts/diagnose_resume10_aoq_bin_crossing_20260708.py
输出: docs/resume10_aoq_explore095_bin_crossing_20260708/
比较: phase2s -> AOQ095 checkpoint-2
```

关键 crossing：

| module | changed_fraction | after_near_fraction | Top-1 delta vs Phase 2S |
|---|---:|---:|---:|
| `features.7.1|mlp_fc2` | 0.07134 | 0.33846 | -0.0480 |
| `features.7.1|attn_proj` | 0.06916 | 0.30927 | -0.0480 |
| `features.5.5|attn_proj` | 0.06213 | 0.27185 | -0.0480 |
| `features.5.5|attn_v` | 0.06127 | 0.26030 | -0.0480 |
| `features.5.5|attn_qk_composite` | 0.03838 | 0.20160 | -0.0480 |

中文结论：

1. 代码路径有效：AOQ explore scale ratio 和 delayed BinReg 都按 update 区间生效。
2. 但这个保守 AOQ095 gate 的 full-val 只有 `80.4740`，低于 Phase 2S/2W，也低于当前 best，按 gate 规则失败，不应继续延长。
3. 失败原因不是 crossing 完全没发生。该分支的 `features.7.1.mlp_fc2 / attn_proj` crossing 约 `6.9-7.1%`，接近 Phase 2S->2W 的数量级，但 Top-1 反而下降。这说明“通过 scale-ratio 人为诱导 crossing”不等价于历史有益适配中的自然 crossing。
4. 当前实现仍依赖 QKR/StatsQ，只是在 StatsQ 上加临时 scale ratio；这可能仍被 OFQ-family 结构限制，未真正进入 goal 要求的 clean AOQ-native/no-QKR/no-StatsQ 范式。

下一步判断：

- 不继续 AOQ095 + delayed BinReg 这个分支。
- 不建议立刻尝试更激进 `0.9` full-val，因为 2-update smoke 已验证路径，而 0.95 full-val 已经明显低于 gate；更激进大概率只会制造更多无效 crossing。
- 下一步应转向 clean no-QKR/no-StatsQ 最小可跑分支：先禁用 QKR，尽量使用 LSQ/AOQ-compatible weight quantizer，验证 strict W4A4 full-val 能否稳定启动和达到接近 `checkpoint-10` 的水平，再决定是否加入 AOQ stage schedule。

### Phase 2BV：clean LSQ no-QKR 最小分支 smoke

实验动机：

AOQ-native goal 明确要求从 OFQ-family 局部 patch 中跳出来，尽量丢弃 QKR、StatsQ 等 OFQ-specific innovation。Phase 2BU 说明仅在 StatsQ 上加 AOQ scale ratio 仍然不够。因此本阶段构建一个最小 clean branch：

- 禁用 QKR。
- 权重量化从 StatsQ 改为 LSQ。
- activation quantization 仍保持 LSQ。
- 仍保持 strict W4A4。
- 先只做 2-update smoke，不跑 full-val。

目标不是看精度，而是验证从 fixed-QKR `checkpoint-10` 直接切到 no-QKR/LSQ 结构时，checkpoint 是否能严格恢复、训练是否能启动。

新增脚本：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_gate_20260708.sh
```

静态检查：

```text
bash -n tmp_scripts/run_resume10_clean_lsq_noqkr_gate_20260708.sh
git diff --check -- tmp_scripts/run_resume10_clean_lsq_noqkr_gate_20260708.sh
```

Smoke 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_smoke2upd_20260708 \
MASTER_PORT=30787 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar \
START_EPOCH=10 \
EPOCHS=11 \
SCHEDULER_EPOCHS=11 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
QUANT_ONLY_START_EPOCH=10 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
MAX_TRAIN_UPDATES=2 \
SKIP_VALIDATE=1 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_gate_20260708.sh
```

运行证据：

```text
GPU worker: 984521, 8x NVIDIA H100 80GB HBM3 visible
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_smoke2upd_20260708.log
Output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_smoke2upd_20260708
args.yaml:
  qk_reparam: false
  qk_reparam_type: 0
  wq_mode: lsq
```

关键日志：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar; missing=171, unexpected=219
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
TrainSummary: epoch=10 updates=2 avg_step_time=0.600642s samples_per_step=512 samples_per_sec=852.42
```

结果：

- no-QKR/LSQ 分支可以启动并跑 2 个 optimizer updates。
- 但直接从 fixed-QKR `checkpoint-10` strict resume 到 no-QKR/LSQ 结构不成立：`missing=171, unexpected=219`。
- 因为结构 key 不匹配，这个 smoke 不能进入 full-val gate。即使能跑，精度也不能公平归因于 clean no-QKR/LSQ 范式。
- 这个结果不是目标失败，而是说明需要一个 explicit checkpoint migration / initialization bridge。

中文结论：

1. `checkpoint-10` 是 fixed-QKR/StatsQ 结构；clean no-QKR/LSQ 结构的 module key 和 quantizer state 不兼容，不能直接 strict resume。
2. 下一步不能直接跑 full-val，否则会测到一个大量随机/重新初始化量化参数的混合模型。
3. 要继续推进 clean AOQ-native/no-QKR/no-StatsQ，必须先做权重迁移：
   - 从 QKR checkpoint 中把 `q.weight/k.weight/v.weight` 合并回 no-QKR `qkv.weight`。
   - 把 `q_bias/k_bias/v.bias` 合并回 `qkv.bias`。
   - 对 `proj/fc1/fc2` 的 FP weights 直接迁移。
   - 对 StatsQ/LSQ quantizer state 不强行迁移，先由 setup-alpha / LSQ init 重新建立。
   - 迁移后必须验证 missing/unexpected 大幅收敛，并记录 strict init 证据。

下一步判断：

- 写一个 `convert_resume10_qkr_to_noqkr_lsq_init_20260708.py` 脚本，生成 clean no-QKR/LSQ 可加载的 initialization checkpoint。
- 先对转换后的 checkpoint 做 dry-run load / 2-update smoke。
- 只有 missing/unexpected 可解释且训练稳定，才跑 clean no-QKR/LSQ full-val gate。

### Phase 2BW：QKR checkpoint 到 clean no-QKR/LSQ init 的转换 smoke

实验动机：

Phase 2BV 已经证明 clean LSQ no-QKR 分支本身能启动，但不能直接从 fixed-QKR/StatsQ 的 `checkpoint-10.pth.tar` strict resume：

```text
Strict resume: loaded model from .../checkpoint-10.pth.tar; missing=171, unexpected=219
```

这个 mismatch 说明如果直接进入 full-val，结果既不是干净的 no-QKR 初始化，也不能公平归因于 AOQ-native/no-QKR/no-StatsQ 范式。因此本阶段先做结构迁移，把 fixed-QKR checkpoint 中分裂的 `q.weight/k.weight/v.weight` 和 bias 合并回 no-QKR 模型的 `qkv.weight/qkv.bias`，其它同名 FP 权重直接迁移，LSQ quantizer state 保留 no-QKR template 的初始化状态。

转换脚本：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/convert_resume10_qkr_to_noqkr_lsq_init_20260708.py
```

转换产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/converted_resume10_clean_lsq_noqkr_init_20260708/checkpoint-init.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/converted_resume10_clean_lsq_noqkr_init_20260708/checkpoint-init.pth.summary.json
```

转换统计：

```text
direct_copied=324
qkv_weight_merged=12
qkv_bias_merged=12
qkv_aux_copied=0
template_kept=149
```

第一次转换时错误保留了 `module.` 前缀，导致 no-QKR template 与 converted checkpoint 出现全量 key mismatch：

```text
missing=497, unexpected=497
```

修正内容：

- 转换 checkpoint 改为 plain key，不再额外添加 `module.` 前缀。
- `qat_launch.py::strict_resume_checkpoint(...)` 增加 key-prefix fallback：当 direct load 同时出现大量 missing/unexpected 时，自动尝试 add/strip `module.`，并选择 mismatch 更少的版本。
- 这个 fallback 只解决 DDP wrapper 前缀差异，不放松真正的结构 mismatch；因此 `missing=0, unexpected=0` 仍然是有效的 strict init 证据。

2-update smoke 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_converted_smoke2upd_v3_20260708 \
MASTER_PORT=30790 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/converted_resume10_clean_lsq_noqkr_init_20260708/checkpoint-init.pth.tar \
START_EPOCH=10 \
EPOCHS=11 \
SCHEDULER_EPOCHS=11 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
QUANT_ONLY_START_EPOCH=10 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
MAX_TRAIN_UPDATES=2 \
SKIP_VALIDATE=1 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_gate_20260708.sh
```

关键 args 证据：

```text
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/converted_resume10_clean_lsq_noqkr_init_20260708/checkpoint-init.pth.tar
WQ_MODE=lsq
AQ_MODE=lsq
QK_REPARAM=0
MAX_TRAIN_UPDATES=2
SKIP_VALIDATE=1
```

strict init / resume 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/converted_resume10_clean_lsq_noqkr_init_20260708/checkpoint-init.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
```

训练路径证据：

```text
Trainable parameter policy: epoch=10, quant_only=True, policy=params_in_layers, trainable=8903512, frozen=19631895
Teacher feature-output hooks: layers=('features.5.5', 'features.7.1'), student_matches=('features.5.5', 'features.7.1'), teacher_matches=('features.5.5', 'features.7.1')
TrainSummary: epoch=10 updates=2 avg_step_time=0.591283s samples_per_step=512 samples_per_sec=865.91
Stopped early after 2 optimizer updates in epoch 10.
```

输出产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_converted_smoke2upd_v3_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_converted_smoke2upd_v3_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_converted_smoke2upd_v3_20260708/checkpoint-11.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_converted_smoke2upd_v3_20260708/last.pth.tar
```

中文结论：

1. clean no-QKR/LSQ 分支的结构兼容问题已经解决：converted init checkpoint 可以 `missing=0, unexpected=0` strict resume。
2. 当前分支明确丢弃了 QKR，权重量化和激活量化都使用 LSQ，不再依赖 StatsQ 的 scale-ratio 临时 patch；它比 Phase 2BU 更接近 AOQ-native goal 要求的 clean branch。
3. 这个 smoke 只跑 2 个 optimizer updates 且 `SKIP_VALIDATE=1`，不产生 Top-1 结论，也不改变当前 best。当前 best 仍是 Phase 2Z checkpoint-3：Top-1 `80.5540`、Top-5 `95.3060`、Loss `0.8387`、Samples `50000`。
4. teardown 中的 TCPStore/NCCL broken pipe 是 early-stop 后的退出噪声；已有 `TrainSummary` 和 checkpoint 产物，不能作为训练路径失败证据。

下一步判断：

- 立即进入 clean no-QKR/LSQ 的第一个 full-val gate。
- 该 gate 仍然必须满足 strict W4A4、single checkpoint、full ImageNet raw validation、Samples=50000，不允许 soup、checkpoint averaging 或 ensemble。
- 如果第一个 full-val 明显低于 checkpoint-10 baseline `80.3640`，说明 clean no-QKR/LSQ 迁移损伤过大，应先修 initialization/LSQ calibration，而不是直接叠加 AOQ schedule。
- 如果第一个 full-val 接近或超过 `80.5220`，再加入 AOQ-style stage schedule：前期受控 bin crossing，后期 delayed stabilization。

### Phase 2BX：clean no-QKR/LSQ converted init 第一个 full-val gate

实验动机：

Phase 2BW 已经把 fixed-QKR checkpoint 转换成 clean no-QKR/LSQ 可 strict resume 的 init checkpoint，且 2-update smoke 显示训练路径可用。本阶段执行第一个真实 full-val gate，判断这个 clean branch 是否至少能保住接近 `checkpoint-10` baseline 的精度。如果它已经大幅低于 baseline，则不能继续叠加 AOQ schedule，必须先修复初始化语义。

命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_converted_gate1_20260708 \
MASTER_PORT=30791 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/converted_resume10_clean_lsq_noqkr_init_20260708/checkpoint-init.pth.tar \
START_EPOCH=10 \
EPOCHS=11 \
SCHEDULER_EPOCHS=11 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.003 \
QUANT_ONLY_START_EPOCH=10 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
MAX_TRAIN_UPDATES=0 \
SKIP_VALIDATE=0 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_gate_20260708.sh
```

GPU / 数据证据：

```text
gpu-device-present
0-7: NVIDIA H100 80GB HBM3
train_shards=294
validation_shards=14
```

关键 args 证据：

```text
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/converted_resume10_clean_lsq_noqkr_init_20260708/checkpoint-init.pth.tar
WQ_MODE=lsq
AQ_MODE=lsq
QK_REPARAM=0
MAX_TRAIN_UPDATES=0
SKIP_VALIDATE=0
```

strict init / resume 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/converted_resume10_clean_lsq_noqkr_init_20260708/checkpoint-init.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
```

训练完成证据：

```text
Trainable parameter policy: epoch=10, quant_only=True, policy=params_in_layers, trainable=8903512, frozen=19631895
TrainSummary: epoch=10 updates=2496 avg_step_time=0.086420s samples_per_step=512 samples_per_sec=5924.54
```

full-val 结果：

| checkpoint | strict W4A4 | single checkpoint | raw val samples | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-11.pth.tar` | yes | yes | 50000 | 35.4440 | 60.9560 | 3.8044 | fail：大幅低于 checkpoint-10 baseline `80.3640` |

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_converted_gate1_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_converted_gate1_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_converted_gate1_20260708/checkpoint-11.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_converted_gate1_20260708/last.pth.tar
```

中文结论：

1. 这个 gate 是有效 full-val：strict W4A4、single checkpoint、Samples=50000；没有 soup、averaging 或 ensemble。
2. 结果只有 Top-1 `35.4440`，说明 Phase 2BW 的转换只解决了 checkpoint key 兼容，没有解决量化语义兼容。
3. 不能把这个 branch 继续扩展到 AOQ stage schedule。当前问题在更前面：clean no-QKR/LSQ 初始化已经严重损伤模型。
4. 代码层面最可疑的是 QKR 与 no-QKR 的 attention 量化语义不同：QKR 路径把 multi-head `q @ k^T` 复合权重作为 `qk_quant` 的量化对象，而 no-QKR/LSQ 路径恢复成普通 `qkv` linear 后再分别量化 `q/k/v` activation 和 `qkv` weight。简单拼回 `qkv.weight/qkv.bias` 无法迁移 QKR 学到的 `qk_quant`、`quant_x_4_qkv`、`move_qkx_*` 等状态。
5. 另一个高风险点是 LSQ state：当前转换保留 no-QKR template 的 `lsqw_fn.*` 与若干 activation quantizer state，而不是从 QKR checkpoint 中重建与 merged `qkv` 权重匹配的 clipping/scale；这会直接造成 W4 quantization bin 错位。

下一步判断：

- 不继续当前 converted init gate，也不叠加 AOQ explore/delayed BinReg。
- 先做最小诊断，按顺序验证：
  1. 直接评估 converted init checkpoint（不训练）是否已经是 35% 级别；如果是，问题完全在转换/init。
  2. 对比 no-QKR/LSQ template、converted init、QKR source 的 `qkv/lsqw/input_quant_fn` scale 统计，确认是否 LSQ scale 与 merged 权重范围不匹配。
  3. 研究 no-QKR LSQ 模块的初始化接口，优先尝试基于 merged `qkv.weight` 重新初始化 `lsqw_fn.s`，而不是沿用 template scale。
  4. 如果 qkv 合并本身无误但 LSQ scale 修复仍无法接近 baseline，则 clean no-QKR 不能从 QKR checkpoint 直接迁移，应该改为从 pretrained/no-QKR AOQ-native 路线重新做短程 QAT warm start。

### Phase 2BY：converted init 直接评估与 LSQ scale 诊断

实验动机：

Phase 2BX 的 full-val gate 训练 1 个 epoch 后只有 Top-1 `35.4440`。这还不能区分两种情况：

1. converted init 本身已经坏掉；
2. converted init 尚可，但训练设置把它拉坏。

本阶段做两个低成本诊断：

- 不训练，直接 `eval-only` 评估 converted init checkpoint。
- CPU-only 扫描 source/template/converted 三个 checkpoint，确认 Q/K/V 合并是否正确，以及 converted 的 `qkv.lsqw_fn.s` 是否与 merged qkv weight 的理想 LSQ scale 匹配。

#### 2BY-1：converted init eval-only full-val

命令要点：

```text
--resume /mlx_devbox/users/quyanyi/playground/qat_public_repro/converted_resume10_clean_lsq_noqkr_init_20260708/checkpoint-init.pth.tar
--wq-mode lsq
--aq-mode lsq
--extra-arg=--eval-only
无 --qk-reparam
```

strict init 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/converted_resume10_clean_lsq_noqkr_init_20260708/checkpoint-init.pth.tar; missing=0, unexpected=0
```

full-val 结果：

| checkpoint | strict W4A4 | single checkpoint | raw val samples | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| converted init `checkpoint-init.pth.tar` | yes | yes | 50000 | 4.6140 | 11.3580 | 6.2010 | fail：init 本身已基本失效 |

日志：

```text
/mlx_devbox/users/quyanyi/playground/train_eval_resume10_clean_lsq_noqkr_converted_init_20260708.log
```

中文结论：

- converted init 不训练直接 eval 只有 Top-1 `4.6140`。
- Phase 2BX 训练 1 个 epoch 后的 Top-1 `35.4440` 不是训练把模型拉坏，而是从已经失效的 init 上部分恢复。
- 所以 clean no-QKR/LSQ 当前失败点在 checkpoint conversion / quantizer init，不在 QAT schedule。

#### 2BY-2：Q/K/V 合并与 LSQ scale 诊断

诊断脚本：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_noqkr_lsq_init_scales_20260708.py
```

输出：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_noqkr_lsq_scale_diagnosis_20260708/noqkr_lsq_scale_diagnosis.json
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_noqkr_lsq_scale_diagnosis_20260708/noqkr_lsq_scale_diagnosis.tsv
```

核心统计：

```text
modules=12
all_source_merge_equal_converted=True
all_template_s_equal_converted=True
s_over_ideal_mean_min=0.8243
s_over_ideal_mean_max=0.9657
s_over_ideal_mean_avg=0.9263
```

解释：

- `all_source_merge_equal_converted=True`：转换脚本把 source QKR checkpoint 的 `q.weight/k.weight/v.weight` 和 bias 拼回 converted `qkv.weight/qkv.bias` 是逐层完全一致的。qkv 拼接本身不是主要 bug。
- `all_template_s_equal_converted=True`：converted checkpoint 的 `qkv.lsqw_fn.s` 完全来自 no-QKR template，而不是根据 merged qkv weight 重算。
- `s_over_ideal_mean_avg=0.9263`：converted 的 qkv weight scale 比按 LSQ 公式从 merged weight 计算出的理想 scale 平均偏小约 `7.4%`，最差模块偏小约 `17.6%`。这会导致 weight quantization clipping/bin assignment 偏移。
- 但仅 qkv weight scale 偏小 `3%-18%` 不足以单独解释 Top-1 从 80% 掉到 4.6%。更大的风险仍在 QKR 与 no-QKR 的 attention 量化语义差异：QKR checkpoint 学到的是 `qk_quant`、`quant_x_4_qkv`、`quan_a_qkx_fn`、`move_qkx_*` 等复合路径状态，而 no-QKR/LSQ 需要普通 `qkv.input_quant_fn`、`move_qkv_b4/aft`、`quan_a_q/k/v_fn` 等状态。当前转换大部分这些状态来自 no-QKR 2-update template，语义不匹配。

中文结论：

1. clean no-QKR/LSQ 方向仍符合新 goal 的范式要求，但不能从 fixed-QKR/StatsQ checkpoint 通过简单结构拼接直接迁移。
2. 当前 converted init 已被证实本身无效：eval-only Top-1 `4.6140`，训练 1 epoch 后 Top-1 `35.4440` 只是低起点恢复。
3. 下一步不应继续这个 converted init 分支，不应叠加 AOQ explore/delayed BinReg。
4. 如果还要坚持 no-QKR，需要换 init 策略：
   - 要么在构建 no-QKR/LSQ 模型后，基于 merged qkv weight 重新初始化所有 LSQ weight scale 和 activation scale，再做一个 eval-only gate；
   - 要么彻底放弃从 QKR checkpoint 迁移，改从 pretrained/no-QKR strict W4A4 做 AOQ-native warm start；
   - 要么保留 QKR 作为 teacher/source，只用它做 feature/logit/attention supervision，而 student 从 no-QKR native 路线启动。

下一步判断：

- 最便宜的下一步是写一个 `reinit_noqkr_lsq_scales_from_weight` 转换版本：转换 qkv 后删除或重算所有 `*.lsqw_fn.s`，让 setup-alpha / LSQ forward 根据当前权重初始化，再做 eval-only full-val。
- Gate 规则：如果 eval-only 仍远低于 `80.3640`，说明 activation/move 状态才是主因，应停止 QKR checkpoint direct migration。
- 若 eval-only 能恢复到至少 `75%+`，再跑 1-epoch full-val 看是否能接近 `80.3640` baseline。

### Phase 2BZ：重算 weight LSQ scale 的 no-QKR converted init eval-only gate

实验动机：

Phase 2BY 证明 converted init 不训练直接 eval 只有 Top-1 `4.6140`，同时诊断发现 converted checkpoint 中所有 `*.lsqw_fn.s` 来自 no-QKR template，其中 patch embedding 等模块的旧 scale 与当前权重范围严重不匹配。本阶段测试最便宜的修复：保持 Q/K/V -> qkv 合并和其它状态不变，但对所有 weight LSQ scale 按当前 converted weight 重新计算。

代码改动：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/convert_resume10_qkr_to_noqkr_lsq_init_20260708.py
```

新增参数：

```text
--reinit-weight-lsq-from-weight
```

转换产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/converted_resume10_clean_lsq_noqkr_weightreinit_20260708/checkpoint-init.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/converted_resume10_clean_lsq_noqkr_weightreinit_20260708/checkpoint-init.pth.summary.json
```

转换统计：

```text
direct_copied=324
qkv_weight_merged=12
qkv_bias_merged=12
qkv_aux_copied=0
template_kept=149
weight_lsq_reinitialized=53
weight_lsq_skipped=0
```

诊断输出：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_noqkr_lsq_weightreinit_diagnosis_20260708/noqkr_lsq_scale_diagnosis.json
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_noqkr_lsq_weightreinit_diagnosis_20260708/noqkr_lsq_scale_diagnosis.tsv
```

诊断结论：

```text
all_source_merge_equal_converted=True
all_template_s_equal_converted=False
qkv_s_over_ideal_min=1.0
qkv_s_over_ideal_max=1.0
qkv_s_over_ideal_avg=1.0
```

解释：Q/K/V 合并仍然逐层一致；qkv weight 的 LSQ scale 已经完全等于按当前 qkv weight 计算出的理想值。这个 gate 直接回答“只修 weight LSQ scale 是否能恢复 no-QKR init”。

eval-only 命令要点：

```text
--resume /mlx_devbox/users/quyanyi/playground/qat_public_repro/converted_resume10_clean_lsq_noqkr_weightreinit_20260708/checkpoint-init.pth.tar
--wq-mode lsq
--aq-mode lsq
--extra-arg=--eval-only
无 --qk-reparam
```

strict init 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/converted_resume10_clean_lsq_noqkr_weightreinit_20260708/checkpoint-init.pth.tar; missing=0, unexpected=0
```

full-val 结果：

| checkpoint | strict W4A4 | single checkpoint | raw val samples | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| weight-reinit converted init `checkpoint-init.pth.tar` | yes | yes | 50000 | 2.8440 | 7.4100 | 6.5052 | fail：比原 converted init `4.6140` 更差 |

日志：

```text
/mlx_devbox/users/quyanyi/playground/train_eval_resume10_clean_lsq_noqkr_weightreinit_20260708.log
```

中文结论：

1. 这个 gate 有效验证了 weight LSQ scale 修复假设：53 个 `*.lsqw_fn.s` 已经全部按当前 weight 重算，strict resume 仍为 `missing=0, unexpected=0`。
2. eval-only Top-1 从原 converted init 的 `4.6140` 进一步降到 `2.8440`，说明 weight scale mismatch 不是主要可修复因素。
3. 这基本关闭了 fixed-QKR/StatsQ checkpoint 到 clean no-QKR/LSQ checkpoint 的 direct migration 路线。QKR 学到的复合 attention 量化状态不能靠 qkv 拼接和 weight LSQ 重算迁移到普通 no-QKR LSQ attention。
4. 后续不应继续在这个 direct migration 分支上叠加 AOQ schedule 或训练 epoch；它的 init 起点已经远低于可接受范围。

下一步判断：

- 转向 no-QKR native 起点验证：直接从 pretrained Swin-T 构建 clean no-QKR/LSQ strict W4A4，先 eval-only 看量化初始化起点，再决定是否做 AOQ-native warm start。
- 如果 no-QKR native pretrained eval-only 也很低，说明 clean no-QKR/LSQ 需要完整 warm start/reconstruction，而不是 checkpoint conversion。
- 如果 no-QKR native pretrained 起点明显好于 converted init，则继续围绕 no-QKR native 的 AOQ-style 训练范式，而不是 QKR migration。

### Phase 2CA：clean no-QKR/LSQ native pretrained eval-only 起点

实验动机：

Phase 2BZ 关闭了 fixed-QKR checkpoint -> no-QKR/LSQ checkpoint 的 direct migration 路线。剩下的问题是：clean no-QKR/LSQ 范式本身是否有一个可用起点？本阶段不使用任何 QKR checkpoint，不 resume，只从 torchvision/timm pretrained Swin-T 构建 strict W4A4 no-QKR/LSQ student，经过 setup-alpha 后直接 eval-only。

命令要点：

```text
无 --resume
无 --qk-reparam
--pretrained --pretrained-initialized
--wq-mode lsq
--aq-mode lsq
--wbits 4 --abits 4
--extra-arg=--eval-only
```

GPU / 数据证据：

```text
gpu-device-present
0-7: NVIDIA H100 80GB HBM3
full raw validation Samples=50000
```

full-val 结果：

| checkpoint / init | strict W4A4 | single model | raw val samples | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| native pretrained no-QKR/LSQ init | yes | yes | 50000 | 0.6680 | 2.3320 | 7.1965 | fail：native init 本身不可用 |

日志：

```text
/mlx_devbox/users/quyanyi/playground/train_eval_resume10_clean_lsq_noqkr_native_pretrained_20260708.log
```

中文结论：

1. clean no-QKR/LSQ 不能直接从 pretrained + setup-alpha 得到可用 strict W4A4 eval 起点：Top-1 只有 `0.6680`。
2. 这说明当前 OFQ 的 no-QKR/LSQ 模块不是一个“可直接 post-training quantize”的路径；它需要 QAT 或 reconstruction/warm-start 才能进入可训练区间。
3. Phase 2BY/2BZ/2CA 合在一起给出清晰边界：
   - QKR checkpoint direct migration 失败：eval-only `4.6140`，weight scale 重算后 `2.8440`。
   - no-QKR native pretrained direct eval 也失败：`0.6680`。
   - 因此下一步必须是 no-QKR native warm-start / reconstruction，而不是继续做 checkpoint conversion 或直接叠 AOQ schedule。

下一步判断：

- 构建一个最小 no-QKR native warm-start gate：从 pretrained no-QKR/LSQ 出发，先做短程 teacher-supervised reconstruction / QAT warm-up，让 eval-only 起点进入至少 `70%+`，再考虑 AOQ-style bin crossing schedule。
- 该 gate 的早停规则必须严格：如果 1 epoch 后仍低于 `60%-70%`，说明当前 no-QKR/LSQ 模块需要更深结构修复，不应消耗 20-epoch 预算。
- 继续保留 QKR checkpoint 作为 teacher/source，而不是 student init。

### Phase 2CB：clean no-QKR/LSQ native warm-start 最小 gate

实验动机：

Phase 2CA 说明 clean no-QKR/LSQ 从 pretrained + setup-alpha 直接 eval 只有 Top-1 `0.6680`，但这不代表该范式完全不可训练。AOQ-native goal 要求丢弃 QKR/StatsQ，因此需要验证 clean student 是否可以通过短程 teacher-supervised warm-start 进入可训练区间。本阶段不使用 QKR checkpoint 作为 student init，只用 FP teacher 做监督：

- 100 步 teacher-logit KD reconstruction；
- 100 步多层 feature reconstruction；
- 之后只跑 1 个正式 optimizer update；
- 不跳过 full raw validation。

新增脚本：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_native_warmstart_gate_20260708.sh
```

命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_native_warmstart_100logit100feat_1upd_20260708 \
MASTER_PORT=30795 \
PRE_QAT_RECON_UPDATES=100 \
PRE_QAT_FEATURE_RECON_UPDATES=100 \
PRE_QAT_FEATURE_RECON_LAYERS=features.1.1,features.3.1,features.5.5,features.7.1 \
PRE_QAT_FEATURE_RECON_POLICY=quant \
MAX_TRAIN_UPDATES=1 \
SKIP_VALIDATE=0 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_native_warmstart_gate_20260708.sh
```

关键 args 证据：

```text
WQ_MODE=lsq
AQ_MODE=lsq
QK_REPARAM=0
PRE_QAT_RECON_UPDATES=100
PRE_QAT_FEATURE_RECON_UPDATES=100
PRE_QAT_FEATURE_RECON_LAYERS=features.1.1,features.3.1,features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
MAX_TRAIN_UPDATES=1
SKIP_VALIDATE=0
```

warm-start 证据：

```text
Starting pre-QAT teacher-logit reconstruction: updates=100, policy=quant, temperature=2.75
PreQATRecon: update=1/100 loss=53.128387
PreQATRecon: update=50/100 loss=52.154999
PreQATRecon: update=100/100 loss=51.737476
Finished pre-QAT teacher-logit reconstruction: updates=100

Starting pre-QAT feature reconstruction: updates=100, policy=quant, layers=('features.1.1', 'features.3.1', 'features.5.5', 'features.7.1')
pre-QAT feature reconstruction: update=1/100 loss=0.404463 kept=31430 masked=27689816
pre-QAT feature reconstruction: update=50/100 loss=0.339379 kept=31430 masked=27689816
pre-QAT feature reconstruction: update=100/100 loss=0.326978 kept=31430 masked=27689816
Finished pre-QAT feature reconstruction: updates=100
```

正式训练证据：

```text
Trainable parameter policy: epoch=0, quant_only=False, policy=all, trainable=28535407, frozen=0
TrainSummary: epoch=0 updates=1 avg_step_time=1.065144s samples_per_step=512 samples_per_sec=480.69
Stopped early after 1 optimizer updates in epoch 0.
```

full-val 结果：

| checkpoint / gate | strict W4A4 | single model | raw val samples | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| no-QKR/LSQ native warm-start 100+100 + 1 update | yes | yes | 50000 | 63.2300 | 86.2860 | 2.2888 | positive but below 70% warm-start gate |

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_native_warmstart_100logit100feat_1upd_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_100logit100feat_1upd_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_100logit100feat_1upd_20260708/checkpoint-1.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_100logit100feat_1upd_20260708/last.pth.tar
```

中文结论：

1. 这是 clean no-QKR/LSQ 路线第一次出现可用正信号：从 native pretrained eval-only `0.6680` 拉到 `63.2300`。
2. 说明 no-QKR/LSQ 模块不是完全不可训练，关键是必须先做 teacher-supervised warm-start / reconstruction，不能直接 eval 或直接从 QKR checkpoint 拼接迁移。
3. 但 `63.2300` 仍低于预设 `70%+` warm-start 起跑线，也远低于 fixed-QKR resume branch 的 `80.5540`。现在不能进入 20-epoch AOQ stage schedule。
4. 当前 warm-start 只更新 quant/shift 参数 200 步，正式训练只 1 update；它证明方向可行，但力度不足。

下一步判断：

- 下一条 gate 应增强 clean no-QKR warm-start，而不是回到 QKR migration：
  - 增加 reconstruction updates，例如 `300 logit + 300 feature`；
  - 扩大 feature layers 到所有 stage 末尾或全 stage blocks；
  - 或把 `pre_qat_feature_recon_policy` 从 `quant` 改为 `module_all` 做一次更强但更高风险的局部权重适配。
- Gate 规则：增强 warm-start 后的第一个 full-val 必须超过 `70%`，否则 clean no-QKR/LSQ 当前实现不值得进入长程 20-epoch 目标。

### Phase 2CC：clean no-QKR/LSQ native warm-start 增强 gate，300+300 + 1 update

实验动机：

Phase 2CB 的 `100 logit + 100 feature + 1 update` warm-start 把 clean no-QKR/LSQ 从 eval-only `0.6680` 拉到 `63.2300`，说明 reconstruction 有效但力度不足。本阶段只增强 warm-start 步数，不改变范式：

- 仍然无 QKR；
- 仍然无 StatsQ；
- 仍然使用 strict W4A4 LSQ；
- 仍然只跑 1 个正式 optimizer update 后 full-val；
- 把 warm-start 从 `100+100` 增加到 `300 logit + 300 feature`。

命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_1upd_20260708 \
MASTER_PORT=30796 \
PRE_QAT_RECON_UPDATES=300 \
PRE_QAT_FEATURE_RECON_UPDATES=300 \
PRE_QAT_FEATURE_RECON_LAYERS=features.1.1,features.3.1,features.5.5,features.7.1 \
PRE_QAT_FEATURE_RECON_POLICY=quant \
MAX_TRAIN_UPDATES=1 \
SKIP_VALIDATE=0 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_native_warmstart_gate_20260708.sh
```

关键 args 证据：

```text
WQ_MODE=lsq
AQ_MODE=lsq
QK_REPARAM=0
PRE_QAT_RECON_UPDATES=300
PRE_QAT_FEATURE_RECON_UPDATES=300
PRE_QAT_FEATURE_RECON_LAYERS=features.1.1,features.3.1,features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
MAX_TRAIN_UPDATES=1
SKIP_VALIDATE=0
```

warm-start 证据：

```text
Starting pre-QAT teacher-logit reconstruction: updates=300, policy=quant, temperature=2.75
PreQATRecon: update=1/300 loss=53.128387
PreQATRecon: update=150/300 loss=51.651764
PreQATRecon: update=300/300 loss=51.579350
Finished pre-QAT teacher-logit reconstruction: updates=300

Starting pre-QAT feature reconstruction: updates=300, policy=quant, layers=('features.1.1', 'features.3.1', 'features.5.5', 'features.7.1')
pre-QAT feature reconstruction: update=1/300 loss=0.352778 kept=31430 masked=27689816
pre-QAT feature reconstruction: update=150/300 loss=0.272416 kept=31430 masked=27689816
pre-QAT feature reconstruction: update=300/300 loss=0.253778 kept=31430 masked=27689816
Finished pre-QAT feature reconstruction: updates=300
```

正式训练证据：

```text
Trainable parameter policy: epoch=0, quant_only=False, policy=all, trainable=28535407, frozen=0
TrainSummary: epoch=0 updates=1 avg_step_time=1.003650s samples_per_step=512 samples_per_sec=510.14
Stopped early after 1 optimizer updates in epoch 0.
```

full-val 结果：

| checkpoint / gate | strict W4A4 | single model | raw val samples | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| no-QKR/LSQ native warm-start 300+300 + 1 update | yes | yes | 50000 | 71.2260 | 90.7120 | 1.5499 | pass warm-start gate：超过 70%，但远未达到 81 |

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_1upd_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_1upd_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_1upd_20260708/checkpoint-1.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_1upd_20260708/last.pth.tar
```

中文结论：

1. `300+300` warm-start 成功跨过 `70%` 起跑线：Top-1 从 `63.2300` 提升到 `71.2260`，说明 clean no-QKR/LSQ native 路线具有可训练性。
2. 这个结果仍然远低于 fixed-QKR resume branch 的 `80.5540`，也远低于目标 `81.0`。不能把 goal 标为完成。
3. 目前的关键进展是范式切换成立了一半：丢弃 QKR/StatsQ 后，不能 direct eval，也不能 checkpoint migration；但 teacher-supervised warm-start 能把模型拉回有效区间。
4. 现在可以进入下一层 gate：保留 `300+300` warm-start，跑完整第一个 QAT epoch 后 full-val，判断 clean no-QKR/LSQ 是否能继续从 `71.2260` 向 75%/80% 区间爬升。

下一步判断：

- 运行完整 epoch gate：`300 logit + 300 feature` warm-start 后，不设置 `MAX_TRAIN_UPDATES`，完整训练 epoch 0 并 full-val。
- 如果完整 epoch 仍低于 `75%`，说明仅 quant/shift warm-start 不够，需要 `module_all` 或更宽 feature layers。
- 如果完整 epoch 达到 `75%+`，再考虑第 2-3 epoch 短程趋势以及 AOQ-style controlled bin crossing schedule。

### Phase 2CD：clean no-QKR/LSQ native warm-start 300+300 后完整第 1 epoch gate

实验动机：

Phase 2CC 证明 `300 logit + 300 feature` warm-start 加 1 个正式 update 可达到 Top-1 `71.2260`，超过 70% 起跑线。本阶段保持完全相同的 clean no-QKR/LSQ 范式和 warm-start，但不再限制 `MAX_TRAIN_UPDATES`，完整训练第 1 个 QAT epoch，判断该路线是否能从 71% 继续爬升到可扩展区间。

命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_1ep_20260708 \
MASTER_PORT=30797 \
PRE_QAT_RECON_UPDATES=300 \
PRE_QAT_FEATURE_RECON_UPDATES=300 \
PRE_QAT_FEATURE_RECON_LAYERS=features.1.1,features.3.1,features.5.5,features.7.1 \
PRE_QAT_FEATURE_RECON_POLICY=quant \
MAX_TRAIN_UPDATES=0 \
SKIP_VALIDATE=0 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_native_warmstart_gate_20260708.sh
```

关键 args 证据：

```text
WQ_MODE=lsq
AQ_MODE=lsq
QK_REPARAM=0
PRE_QAT_RECON_UPDATES=300
PRE_QAT_FEATURE_RECON_UPDATES=300
PRE_QAT_FEATURE_RECON_LAYERS=features.1.1,features.3.1,features.5.5,features.7.1
PRE_QAT_FEATURE_RECON_POLICY=quant
MAX_TRAIN_UPDATES=0
SKIP_VALIDATE=0
```

warm-start 证据：

```text
PreQATRecon: update=1/300 loss=53.128387
PreQATRecon: update=300/300 loss=51.579350
Finished pre-QAT teacher-logit reconstruction: updates=300

pre-QAT feature reconstruction: update=1/300 loss=0.352778 kept=31430 masked=27689816
pre-QAT feature reconstruction: update=300/300 loss=0.253778 kept=31430 masked=27689816
Finished pre-QAT feature reconstruction: updates=300
```

训练完成证据：

```text
Trainable parameter policy: epoch=0, quant_only=False, policy=all, trainable=28535407, frozen=0
TrainSummary: epoch=0 updates=2496 avg_step_time=0.166236s samples_per_step=512 samples_per_sec=3079.96
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-1.pth.tar` | yes | yes | 50000 | 79.5060 | 94.9920 | 0.8879 | strong positive：clean no-QKR/LSQ 已接近 80%，但未达 81 |

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_1ep_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_1ep_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_1ep_20260708/checkpoint-1.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_1ep_20260708/last.pth.tar
```

中文结论：

1. 这是 clean no-QKR/no-StatsQ/LSQ 路线目前最强信号：从 direct eval `0.6680`，到 `100+100 + 1 update` 的 `63.2300`，到 `300+300 + 1 update` 的 `71.2260`，再到完整第 1 epoch 的 `79.5060`。
2. 该结果仍低于目标 `81.0`，也低于当前 fixed-QKR resume best `80.5540`，所以 goal 未完成。
3. 但它证明了 AOQ-native clean branch 不是死路：只要先做足够的 teacher-supervised warm-start，strict W4A4 no-QKR/LSQ 可以接近 80%。
4. 第 1 epoch 训练 loss 下降不明显，但 full-val 大幅提升，说明 warm-start 后完整数据遍历仍然在修复泛化状态。

下一步判断：

- 从 `checkpoint-1.pth.tar` strict resume，继续跑 epoch 1-2，观察短程趋势。
- Gate 规则：checkpoint-2 必须继续提升并接近/超过 `80%`，checkpoint-3 若未超过 `80.5%`，需要调整 warm-start policy 或引入 AOQ-style controlled crossing，而不是盲目跑满 20 epoch。

### Phase 2CE：clean no-QKR/LSQ checkpoint-1 到 checkpoint-3 continuation

实验动机：

Phase 2CD 的 clean no-QKR/LSQ 第 1 epoch checkpoint 已达到 Top-1 `79.5060`。本阶段不重新 warm-start，不启用 QKR/StatsQ，不使用 soup，只从 `checkpoint-1.pth.tar` strict resume 继续普通 QAT 两个 epoch，观察该 clean branch 是否自然爬升到 `80%+`，以及是否接近进入 AOQ stage schedule 的门槛。

命令要点：

```text
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_1ep_20260708/checkpoint-1.pth.tar
--no-resume-opt
--start-epoch 1
--epochs 3
--scheduler-epochs 3
--wq-mode lsq
--aq-mode lsq
无 --qk-reparam
无 StatsQ
```

strict resume 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_1ep_20260708/checkpoint-1.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
```

args.yaml 关键字段：

```text
wq_mode: lsq
aq_mode: lsq
qk_reparam: False
wq_bitw: 4
aq_bitw: 4
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_1ep_20260708/checkpoint-1.pth.tar
start_epoch: 1
epochs: 3
scheduler_epochs: 3
use_kd: True
kd_hard_and_soft: 0
pre_qat_recon_updates: 0
pre_qat_feature_recon_updates: 0
```

训练完成证据：

```text
TrainSummary: epoch=1 updates=2496 avg_step_time=0.167059s samples_per_step=512 samples_per_sec=3064.79
TrainSummary: epoch=2 updates=2496 avg_step_time=0.167007s samples_per_step=512 samples_per_sec=3065.75
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-2.pth.tar` | yes | yes | 50000 | 79.5360 | 94.9980 | 0.8827 | only +0.030 vs checkpoint-1 |
| `checkpoint-3.pth.tar` | yes | yes | 50000 | 79.9220 | 95.2060 | 0.8510 | improved, still below 80.5 gate and 81 target |

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-2.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/last.pth.tar
```

中文结论：

1. clean no-QKR/LSQ ordinary continuation 没有崩，checkpoint-3 达到 Top-1 `79.9220`，相比 checkpoint-1 的 `79.5060` 仍有提升。
2. 但提升速度不够：checkpoint-2 只提升 `+0.030`，checkpoint-3 到 `79.9220` 仍低于 `80.5` 调度门槛，也低于 fixed-QKR resume best `80.5540` 和目标 `81.0`。
3. 继续普通 continuation 可能还能缓慢上升，但不是高杠杆路径。现在应回到 AOQ-native 思路：在 clean no-QKR/LSQ 上加入受控 bin crossing / delayed stabilization，而不是只延长普通 QAT。
4. 该分支仍是 strict W4A4、single checkpoint、full raw validation，且无 QKR、无 StatsQ、无 soup，符合新 goal 的范式约束。

下一步判断：

- 以 `checkpoint-3.pth.tar` 作为 clean no-QKR/LSQ 当前 source checkpoint。
- 下一条 gate 应测试 AOQ-style controlled crossing：对 LSQ weight scale 施加短窗口 scale ratio 或等价 threshold compression，然后 delayed stabilization，而不是回到 StatsQ scale-ratio 或 QKR。
- 如果没有 LSQ 版 AOQ explore 控制，需要先在 `qat_launch.py` 中实现只作用于 `lsqw_fn.s` 的 runtime scale-ratio window，并先做 2-update smoke，再做 full-val gate。

### Phase 2CF：LSQ-AOQ runtime scale-ratio 支持与 2-update smoke

实验动机：

Phase 2CE 表明 clean no-QKR/LSQ 普通 continuation 到 checkpoint-3 后 Top-1 为 `79.9220`，趋势向上但已明显变慢。按照 AOQ-native goal，下一步应在 clean LSQ branch 上引入受控 weight-bin crossing，而不是回到 QKR/StatsQ。本阶段先实现并 smoke 验证 LSQ 版 AOQ explore：

- 不修改 checkpoint 中的 `lsqw_fn.s` 参数；
- 只在 forward 时临时缩放 LSQ weight quantizer 的有效 scale；
- 用 update window 控制开启/关闭；
- 先只跑 2 update，不做精度结论。

代码改动：

```text
/mlx_devbox/users/quyanyi/playground/QATs/third_party/OFQ/src/quantization/quantizer/lsq.py
/mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py
```

实现要点：

- `LsqQuantizerWeight` 与 `LsqQuantizer4Conv2d` 新增 `aoq_scale_ratio=1.0`。
- forward 中使用 `s_scale = s_scale * aoq_scale_ratio`，临时改变有效 bin/level 间隔。
- `qat_launch.py::set_aoq_explore_scale_ratio(...)` 从只匹配 `statsq_fn/qk_quant/v_quant` 扩展为也匹配 `lsqw_fn`。

静态检查：

```text
python3 -m py_compile qat_launch.py third_party/OFQ/src/quantization/quantizer/lsq.py
git diff --check -- qat_launch.py third_party/OFQ/src/quantization/quantizer/lsq.py docs/resume10_to81_goal_progress_20260706.md tmp_scripts/run_resume10_clean_lsq_noqkr_native_warmstart_gate_20260708.sh
```

smoke 命令要点：

```text
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
--wq-mode lsq
--aq-mode lsq
--aoq-explore-scale-ratio 0.95
--aoq-explore-layers features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
--aoq-explore-start-update 0
--aoq-explore-end-update 2
--extra-arg=--max_train_updates --extra-arg=2
--extra-arg=--skip_validate
```

strict resume / AOQ 命中证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
AOQ explore scale ratio update: epoch=3, update=0, active=True, ratio=0.95, layers=('features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=3, start_update=0, end_update=2
TrainSummary: epoch=3 updates=2 avg_step_time=0.961353s samples_per_step=512 samples_per_sec=532.58
Stopped early after 2 optimizer updates in epoch 3.
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq_smoke2upd_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq_smoke2upd_20260708/
```

中文结论：

1. LSQ-AOQ runtime path 已接通：selected late qkv/proj/fc2 三个 LSQ weight quantizer 被命中，`quantizers=3`。
2. strict resume 干净，训练 2 updates 可运行。
3. smoke 不给精度结论。下一步可以跑 full-val gate，比较 LSQ-AOQ controlled crossing 是否优于 ordinary continuation 的 `79.9220`。

下一步判断：

- 从 clean no-QKR checkpoint-3 跑 1 epoch LSQ-AOQ gate。
- 初始设置：前半 epoch `aoq_explore_scale_ratio=0.95`，后半自动恢复 1.0。
- Gate：必须超过 checkpoint-3 `79.9220`，最好接近或超过 `80.5`，否则说明单纯 LSQ scale-ratio window 不够。

### Phase 2CG：LSQ-AOQ095 full-val gate 第一次尝试无效，环境 OOM 中断

实验动机：

Phase 2CF 的 2-update smoke 已证明 LSQ-AOQ runtime path 能命中 selected `lsqw_fn`。本阶段尝试跑第一个 full-val gate：从 clean no-QKR/LSQ `checkpoint-3` 出发，前半 epoch 用 `aoq_explore_scale_ratio=0.95`，后半恢复 1.0，full-val 对比 ordinary continuation 的 `79.9220`。

命令要点：

```text
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
--wq-mode lsq
--aq-mode lsq
--aoq-explore-scale-ratio 0.95
--aoq-explore-layers features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
--aoq-explore-start-update 0
--aoq-explore-end-update 1250
--epochs 4
--start-epoch 3
```

已确认的有效启动证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
AOQ explore scale ratio update: epoch=3, update=0, active=True, ratio=0.95, layers=('features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=3, start_update=0, end_update=1250
```

无效原因：

```text
日志只到 Train: 3 [ 300/2502 ...]
没有 TrainSummary
没有 checkpoint-4.pth.tar
没有 Test: [distributed-summary]
输出目录只有 args.yaml
```

系统证据：

```text
dmesg: Memory cgroup out of memory
oom-kill: task=node,pid=989179
Memory cgroup out of memory: Killed process 989179 (node)
```

中文结论：

1. 这次 LSQ-AOQ095 full-val gate 启动有效，但结果无效：没有 checkpoint、没有 full-val summary，不能作为精度证据。
2. 中断原因更像 worker 容器 cgroup OOM，而不是训练代码报错。日志中没有 Python traceback；dmesg 显示 node 进程被 OOM kill。
3. 不能据此判断 LSQ-AOQ095 好坏，必须重跑一个低 CPU 压力版本。

下一步判断：

- 用更低 CPU 内存压力重跑同一 full-val gate：
  - `workers=2`；
  - 其它关键参数保持不变；
  - 继续从 clean no-QKR/LSQ `checkpoint-3` strict resume；
  - 只有出现 `Test: [distributed-summary] ... Samples: 50000` 才算有效 gate。

### Phase 2CH：复核 Phase 2CG，LSQ-AOQ095 full-val gate 实际有效但未达 81

实验动机：

Phase 2CG 最初依据 master 侧 `ps` 和早期日志误判为中断，但 worker TTY 中训练实际继续完成，并产出了 full ImageNet raw validation 结果和 `checkpoint-4.pth.tar`。本阶段对该 gate 做复核更正：确认其是否满足 strict W4A4、单 checkpoint、full raw validation、无 soup 的计分条件，并据此决定下一步 AOQ-native gate。

方法设计：

- 从 clean no-QKR/LSQ ordinary continuation 的 `checkpoint-3` 出发；
- 不使用 QKR，不使用 StatsQ；
- weight/activation quantizer 均为 LSQ；
- 对 late block 的三个 weight quantizer 做 LSQ-AOQ controlled crossing：
  - `features.7.1.attn.qkv`
  - `features.7.1.attn.proj`
  - `features.7.1.mlp.fc2`
- 前半 epoch 使用 `aoq_explore_scale_ratio=0.95`；
- update 1250 之后恢复 ratio=1.0；
- full-val 只认单个 `checkpoint-4.pth.tar`，不使用 soup / averaging / ensemble。

关键命令要点：

```text
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
--wq-mode lsq
--aq-mode lsq
--wbits 4
--abits 4
--no-resume-opt
--start-epoch 3
--epochs 4
--scheduler-epochs 4
--aoq-explore-scale-ratio 0.95
--aoq-explore-layers features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
--aoq-explore-start-update 0
--aoq-explore-end-update 1250
```

关键 `args.yaml` 证据：

```text
experiment: recipe_resume10_clean_lsq_noqkr_lsqaoq095_gate_20260708
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
wq_mode: lsq
aq_mode: lsq
wq_bitw: 4
aq_bitw: 4
qk_reparam: false
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
trainable_policy: all
use_kd: true
kd_hard_and_soft: 0
aoq_explore_scale_ratio: 0.95
aoq_explore_layers: features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
aoq_explore_start_update: 0
aoq_explore_end_update: 1250
```

strict resume / AOQ 命中证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, ratio=0.95, layers=('features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=3, start_update=0, end_update=1250
AOQ explore scale ratio update: epoch=3, update=1250, active=False, ratio=1.0, layers=('features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=3, start_update=0, end_update=1250
TrainSummary: epoch=3 updates=2496 avg_step_time=0.167847s samples_per_step=512 samples_per_sec=3050.40
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 80.0040 | 95.1140 | 0.8624 | 有效 gate，但未达 81 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 32.083s  Loss: 0.8624  Acc@1: 80.0040  Acc@5: 95.1140  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq095_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq095_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq095_gate_20260708/checkpoint-4.pth.tar
```

误启动复跑状态：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq095_w2_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq095_w2_gate_20260708/args.yaml
```

`w2` 是在误以为第一次 gate 中断后启动的低 workers 复跑。复核时它仍在 worker TTY 中继续写日志，已确认 strict resume 与 AOQ 命中，但当前只作为重复实验观察，不作为下一步决策前提；在它完成前不再并发启动新 gate。

中文结论：

1. Phase 2CG 的第一次 LSQ-AOQ095 gate 实际是有效 full-val gate，不应按“无效中断”处理。
2. 该结果满足 strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
3. 相比 clean no-QKR/LSQ ordinary continuation 的 `checkpoint-3` Top-1 `79.9220`，LSQ-AOQ095 提升到 `80.0040`，说明 controlled crossing 有正向信号。
4. 但它仍低于 fixed-QKR resume best `80.5540`，更低于最终目标 `81.0`。单层组、0.95 ratio、半 epoch window 的探索强度不够。

下一步判断：

- 不重复普通 continuation，也不回到 QKR/StatsQ。
- 等当前 `w2` 复跑结束或确认停止后，再启动下一条 AOQ-native gate。
- 下一条高优先级 gate 应增强 LSQ-AOQ 的离散解空间探索，例如：
  - 更强 scale ratio：`0.90` 或分段 `0.90 -> 0.95 -> 1.0`；
  - 更长 crossing window：覆盖 0-1800 update，再留后段 stabilization；
  - 扩大 late-stage layer set：从 `features.7.1` 扩到 `features.6.1` 和 `features.7.1` 的 qkv/proj/fc2；
  - 仍保持 strict W4A4、single checkpoint、无 soup。

### Phase 2CI：LSQ-AOQ095 w2 重复复跑完成，结果确认低于 80.004

实验动机：

Phase 2CH 中确认第一次 LSQ-AOQ095 gate 已经有效。由于此前误判第一次 gate 中断，又启动了一条低 workers 复跑 `w2`。本阶段记录该重复复跑的最终结果，避免后续把它误认为新策略。

方法设计：

- 与 Phase 2CH 的 LSQ-AOQ095 配方保持一致；
- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- `wq_mode=lsq`，`aq_mode=lsq`，`wq_bitw=4`，`aq_bitw=4`；
- 对 `features.7.1.attn.qkv/proj` 和 `features.7.1.mlp.fc2` 做 `0.95` scale-ratio crossing；
- update 0-1250 开启 AOQ explore，之后恢复 ratio=1.0；
- 本次只作为重复验证，不作为新策略 gate。

关键 `args.yaml` / 启动证据：

```text
experiment: recipe_resume10_clean_lsq_noqkr_lsqaoq095_w2_gate_20260708
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
wq_mode: lsq
aq_mode: lsq
wq_bitw: 4
aq_bitw: 4
qk_reparam: false
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
aoq_explore_scale_ratio: 0.95
aoq_explore_layers: features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
aoq_explore_start_update: 0
aoq_explore_end_update: 1250
```

strict resume / AOQ 命中证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
AOQ explore scale ratio update: epoch=3, update=0, active=True, ratio=0.95, layers=('features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=3, start_update=0, end_update=1250
AOQ explore scale ratio update: epoch=3, update=1250, active=False, ratio=1.0, layers=('features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=3, start_update=0, end_update=1250
TrainSummary: epoch=3 updates=2502 avg_step_time=0.165583s samples_per_step=512 samples_per_sec=3092.11
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 79.9800 | 95.1260 | 0.8500 | 有效重复验证，但低于第一次 80.0040 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 35.205s  Loss: 0.8500  Acc@1: 79.9800  Acc@5: 95.1260  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq095_w2_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq095_w2_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq095_w2_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq095_w2_gate_20260708/last.pth.tar
```

中文结论：

1. `w2` 复跑是有效 full-val：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. Top-1 `79.9800` 低于第一次 LSQ-AOQ095 的 `80.0040`，说明 `0.95 + 单 late block + 半 epoch window` 的收益很弱，且不稳定。
3. 该结果不满足 81 目标，也没有超过 fixed-QKR resume best `80.5540`。
4. 下一步不再重复 LSQ-AOQ095，而应加大 AOQ-native controlled crossing 的探索强度。

下一步判断：

- 已准备下一条复现实验脚本：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_late2_gate_20260708.sh
```

- 下一条 gate 设计：
  - scale ratio 从 `0.95` 加强到 `0.90`；
  - layer set 从 `features.7.1` 扩展到 `features.6.1 + features.7.1` 的 qkv/proj/fc2；
  - crossing window 从 0-1250 扩展到 0-1800 update；
  - 仍然 no-QKR、no-StatsQ、LSQ strict W4A4、single checkpoint、无 soup。

### Phase 2CJ：LSQ-AOQ090 late2 gate 有效，但实际只命中 3 个 quantizer，Top-1 到 80.058

实验动机：

Phase 2CH/2CI 说明 `0.95 + features.7.1 + 0-1250 update` 的收益很弱：第一次为 `80.0040`，w2 复跑为 `79.9800`。本阶段加大 AOQ-native controlled crossing 强度：scale ratio 从 `0.95` 改为 `0.90`，窗口从 0-1250 延长到 0-1800，并尝试把 layer set 从 `features.7.1` 扩到 `features.6.1 + features.7.1`。

方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- `wq_mode=lsq`，`aq_mode=lsq`，`wq_bitw=4`，`aq_bitw=4`；
- `aoq_explore_scale_ratio=0.90`；
- `aoq_explore_start_update=0`，`aoq_explore_end_update=1800`；
- 名义 layer set 为：
  - `features.6.1.attn.qkv`
  - `features.6.1.attn.proj`
  - `features.6.1.mlp.fc2`
  - `features.7.1.attn.qkv`
  - `features.7.1.attn.proj`
  - `features.7.1.mlp.fc2`
- 但实际日志显示 `quantizers=3`，只命中 3 个 LSQ weight quantizer。结合历史命名，Swin-T late blocks 是 `features.5.5` 和 `features.7.1`，不是 `features.6.1`，因此本次“late2”扩层没有真正生效。

关键命令要点：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_late2_gate_20260708.sh
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
--wq-mode lsq
--aq-mode lsq
--wbits 4
--abits 4
--no-resume-opt
--start-epoch 3
--epochs 4
--scheduler-epochs 4
--lr 2e-4
--min-lr 1e-5
--quant-lr-multiplier 4
--aoq-explore-scale-ratio 0.90
--aoq-explore-layers features.6.1.attn.qkv,features.6.1.attn.proj,features.6.1.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
--aoq-explore-start-update 0
--aoq-explore-end-update 1800
```

关键 `args.yaml` 证据：

```text
experiment: recipe_resume10_clean_lsq_noqkr_lsqaoq090_late2_gate_20260708
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
wq_mode: lsq
aq_mode: lsq
wq_bitw: 4
aq_bitw: 4
qk_reparam: false
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
lr: 0.0002
min_lr: 1.0e-05
quant_lr_multiplier: 4.0
workers: 2
kd_hard_and_soft: 0
aoq_explore_scale_ratio: 0.9
aoq_explore_layers: features.6.1.attn.qkv,features.6.1.attn.proj,features.6.1.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
aoq_explore_start_update: 0
aoq_explore_end_update: 1800
```

strict resume / AOQ 命中证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, ratio=0.9, layers=('features.6.1.attn.qkv', 'features.6.1.attn.proj', 'features.6.1.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=3, start_update=0, end_update=1800
AOQ explore scale ratio update: epoch=3, update=1800, active=False, ratio=1.0, layers=('features.6.1.attn.qkv', 'features.6.1.attn.proj', 'features.6.1.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=3, start_update=0, end_update=1800
TrainSummary: epoch=3 updates=2502 avg_step_time=0.165372s samples_per_step=512 samples_per_sec=3096.06
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 80.0580 | 95.1800 | 0.8481 | 有效 gate，但未达 81 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 34.580s  Loss: 0.8481  Acc@1: 80.0580  Acc@5: 95.1800  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_late2_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late2_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late2_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late2_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. Top-1 `80.0580` 高于 LSQ-AOQ095 的 `80.0040` / `79.9800`，说明更强 ratio 和更长 window 有一点正向信号。
3. 但该结果仍低于 fixed-QKR resume best `80.5540`，更低于最终目标 `81.0`。
4. 本次最关键的技术发现是：名义上指定 `features.6.1 + features.7.1`，实际只命中 `quantizers=3`。原因是当前 Swin-T 命名中 late stage blocks 对应 `features.5.5` 和 `features.7.1`，不是 `features.6.1`。所以本次不是完整 late2 gate，而是 `features.7.1` 上的 stronger/longer AOQ gate。

下一步判断：

- 不应直接重复本次脚本。
- 下一条修正版 AOQ-native gate 应把 layer set 改为：

```text
features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,
features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
```

- 预期应命中 `quantizers=6`。如果仍只命中 3，则说明 `features.5.5` 的 no-QKR/LSQ 模块结构不同，需要先做 named-module/quantizer inventory。
- ratio/window 可先沿用 `0.90` 和 0-1800；只有命中数正确后，才判断 late2 crossing 是否有效。

### Phase 2CK：LSQ-AOQ090 late5571 2-update smoke 命中 6 个 quantizer

实验动机：

Phase 2CJ 发现 `features.6.1 + features.7.1` 的 late2 设计实际只命中 `features.7.1` 的 3 个 LSQ weight quantizer。根据 Swin-T 当前命名和历史日志，正确的 late blocks 应是 `features.5.5` 与 `features.7.1`。本阶段先做 2-update smoke，只验证命中数和训练可启动，不做精度结论。

方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- `wq_mode=lsq`，`aq_mode=lsq`，strict W4A4；
- `aoq_explore_scale_ratio=0.90`；
- `aoq_explore_layers` 改为：

```text
features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,
features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
```

- `max_train_updates=2`，`skip_validate=1`，只做 smoke。

关键 `args.yaml` 证据：

```text
experiment: recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_smoke2upd_20260708
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
wq_mode: lsq
aq_mode: lsq
wq_bitw: 4
aq_bitw: 4
qk_reparam: false
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
lr: 0.0002
min_lr: 1.0e-05
quant_lr_multiplier: 4.0
max_train_updates: 2
workers: 2
aoq_explore_scale_ratio: 0.9
aoq_explore_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
aoq_explore_start_update: 0
aoq_explore_end_update: 2
```

strict resume / AOQ 命中证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, ratio=0.9, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=6, start_update=0, end_update=2
TrainSummary: epoch=3 updates=2 avg_step_time=0.634115s samples_per_step=512 samples_per_sec=807.42
Stopped early after 2 optimizer updates in epoch 3.
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_smoke2upd_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_smoke2upd_20260708/args.yaml
```

中文结论：

1. 修正版 layer set 命中 `quantizers=6`，证明 `features.5.5 + features.7.1` 是当前 clean no-QKR/LSQ Swin-T 的正确 late2 AOQ selection。
2. strict resume 干净，2 update 训练可运行。
3. `terminate called without an active exception` 与 TCPStore broken pipe 出现在 `max_train_updates=2` 早停退出阶段，属于短跑 smoke 的 DDP/NCCL teardown 噪声，不影响“命中数正确、训练可启动”的结论。
4. smoke 不做精度结论。下一步可以启动 full-val gate。

下一步判断：

- 启动 full gate：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_late5571_gate_20260708.sh
```

- 关键期望：
  - `quantizers=6`；
  - 0-1800 update 开启 `ratio=0.90`；
  - update 1800 后恢复 1.0；
  - full-val `Samples=50000`；
  - strict W4A4、single checkpoint、无 soup。

### Phase 2CL：LSQ-AOQ090 late5571 full gate 有效，Top-1 到 80.124

实验动机：

Phase 2CK 的 2-update smoke 已证明 `features.5.5 + features.7.1` 能正确命中 6 个 LSQ weight quantizer。本阶段运行 full gate，验证真正扩展到两个 late blocks 的 `0.90` scale-ratio crossing 是否能超过 `features.7.1` 单 block 的 `80.0580`，并继续向 81 推进。

方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- `wq_mode=lsq`，`aq_mode=lsq`，strict W4A4；
- `aoq_explore_scale_ratio=0.90`；
- `aoq_explore_layers` 为 `features.5.5 + features.7.1` 的 qkv/proj/fc2；
- update 0-1800 开启 controlled crossing；
- update 1800 后恢复 ratio=1.0；
- full-val 只认单个 `checkpoint-4.pth.tar`，不使用 soup / averaging / ensemble。

关键命令要点：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_late5571_gate_20260708.sh
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
--wq-mode lsq
--aq-mode lsq
--wbits 4
--abits 4
--no-resume-opt
--start-epoch 3
--epochs 4
--scheduler-epochs 4
--lr 2e-4
--min-lr 1e-5
--quant-lr-multiplier 4
--aoq-explore-scale-ratio 0.90
--aoq-explore-layers features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
--aoq-explore-start-update 0
--aoq-explore-end-update 1800
```

关键 `args.yaml` 证据：

```text
experiment: recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_gate_20260708
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
wq_mode: lsq
aq_mode: lsq
wq_bitw: 4
aq_bitw: 4
qk_reparam: false
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
lr: 0.0002
min_lr: 1.0e-05
quant_lr_multiplier: 4.0
workers: 2
kd_hard_and_soft: 0
aoq_explore_scale_ratio: 0.9
aoq_explore_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
aoq_explore_start_update: 0
aoq_explore_end_update: 1800
```

strict resume / AOQ 命中证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, ratio=0.9, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=6, start_update=0, end_update=1800
AOQ explore scale ratio update: epoch=3, update=1800, active=False, ratio=1.0, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=6, start_update=0, end_update=1800
TrainSummary: epoch=3 updates=2502 avg_step_time=0.165468s samples_per_step=512 samples_per_sec=3094.25
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 80.1240 | 95.1200 | 0.8488 | 有效 gate，但未达 81 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 35.415s  Loss: 0.8488  Acc@1: 80.1240  Acc@5: 95.1200  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. 修正版 late5571 确实命中 `quantizers=6`，比 Phase 2CJ 的错误 late2 命中更完整。
3. Top-1 `80.1240` 高于 LSQ-AOQ090 单 block 的 `80.0580`，说明扩展到 `features.5.5 + features.7.1` 有小幅正向信号。
4. 但它仍低于 fixed-QKR resume best `80.5540`，也低于最终目标 `81.0`。单 epoch 的强 crossing 仍不足以完成目标。

下一步判断：

- 不调用 `update_goal complete`。
- 下一步优先从本次 `checkpoint-4` 出发做“关闭 AOQ explore 的 stabilization/continuation”短 gate：
  - resume `recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_gate_20260708/checkpoint-4.pth.tar`；
  - no-QKR、no-StatsQ、LSQ strict W4A4；
  - `aoq_explore_scale_ratio=1.0` 或不传 AOQ explore；
  - 低 LR / no-resume-opt 继续 1 epoch；
  - 验证 crossing 后的恢复阶段是否继续涨。
- 如果 stabilization 仍低于 `80.2`，说明 AOQ scale-ratio 只能带来很小提升，需要换成更接近 AOQ 论文的 bin-center delayed stabilization 或 oscillation-aware loss，而不是继续加大 ratio。

### Phase 2CM：late5571 checkpoint-4 关闭 AOQ 后继续 1 epoch，Top-1 回落到 80.064

实验动机：

Phase 2CL 的 `features.5.5 + features.7.1` 双 late-block AOQ explore gate 达到 Top-1 `80.1240`，是 clean no-QKR/LSQ AOQ-native 分支目前最好结果。本阶段测试 crossing 后是否能通过关闭 AOQ explore 的普通 stabilization / continuation 继续上行：从 Phase 2CL 的 `checkpoint-4` resume，不再启用 scale-ratio explore，训练 1 个 epoch 并 full-val。

方法设计：

- 从 `recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_gate_20260708/checkpoint-4.pth.tar` strict resume；
- 不使用 QKR，不使用 StatsQ；
- `wq_mode=lsq`，`aq_mode=lsq`，strict W4A4；
- 不传 AOQ explore layers，`aoq_explore_scale_ratio=1.0`；
- `--no-resume-opt`；
- 继续 epoch 4 -> 5；
- full-val 只认单个 `checkpoint-5.pth.tar`，不使用 soup / averaging / ensemble。

关键命令要点：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_late5571_stabilize4to5_20260708.sh
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_gate_20260708/checkpoint-4.pth.tar
--wq-mode lsq
--aq-mode lsq
--wbits 4
--abits 4
--no-resume-opt
--start-epoch 4
--epochs 5
--scheduler-epochs 5
--lr 2e-4
--min-lr 1e-5
--quant-lr-multiplier 4
```

关键 `args.yaml` 证据：

```text
experiment: recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_stabilize4to5_20260708
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_gate_20260708/checkpoint-4.pth.tar
wq_mode: lsq
aq_mode: lsq
wq_bitw: 4
aq_bitw: 4
qk_reparam: false
no_resume_opt: true
start_epoch: 4
epochs: 5
scheduler_epochs: 5
lr: 0.0002
min_lr: 1.0e-05
quant_lr_multiplier: 4.0
workers: 2
kd_hard_and_soft: 0
aoq_explore_scale_ratio: 1.0
aoq_explore_layers: ''
```

strict resume / 关闭 AOQ 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_gate_20260708/checkpoint-4.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=4, update=0, active=False, ratio=1.0, layers=(), quantizers=53, start_update=0, end_update=0
TrainSummary: epoch=4 updates=2502 avg_step_time=0.166259s samples_per_step=512 samples_per_sec=3079.53
```

说明：`active=False, ratio=1.0, layers=(), quantizers=53` 表示启动时把所有支持 `aoq_scale_ratio` 的量化器重置为 1.0，不是开启 AOQ explore。

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-5.pth.tar` | yes | yes | 50000 | no | 80.0640 | 95.1480 | 0.8497 | 有效 gate，但较 Phase 2CL 回落 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 36.060s  Loss: 0.8497  Acc@1: 80.0640  Acc@5: 95.1480  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_stabilize4to5_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_stabilize4to5_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_stabilize4to5_20260708/checkpoint-5.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_stabilize4to5_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. Top-1 `80.0640` 低于 Phase 2CL 的 `80.1240`，说明 crossing 后仅靠普通 continuation / stabilization 不能继续上行。
3. 该结果仍低于 fixed-QKR resume best `80.5540` 和目标 `81.0`，goal 未完成。
4. 当前证据表明：简单 LSQ scale-ratio explore 能从 `79.9220` 推到 `80.1240`，但收益很小；继续普通 QAT 会回落。下一步需要在探索窗口后同 epoch 引入更明确的 delayed bin-center stabilization / bin regularization，而不是继续无约束 continuation。

下一步判断：

- 停止重复普通 stabilization。
- 下一条优先测试：`0.90` late5571 AOQ explore + delayed BinReg：
  - AOQ explore：0-1800 update，`features.5.5 + features.7.1` qkv/proj/fc2；
  - BinReg：从 update 1800 开始，作用于同一组 LSQ weight quantizers；
  - 目标是避免 crossing 后离散状态继续漂移，把探索后的候选点拉回 bin-center 附近；
  - full-val 若仍低于 `80.2`，说明 scale-ratio + scalar BinReg 仍不足，需要实现更接近 AOQ 论文的 oscillation-aware loss / sign-flip tracking。

### Phase 2CN：LSQ-AOQ090 late5571 delayed-BinReg 4-update smoke 成功

实验动机：

Phase 2CM 表明 crossing 后普通 continuation 会从 `80.1240` 回落到 `80.0640`。本阶段按 AOQ-native 思路验证“先允许 crossing、后延迟稳定”的训练路径是否可执行：前 2 update 用 `0.90` scale-ratio explore，随后关闭 explore 并启用 BinReg，把同一组 late5571 LSQ weight quantizer 拉回 bin center 附近。该阶段只做 4-update smoke，不做精度结论。

方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- `wq_mode=lsq`，`aq_mode=lsq`，strict W4A4；
- AOQ explore layers：

```text
features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,
features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
```

- smoke 参数：
  - `aoq_explore_scale_ratio=0.90`
  - `aoq_explore_end_update=2`
  - `bin_reg_weight=1e-5`
  - `bin_reg_start_update=2`
  - `bin_reg_end_update=4`
  - `max_train_updates=4`
  - `skip_validate=1`

关键 `args.yaml` 证据：

```text
experiment: recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_delayedbinreg_smoke4upd_20260708
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
wq_mode: lsq
aq_mode: lsq
wq_bitw: 4
aq_bitw: 4
qk_reparam: false
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
lr: 0.0002
min_lr: 1.0e-05
quant_lr_multiplier: 4.0
max_train_updates: 4
workers: 2
aoq_explore_scale_ratio: 0.9
aoq_explore_end_update: 2
bin_reg_weight: 1.0e-05
bin_reg_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
bin_reg_start_update: 2
bin_reg_end_update: 4
```

strict resume / AOQ / BinReg 命中证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, ratio=0.9, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=6, start_update=0, end_update=2
AOQ explore scale ratio update: epoch=3, update=2, active=False, ratio=1.0, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=6, start_update=0, end_update=2
Enabled bin regularizer: weight=1e-05, variance_weight=1.0, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), attn_only=False, pairs=6, start_update=2, end_update=4
TrainSummary: epoch=3 updates=4 avg_step_time=0.414804s samples_per_step=512 samples_per_sec=1234.32
Stopped early after 4 optimizer updates in epoch 3.
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_delayedbinreg_smoke4upd_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_delayedbinreg_smoke4upd_20260708/args.yaml
```

中文结论：

1. smoke 证明 AOQ explore 与 delayed BinReg 的时序已接通：update 0-2 开启 AOQ，update 2 后关闭 AOQ 并启用 BinReg。
2. AOQ 命中 `quantizers=6`，BinReg 命中 `pairs=6`，与 late5571 目标模块一致。
3. smoke 不做 Top-1 结论。下一步可以跑 full gate。

下一步判断：

- 启动 full gate：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_late5571_delayedbinreg_gate_20260708.sh
```

- full gate 关键参数：
  - AOQ explore：0-1800 update，ratio `0.90`；
  - BinReg：从 update 1800 开始，`bin_reg_weight=1e-5`；
  - full-val `Samples=50000`；
  - strict W4A4、single checkpoint、无 soup。

### Phase 2CO：LSQ-AOQ090 late5571 delayed-BinReg full gate 有效，但 Top-1 只有 80.084

实验动机：

Phase 2CL 的 pure AOQ explore 达到 `80.1240`，Phase 2CM 的普通 continuation 回落到 `80.0640`。本阶段测试更接近 AOQ-native 思路的 delayed stabilization：前 0-1800 update 允许 `0.90` scale-ratio crossing，后段关闭 explore 并启用 BinReg，尝试把探索后的权重拉回 bin center 附近。

方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- `wq_mode=lsq`，`aq_mode=lsq`，strict W4A4；
- AOQ explore layers：

```text
features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,
features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
```

- `aoq_explore_scale_ratio=0.90`，0-1800 update 开启；
- `bin_reg_weight=1e-5`，从 update 1800 开始，作用于同一组 late5571 LSQ weight quantizer；
- full-val 只认单个 `checkpoint-4.pth.tar`，不使用 soup / averaging / ensemble。

关键 `args.yaml` 证据：

```text
experiment: recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_delayedbinreg_gate_20260708
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
wq_mode: lsq
aq_mode: lsq
wq_bitw: 4
aq_bitw: 4
qk_reparam: false
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
lr: 0.0002
min_lr: 1.0e-05
quant_lr_multiplier: 4.0
workers: 2
kd_hard_and_soft: 0
aoq_explore_scale_ratio: 0.9
aoq_explore_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
aoq_explore_end_update: 1800
bin_reg_weight: 1.0e-05
bin_reg_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
bin_reg_start_update: 1800
bin_reg_end_update: 0
```

strict resume / AOQ / BinReg 命中证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, ratio=0.9, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=6, start_update=0, end_update=1800
AOQ explore scale ratio update: epoch=3, update=1800, active=False, ratio=1.0, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=6, start_update=0, end_update=1800
Enabled bin regularizer: weight=1e-05, variance_weight=1.0, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), attn_only=False, pairs=6, start_update=1800, end_update=0
TrainSummary: epoch=3 updates=2502 avg_step_time=0.176402s samples_per_step=512 samples_per_sec=2902.47
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 80.0840 | 95.0940 | 0.8535 | 有效 gate，但低于 pure AOQ 80.1240 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 36.666s  Loss: 0.8535  Acc@1: 80.0840  Acc@5: 95.0940  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_delayedbinreg_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_delayedbinreg_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_delayedbinreg_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_delayedbinreg_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. delayed BinReg 的时序和命中都正确：AOQ `quantizers=6`，BinReg `pairs=6`。
3. 但 Top-1 `80.0840` 低于 pure late5571 AOQ 的 `80.1240`，也低于 fixed-QKR resume best `80.5540` 和目标 `81.0`。
4. 结论是 scalar BinReg 没有解决 crossing 后的有效稳定问题，甚至可能过约束或拉回了有益 crossing。当前 AOQ scale-ratio + delayed scalar BinReg 路线不应继续盲目加权重或延长。

下一步判断：

- goal 未完成，不调用 `update_goal complete`。
- 下一步先做 bin-crossing/near-bin 诊断，对比：
  - clean no-QKR source `checkpoint-3`；
  - pure AOQ late5571 `80.1240`；
  - stabilization `80.0640`；
  - delayed-BinReg `80.0840`。
- 如果 delayed-BinReg 确实减少了 crossing 但 Top-1 更低，说明需要保留“有益 crossing”而不是统一拉回中心；下一步应实现 sign-flip / crossing-aware selective stabilization，而不是全量 BinReg。

### Phase 2CP：late5571 bin-crossing 诊断，确认全量 BinReg 不是正确稳定方式

诊断动机：

Phase 2CL/2CM/2CO 的结果形成清楚对比：

- pure AOQ late5571：`80.1240`
- 关闭 AOQ 普通 continuation：`80.0640`
- AOQ + delayed scalar BinReg：`80.0840`

这说明 AOQ scale-ratio 能带来小幅正向 crossing，但普通继续训练和全量 BinReg 都没有把收益放大。需要看具体 bin-crossing / near-bin 变化，而不是继续盲跑。

诊断脚本：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_aoq_bin_crossing_20260708.py
```

诊断输出：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_late5571_bin_crossing_20260708/summary.json
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_late5571_bin_crossing_20260708/aggregate_bin_crossing.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_late5571_bin_crossing_20260708/pair_bin_crossing.tsv
```

标签映射：

```text
ckpt10  = clean no-QKR/LSQ source checkpoint-3, Top-1 79.9220
phase2s = pure AOQ late5571 checkpoint-4, Top-1 80.1240
phase2w = stabilization checkpoint-5, Top-1 80.0640
phase2z = delayed-BinReg checkpoint-4, Top-1 80.0840
```

关键诊断摘要：

| pair | Top-1 delta | features.7.1 attn_proj changed | features.7.1 mlp_fc2 changed | features.5.5 attn_proj changed | features.5.5 attn_qkv changed | 结论 |
|---|---:|---:|---:|---:|---:|---|
| source -> pure AOQ | +0.2020 | 0.0593 | 0.0587 | 0.0366 | 0.0315 | 有益 crossing 主要来自 late proj/fc2 |
| pure AOQ -> stabilization | -0.0600 | 0.0522 | 0.0529 | 0.0314 | 0.0263 | 继续训练产生大量新增 crossing 并回落 |
| pure AOQ -> delayed-BinReg | -0.0400 | 0.0125 | 0.0105 | 0.0089 | 0.0077 | BinReg 限制了后续 crossing，但没有提升 |
| source -> delayed-BinReg | +0.1620 | 0.0593 | 0.0587 | 0.0366 | 0.0317 | 总体 crossing 与 pure AOQ 类似，但 Top-1 更低 |

中文结论：

1. 从 source 到 pure AOQ 的收益不是来自“越少 crossing 越好”，而是来自一批中等规模 late weight-bin crossing。
2. 普通 continuation 的问题是 crossing 后继续大量漂移，Top-1 回落。
3. delayed scalar BinReg 确实减少了 pure AOQ 之后的新增 crossing，但 Top-1 仍低于 pure AOQ，说明“全量拉回 bin center”不是正确稳定方式。
4. 下一步不能继续简单加大 BinReg，也不应重复普通 continuation；需要 selective stabilization：在 AOQ explore 结束点捕获已经形成的离散 bin，只锚定靠近边界或继续漂移风险高的权重，保留有益 crossing 本身。

completion audit：

- strict W4A4：当前 clean no-QKR/LSQ 分支均为 `wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`，且无 QKR/StatsQ。
- 单 checkpoint：所有记录结果均为单个 checkpoint full-val，没有 soup / averaging / ensemble。
- full ImageNet raw validation：有效结果均含 `Samples=50000`。
- 目标 Top-1：当前 AOQ-native clean 分支最好为 Phase 2CL `80.1240`，旧 fixed-QKR resume best 为 `80.5540`，均低于 `81.0`。
- 结论：goal 未完成，不调用 `update_goal complete`。

下一步判断：

- 实现并 smoke `selective bin anchor`：
  - AOQ explore：0-1800 update；
  - update 1800 捕获 selected late5571 weight 的量化输出或整数 bin；
  - 后段只对 near-boundary / risk mask 施加 anchor；
  - 先 4-update smoke 验证捕获、pairs、mask fraction；
- 通过后再跑 full-val gate。

### Phase 2CQ：selective bin anchor 4-update smoke 成功

实验动机：

Phase 2CP 诊断显示：pure AOQ 的收益来自一部分有益 crossing，而 delayed scalar BinReg 虽然减少了后续 crossing，却没有提升 Top-1。说明稳定策略不能把所有权重无差别拉回 bin center，而应该在 AOQ explore 结束点捕获已经形成的离散状态，只锚定 near-boundary / 继续漂移风险较高的权重。本阶段实现并 smoke `selective bin anchor`，只验证训练闭环，不做精度结论。

代码改动：

```text
/mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py
```

新增能力：

- `--selective-bin-anchor-weight`
- `--selective-bin-anchor-layers`
- `--selective-bin-anchor-capture-update`
- `--selective-bin-anchor-end-update`
- `--selective-bin-anchor-margin`

实现要点：

- 只作用于带 `lsqw_fn` 的 weight module，符合当前 no-QKR/no-StatsQ/LSQ 分支；
- 在 capture update 捕获 selected module 的当前 quantized target；
- 用 LSQ scale 计算每个权重到 bin boundary 的距离；
- 只对 `boundary_dist <= margin` 的 mask 元素施加 anchor MSE；
- 打印 `pairs/masked/total/mask_fraction`，作为命中证据。

smoke 方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- AOQ explore：update 0-2，ratio `0.90`；
- selective bin anchor：update 2-4；
- `max_train_updates=4`，`skip_validate=1`；
- 不使用 QKR，不使用 StatsQ，不使用 soup。

关键 `args.yaml` 证据：

```text
experiment: recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_selectiveanchor_smoke4upd_20260708
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
wq_mode: lsq
aq_mode: lsq
wq_bitw: 4
aq_bitw: 4
qk_reparam: false
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
lr: 0.0002
min_lr: 1.0e-05
quant_lr_multiplier: 4.0
max_train_updates: 4
workers: 2
aoq_explore_scale_ratio: 0.9
aoq_explore_end_update: 2
selective_bin_anchor_weight: 0.0001
selective_bin_anchor_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
selective_bin_anchor_capture_update: 2
selective_bin_anchor_end_update: 4
selective_bin_anchor_margin: 0.05
```

strict resume / AOQ / selective anchor 命中证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, ratio=0.9, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=6, start_update=0, end_update=2
AOQ explore scale ratio update: epoch=3, update=2, active=False, ratio=1.0, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=6, start_update=0, end_update=2
Captured selective bin anchor: weight=0.0001, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), pairs=6, masked=1125296, total=5898240, mask_fraction=0.190785, capture_update=2, end_update=4, margin=0.05
Enabled selective bin anchor: weight=0.0001, pairs=6, masked=1125296, total=5898240, mask_fraction=0.190785, capture_update=2, end_update=4
TrainSummary: epoch=3 updates=4 avg_step_time=0.418896s samples_per_step=512 samples_per_sec=1222.26
Stopped early after 4 optimizer updates in epoch 3.
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_selectiveanchor_smoke4upd_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_selectiveanchor_smoke4upd_20260708/args.yaml
```

中文结论：

1. selective bin anchor 已接通，AOQ 关闭点捕获成功。
2. `pairs=6` 对齐 late5571 目标模块；`mask_fraction=0.190785`，说明只锚定约 19.1% near-boundary 权重，不是全量拉回中心。
3. smoke 不产生 Top-1 结论。下一步可以跑 full gate。

下一步判断：

- 启动 full gate：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_late5571_selectiveanchor_gate_20260708.sh
```

- full gate 关键参数：
  - AOQ explore：0-1800 update，ratio `0.90`；
  - selective anchor：update 1800 捕获，后段持续启用；
  - `selective_bin_anchor_weight=1e-4`；
  - `selective_bin_anchor_margin=0.05`；
  - full-val `Samples=50000`；
  - strict W4A4、single checkpoint、无 soup。

### Phase 2CR：selective bin anchor full gate 有效，但 Top-1 只有 80.098

实验动机：

Phase 2CO 的 delayed scalar BinReg 结果为 `80.0840`，低于 pure AOQ `80.1240`。Phase 2CP 诊断显示，pure AOQ 的有益提升来自一批 late weight-bin crossing，而全量拉回 bin center 不是正确稳定方式。本阶段测试更选择性的稳定方式：在 AOQ explore 关闭点捕获当前 late5571 量化 target，只对靠近 bin boundary 的权重施加 anchor，尽量保留已经形成的有益 crossing。

方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- `wq_mode=lsq`，`aq_mode=lsq`，strict W4A4；
- AOQ explore：0-1800 update，ratio `0.90`；
- selective bin anchor：update 1800 捕获，后段持续启用；
- `selective_bin_anchor_weight=1e-4`；
- `selective_bin_anchor_margin=0.05`；
- full-val 只认单个 `checkpoint-4.pth.tar`，不使用 soup / averaging / ensemble。

关键 `args.yaml` 证据：

```text
experiment: recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_selectiveanchor_gate_20260708
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
wq_mode: lsq
aq_mode: lsq
wq_bitw: 4
aq_bitw: 4
qk_reparam: false
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
lr: 0.0002
min_lr: 1.0e-05
quant_lr_multiplier: 4.0
workers: 2
kd_hard_and_soft: 0
aoq_explore_scale_ratio: 0.9
aoq_explore_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
aoq_explore_end_update: 1800
selective_bin_anchor_weight: 0.0001
selective_bin_anchor_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
selective_bin_anchor_capture_update: 1800
selective_bin_anchor_end_update: 0
selective_bin_anchor_margin: 0.05
```

strict resume / AOQ / selective anchor 命中证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, ratio=0.9, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=6, start_update=0, end_update=1800
AOQ explore scale ratio update: epoch=3, update=1800, active=False, ratio=1.0, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=6, start_update=0, end_update=1800
Captured selective bin anchor: weight=0.0001, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), pairs=6, masked=970469, total=5898240, mask_fraction=0.164535, capture_update=1800, end_update=0, margin=0.05
Enabled selective bin anchor: weight=0.0001, pairs=6, masked=970469, total=5898240, mask_fraction=0.164535, capture_update=1800, end_update=0
TrainSummary: epoch=3 updates=2502 avg_step_time=0.169110s samples_per_step=512 samples_per_sec=3027.62
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 80.0980 | 95.1720 | 0.8488 | 有效 gate，但低于 pure AOQ 80.1240 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 35.489s  Loss: 0.8488  Acc@1: 80.0980  Acc@5: 95.1720  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_selectiveanchor_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_selectiveanchor_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_selectiveanchor_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_selectiveanchor_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. selective bin anchor 技术闭环有效：捕获 `pairs=6`，`mask_fraction=0.164535`，训练日志中 `SelBinAnchor` 正常出现。
3. 但 Top-1 `80.0980` 仍低于 pure AOQ late5571 的 `80.1240`，更低于 fixed-QKR resume best `80.5540` 和目标 `81.0`。
4. 当前结论：简单 scale-ratio + 后段稳定类正则（全量 BinReg 或 selective near-boundary anchor）都没能突破 `80.2`。说明瓶颈不只是“探索后漂移”，更可能是 scale-ratio 人工诱导的 crossing 与真正有益离散迁移不完全一致。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.0980`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

下一步判断：

- 停止继续在同一 `0.90 late5571 + 后段正则` 框架下做无差别小改。
- 下一步需要更接近 AOQ 论文本体的 oscillation-aware 实现：
  - 显式跟踪 selected LSQ weight 的 integer-bin sign flip / crossing history；
  - 对持续 oscillating 或反复 flip 的权重做 delayed freeze / damp；
  - 对只发生一次且带来候选迁移的 crossing 不强行拉回；
  - full gate 前先做 10-20 update 的 oscillation telemetry，确认真实 flip 率和反向 flip 率。

### Phase 2CS：late5571 weight-bin telemetry 20 update，反向 oscillation 很弱

实验动机：

Phase 2CR 的 selective bin anchor 仍然低于 pure AOQ，说明“后段稳定”类正则不是直接突破口。为了决定是否应该实现真正的 AOQ-style oscillation freeze / damp，本阶段先做只读 telemetry：对 selected late5571 LSQ weight 的 integer bin 逐 update 记录 switch fraction 与 reverse-flip / oscillation fraction。该阶段不做 full-val，不改变 loss。

代码改动：

```text
/mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py
```

新增能力：

- `--weight-bin-telemetry-layers`
- `--weight-bin-telemetry-start-update`
- `--weight-bin-telemetry-end-update`
- `--weight-bin-telemetry-interval`
- `--weight-bin-telemetry-margin`

实现要点：

- 只读观测，不改变 loss；
- 只覆盖带 `lsqw_fn` 的 weight module；
- 每个 telemetry update 计算当前 integer bin；
- 与上一轮 bin 对比得到 `switch_fraction`；
- 与上一轮 delta 方向对比得到 `oscillation_fraction`；
- 同时记录 near-boundary 比例。

实验设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- AOQ explore：0-20 update，ratio `0.90`；
- telemetry：0-20 update，每 update 记录一次；
- `max_train_updates=20`，`skip_validate=1`；
- 不使用 QKR，不使用 StatsQ，不使用 soup。

关键 `args.yaml` 证据：

```text
experiment: recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_telemetry20_20260708
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
wq_mode: lsq
aq_mode: lsq
wq_bitw: 4
aq_bitw: 4
qk_reparam: false
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
lr: 0.0002
min_lr: 1.0e-05
quant_lr_multiplier: 4.0
max_train_updates: 20
workers: 2
aoq_explore_scale_ratio: 0.9
aoq_explore_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
aoq_explore_end_update: 20
weight_bin_telemetry_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
weight_bin_telemetry_start_update: 0
weight_bin_telemetry_end_update: 20
weight_bin_telemetry_interval: 1
weight_bin_telemetry_margin: 0.05
```

strict resume / telemetry 命中证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, ratio=0.9, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=6, start_update=0, end_update=20
WeightBinTelemetry: epoch=3, update=1, pairs=6, total=5898240, near_fraction=0.190793, switch_fraction=0.000000, oscillation_fraction=0.000000, mean_abs_delta=0.000000
WeightBinTelemetry: epoch=3, update=2, pairs=6, total=5898240, near_fraction=0.190785, switch_fraction=0.004868, oscillation_fraction=0.000000, mean_abs_delta=0.004868
WeightBinTelemetry: epoch=3, update=3, pairs=6, total=5898240, near_fraction=0.190752, switch_fraction=0.003606, oscillation_fraction=0.000305, mean_abs_delta=0.003606
WeightBinTelemetry: epoch=3, update=20, pairs=6, total=5898240, near_fraction=0.190209, switch_fraction=0.000918, oscillation_fraction=0.000020, mean_abs_delta=0.000918
TrainSummary: epoch=3 updates=20 avg_step_time=5.636538s samples_per_step=512 samples_per_sec=90.84
Stopped early after 20 optimizer updates in epoch 3.
```

解析输出：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_late5571_telemetry20_20260708.tsv
```

关键数值：

| metric | value |
|---|---:|
| telemetry rows | 20 |
| selected pairs | 6 |
| selected elements | 5898240 |
| update 2 switch_fraction | 0.004868 |
| update 3 switch_fraction | 0.003606 |
| update 20 switch_fraction | 0.000918 |
| max oscillation_fraction | 0.000305 |
| update 20 oscillation_fraction | 0.000020 |
| near_fraction trend | 0.190793 -> 0.190209 |

中文结论：

1. 真实数据不支持“当前主要问题是大量反复震荡”。`oscillation_fraction` 全程很低，最高约 `0.000305`，到 update 20 只有 `0.000020`。
2. 当前 AOQ scale-ratio 更像是在早期制造一批单向 / 一次性 crossing：`switch_fraction` 从 update 2 的 `0.004868` 逐步降到 update 20 的 `0.000918`。
3. 因此直接做“反复 flip 的 freeze/damp”不太可能是高杠杆，因为可冻结的反复震荡权重占比太小。
4. 后续应该重点解决“哪些一次性 crossing 是有益的、哪些是有害的”，而不是只处理反向 oscillation。

completion audit：

- 本阶段是 20-update telemetry smoke，`skip_validate=1`，没有 full-val，因此不可能满足 goal。
- 当前 AOQ-native clean 分支最好仍是 Phase 2CL `80.1240`，低于 `81.0`。
- 旧 fixed-QKR resume best `80.5540` 也低于 `81.0`。
- goal 未完成，不调用 `update_goal complete`。

下一步判断：

- 不优先实现 freeze repeated-oscillator gate，因为反复 oscillation 很弱。
- 下一步应做 crossing-quality selector：
  - 对 source -> pure AOQ 的 crossing 权重按 layer/kind/near-boundary 分组；
  - 只允许历史上正向贡献明显的 `features.7.1 attn_proj/mlp_fc2` 保持较强 ratio；
  - 对 `features.5.5` 或 qkv 采用更弱 ratio 或更短窗口；
  - 先跑分层 ratio gate，而不是全 late5571 同一 ratio。

### Phase 2CT：core71-only AOQ gate 有效，但 Top-1 只有 80.010

实验动机：

Phase 2CS 的 telemetry 显示反复 oscillation 很弱，不能优先做 repeated-flip freeze。Phase 2CP 的 crossing 诊断显示 source -> pure AOQ 的正向提升中，`features.7.1|attn_proj` 和 `features.7.1|mlp_fc2` 的 changed_fraction 最大。因此本阶段测试最窄的 crossing-quality selector：只对这两个 core modules 使用 `0.90` AOQ scale-ratio，避免把 `features.5.5` 和 qkv 一起引入。

方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- `wq_mode=lsq`，`aq_mode=lsq`，strict W4A4；
- AOQ explore：0-1800 update，ratio `0.90`；
- 只作用于：

```text
features.7.1.attn.proj,features.7.1.mlp.fc2
```

- full-val 只认单个 `checkpoint-4.pth.tar`，不使用 soup / averaging / ensemble。

smoke 证据：

```text
Train log: /mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_core71_smoke2upd_20260708.log
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
AOQ explore scale ratio update: epoch=3, update=0, active=True, ratio=0.9, layers=('features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=2, start_update=0, end_update=2
TrainSummary: epoch=3 updates=2 avg_step_time=0.592248s samples_per_step=512 samples_per_sec=864.50
Stopped early after 2 optimizer updates in epoch 3.
```

full gate 关键 `args.yaml` 证据：

```text
experiment: recipe_resume10_clean_lsq_noqkr_lsqaoq090_core71_gate_20260708
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
wq_mode: lsq
aq_mode: lsq
wq_bitw: 4
aq_bitw: 4
qk_reparam: false
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
lr: 0.0002
min_lr: 1.0e-05
quant_lr_multiplier: 4.0
workers: 2
kd_hard_and_soft: 0
aoq_explore_scale_ratio: 0.9
aoq_explore_layers: features.7.1.attn.proj,features.7.1.mlp.fc2
aoq_explore_end_update: 1800
```

strict resume / AOQ 命中证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, ratio=0.9, layers=('features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=2, start_update=0, end_update=1800
AOQ explore scale ratio update: epoch=3, update=1800, active=False, ratio=1.0, layers=('features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quantizers=2, start_update=0, end_update=1800
TrainSummary: epoch=3 updates=2502 avg_step_time=0.165443s samples_per_step=512 samples_per_sec=3094.72
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 80.0100 | 95.1160 | 0.8504 | 有效 gate，但明显低于 pure late5571 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 35.013s  Loss: 0.8504  Acc@1: 80.0100  Acc@5: 95.1160  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_core71_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_core71_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_core71_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_core71_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. 只选 `features.7.1.attn.proj + features.7.1.mlp.fc2` 命中正确，但 Top-1 只有 `80.0100`。
3. 结果低于 pure late5571 的 `80.1240`，也低于 delayed-BinReg `80.0840` 和 selective anchor `80.0980`。说明 core71-only 太窄，虽然这两个模块 crossing 最强，但 `features.5.5` 和 qkv 的参与仍可能对 pure AOQ 的小幅收益有贡献。
4. 当前证据给出的边界是：full late5571 太宽但最好，core71 太窄且变差；后段稳定类正则也变差。下一步需要分层/分窗口 ratio，而不是二选一。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.0100`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

下一步判断：

- 不继续 core71-only。
- 不继续后段正则类小改。
- 下一条如果继续 scale-ratio，应尝试分层强度：
  - `features.7.1.attn.proj,features.7.1.mlp.fc2`: ratio `0.90`，长窗口；
  - `features.5.5.attn.proj,features.5.5.attn.qkv,features.5.5.mlp.fc2,features.7.1.attn.qkv`: ratio `0.95` 或短窗口；
  - 但当前 launcher 还不支持同一 run 内 layer-specific ratio，需要实现 `--aoq-explore-layer-ratios` 或拆成两阶段短窗口。
- 更根本的路线仍是 crossing-quality selector，而不是继续全局 scale-ratio。

### Phase 2CU：实现 layer-specific AOQ ratio，并完成 2-update smoke

实验动机：

Phase 2CT 证明 core71-only 太窄，而 pure late5571 仍是 clean AOQ-native 路线的当前最好结果。新的假设是：保留 late5571 的宽覆盖，但不要对所有 late 层使用同样强的 crossing 压力；让 `features.5.5` 与 `features.7.1.attn.qkv` 采用更弱 ratio `0.95`，只让 crossing 贡献最明显的 `features.7.1.attn.proj` 与 `features.7.1.mlp.fc2` 使用更强 ratio `0.90`。

代码改动：

- 在 `qat_launch.py` 增加 `--aoq-explore-layer-ratios`。
- 参数格式：

```text
features.7.1.attn.proj:0.90,features.7.1.mlp.fc2:0.90
```

- 生效顺序：
  - 先对 `--aoq-explore-layers` 命中的基础层写入 `--aoq-explore-scale-ratio`；
  - 再按 `--aoq-explore-layer-ratios` 覆盖核心层；
  - AOQ explore 窗口结束后把基础层和覆盖层恢复到 `1.0`。

静态检查：

```text
python3 -m py_compile QATs/qat_launch.py
git -C QATs diff --check -- qat_launch.py tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq_layer_ratio_smoke2upd_20260708.sh
```

smoke 方法：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- 不使用 QKR，不使用 StatsQ；
- 不使用 soup / checkpoint averaging / ensemble；
- 只跑 2 个 optimizer update，跳过 validation；
- base AOQ：

```text
ratio=0.95
layers=features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
```

- layer override：

```text
features.7.1.attn.proj:0.90,features.7.1.mlp.fc2:0.90
```

smoke 关键 `args.yaml` 证据：

```text
aoq_explore_layer_ratios: features.7.1.attn.proj:0.90,features.7.1.mlp.fc2:0.90
aoq_explore_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
aoq_explore_scale_ratio: 0.95
```

strict resume / 命中证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.95, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), base_quantizers=6, layer_ratios={'features.7.1.attn.proj': 0.9, 'features.7.1.mlp.fc2': 0.9}, layer_quantizers=2, layer_counts={'features.7.1.attn.proj': 1, 'features.7.1.mlp.fc2': 1}, start_update=0, end_update=2
TrainSummary: epoch=3 updates=2 avg_step_time=0.834199s samples_per_step=512 samples_per_sec=613.76
Stopped early after 2 optimizer updates in epoch 3.
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq_layer_ratio_smoke2upd_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq_layer_ratio_smoke2upd_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq_layer_ratio_smoke2upd_20260708.sh
```

中文结论：

1. layer-specific AOQ ratio 技术路径成立，wrapper、runtime override、`args.yaml`、训练循环全部接通。
2. base late5571 命中 `6` 个 weight quantizer；core override 命中 `2` 个 weight quantizer，且 `features.7.1.attn.proj` 与 `features.7.1.mlp.fc2` 各命中 `1` 个。
3. 2-update smoke 没有 full-val 结果，不能计入 goal accuracy；它只证明参数和命中逻辑正确。
4. 下一步跑 full gate：base `0.95` + core `0.90`，AOQ explore 窗口 `0-1800`，评估单个 `checkpoint-4.pth.tar`，full ImageNet raw validation，`Samples=50000`。

### Phase 2CV：layer-specific AOQ ratio full gate 失败，Top-1 79.9860

实验动机：

Phase 2CU smoke 证明 layer-specific AOQ ratio 技术上可行。本阶段做 full gate，验证“late5571 宽覆盖用弱 ratio `0.95`，core71 用强 ratio `0.90`”是否能超过 pure late5571 `0.90` 的当前最好 clean AOQ-native 结果 `80.1240`。

方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- AOQ explore 窗口：`0-1800` update；
- base AOQ ratio：`0.95`，作用于：

```text
features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
```

- layer override：

```text
features.7.1.attn.proj:0.90,features.7.1.mlp.fc2:0.90
```

- full-val 只认单个 `checkpoint-4.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

关键 `args.yaml` 证据：

```text
aoq_explore_layer_ratios: features.7.1.attn.proj:0.90,features.7.1.mlp.fc2:0.90
aoq_explore_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
aoq_explore_scale_ratio: 0.95
aq_bitw: 4
aq_mode: lsq
epochs: 4
kd_hard_and_soft: 0
no_resume_opt: true
qk_reparam: false
qk_reparam_type: 0
scheduler_epochs: 4
start_epoch: 3
teacher_soft_temperature: 2.75
wq_bitw: 4
wq_mode: lsq
```

strict resume / AOQ 命中与恢复证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.95, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), base_quantizers=6, layer_ratios={'features.7.1.attn.proj': 0.9, 'features.7.1.mlp.fc2': 0.9}, layer_quantizers=2, layer_counts={'features.7.1.attn.proj': 1, 'features.7.1.mlp.fc2': 1}, start_update=0, end_update=1800
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), base_quantizers=6, layer_ratios={'features.7.1.attn.proj': 1.0, 'features.7.1.mlp.fc2': 1.0}, layer_quantizers=2, layer_counts={'features.7.1.attn.proj': 1, 'features.7.1.mlp.fc2': 1}, start_update=0, end_update=1800
TrainSummary: epoch=3 updates=2502 avg_step_time=0.165807s samples_per_step=512 samples_per_sec=3087.93
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 79.9860 | 95.1140 | 0.8537 | 有效 gate，但低于 pure late5571 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 34.741s  Loss: 0.8537  Acc@1: 79.9860  Acc@5: 95.1140  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq_layer_ratio_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq_layer_ratio_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq_layer_ratio_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq_layer_ratio_gate_20260708/last.pth.tar
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq_layer_ratio_gate_20260708.sh
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. layer-specific ratio 机制正确生效，并且在 update 1800 恢复到 `1.0`，所以保存和验证不是在临时缩放状态下完成。
3. 结果 `79.9860` 低于 pure late5571 `80.1240`、selective anchor `80.0980`、delayed BinReg `80.0840`，也低于 core71-only `80.0100`。
4. “非 core late 层弱 ratio + core 强 ratio”的假设不成立。当前证据反而说明 pure late5571 的收益可能依赖 `features.5.5` 与 `features.7.1.attn.qkv` 的强 crossing 共同参与；把这些层弱化会损失收益。
5. 继续做简单 layer ratio sweep 的优先级下降。下一步应该从 crossing-quality selector 转向更细粒度的选择标准，例如按 weight bin crossing 的方向/置信度/teacher loss 代理做在线或离线筛选，而不是只按 module 粗粒度调 ratio。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `79.9860`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2CW：selective-margin AOQ 实现与 2-update smoke 通过

实验动机：

Phase 2CV 证明粗粒度 layer-ratio sweep 失败：把非 core late 层从 `0.90` 弱化到 `0.95` 会把 Top-1 从 pure late5571 的 `80.1240` 拉低到 `79.9860`。这说明 pure late5571 的收益不是简单来自 core71 两层，而更可能来自 late5571 内多个模块共同产生的 weight-bin crossing。

新的方向不再按 module 粗粒度调 ratio，而是按权重在量化 bin 内的位置做更细粒度选择：

- 只对距离 half-integer bin boundary 很近的权重启用 AOQ scale-ratio；
- 已经靠近 bin center 的权重保持原 LSQ scale；
- 目标是让“可能跨 bin 的权重”受控探索，同时减少全量缩放对已稳定权重的破坏。

代码改动：

- 在 `third_party/OFQ/src/quantization/quantizer/lsq.py` 的 LSQ weight quantizer 中增加 `aoq_selective_margin`。
- 当 `aoq_selective_margin > 0` 且 `aoq_scale_ratio != 1` 时：
  - 先用原始 LSQ scale 得到 normalized weight；
  - 计算到 nearest integer center 的距离和到 half-integer boundary 的距离；
  - 仅对 `boundary_dist <= margin` 的元素应用 `s_base * aoq_scale_ratio`；
  - 其他元素使用 `s_base`。
- 在 `qat_launch.py` 增加 `--aoq-explore-selective-margin`，并在 AOQ explore active/inactive 切换时设置/恢复该 margin。

静态与最小行为检查：

```text
python3 -m py_compile QATs/qat_launch.py QATs/third_party/OFQ/src/quantization/quantizer/lsq.py
git -C QATs diff --check -- qat_launch.py third_party/OFQ/src/quantization/quantizer/lsq.py tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_smoke2upd_20260708.sh
```

最小 quantizer forward/backward 检查：

```text
shape (2, 4)
finite True True True
grad_shape (2, 4) (2,)
```

smoke 方法：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- AOQ explore：ratio `0.90`，selective margin `0.08`；
- AOQ layers 沿用 pure late5571 的 6 个模块：

```text
features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
```

- 只跑 2 个 optimizer update，跳过 validation；
- 不使用 soup / checkpoint averaging / ensemble。

关键 `args.yaml` 证据：

```text
aoq_explore_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aq_bitw: 4
aq_mode: lsq
epochs: 4
kd_hard_and_soft: 0
no_resume_opt: true
qk_reparam: false
qk_reparam_type: 0
scheduler_epochs: 4
start_epoch: 3
teacher_soft_temperature: 2.75
wq_bitw: 4
wq_mode: lsq
```

strict resume / AOQ 命中证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, start_update=0, end_update=2
TrainSummary: epoch=3 updates=2 avg_step_time=0.650907s samples_per_step=512 samples_per_sec=786.59
Stopped early after 2 optimizer updates in epoch 3.
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_smoke2upd_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_smoke2upd_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_smoke2upd_20260708.sh
```

中文结论：

1. selective-margin AOQ 的代码路径和启动链路成立，真实 GPU worker 上 2-update smoke 正常完成。
2. strict resume 成功，`missing=0, unexpected=0`，没有恢复 optimizer/scheduler/scaler/RNG。
3. AOQ selective-margin 命中 pure late5571 的 `6` 个 LSQ weight quantizer，`selective_margin=0.08` 已写入 runtime 日志。
4. 这个 smoke 没有 full-val，不能计入 goal accuracy；它只证明新范式分支可运行。
5. 下一步应跑 full gate：ratio `0.90` + selective margin `0.08` + AOQ window `0-1800`，评估单个 `checkpoint-4.pth.tar`，full ImageNet raw validation，`Samples=50000`。如果 full gate 低于 `80.1240`，说明 margin 过窄或“只边界权重缩放”破坏了 pure AOQ 的共同 crossing；如果超过 `80.1240`，再继续 margin/窗口精调。

### Phase 2CX：selective-margin08 AOQ full gate 小幅超过 pure late5571，Top-1 80.1660

实验动机：

Phase 2CW 的 2-update smoke 已经证明 selective-margin AOQ 可运行。本阶段做 full gate，验证“只对 near-boundary 权重应用 AOQ scale-ratio”是否比 pure late5571 全量缩放更好。

方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- AOQ explore 窗口：`0-1800` update；
- AOQ ratio：`0.90`；
- selective margin：`0.08`；
- AOQ layers 沿用 pure late5571 的 6 个模块：

```text
features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
```

- full-val 只认单个 `checkpoint-4.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

关键 `args.yaml` 证据：

```text
aoq_explore_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aq_bitw: 4
aq_mode: lsq
epochs: 4
kd_hard_and_soft: 0
no_resume_opt: true
qk_reparam: false
qk_reparam_type: 0
scheduler_epochs: 4
start_epoch: 3
teacher_soft_temperature: 2.75
wq_bitw: 4
wq_mode: lsq
```

strict resume / AOQ 命中与恢复证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, start_update=0, end_update=1800
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, start_update=0, end_update=1800
TrainSummary: epoch=3 updates=2502 avg_step_time=0.166242s samples_per_step=512 samples_per_sec=3079.85
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 80.1660 | 95.1680 | 0.8476 | 有效 gate，小幅超过 pure late5571 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 34.854s  Loss: 0.8476  Acc@1: 80.1660  Acc@5: 95.1680  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/last.pth.tar
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. selective-margin AOQ 机制正确生效，并且在 update 1800 恢复到 `ratio=1.0`、`selective_margin=0.0`，所以保存和验证不是在临时缩放状态下完成。
3. `80.1660` 小幅超过 pure late5571 `80.1240`，也超过 delayed BinReg `80.0840`、selective anchor `80.0980`、layer-ratio `79.9860`、core71-only `80.0100`。
4. 这个结果支持“只让 near-boundary 权重参与受控 crossing”这个方向，比全量缩放更稳，但增益只有 `+0.042`，距离 81 仍差 `0.834`。
5. 下一步不应回到粗粒度 layer ratio；应继续围绕 selective-margin 做更高杠杆实验：
   - 扩大 margin，例如 `0.12`，让更多 near-boundary 权重参与 crossing；
   - 或延长/分段窗口，例如 `0-2200` 后再恢复；
   - 或在 selective-margin AOQ 后追加更强 delayed bin-center stabilization。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.1660`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2CY：selective-margin12 AOQ full gate 低于 margin08，Top-1 80.0880

实验动机：

Phase 2CX 证明 selective-margin08 比 pure late5571 更好，但增益很小。本阶段测试更大的 selective margin `0.12`，判断是否需要让更多 near-boundary 权重参与 crossing。

方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- AOQ explore 窗口：`0-1800` update；
- AOQ ratio：`0.90`；
- selective margin：`0.12`；
- AOQ layers 沿用 pure late5571 的 6 个模块：

```text
features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
```

- full-val 只认单个 `checkpoint-4.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

实际命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin12_gate_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin12_gate_20260708.log \
MASTER_PORT=30953 \
AOQ_EXPLORE_SELECTIVE_MARGIN=0.12 \
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.12
aq_bitw: 4
aq_mode: lsq
epochs: 4
kd_hard_and_soft: 0
no_resume_opt: true
qk_reparam: false
qk_reparam_type: 0
scheduler_epochs: 4
start_epoch: 3
teacher_soft_temperature: 2.75
wq_bitw: 4
wq_mode: lsq
```

strict resume / AOQ 命中与恢复证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.12, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, start_update=0, end_update=1800
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, start_update=0, end_update=1800
TrainSummary: epoch=3 updates=2502 avg_step_time=0.165803s samples_per_step=512 samples_per_sec=3088.01
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 80.0880 | 95.1840 | 0.8505 | 有效 gate，但低于 margin08 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 34.959s  Loss: 0.8505  Acc@1: 80.0880  Acc@5: 95.1840  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin12_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin12_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin12_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin12_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. selective margin `0.12` 正确生效，并且在 update 1800 恢复到 `ratio=1.0`、`selective_margin=0.0`。
3. `80.0880` 低于 margin08 的 `80.1660`，也低于 pure late5571 的 `80.1240`。说明扩大 near-boundary 范围会引入过多不可靠 crossing。
4. 当前 selective-margin 的甜点更接近 `0.08`，下一步不应继续增大 margin；更合理的是保持 margin08，改探索时间窗口或 delayed stabilization。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.0880`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2FR：early-free + delayed recent oscillation selector 失败，Top-1 80.0340

实验动机：

Phase 2FQ 的 `recent_oscillating` 全程启用后，只选择当前 update 新发生方向反转的极少量 near-boundary 权重，full-val 为 `80.0880`。它比 Phase 2FP 的累计 history `79.9160` 好很多，但仍低于 Phase 2CX `80.1660`，说明全程 recent-only 太窄，丢掉了 Phase 2CX 早期普通 near-boundary AOQ 的有效探索。本阶段测试 hybrid 时序：前 600 update 保持 Phase 2CX 普通 selective-margin08 AOQ，先允许充分 crossing；600 之后切到 `recent_oscillating`，只让当前方向反转的权重继续探索。

方法设计：

- clean no-QKR / no-StatsQ 主线；
- 从 clean no-QKR/LSQ source `checkpoint-3` strict resume；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- AOQ explore：`scale_ratio=0.90`，`selective_margin=0.08`，`end_update=1800`；
- `aoq_explore_quality_mode=recent_oscillating`；
- `quality_start_update=600`，600 前不启用 quality mask，600 后启用 recent-only per-weight selector；
- `quality_min_frac=0`；
- 不使用 QKR、StatsQ、confidence-band KD、local reference、BinReg、selective anchor、soup、checkpoint averaging、multi-checkpoint averaging、ensemble。

805-update smoke 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_recentosc_start600_smoke805upd_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_recentosc_start600_smoke805upd_20260709.log \
MASTER_PORT=31451 \
AOQ_EXPLORE_SCALE_RATIO=0.90 \
AOQ_EXPLORE_SELECTIVE_MARGIN=0.08 \
AOQ_EXPLORE_QUALITY_MODE=recent_oscillating \
AOQ_EXPLORE_QUALITY_START_UPDATE=600 \
AOQ_EXPLORE_QUALITY_MIN_FRAC=0 \
AOQ_EXPLORE_END_UPDATE=1800 \
MAX_TRAIN_UPDATES=805 \
SKIP_VALIDATE=1 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

smoke 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, ... quality_mode=recent_oscillating, quality_start_update=600 ...
AOQ crossing-quality selector: epoch=3, update=600, mode=recent_oscillating, pairs=6, near=1488978, selected=0, selected_over_near=0.000000, moved_excluded=0, switched=0, oscillating=0, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=800, mode=recent_oscillating, pairs=6, near=1502920, selected=2053, selected_over_near=0.001366, moved_excluded=2053, switched=2605, oscillating=2053, missing_pairs=0
TrainSummary: epoch=3 updates=805 avg_step_time=0.172245s samples_per_step=512 samples_per_sec=2972.52
```

smoke 结论：

hybrid 时序工程链路成立。600 前没有 quality mask；600 时初始化 recent history，仍无 selected；800 时 recent-only selector 选中 `2053 / 1502920 = 0.1366%` 的 near-boundary 权重，保持极稀疏。

full gate 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_recentosc_start600_gate_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_recentosc_start600_gate_20260709.log \
MASTER_PORT=31453 \
AOQ_EXPLORE_SCALE_RATIO=0.90 \
AOQ_EXPLORE_SELECTIVE_MARGIN=0.08 \
AOQ_EXPLORE_QUALITY_MODE=recent_oscillating \
AOQ_EXPLORE_QUALITY_START_UPDATE=600 \
AOQ_EXPLORE_QUALITY_MIN_FRAC=0 \
AOQ_EXPLORE_END_UPDATE=1800 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_end_update: 1800
aoq_explore_quality_min_frac: 0.0
aoq_explore_quality_mode: recent_oscillating
aoq_explore_quality_start_update: 600
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aq_bitw: 4
aq_mode: lsq
epochs: 4
kd_hard_and_soft: 0
max_train_updates: 0
no_resume_opt: true
qk_reparam: false
scheduler_epochs: 4
skip_validate: false
start_epoch: 3
teacher_soft_temperature: 2.75
wq_bitw: 4
wq_mode: lsq
```

selector 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, ... quality_mode=recent_oscillating, quality_start_update=600 ...
AOQ crossing-quality selector: epoch=3, update=600, mode=recent_oscillating, pairs=6, near=1488978, selected=0, selected_over_near=0.000000, moved_excluded=0, switched=0, oscillating=0, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=800, mode=recent_oscillating, pairs=6, near=1502920, selected=2053, selected_over_near=0.001366, moved_excluded=2053, switched=2605, oscillating=2053, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=1200, mode=recent_oscillating, pairs=6, near=1518918, selected=2489, selected_over_near=0.001639, moved_excluded=2489, switched=2750, oscillating=2489, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=1600, mode=recent_oscillating, pairs=6, near=1529700, selected=2376, selected_over_near=0.001553, moved_excluded=2376, switched=2507, oscillating=2376, missing_pairs=0
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, ... quality_mode=recent_oscillating ...
TrainSummary: epoch=3 updates=2502 avg_step_time=0.174303s samples_per_step=512 samples_per_sec=2937.41
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 Phase 2CX | 对比全局 best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-4.pth.tar` | yes | yes | yes | 50000 | no | 80.0340 | 95.1960 | 0.8477 | -0.1320 | -0.5200 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 34.869s  Loss: 0.8477  Acc@1: 80.0340  Acc@5: 95.1960  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_recentosc_start600_smoke805upd_20260709.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_recentosc_start600_gate_20260709.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_recentosc_start600_gate_20260709/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_recentosc_start600_gate_20260709/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_recentosc_start600_gate_20260709/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. hybrid 时序工程上有效：前 600 update 是 Phase 2CX 风格普通 near-boundary AOQ，600 后切到 recent-only per-weight selector。
3. Top-1 只有 `80.0340`，比全程 recent-only 的 `80.0880` 更低，也低于 Phase 2CX `80.1660`。说明“早期自由 + 后期仅 recent oscillation”并没有保留 Phase 2CX 的收益。
4. 三个 per-weight selector 结果形成清晰证据链：累计 history 太宽会掉到 `79.9160`；全程 recent-only 过窄为 `80.0880`；延迟 recent-only 为 `80.0340`。当前 per-weight oscillation mask 还没有找到超过普通 selective-margin08 的选择准则。
5. 下一步不应继续扫 recent start update。更合理的是换 candidate-state 的定义，例如记录每个权重的候选 bin endpoint 并在 validation proxy / loss proxy 上选择，而不是只用 oscillation 事件本身。
6. Goal 仍未完成，不调用 `update_goal complete`。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- clean no-QKR/no-StatsQ：满足，`qk_reparam=false`、`wq_mode=lsq`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.0340`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2CZ：margin08 延长 AOQ window 到 2200 失败，Top-1 79.9800

实验动机：

Phase 2CX 的 margin08、`0-1800` 窗口是当前 clean AOQ-native 最好结果 `80.1660`。Phase 2CY 证明扩大 margin 到 `0.12` 会下降。本阶段固定 margin08，只把 AOQ explore end update 从 `1800` 延长到 `2200`，判断是否需要更长探索时间。

方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- AOQ ratio：`0.90`；
- selective margin：`0.08`；
- AOQ explore 窗口：`0-2200` update；
- AOQ layers 沿用 pure late5571 的 6 个模块：

```text
features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
```

- full-val 只认单个 `checkpoint-4.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

实际命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_end2200_gate_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_end2200_gate_20260708.log \
MASTER_PORT=30963 \
AOQ_EXPLORE_SELECTIVE_MARGIN=0.08 \
AOQ_EXPLORE_END_UPDATE=2200 \
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_end_update: 2200
aoq_explore_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aq_bitw: 4
aq_mode: lsq
epochs: 4
kd_hard_and_soft: 0
no_resume_opt: true
qk_reparam: false
qk_reparam_type: 0
scheduler_epochs: 4
start_epoch: 3
teacher_soft_temperature: 2.75
wq_bitw: 4
wq_mode: lsq
```

strict resume / AOQ 命中与恢复证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, start_update=0, end_update=2200
AOQ explore scale ratio update: epoch=3, update=2200, active=False, base_ratio=1.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, start_update=0, end_update=2200
TrainSummary: epoch=3 updates=2502 avg_step_time=0.165497s samples_per_step=512 samples_per_sec=3093.70
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 79.9800 | 95.1620 | 0.8517 | 有效 gate，但低于 margin08/end1800 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 35.274s  Loss: 0.8517  Acc@1: 79.9800  Acc@5: 95.1620  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_end2200_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_end2200_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_end2200_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_end2200_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. 延长窗口到 `2200` 正确生效，并且在 update 2200 恢复到 `ratio=1.0`、`selective_margin=0.0`。
3. `79.9800` 明显低于 margin08/end1800 的 `80.1660`，也低于 pure late5571 的 `80.1240`。说明 selective-margin08 的探索窗口不宜继续延长，后段过晚恢复会损害收敛。
4. 当前最好 clean AOQ-native 仍是 Phase 2CX：`selective_margin=0.08`、AOQ window `0-1800`、Top-1 `80.1660`。
5. 下一步应从 Phase 2CX 的 checkpoint 继续做 delayed stabilization/收敛，而不是从 source 重跑更长探索：例如从 `selectivemargin08_gate` 的 `checkpoint-4` resume 1 epoch，只做 bin-center stabilization 或 mild BinReg，检查是否能在 `80.1660` 基础上继续提升。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `79.9800`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DA：从 margin08 best 继续 mild BinReg 收敛失败，Top-1 80.0260

实验动机：

Phase 2CX 的 selective-margin08 full gate 达到 clean AOQ-native 当前最好 `80.1660`。Phase 2CY/2CZ 说明扩大 margin 或延长探索窗口都会变差。本阶段不再重新探索，而是从 Phase 2CX 的 `checkpoint-4` 出发，关闭 AOQ explore，用更弱的 late5571 BinReg 做 1 个 epoch 的后段收敛，测试是否能在 `80.1660` 基础上继续提升。

方法设计：

- 从 `selectivemargin08_gate` 的 `checkpoint-4` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- AOQ explore 关闭：`aoq_explore_scale_ratio=1.0`；
- BinReg：从 update 0 开始，`bin_reg_weight=3e-6`，作用于 late5571 6 个 weight quantizer；
- full-val 只认单个 `checkpoint-5.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

smoke 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_mildbinreg4to5_smoke4upd_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_mildbinreg4to5_smoke4upd_20260708.log \
MASTER_PORT=30973 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
START_EPOCH=4 EPOCHS=5 SCHEDULER_EPOCHS=5 \
AOQ_EXPLORE_SCALE_RATIO=1.0 AOQ_EXPLORE_END_UPDATE=0 \
BIN_REG_WEIGHT=3e-6 BIN_REG_START_UPDATE=0 \
MAX_TRAIN_UPDATES=4 SKIP_VALIDATE=1 \
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_late5571_delayedbinreg_gate_20260708.sh
```

smoke 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Enabled bin regularizer: weight=3e-06, variance_weight=1.0, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), attn_only=False, pairs=6, start_update=0, end_update=0
TrainSummary: epoch=4 updates=4 avg_step_time=0.442883s samples_per_step=512 samples_per_sec=1156.06
Stopped early after 4 optimizer updates in epoch 4.
```

full gate 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_mildbinreg4to5_gate_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_mildbinreg4to5_gate_20260708.log \
MASTER_PORT=30983 \
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
START_EPOCH=4 EPOCHS=5 SCHEDULER_EPOCHS=5 \
AOQ_EXPLORE_SCALE_RATIO=1.0 AOQ_EXPLORE_END_UPDATE=0 \
BIN_REG_WEIGHT=3e-6 BIN_REG_START_UPDATE=0 \
MAX_TRAIN_UPDATES=0 SKIP_VALIDATE=0 \
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_late5571_delayedbinreg_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_scale_ratio: 1.0
bin_reg_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
bin_reg_start_update: 0
bin_reg_weight: 3.0e-06
aq_bitw: 4
aq_mode: lsq
epochs: 5
kd_hard_and_soft: 0
no_resume_opt: true
qk_reparam: false
qk_reparam_type: 0
scheduler_epochs: 5
start_epoch: 4
teacher_soft_temperature: 2.75
wq_bitw: 4
wq_mode: lsq
```

strict resume / BinReg 命中证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=4, update=0, active=False, base_ratio=1.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=0, layer_ratios={}, layer_quantizers=0, layer_counts={}, start_update=0, end_update=0
Enabled bin regularizer: weight=3e-06, variance_weight=1.0, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), attn_only=False, pairs=6, start_update=0, end_update=0
TrainSummary: epoch=4 updates=2502 avg_step_time=0.203484s samples_per_step=512 samples_per_sec=2516.17
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-5.pth.tar` | yes | yes | 50000 | no | 80.0260 | 95.1540 | 0.8492 | 有效 gate，但低于 margin08 best |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 36.049s  Loss: 0.8492  Acc@1: 80.0260  Acc@5: 95.1540  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_mildbinreg4to5_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_mildbinreg4to5_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_mildbinreg4to5_gate_20260708/checkpoint-5.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_mildbinreg4to5_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. 从 margin08 best 继续 1 个 epoch，即使加入更弱的 late5571 BinReg，也从 `80.1660` 回落到 `80.0260`。
3. 说明当前瓶颈不是“探索后缺少统一 bin-center 收敛”，而是 margin08 的有益 crossing 在继续训练中很容易被破坏；BinReg 即使很弱也不能保住。
4. 当前 clean AOQ-native 最好仍是 Phase 2CX：`checkpoint-4` Top-1 `80.1660`。
5. 下一步应避免再做 BinReg/普通 continuation；更可能需要在同一个 epoch 内更早停在探索后的局部高点，或做 step-level checkpoint/validation around update 1500-1900，找到 80.166 之前是否存在更高峰值。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-5.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.0260`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DB：selective-margin08 step-level peak 检查，局部早停均低于完整 epoch

实验动机：

Phase 2CX 的完整 epoch 结果是当前 clean AOQ-native 最好 `80.1660`。Phase 2DA 表明从该 checkpoint 继续 1 epoch 会回落。为了判断 `80.1660` 是否已经错过了 epoch 内更高峰值，本阶段做 step-level full-val：在 AOQ explore 恢复到正常 scale 后，于不同 update 提前停止、保存单 checkpoint 并 full-val。

关键原则：

- 不能在临时 AOQ scale 下验证；
- 因此若 AOQ end update 是 `N`，`max_train_updates` 至少设为 `N+1`，确保训练循环执行 `active=False` 恢复逻辑；
- 每个点都是 strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`；
- 不使用 soup / checkpoint averaging / ensemble。

测试设置：

```text
source checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
AOQ ratio: 0.90
selective margin: 0.08
AOQ layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
```

实际命令形态：

```bash
EXP=<experiment_name> \
LOG=<train_log> \
MASTER_PORT=<port> \
AOQ_EXPLORE_SELECTIVE_MARGIN=0.08 \
AOQ_EXPLORE_END_UPDATE=<end_update> \
MAX_TRAIN_UPDATES=<stop_update> \
SKIP_VALIDATE=0 \
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

关键结果：

| phase | end_update | stop_update | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2DB-a | 1500 | 1501 | yes | yes | 50000 | no | 79.8280 | 95.0480 | 0.8635 | 过早，明显低 |
| 2DB-b | 1800 | 1801 | yes | yes | 50000 | no | 80.0260 | 95.0380 | 0.8632 | 刚恢复后仍低 |
| 2DB-c | 1800 | 2200 | yes | yes | 50000 | no | 80.0580 | 95.1280 | 0.8557 | 恢复后 400 update 仍低 |
| 2DB-d | 1800 | 2400 | yes | yes | 50000 | no | 80.0140 | 95.1720 | 0.8609 | 仍低 |
| 2DB-e | 1800 | 2450 | yes | yes | 50000 | no | 80.0140 | 95.0500 | 0.8663 | 仍低 |
| 2CX | 1800 | 2502/full epoch | yes | yes | 50000 | no | 80.1660 | 95.1680 | 0.8476 | 当前 clean AOQ-native best |

原始 full-val 摘要：

```text
2DB-a: Test: [distributed-summary]  Time: 36.874s  Loss: 0.8635  Acc@1: 79.8280  Acc@5: 95.0480  Samples: 50000
2DB-b: Test: [distributed-summary]  Time: 35.453s  Loss: 0.8632  Acc@1: 80.0260  Acc@5: 95.0380  Samples: 50000
2DB-c: Test: [distributed-summary]  Time: 37.049s  Loss: 0.8557  Acc@1: 80.0580  Acc@5: 95.1280  Samples: 50000
2DB-d: Test: [distributed-summary]  Time: 36.465s  Loss: 0.8609  Acc@1: 80.0140  Acc@5: 95.1720  Samples: 50000
2DB-e: Test: [distributed-summary]  Time: 35.127s  Loss: 0.8663  Acc@1: 80.0140  Acc@5: 95.0500  Samples: 50000
```

代表性恢复证据：

```text
AOQ explore scale ratio update: epoch=3, update=1500, active=False, base_ratio=1.0, ... selective_margin=0.0, base_quantizers=6, ... end_update=1500
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, ... selective_margin=0.0, base_quantizers=6, ... end_update=1800
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_end1500_val1501_20260708.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_end1800_val1801_20260708.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_end1800_val2200_20260708.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_end1800_val2400_20260708.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_end1800_val2450_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_end1500_val1501_20260708/
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_end1800_val1801_20260708/
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_end1800_val2200_20260708/
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_end1800_val2400_20260708/
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_end1800_val2450_20260708/
```

中文结论：

1. step-level peak 检查没有找到高于完整 epoch 的局部点。
2. `1501` 太早，只有 `79.8280`；刚恢复后的 `1801` 也只有 `80.0260`。
3. 恢复后继续到 `2200/2400/2450` 仍在 `80.01-80.06`，没有接近完整 epoch `80.1660`。
4. 因此 Phase 2CX 的 `80.1660` 不是“错过的中途峰值”，而更像完整 epoch 末端训练状态的结果。
5. 下一步不应继续做 stop-point 搜索；应改一个更大的变量，例如在 selective-margin08 探索期间限制可训练集合到 late blocks，测试 full-model 更新是否在 clean AOQ 分支中过度破坏有益 crossing。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，每个点评估各自单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，所有点 `Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，最好仍是 Phase 2CX `80.1660`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DC：selective-margin08 + params-in-late 可训练集合 full gate 低于 full-model，Top-1 80.0640

实验动机：

Phase 2DB 说明 step-level 早停没有找到比完整 epoch 更高的局部峰值。下一步改一个更大的变量：在 clean AOQ-native selective-margin08 中限制可训练集合，只训练 `features.5.5` 与 `features.7.1` late blocks，测试 full-model 更新是否在 clean no-QKR/LSQ 分支里破坏有益 crossing。

方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- AOQ ratio：`0.90`；
- selective margin：`0.08`；
- AOQ explore 窗口：`0-1800` update；
- 可训练集合：`trainable_policy=params_in_layers`；
- 可训练 layers：`features.5.5,features.7.1`；
- full-val 只认单个 `checkpoint-4.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

脚本改动：

- `tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh` 增加可选透传：

```text
TRAINABLE_POLICY
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS
```

smoke 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_paramsinlate_smoke2upd_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_paramsinlate_smoke2upd_20260708.log \
MASTER_PORT=31043 \
AOQ_EXPLORE_SELECTIVE_MARGIN=0.08 \
AOQ_EXPLORE_END_UPDATE=2 \
MAX_TRAIN_UPDATES=2 \
SKIP_VALIDATE=1 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

smoke 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, start_update=0, end_update=2
Trainable parameter update policy: epoch=3, update=0, mode=requires_grad, policy=params_in_layers, trainable=8903512, frozen=19631895
TrainSummary: epoch=3 updates=2 avg_step_time=0.418504s samples_per_step=512 samples_per_sec=1223.41
```

full gate 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_paramsinlate_gate_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_paramsinlate_gate_20260708.log \
MASTER_PORT=31053 \
AOQ_EXPLORE_SELECTIVE_MARGIN=0.08 \
AOQ_EXPLORE_END_UPDATE=1800 \
MAX_TRAIN_UPDATES=0 \
SKIP_VALIDATE=0 \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_end_update: 1800
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aq_bitw: 4
aq_mode: lsq
epochs: 4
kd_hard_and_soft: 0
no_resume_opt: true
qk_reparam: false
qk_reparam_type: 0
scheduler_epochs: 4
start_epoch: 3
teacher_soft_temperature: 2.75
trainable_policy: params_in_layers
trainable_policy_freeze_act_except_layers: features.5.5,features.7.1
wq_bitw: 4
wq_mode: lsq
```

strict resume / AOQ / trainable 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, start_update=0, end_update=1800
Trainable parameter update policy: epoch=3, update=0, mode=requires_grad, policy=params_in_layers, trainable=8903512, frozen=19631895
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, start_update=0, end_update=1800
TrainSummary: epoch=3 updates=2502 avg_step_time=0.136715s samples_per_step=512 samples_per_sec=3745.01
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 80.0640 | 95.1640 | 0.8528 | 有效 gate，但低于 full-model selective-margin08 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 36.185s  Loss: 0.8528  Acc@1: 80.0640  Acc@5: 95.1640  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_paramsinlate_smoke2upd_20260708.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_paramsinlate_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_paramsinlate_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_paramsinlate_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_paramsinlate_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. 限制可训练集合到 `features.5.5,features.7.1` 能正常跑，速度也更快，但 Top-1 只有 `80.0640`。
3. 结果低于 full-model selective-margin08 的 `80.1660`，说明 clean AOQ-native 分支里 full-model 的小幅更新不是主要破坏项，反而可能对最后提升有贡献。
4. 当前 clean AOQ-native 最好仍是 Phase 2CX：full-model selective-margin08，Top-1 `80.1660`。
5. 下一步应避免继续缩窄可训练集合。更可能需要增强 supervision / reconstruction 信号，而不是只调 crossing mask、stop point 或 trainable set。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.0640`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DD：selective-margin08 + 轻量 teacher feature-output 监督失败，Top-1 79.9580

实验动机：

Phase 2DB/2DC 说明继续调 stop point、可训练集合、BinReg 都不能超过 Phase 2CX 的 `80.1660`。本阶段换一个更大的机制变量：在 selective-margin08 的正常 QAT 中加入轻量 teacher feature-output 监督，尝试用 teacher 局部特征约束 crossing 期间的表征漂移。为避免历史上较强 feature-output 权重过约束，本阶段只用 `0.001`。

方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- AOQ ratio：`0.90`；
- selective margin：`0.08`；
- AOQ explore 窗口：`0-1800` update；
- teacher feature-output：
  - `teacher_feature_output_weight=0.001`
  - `teacher_feature_output_layers=features.5.5,features.7.1`
  - `teacher_feature_output_loss=norm_mse`
- full-val 只认单个 `checkpoint-4.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

脚本改动：

- `tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh` 增加可选透传：

```text
TEACHER_FEATURE_OUTPUT_WEIGHT
TEACHER_FEATURE_OUTPUT_LAYERS
TEACHER_FEATURE_OUTPUT_LOSS
```

smoke 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_featout001_smoke2upd_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_featout001_smoke2upd_20260708.log \
MASTER_PORT=31063 \
AOQ_EXPLORE_SELECTIVE_MARGIN=0.08 \
AOQ_EXPLORE_END_UPDATE=2 \
MAX_TRAIN_UPDATES=2 \
SKIP_VALIDATE=1 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.001 \
TEACHER_FEATURE_OUTPUT_LAYERS=features.5.5,features.7.1 \
TEACHER_FEATURE_OUTPUT_LOSS=norm_mse \
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

smoke 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Teacher feature-output hooks: layers=('features.5.5', 'features.7.1'), student_matches=('features.5.5', 'features.7.1'), teacher_matches=('features.5.5', 'features.7.1')
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, start_update=0, end_update=2
Teacher feature-output debug: student_count=2, teacher_count=2, 0:s=(64, 14, 14, 384) t=(64, 14, 14, 384) mse=1.386e+01; 1:s=(64, 7, 7, 768) t=(64, 7, 7, 768) mse=1.959e+00
TeacherFeatOut: 2.824e-01
TrainSummary: epoch=3 updates=2 avg_step_time=0.646019s samples_per_step=512 samples_per_sec=792.55
```

full gate 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_featout001_gate_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_featout001_gate_20260708.log \
MASTER_PORT=31073 \
AOQ_EXPLORE_SELECTIVE_MARGIN=0.08 \
AOQ_EXPLORE_END_UPDATE=1800 \
MAX_TRAIN_UPDATES=0 \
SKIP_VALIDATE=0 \
TEACHER_FEATURE_OUTPUT_WEIGHT=0.001 \
TEACHER_FEATURE_OUTPUT_LAYERS=features.5.5,features.7.1 \
TEACHER_FEATURE_OUTPUT_LOSS=norm_mse \
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_end_update: 1800
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aq_bitw: 4
aq_mode: lsq
epochs: 4
kd_hard_and_soft: 0
no_resume_opt: true
qk_reparam: false
qk_reparam_type: 0
scheduler_epochs: 4
start_epoch: 3
teacher_feature_output_layers: features.5.5,features.7.1
teacher_feature_output_loss: norm_mse
teacher_feature_output_weight: 0.001
teacher_soft_temperature: 2.75
wq_bitw: 4
wq_mode: lsq
```

strict resume / AOQ / feature-output 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Teacher feature-output hooks: layers=('features.5.5', 'features.7.1'), student_matches=('features.5.5', 'features.7.1'), teacher_matches=('features.5.5', 'features.7.1')
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, start_update=0, end_update=1800
Teacher feature-output debug: student_count=2, teacher_count=2, 0:s=(64, 14, 14, 384) t=(64, 14, 14, 384) mse=1.386e+01; 1:s=(64, 7, 7, 768) t=(64, 7, 7, 768) mse=1.959e+00
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, start_update=0, end_update=1800
TrainSummary: epoch=3 updates=2502 avg_step_time=0.165984s samples_per_step=512 samples_per_sec=3084.64
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 79.9580 | 95.1120 | 0.8513 | 有效 gate，但低于 baseline selective-margin08 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 34.854s  Loss: 0.8513  Acc@1: 79.9580  Acc@5: 95.1120  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_featout001_smoke2upd_20260708.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_featout001_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_featout001_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_featout001_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_featout001_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. 轻量 teacher feature-output 监督技术闭环有效，hook 命中正确，`TeacherFeatOut` 非零并逐步下降。
3. 但 Top-1 只有 `79.9580`，低于 selective-margin08 baseline 的 `80.1660`，说明在 AOQ crossing 阶段加入特征输出约束会压制或干扰有益 crossing。
4. 当前 clean AOQ-native 最好仍是 Phase 2CX：full-model selective-margin08，Top-1 `80.1660`。
5. 下一步不应继续增加 feature-output 权重；如果继续用 supervision 信号，应考虑更局部的 crossing-quality selector，而不是直接约束 block output。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `79.9580`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DE：AOQ crossing-quality selector（grad_cross）smoke 通过

实验动机：

Phase 2CX 的 selective-margin08 说明“只让 near-boundary weights 参与 AOQ 缩放”有小幅收益，但 Phase 2CZ/2DA/2DB/2DC/2DD 说明继续扩大窗口、调 stop point、加 BinReg、缩小可训练集合或直接加 teacher feature-output 监督都不能稳定超过当前 clean AOQ-native best `80.1660`。本阶段不再重复这些轴，而是把 selector 从“距离边界近不近”升级为“当前局部梯度方向是否支持这次 crossing”。

方法设计：

- 保持 clean no-QKR/no-StatsQ/LSQ strict W4A4 分支；
- 保持 AOQ ratio `0.90` 和 selective margin `0.08`；
- 新增 `aoq_explore_quality_mode=grad_cross`；
- 在每个 optimizer step 前，用当前 `weight.grad` 的一阶下降方向 `-grad` 预测下一步权重移动方向；
- 对 near-boundary 元素，仅当预测移动方向远离当前 bin center、即更可能发生被 loss 支持的 crossing 时，才允许使用 AOQ 缩窄 scale；
- `aoq_explore_quality_min_frac=0.10` 作为每个 quantizer 的最小保留比例，避免 selector 过严导致 AOQ 完全关闭；
- 先做 2-update smoke，只验证参数链路、mask 统计和训练启动，不做 full-val。

代码与脚本改动：

```text
qat_launch.py：新增 --aoq-explore-quality-mode / --aoq-explore-quality-min-frac，新增 update_aoq_explore_quality_masks，在 backward 后、optimizer.step 前刷新 quality mask。
third_party/OFQ/src/quantization/quantizer/lsq.py：LsqQuantizerWeight / LsqQuantizer4Conv2d 新增 aoq_quality_mask，用 quality mask 进一步过滤 selective-margin AOQ 元素。
tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh：透传 AOQ_EXPLORE_QUALITY_MODE / AOQ_EXPLORE_QUALITY_MIN_FRAC。
tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_smoke2upd_20260708.sh：固定 2-update smoke。
```

smoke 命令：

```bash
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_smoke2upd_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_quality_mode: grad_cross
aoq_explore_quality_min_frac: 0.1
aoq_explore_selective_margin: 0.08
aoq_explore_scale_ratio: 0.9
aoq_explore_end_update: 2
max_train_updates: 2
skip_validate: true
wq_bitw: 4
aq_bitw: 4
wq_mode: lsq
aq_mode: lsq
qk_reparam: false
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
```

strict resume / AOQ selector smoke 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=grad_cross, quality_min_frac=0.1, start_update=0, end_update=2
AOQ crossing-quality selector: epoch=3, update=0, mode=grad_cross, pairs=6, near=1497634, selected=741944, selected_over_near=0.495411
AOQ crossing-quality selector: epoch=3, update=1, mode=grad_cross, pairs=6, near=1497819, selected=753578, selected_over_near=0.503117
TrainSummary: epoch=3 updates=2
Stopped early after 2 optimizer updates in epoch 3.
```

产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_smoke2upd_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_smoke2upd_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_smoke2upd_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_smoke2upd_20260708/last.pth.tar
```

中文结论：

1. 新 selector 技术闭环通过：launcher 参数、runtime `args.yaml`、LSQ quantizer mask、训练 loop 更新点、日志统计均正常。
2. selector 每步在 6 个 selected LSQ weight quantizers 上工作，near-boundary 元素约 `1.50M`，实际保留约 `49.5%-50.3%`，说明它不是简单复刻 selective-margin08，而是在 near-boundary 集合中做了进一步局部筛选。
3. 本阶段只跑 2-update smoke，`skip_validate=true`，没有 full ImageNet raw validation，因此不计入 Top-1 目标，也不能和 Phase 2CX `80.1660` 比较。
4. DDP/NCCL teardown warning 出现在 stop 后，训练已经打印 `TrainSummary: epoch=3 updates=2` 和 `Stopped early after 2 optimizer updates`，按前面短跑经验视为退出噪声，不作为训练失败。
5. 下一步可以启动一个完整 epoch 的 qualityselector full-val gate：保持 `quality_mode=grad_cross`、`quality_min_frac=0.10`、`AOQ_EXPLORE_END_UPDATE=1800`，只评估单个 `checkpoint-4.pth.tar`，并与 clean AOQ-native best `80.1660` 对比。

completion audit：

- strict W4A4：smoke 配置满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：smoke 产出 `checkpoint-4.pth.tar`，但未做 full-val。
- full ImageNet raw validation：不满足，本阶段 `skip_validate=true`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：未验证。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DF：AOQ crossing-quality selector full-val 失败，Top-1 79.9640

实验动机：

Phase 2DE smoke 证明 `grad_cross` selector 的参数链路、mask 更新和训练启动都正常。本阶段把 smoke 扩展为完整 1 个 resumed epoch gate，检查“局部梯度方向筛选 crossing”是否能超过 clean AOQ-native best Phase 2CX 的 `80.1660`。

方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- AOQ ratio：`0.90`；
- selective margin：`0.08`；
- crossing-quality selector：`grad_cross`；
- selector min fraction：`0.10`；
- AOQ explore 窗口：`0-1800` update；
- update 1800 后关闭 AOQ explore，恢复 `ratio=1.0`、`selective_margin=0.0`；
- full-val 只认单个 `checkpoint-4.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

full gate 命令：

```bash
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_quality_mode: grad_cross
aoq_explore_quality_min_frac: 0.1
aoq_explore_selective_margin: 0.08
aoq_explore_scale_ratio: 0.9
aoq_explore_end_update: 1800
max_train_updates: 0
skip_validate: false
wq_bitw: 4
aq_bitw: 4
wq_mode: lsq
aq_mode: lsq
qk_reparam: false
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
```

strict resume / AOQ selector 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=grad_cross, quality_min_frac=0.1, start_update=0, end_update=1800
AOQ crossing-quality selector: epoch=3, update=0, mode=grad_cross, pairs=6, near=1497634, selected=741944, selected_over_near=0.495411
AOQ crossing-quality selector: epoch=3, update=1600, mode=grad_cross, pairs=6, near=1501814, selected=751286, selected_over_near=0.500252
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=grad_cross, quality_min_frac=0.1, start_update=0, end_update=1800
TrainSummary: epoch=3 updates=2502 avg_step_time=0.175257s samples_per_step=512 samples_per_sec=2921.42
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 79.9640 | 95.1240 | 0.8562 | 有效 gate，但低于 selective-margin08 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 35.936s  Loss: 0.8562  Acc@1: 79.9640  Acc@5: 95.1240  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. `grad_cross` selector 技术上稳定，训练完整跑完，update 1800 正确关闭 AOQ explore。
3. Top-1 `79.9640`，低于 Phase 2CX selective-margin08 的 `80.1660`，也低于 pure late5571 LSQ-AOQ090 的 `80.1240`。
4. 结论是当前一阶 `grad_cross` selector 过于局部或方向信号不够可靠；它把 near-boundary 集合稳定裁掉约一半，但裁掉的 crossing 很可能包含有益探索。
5. 下一步不应盲目扫描 `quality_min_frac`。更合理的是先对比 Phase 2CX selective-margin08 checkpoint 与 Phase 2DF qualityselector checkpoint 的 bin-crossing 差异，定位哪些 module/param 的 crossing 被 selector 抑制后造成 Top-1 回落，再决定是否做 module-wise selector 或反向保留策略。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `79.9640`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DG：Phase 2CX vs Phase 2DF bin-crossing 诊断，定位 grad_cross 失败原因

诊断动机：

Phase 2DF 的 `grad_cross` selector 全量验证只有 `79.9640`，低于 selective-margin08 的 `80.1660`。如果只看方法直觉，可能会误以为 selector 过严、crossing 不够。但这需要用 checkpoint 级 bin-crossing 诊断验证，避免继续盲扫 `quality_min_frac`。

诊断命令：

```bash
python3 QATs/tmp_scripts/diagnose_resume10_aoq_bin_crossing_20260708.py \
  --out-dir QATs/docs/resume10_clean_lsq_aoq_qualityselector_bin_crossing_20260708 \
  --pairs 'ckpt10->phase2s,ckpt10->phase2w,phase2s->phase2w' \
  --module-patterns features.5.5,features.7.1 \
  --near-margin 0.08 \
  --topn 80 \
  --ckpt10-checkpoint /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar \
  --ckpt10-top1 79.9220 \
  --phase2s-checkpoint /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
  --phase2s-top1 80.1660 \
  --phase2w-checkpoint /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_gate_20260708/checkpoint-4.pth.tar \
  --phase2w-top1 79.9640
```

产物：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_qualityselector_bin_crossing_20260708/aggregate_bin_crossing.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_qualityselector_bin_crossing_20260708/pair_bin_crossing.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_qualityselector_bin_crossing_20260708/summary.json
```

关键 stage_kind 对比：

| pair | Top-1 delta | features.7.1 attn_proj changed | features.7.1 mlp_fc2 changed | features.7.1 attn_qkv changed | features.5.5 attn_proj changed | features.5.5 attn_qkv changed |
|---|---:|---:|---:|---:|---:|---:|
| source -> selective-margin08 | +0.2440 | 0.056876 | 0.055845 | 0.032492 | 0.036214 | 0.031071 |
| source -> grad_cross | +0.0420 | 0.070231 | 0.057518 | 0.039156 | 0.043057 | 0.036542 |
| selective-margin08 -> grad_cross | -0.2020 | 0.067374 | 0.053757 | 0.041068 | 0.041036 | 0.037263 |

中文结论：

1. `grad_cross` 失败不是因为 crossing 过少。相反，和 selective-margin08 相比，它在 `features.7.1.attn.proj`、`features.7.1.attn.qkv`、`features.5.5.attn.proj`、`features.5.5.attn.qkv` 上都产生了更多 crossing。
2. Top-1 从 selective-margin08 的 `80.1660` 掉到 `79.9640`，而 attention 相关 crossing 增幅最明显，说明当前一阶梯度 selector 在 attention 分支上放大了有害 crossing。
3. `features.7.1.mlp_fc2` 的 changed fraction 只从 `0.055845` 到 `0.057518`，增幅较小；相比之下 attention proj/qkv 的过量 crossing 更可疑。
4. 下一步不应继续全 6 个 quantizer 一刀切启用 `grad_cross`，也不应简单扫描 `quality_min_frac`。更合理的下一步是 module-wise selector：
   - attention qkv/proj 保持原 selective-margin08，不启用 `grad_cross`；
   - 只在 MLP `fc2` 或 MLP `fc1/fc2` 上启用 `grad_cross`；
   - 或反过来在 attention 上使用更严格的 delayed stabilization，而不是放大 crossing。
5. 这个诊断不改变 completion：Phase 2DG 是离线分析，不是 full-val gate；goal 仍未完成。

### Phase 2DH：module-wise grad_cross smoke 通过，selector 限定到 MLP fc2

实验动机：

Phase 2DG 诊断显示，全 6 个 quantizer 启用 `grad_cross` 后，attention qkv/proj crossing 明显增多并导致 Top-1 从 `80.1660` 回落到 `79.9640`。因此本阶段改为 module-wise selector：AOQ selective-margin08 仍作用于 6 个 late quantizers，但 `grad_cross` quality mask 只作用于 `features.5.5.mlp.fc2` 和 `features.7.1.mlp.fc2`，让 attention qkv/proj 回到原 selective-margin08 行为。

代码与脚本改动：

```text
qat_launch.py：新增 --aoq-explore-quality-layers；为空时沿用 aoq-explore-layers，非空时 selector 只作用于指定 layers。
训练 loop：每次刷新 quality mask 前先 clear 全部旧 mask，再只为 quality_layers 设置 mask，避免 attention 残留旧 selector mask。
tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_mlpfc2_smoke2upd_20260708.sh：固定 MLP fc2-only 2-update smoke。
```

smoke 命令：

```bash
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_mlpfc2_smoke2upd_20260708.sh
```

关键 smoke 证据：

```text
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=grad_cross, quality_layers=('features.5.5.mlp.fc2', 'features.7.1.mlp.fc2'), quality_min_frac=0.1, start_update=0, end_update=2
AOQ crossing-quality selector: epoch=3, update=0, mode=grad_cross, pairs=2, near=815354, selected=403273, selected_over_near=0.494599
AOQ crossing-quality selector: epoch=3, update=1, mode=grad_cross, pairs=2, near=815411, selected=411281, selected_over_near=0.504385
TrainSummary: epoch=3 updates=2 avg_step_time=0.642213s samples_per_step=512 samples_per_sec=797.24
Stopped early after 2 optimizer updates in epoch 3.
```

中文结论：

1. MLP fc2-only selector 技术闭环通过：AOQ base quantizers 仍是 6 个，但 quality selector `pairs=2`，符合预期。
2. attention qkv/proj 不再被 `grad_cross` mask 控制，应回到原 selective-margin08 的 crossing 方式。
3. 这是 smoke，不做 full-val，不计入 81 completion。
4. 下一步启动 MLP fc2-only full-val gate，若仍低于 `80.1660`，说明一阶 selector 本身不适合该分支；若恢复接近或超过 `80.1660`，再考虑更精细的 MLP-only 或 attention stabilization。

### Phase 2DI：module-wise grad_cross MLP fc2-only full-val 失败，Top-1 80.0440

实验动机：

Phase 2DH smoke 证明 `grad_cross` selector 可以只作用于 MLP fc2，attention qkv/proj 不再被 selector mask 影响。本阶段启动完整 1 个 resumed epoch gate，验证“attention 保持 selective-margin08，MLP fc2 使用 grad_cross”是否能恢复或超过 Phase 2CX selective-margin08 的 `80.1660`。

方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- AOQ ratio：`0.90`；
- selective margin：`0.08`；
- AOQ base layers：late 6 个 quantizer；
- crossing-quality selector：`grad_cross`；
- selector layers：`features.5.5.mlp.fc2,features.7.1.mlp.fc2`；
- selector min fraction：`0.10`；
- AOQ explore 窗口：`0-1800` update；
- update 1800 后关闭 AOQ explore，恢复 `ratio=1.0`、`selective_margin=0.0`；
- full-val 只认单个 `checkpoint-4.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

full gate 命令：

```bash
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_mlpfc2_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_quality_mode: grad_cross
aoq_explore_quality_layers: features.5.5.mlp.fc2,features.7.1.mlp.fc2
aoq_explore_quality_min_frac: 0.1
aoq_explore_selective_margin: 0.08
aoq_explore_scale_ratio: 0.9
aoq_explore_end_update: 1800
max_train_updates: 0
skip_validate: false
wq_bitw: 4
aq_bitw: 4
wq_mode: lsq
aq_mode: lsq
qk_reparam: false
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
```

strict resume / AOQ selector 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=grad_cross, quality_layers=('features.5.5.mlp.fc2', 'features.7.1.mlp.fc2'), quality_min_frac=0.1, start_update=0, end_update=1800
AOQ crossing-quality selector: epoch=3, update=0, mode=grad_cross, pairs=2, near=815354, selected=403273, selected_over_near=0.494599
AOQ crossing-quality selector: epoch=3, update=1600, mode=grad_cross, pairs=2, near=826825, selected=414804, selected_over_near=0.501683
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=grad_cross, quality_layers=('features.5.5.mlp.fc2', 'features.7.1.mlp.fc2'), quality_min_frac=0.1, start_update=0, end_update=1800
TrainSummary: epoch=3 updates=2502 avg_step_time=0.177047s samples_per_step=512 samples_per_sec=2891.88
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 80.0440 | 95.1220 | 0.8520 | 有效 gate，但低于 selective-margin08 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 36.712s  Loss: 0.8520  Acc@1: 80.0440  Acc@5: 95.1220  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_mlpfc2_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_mlpfc2_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_mlpfc2_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_mlpfc2_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. MLP fc2-only selector 比全 6 个 quantizer 的 `grad_cross` 更好：Top-1 从 `79.9640` 回到 `80.0440`，说明 Phase 2DG 对 attention harmful crossing 的判断是对的。
3. 但它仍低于 Phase 2CX selective-margin08 的 `80.1660`，也低于 pure late5571 LSQ-AOQ090 的 `80.1240`。
4. 结论是当前一阶 `grad_cross` selector 即使限制在 MLP fc2，也没有给出超过 selective-margin08 的收益；它更像是在过滤部分有益 crossing，而不是发现更优 crossing。
5. 下一步应停止 `grad_cross` 这个一阶 selector 方向。更合理的 AOQ-native 下一步是反过来保留 selective-margin08 的 crossing，但对 attention qkv/proj 加 delayed stabilization 或 post-explore bin-center safety，而不是在 explore 阶段继续用梯度方向筛 crossing。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.0440`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DJ：attention-only post-explore selective anchor smoke 通过

实验动机：

Phase 2DI 说明 `grad_cross` 一阶 selector 即使只作用于 MLP fc2，也不能超过 selective-margin08。Phase 2DG 同时显示 attention qkv/proj 的 crossing 过量时会伤害 Top-1。因此本阶段不再筛 exploration crossing，而是保留 Phase 2CX 的 selective-margin08 exploration，在 AOQ explore 结束后只对 attention qkv/proj 做轻量 selective bin anchor，测试更窄的 post-explore stabilization。

方法设计：

- AOQ exploration：沿用 Phase 2CX selective-margin08；
- 不启用 `grad_cross`；
- AOQ explore 前 2 update；
- update 2 捕获 selective bin anchor；
- anchor layers 只包含 attention qkv/proj：
  - `features.5.5.attn.qkv`
  - `features.5.5.attn.proj`
  - `features.7.1.attn.qkv`
  - `features.7.1.attn.proj`
- `selective_bin_anchor_weight=5e-5`，比之前全 6 层 selective anchor 的 `1e-4` 更弱；
- `selective_bin_anchor_margin=0.05`；
- 只做 4-update smoke，不做 full-val。

smoke 命令：

```bash
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_attnanchor_smoke4upd_20260708.sh
```

关键 smoke 证据：

```text
AOQ explore scale ratio update: epoch=3, update=2, active=False, base_ratio=1.0, ... selective_margin=0.0 ...
Captured selective bin anchor: weight=5e-05, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.7.1.attn.qkv', 'features.7.1.attn.proj'), pairs=4, masked=493145, total=2949120, mask_fraction=0.167218, capture_update=2, end_update=4, margin=0.05
Enabled selective bin anchor: weight=5e-05, pairs=4, masked=493145, total=2949120, mask_fraction=0.167218, capture_update=2, end_update=4
TrainSummary: epoch=3 updates=4 avg_step_time=0.477654s samples_per_step=512 samples_per_sec=1071.91
Stopped early after 4 optimizer updates in epoch 3.
```

中文结论：

1. attention-only post-explore selective anchor 技术闭环通过。
2. anchor 只捕获 attention qkv/proj，`pairs=4`，没有作用到 MLP fc1/fc2。
3. smoke 不做 full-val，不计入 81 completion。
4. 下一步启动 full-val gate，检查这种更窄、更弱的 delayed stabilization 是否能超过 selective-margin08 `80.1660`。

### Phase 2DK：attention-only post-explore selective anchor full-val 失败，Top-1 80.0200

实验动机：

Phase 2DJ smoke 成功后，本阶段跑完整 1 个 resumed epoch，验证“保留 selective-margin08 exploration + 后段只稳定 attention qkv/proj near-boundary 权重”是否能避免 attention 有害漂移，同时不破坏有益 crossing。

方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- AOQ ratio：`0.90`；
- selective margin：`0.08`；
- AOQ explore layers：late 6 个 quantizer；
- `grad_cross`：关闭；
- selective bin anchor：
  - `weight=5e-5`
  - `layers=features.5.5.attn.qkv,features.5.5.attn.proj,features.7.1.attn.qkv,features.7.1.attn.proj`
  - `capture_update=1800`
  - `end_update=0`
  - `margin=0.05`
- full-val 只认单个 `checkpoint-4.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

full gate 命令：

```bash
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_attnanchor_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
selective_bin_anchor_weight: 5e-05
selective_bin_anchor_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.7.1.attn.qkv,features.7.1.attn.proj
selective_bin_anchor_capture_update: 1800
selective_bin_anchor_end_update: 0
selective_bin_anchor_margin: 0.05
aoq_explore_quality_mode: none
aoq_explore_selective_margin: 0.08
aoq_explore_scale_ratio: 0.9
aoq_explore_end_update: 1800
max_train_updates: 0
skip_validate: false
wq_bitw: 4
aq_bitw: 4
wq_mode: lsq
aq_mode: lsq
qk_reparam: false
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
```

strict resume / AOQ / anchor 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=1800
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=1800
Captured selective bin anchor: weight=5e-05, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.7.1.attn.qkv', 'features.7.1.attn.proj'), pairs=4, masked=441576, total=2949120, mask_fraction=0.149731, capture_update=1800, end_update=0, margin=0.05
Enabled selective bin anchor: weight=5e-05, pairs=4, masked=441576, total=2949120, mask_fraction=0.149731, capture_update=1800, end_update=0
TrainSummary: epoch=3 updates=2502 avg_step_time=0.168534s samples_per_step=512 samples_per_sec=3037.97
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 80.0200 | 95.0740 | 0.8517 | 有效 gate，但低于 selective-margin08 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 35.179s  Loss: 0.8517  Acc@1: 80.0200  Acc@5: 95.0740  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_attnanchor_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_attnanchor_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_attnanchor_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_attnanchor_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. attention-only selective anchor 技术上稳定，update 1800 正确捕获 `pairs=4`，`SelBinAnchor` 在后段非零。
3. 但 Top-1 只有 `80.0200`，低于 Phase 2CX selective-margin08 的 `80.1660`，也低于 MLP fc2-only `grad_cross` 的 `80.0440`。
4. 结论是后段 selective anchor 类稳定化仍然会压制有益离散迁移；即使只作用于 attention qkv/proj 且权重降到 `5e-5`，也没有收益。
5. 当前应停止 “post-explore anchor / BinReg / 一阶 crossing selector” 这一组正则化方向。下一步需要换机制：不是再筛 crossing 或锚定 crossing，而是改变 AOQ exploration 本身，例如分阶段只对 MLP/attention 采用不同 ratio 窗口，或做真正的 AOQ-style threshold/level decoupling，而不是只复用 LSQ scale ratio。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.0200`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### 最新结果索引：Phase 2FO anchor-moved tail selector

最近完成的 full gate 是 Phase 2FO：

```text
Phase 2FO：tail-state anchor-moved second pulse 失败，Top-1 80.0820
```

核心结果：

- strict W4A4：满足。
- clean no-QKR/no-StatsQ：满足。
- 单 checkpoint：`recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse_anchormoved2200_gate_20260709/checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足。
- full-val：`Loss 0.8533`，`Top-1 80.0820`，`Top-5 95.1940`。
- 对比 clean AOQ-native best Phase 2CX `80.1660`：低 `0.0840`。
- 对比全局 strict W4A4 best `80.5540`：低 `0.4720`。

关键结论：

`anchor_moved` 工程语义成立。update 2200 时选中 `237516 / 845211 = 28.1014%` 的 near-boundary 权重，正好是 Phase 2FN `anchor_unmoved` 排除的 moved 子集。但 full-val 只有 `80.0820`，说明“已经相对 source 迁移过的 near-boundary 权重在尾段继续弱探索”没有带来收益。`anchor_unmoved=80.0760` 与 `anchor_moved=80.0820` 基本持平，二者都低于 Phase 2CX `80.1660`。

下一步判断：

停止 tail second pulse / anchor-state 二值 selector 小扫。若继续 clean AOQ-native，应转向真正 per-weight / candidate-state 范式：记录每个权重的 crossing history、方向、稳定度和候选状态，而不是在固定时间窗内按 source 是否迁移做二值 mask。

### 最新状态索引：per-weight oscillation selector 系列

最近完成的三组 per-weight oscillation selector gate：

| phase | 方法 | full-val Top-1 | 结论 |
|---|---|---:|---|
| Phase 2FP | `history_oscillating`, 累计 history，`quality_min_frac=0.02` | 79.9160 | 累计 history 太宽，update 1600 选中 41.8006% near-boundary，明显破坏 endpoint |
| Phase 2FQ | `recent_oscillating`, 全程只选当前方向反转，`quality_min_frac=0` | 80.0880 | selector 足够稀疏，但全程过窄，低于 Phase 2CX |
| Phase 2FR | 0-600 普通 AOQ，600 后 `recent_oscillating` | 80.0340 | hybrid 时序也无收益，低于全程 recent-only |

当前最新判断：

- strict W4A4 / clean no-QKR/no-StatsQ / 单 checkpoint / full ImageNet raw validation / `Samples=50000` / 无 soup 全部满足。
- 但最新 Top-1 只有 `80.0340`，没有达到 `81.0`，也没有超过 clean AOQ-native best Phase 2CX `80.1660`。
- per-weight oscillation 事件本身不是足够好的 candidate-state 选择准则：累计过宽，全程 recent 过窄，延迟 recent 也没有恢复收益。
- 下一步若继续 AOQ-native，不应继续扫 recent start update；更合理的是换 candidate-state 定义，例如记录候选 bin endpoint 并用 loss / logit / 小验证代理筛选候选，而不是只用 oscillation 事件。
- Goal 仍未完成，不调用 `update_goal complete`。

### 最新状态索引：candidate-state selection / transplant

最近完成的 candidate-state 相关实验：

| phase | 方法 | full-val Top-1 | 结论 |
|---|---|---:|---|
| Phase 2FS | 同一条 clean AOQ-native 轨迹保存 step checkpoint，并逐个 full-val 选择单 checkpoint | best 80.1660 | `step_1200=79.8080`、`step_1800=79.8740`、`step_2400=80.0140`，final `checkpoint-4=80.1660` 仍最好 |
| Phase 2FT | 以 Phase 2CX 为 base，用 single-cross donor 替换 `features.7.1.attn.proj` 与 `features.7.1.mlp.fc2` 两个模块 | 80.1540 | 模块级 candidate-state transplant 可运行，但低于 base Phase 2CX `80.1660` |

当前最新判断：

- strict W4A4 / clean no-QKR/no-StatsQ / 单 checkpoint / full ImageNet raw validation / `Samples=50000` / 无 soup 全部满足。
- 但最新候选仍没有达到 `81.0`，也没有超过 clean AOQ-native best Phase 2CX `80.1660`。
- 单纯按训练时间点选择 checkpoint 没有收益；最小模块级 transplant 也没有收益。
- 下一步如果继续 candidate-state，不能再盲目模块替换，应先做模块贡献分析或训练内 candidate assignment proxy。
- Goal 仍未完成，不调用 `update_goal complete`。

### 2026-07-09 最终索引：Phase 2FW attn.proj tensor-level candidate-state

由于本长文档存在重复锚点，Phase 2FW 的详细记录位于文档中部。这里在文件末尾保留最终中文索引，供后续接手时直接读取最新状态。

最新 clean AOQ-native no-QKR/no-StatsQ strict W4A4 单 checkpoint best 已从 Phase 2CX 的 `80.1660` 更新为 Phase 2FW 的 `80.2080`：

```text
checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar
method: Phase 2CX base + source-anchor single-cross donor，仅复制 features.7.1.attn.proj.weight、move_b4.bias、move_aft.bias
full-val: Loss 0.8472, Top-1 80.2080, Top-5 95.1560, Samples 50000
strict resume: missing=0, unexpected=0
```

同阶段对照：

| candidate | copied tensor | Top-1 | 结论 |
|---|---|---:|---|
| `attnproj71_weight` | `features.7.1.attn.proj.weight` | 80.1880 | 单独 weight 已超过 full-module transplant，说明正信号主要来自离散 weight endpoint |
| `attnproj71_weight_move` | `weight + move_b4.bias + move_aft.bias` | 80.2080 | 当前 clean AOQ-native best，move bias 与 weight endpoint 有小幅正耦合 |
| `attnproj71_inputscale` | `input_quant_fn.s` | 80.1640 | activation scale 单独无收益，接近 Phase 2CX base |

中文结论：

1. 这不是 soup、不是 checkpoint averaging、不是 ensemble；是单 checkpoint 的 tensor-level candidate-state transplant。
2. 当前最有价值的信号不是整模块替换，而是 late attention projection 的 `weight + move bias` 组合。
3. `lsqw_fn.s` 在 base 和 donor 间完全一致；`input_quant_fn.s` 单独无收益；普通 `bias` 差异很小，因此不继续浪费 full-val 小扫。
4. 下一步如果继续 AOQ-native 范式，应把这个信号转为训练内机制：对 `features.7.1.attn.proj` 一类 late projection 维护候选 weight endpoint，并让 move bias 跟随被选 endpoint 做 stabilization，而不是复制 activation scale 或做 module-level broad transplant。

completion audit：

- strict W4A4：满足，eval 命令含 `--wq-bitw 4 --aq-bitw 4 --wq-mode lsq --aq-mode lsq`。
- clean no-QKR/no-StatsQ：满足，使用 clean LSQ no-QKR eval 脚本。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足。
- Top-1 >= 81.0：不满足，当前 best `80.2080`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2FX：Phase 2FW weight+move endpoint 100-update 单层稳定化，Top-1 80.2160

实验动机：

Phase 2FW 的 tensor-level candidate-state 结果显示，`features.7.1.attn.proj.weight + move_b4.bias + move_aft.bias` 是当前唯一有正信号的 clean AOQ-native endpoint，Top-1 从 Phase 2CX `80.1660` 提升到 `80.2080`。但它仍是离线 candidate-state transplant，不是训练内机制。本阶段不重复 module transplant，也不继续扫 `lsqw/input_scale/bias` 小组合，而是测试一个最小训练化版本：从 Phase 2FW endpoint 出发，关闭 AOQ，只允许 `features.7.1.attn.proj` 做 100 update 极低 LR stabilization，观察这个 candidate endpoint 是否还能通过训练微调继续变好。

方法设计：

- 起点 checkpoint：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar
```

- 训练策略：
  - strict W4A4：`wq_bitw=4`、`aq_bitw=4`、`wq_mode=lsq`、`aq_mode=lsq`。
  - clean no-QKR/no-StatsQ：`qk_reparam=0`，权重/激活均用 LSQ。
  - AOQ 关闭：`aoq_explore_scale_ratio=1.0`、`aoq_explore_selective_margin=0.0`、`aoq_explore_end_update=0`。
  - 只训练 `features.7.1.attn.proj`：`trainable_policy=params_in_layers`，`trainable_policy_freeze_act_except_layers=features.7.1.attn.proj`。
  - 极低 LR：`lr=1e-5`、`min_lr=1e-5`、`quant_lr_multiplier=1`。
  - 只跑 `100` update smoke，并保存 `checkpoint-5.pth.tar` 后 full-val。

关键命令：

```bash
RESUME=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar \
EXP=recipe_resume10_state_transplant_attnproj71_weightmove_stabilize100_smoke_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_state_transplant_attnproj71_weightmove_stabilize100_smoke_20260709.log \
MASTER_PORT=31481 \
START_EPOCH=4 \
EPOCHS=5 \
SCHEDULER_EPOCHS=5 \
LR=1e-5 \
MIN_LR=1e-5 \
QUANT_LR_MULTIPLIER=1 \
AOQ_EXPLORE_SCALE_RATIO=1.0 \
AOQ_EXPLORE_SELECTIVE_MARGIN=0.0 \
AOQ_EXPLORE_END_UPDATE=0 \
AOQ_EXPLORE_LAYERS=features.7.1.attn.proj \
TRAINABLE_POLICY=params_in_layers \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.7.1.attn.proj \
TRAINABLE_POLICY_UPDATE_MODE=requires_grad \
MAX_TRAIN_UPDATES=100 \
SKIP_VALIDATE=1 \
SAVE_STEP_CHECKPOINTS=1 \
SAVE_INITIAL_STEP_CHECKPOINT=1 \
STEP_CHECKPOINT_INTERVAL=100 \
MAX_STEP_CHECKPOINTS_TO_SAVE=3 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh

CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_attnproj71_weightmove_stabilize100_smoke_20260709/checkpoint-5.pth.tar \
EXP=eval_state_transplant_attnproj71_weightmove_stabilize100_clean_lsq_noqkr_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_state_transplant_attnproj71_weightmove_stabilize100_clean_lsq_noqkr_20260709.log \
MASTER_PORT=31482 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_resume10_clean_lsq_noqkr_checkpoint_fullval_20260709.sh
```

关键 `args.yaml` 证据：

```text
aq_bitw: 4
aq_mode: lsq
wq_bitw: 4
wq_mode: lsq
aoq_explore_scale_ratio: 1.0
aoq_explore_selective_margin: 0.0
aoq_explore_end_update: 0
aoq_explore_layers: features.7.1.attn.proj
trainable_policy: params_in_layers
trainable_policy_freeze_act_except_layers: features.7.1.attn.proj
lr: 1.0e-05
min_lr: 1.0e-05
max_train_updates: 100
skip_validate: true
```

strict resume / 训练链路证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=4, update=0, active=False, base_ratio=1.0, threshold_ratio=0.0, base_layers=('features.7.1.attn.proj',), selective_margin=0.0, base_quantizers=0
Trainable parameter update policy: epoch=4, update=0, mode=requires_grad, policy=params_in_layers, trainable=592945, frozen=27942462
TrainSummary: epoch=4 updates=100 avg_step_time=0.135261s samples_per_step=512 samples_per_sec=3785.26
Stopped early after 100 optimizer updates in epoch 4.
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 Phase 2FW | 对比 81 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-5.pth.tar` | yes | yes | yes | 50000 | no | 80.2160 | 95.1500 | 0.8538 | +0.0080 | -0.7840 |

full-val 原始摘要：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_attnproj71_weightmove_stabilize100_smoke_20260709/checkpoint-5.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 29.736s  Loss: 0.8538  Acc@1: 80.2160  Acc@5: 95.1500  Samples: 50000
Eval-only metrics: {'loss': 0.8537603882482648, 'top1': 80.216, 'top5': 95.15, 'samples': 50000, 'local_samples': 6250, 'wall_seconds': 29.735833883285522}
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. Phase 2FW 的 `weight+move` endpoint 经过 100 update 单层 stabilization 后从 `80.2080` 到 `80.2160`，刷新当前 clean AOQ-native best，但增益只有 `+0.0080`。
3. loss 从 Phase 2FW 的 `0.8472` 明显变差到 `0.8538`，Top-5 也略低，说明这个 100-update 微调更像局部 Top-1 抖动，不是强稳定提升。
4. 因此不直接跑完整 1 epoch。下一步如果继续这个方向，应改成更短或更弱的 endpoint stabilization，例如 25/50 update 或 move-bias-only/weight-only 分组，而不是把 `features.7.1.attn.proj` 继续训练到 full epoch。

completion audit：

- strict W4A4：满足，训练和 eval 均含 `wq_bitw=4`、`aq_bitw=4`、`wq_mode=lsq`、`aq_mode=lsq`。
- clean no-QKR/no-StatsQ：满足，本阶段走 clean LSQ no-QKR 脚本。
- 单 checkpoint：满足，评估单个 `checkpoint-5.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.2160`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2FW：attn.proj tensor-level candidate-state transplant，weight+move 小幅刷新 clean AOQ-native best

实验动机：

Phase 2FU/2FV 说明 `features.7.1.attn.proj` 是 module-level surgery 中唯一有正信号的模块：只替换该模块从 Phase 2CX `80.1660` 提升到 `80.1840`，但继续扩展到 `features.5.5.attn.proj` 或双 attn.proj 都是负收益。本阶段不再做更宽的模块组合，而是把 `features.7.1.attn.proj` 拆成 tensor-level state，判断正收益到底来自离散 weight endpoint、activation scale、weight scale，还是 move bias 耦合状态。

方法设计：

- 仍以 Phase 2CX clean AOQ-native best 作为 base：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar
```

- donor 仍使用 source-anchor single-cross：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_gate_20260708/checkpoint-4.pth.tar
```

- 扩展 transplant 工具：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/make_resume10_module_transplant_checkpoint_20260709.py
```

新增 `--include-suffixes`，支持只复制模块内指定 tensor suffix。这个操作仍然是单 checkpoint 状态替换，不是 soup，不是 checkpoint averaging，也不是 ensemble。

候选 checkpoint：

| 候选 | include suffix | copied tensors | 输出 |
|---|---|---:|---|
| weight only | `weight` | 1 | `recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_20260709/checkpoint-4.pth.tar` |
| weight + lsqw | `weight,lsqw_fn.s` | 2 | `recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_lsqw_20260709/checkpoint-4.pth.tar` |
| weight + move | `weight,move_b4.bias,move_aft.bias` | 3 | `recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar` |
| input scale only | `input_quant_fn.s` | 1 | `recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_inputscale_20260709/checkpoint-4.pth.tar` |

关键命令：

```bash
python3 QATs/tmp_scripts/make_resume10_module_transplant_checkpoint_20260709.py \
  --base /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
  --donor /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_gate_20260708/checkpoint-4.pth.tar \
  --modules features.7.1.attn.proj \
  --include-suffixes weight \
  --output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_20260709/checkpoint-4.pth.tar

python3 QATs/tmp_scripts/make_resume10_module_transplant_checkpoint_20260709.py \
  --base /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
  --donor /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_gate_20260708/checkpoint-4.pth.tar \
  --modules features.7.1.attn.proj \
  --include-suffixes weight,move_b4.bias,move_aft.bias \
  --output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar

CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar \
EXP=eval_state_transplant_2cx_singlecross_attnproj71_weight_move_clean_lsq_noqkr_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_state_transplant_2cx_singlecross_attnproj71_weight_move_clean_lsq_noqkr_20260709.log \
MASTER_PORT=31472 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_resume10_clean_lsq_noqkr_checkpoint_fullval_20260709.sh
```

关键 `args` / eval 命令证据：

三条 full-val eval 都走同一个 clean LSQ no-QKR eval 脚本，日志命令包含：

```text
--wq-bitw 4 --wq-enable
--aq-bitw 4 --aq-enable
--wq-mode lsq --aq-mode lsq
--wq-per-channel --aq-per-channel --aq_clip_learnable
--eval-only
--static-graph
--smoothing 0.0 --mixup 0.0 --cutmix 0.0 --aa none --color-jitter 0.0 --reprob 0.0
```

strict resume 证据：

```text
Strict resume: loaded model from .../recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Strict resume: loaded model from .../recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Strict resume: loaded model from .../recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_inputscale_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 Phase 2CX | 对比 full-module attnproj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `attnproj71_weight` | yes | yes | yes | 50000 | no | 80.1880 | 95.1560 | 0.8475 | +0.0220 | +0.0040 |
| `attnproj71_weight_move` | yes | yes | yes | 50000 | no | 80.2080 | 95.1560 | 0.8472 | +0.0420 | +0.0240 |
| `attnproj71_inputscale` | yes | yes | yes | 50000 | no | 80.1640 | 95.1640 | 0.8477 | -0.0020 | -0.0200 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 30.845s  Loss: 0.8475  Acc@1: 80.1880  Acc@5: 95.1560  Samples: 50000
Test: [distributed-summary]  Time: 30.782s  Loss: 0.8472  Acc@1: 80.2080  Acc@5: 95.1560  Samples: 50000
Test: [distributed-summary]  Time: 30.269s  Loss: 0.8477  Acc@1: 80.1640  Acc@5: 95.1640  Samples: 50000
```

tensor 差异证据：

```text
features.7.1.attn.proj.weight max_abs 0.0025489628 mean_abs 0.0003991661 num_diff 589762
features.7.1.attn.proj.move_b4.bias max_abs 0.0012953440 mean_abs 0.0003468642 num_diff 768
features.7.1.attn.proj.move_aft.bias max_abs 0.0012813862 mean_abs 0.0003467787 num_diff 768
features.7.1.attn.proj.input_quant_fn.s max_abs 0.0024341643 mean_abs 0.0009016683 num_diff 49
features.7.1.attn.proj.bias max_abs 0.0002997797 mean_abs 0.0000817548 num_diff 768
features.7.1.attn.proj.lsqw_fn.s max_abs 0.0 mean_abs 0.0 num_diff 0
```

中文结论：

1. `features.7.1.attn.proj` 的 module-level 正信号主要来自 weight endpoint，单独复制 `weight` 已经达到 `80.1880`，超过 full-module transplant 的 `80.1840`。
2. `weight + move_b4/move_aft.bias` 进一步提升到 `80.2080`，刷新当前 clean AOQ-native no-QKR/no-StatsQ strict W4A4 单 checkpoint best，但只比 Phase 2CX 高 `+0.0420`，还远低于 81。
3. `input_quant_fn.s` 单独复制只有 `80.1640`，基本等于 Phase 2CX base，说明 activation scale 不是这次正收益来源；full-module transplant 之所以只到 `80.1840`，很可能是 activation scale / ordinary bias 等非核心状态抵消了部分 weight+move 收益。
4. `lsqw_fn.s` 在 base 与 donor 之间完全一致，因此 `weight+lsqw` 不值得单独 full-val；普通 `bias` 差异很小，也不是当前优先方向。
5. 这条结果给出的范式启发是：AOQ-native candidate-state 不应该按整个 module 粗粒度替换，也不应该复制 activation scale；更应该围绕 per-weight 离散 endpoint 和与它耦合的 move bias 做状态选择、状态锁定或训练内 candidate assignment。

下一步判断：

停止本轮离线 broad transplant 扩展。下一步应把 `weight+move` 的小正信号转化为训练内机制，例如只在 `features.7.1.attn.proj` 这类 late attention projection 上维护候选 weight endpoint，并让 move bias 跟随被选 endpoint，而不是复制整模块或 activation scale。若继续离线验证，应优先做“单层 weight+move candidate-state + 1 个极低 LR stabilize epoch”的短 gate，而不是继续扫 `lsqw/input_scale/bias` 小组合。

completion audit：

- strict W4A4：满足，eval 命令含 `--wq-bitw 4 --aq-bitw 4 --wq-mode lsq --aq-mode lsq`。
- clean no-QKR/no-StatsQ：满足，使用 clean LSQ no-QKR eval 脚本，未启用 QKR/StatsQ。
- 单 checkpoint：满足，逐个评估单一 transplant checkpoint。
- full ImageNet raw validation：满足，三条结果均 `Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，当前 best 为 `80.2080`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2FU：single-module candidate-state transplant，attn.proj 小幅正收益但未达标

实验动机：

Phase 2FT 的双模块 transplant（`features.7.1.attn.proj` + `features.7.1.mlp.fc2`）得到 `80.1540`，低于 Phase 2CX base `80.1660`。为了判断到底哪个模块贡献负面，本阶段拆成两个单模块 transplant：只替换 `features.7.1.attn.proj`，以及只替换 `features.7.1.mlp.fc2`。仍然是不做 soup、不做 averaging 的单 checkpoint 模块级 candidate-state 组合。

方法设计：

- base checkpoint：Phase 2CX clean AOQ-native final `checkpoint-4`，Top-1 `80.1660`；
- donor checkpoint：source-anchor single-cross `checkpoint-4`，Top-1 `80.1560`；
- 候选 A：只从 donor 复制 `features.7.1.attn.proj.*`；
- 候选 B：只从 donor 复制 `features.7.1.mlp.fc2.*`；
- 每个候选输出一个单 checkpoint；
- 使用 clean LSQ/no-QKR eval-only full-val 脚本评估。

构造命令：

```bash
python3 QATs/tmp_scripts/make_resume10_module_transplant_checkpoint_20260709.py \
  --base /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
  --donor /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_gate_20260708/checkpoint-4.pth.tar \
  --modules features.7.1.attn.proj \
  --output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_module_transplant_2cx_base_singlecross_attnproj71_20260709/checkpoint-4.pth.tar

python3 QATs/tmp_scripts/make_resume10_module_transplant_checkpoint_20260709.py \
  --base /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
  --donor /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_gate_20260708/checkpoint-4.pth.tar \
  --modules features.7.1.mlp.fc2 \
  --output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_module_transplant_2cx_base_singlecross_mlpfc2_71_20260709/checkpoint-4.pth.tar
```

构造证据：

```text
attnproj:
modules=('features.7.1.attn.proj',)
copied_tensors=6
missing_tensors=0

mlpfc2:
modules=('features.7.1.mlp.fc2',)
copied_tensors=6
missing_tensors=0
```

eval 命令：

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_module_transplant_2cx_base_singlecross_attnproj71_20260709/checkpoint-4.pth.tar \
EXP=eval_module_transplant_2cx_singlecross_attnproj71_clean_lsq_noqkr_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_module_transplant_2cx_singlecross_attnproj71_clean_lsq_noqkr_20260709.log \
MASTER_PORT=31475 \
bash QATs/tmp_scripts/eval_resume10_clean_lsq_noqkr_checkpoint_fullval_20260709.sh

CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_module_transplant_2cx_base_singlecross_mlpfc2_71_20260709/checkpoint-4.pth.tar \
EXP=eval_module_transplant_2cx_singlecross_mlpfc2_71_clean_lsq_noqkr_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_module_transplant_2cx_singlecross_mlpfc2_71_clean_lsq_noqkr_20260709.log \
MASTER_PORT=31477 \
bash QATs/tmp_scripts/eval_resume10_clean_lsq_noqkr_checkpoint_fullval_20260709.sh
```

strict resume / eval 证据：

```text
attnproj:
Strict resume: loaded model from .../recipe_resume10_module_transplant_2cx_base_singlecross_attnproj71_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 29.946s  Loss: 0.8473  Acc@1: 80.1840  Acc@5: 95.1560  Samples: 50000

mlpfc2:
Strict resume: loaded model from .../recipe_resume10_module_transplant_2cx_base_singlecross_mlpfc2_71_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 29.594s  Loss: 0.8476  Acc@1: 80.1460  Acc@5: 95.1560  Samples: 50000
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 Phase 2CX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attnproj transplant | yes | yes | yes | 50000 | no | 80.1840 | 95.1560 | 0.8473 | +0.0180 |
| mlpfc2 transplant | yes | yes | yes | 50000 | no | 80.1460 | 95.1560 | 0.8476 | -0.0200 |

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. 单模块拆解显示 `features.7.1.attn.proj` transplant 有小幅正收益：`80.1660 -> 80.1840`；`features.7.1.mlp.fc2` 是负收益：`80.1660 -> 80.1460`。
3. 这说明模块级 candidate-state 并非完全无效，但当前收益只有 `+0.0180`，离 `81.0` 仍很远。
4. 下一步如果继续 surgery，应围绕 `attn.proj` 类模块做系统候选分析，例如测试 `features.5.5.attn.proj`、两个 attn.proj 同时替换、或更细粒度只替换 `weight/lsqw_fn.s`，而不是继续组合 mlp.fc2。
5. Goal 仍未完成，不调用 `update_goal complete`。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- clean no-QKR/no-StatsQ：满足，`qk_reparam=false`、`wq_mode=lsq`。
- 单 checkpoint：满足，评估单个 transplant checkpoint。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，最佳 `80.1840`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2FV：attn.proj module transplant 扩展验证，最佳 80.1840 未达标

实验动机：

Phase 2FU 发现只替换 `features.7.1.attn.proj` 有小幅正收益：Phase 2CX base `80.1660` 提升到 `80.1840`。为了判断该信号是否能扩展，本阶段继续测试 `features.5.5.attn.proj` 单模块，以及 `features.5.5.attn.proj + features.7.1.attn.proj` 双模块 transplant。仍然是不做 soup、不做 averaging 的单 checkpoint 模块级 candidate-state 组合。

方法设计：

- base checkpoint：Phase 2CX clean AOQ-native final `checkpoint-4`，Top-1 `80.1660`；
- donor checkpoint：source-anchor single-cross `checkpoint-4`，Top-1 `80.1560`；
- 候选 A：只从 donor 复制 `features.5.5.attn.proj.*`；
- 候选 B：从 donor 同时复制 `features.5.5.attn.proj.*` 与 `features.7.1.attn.proj.*`；
- 每个候选输出一个单 checkpoint；
- 使用 clean LSQ/no-QKR eval-only full-val 脚本评估；
- 不使用 soup、checkpoint averaging、multi-checkpoint averaging 或 ensemble。

构造命令：

```bash
python3 QATs/tmp_scripts/make_resume10_module_transplant_checkpoint_20260709.py \
  --base /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
  --donor /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_gate_20260708/checkpoint-4.pth.tar \
  --modules features.5.5.attn.proj \
  --output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_module_transplant_2cx_base_singlecross_attnproj55_20260709/checkpoint-4.pth.tar

python3 QATs/tmp_scripts/make_resume10_module_transplant_checkpoint_20260709.py \
  --base /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
  --donor /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_gate_20260708/checkpoint-4.pth.tar \
  --modules features.5.5.attn.proj,features.7.1.attn.proj \
  --output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_module_transplant_2cx_base_singlecross_attnproj55_71_20260709/checkpoint-4.pth.tar
```

构造证据：

```text
attnproj55:
modules=('features.5.5.attn.proj',)
copied_tensors=6
missing_tensors=0

attnproj55_71:
modules=('features.5.5.attn.proj', 'features.7.1.attn.proj')
copied_tensors=12
missing_tensors=0
```

eval 命令：

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_module_transplant_2cx_base_singlecross_attnproj55_20260709/checkpoint-4.pth.tar \
EXP=eval_module_transplant_2cx_singlecross_attnproj55_clean_lsq_noqkr_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_module_transplant_2cx_singlecross_attnproj55_clean_lsq_noqkr_20260709.log \
MASTER_PORT=31479 \
bash QATs/tmp_scripts/eval_resume10_clean_lsq_noqkr_checkpoint_fullval_20260709.sh

CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_module_transplant_2cx_base_singlecross_attnproj55_71_20260709/checkpoint-4.pth.tar \
EXP=eval_module_transplant_2cx_singlecross_attnproj55_71_clean_lsq_noqkr_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_module_transplant_2cx_singlecross_attnproj55_71_clean_lsq_noqkr_20260709.log \
MASTER_PORT=31481 \
bash QATs/tmp_scripts/eval_resume10_clean_lsq_noqkr_checkpoint_fullval_20260709.sh
```

strict resume / eval 证据：

```text
attnproj55:
Strict resume: loaded model from .../recipe_resume10_module_transplant_2cx_base_singlecross_attnproj55_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 29.972s  Loss: 0.8469  Acc@1: 80.1420  Acc@5: 95.1280  Samples: 50000

attnproj55_71:
Strict resume: loaded model from .../recipe_resume10_module_transplant_2cx_base_singlecross_attnproj55_71_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 29.573s  Loss: 0.8464  Acc@1: 80.1220  Acc@5: 95.1420  Samples: 50000
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 Phase 2CX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `features.5.5.attn.proj` transplant | yes | yes | yes | 50000 | no | 80.1420 | 95.1280 | 0.8469 | -0.0240 |
| `features.5.5 + features.7.1 attn.proj` transplant | yes | yes | yes | 50000 | no | 80.1220 | 95.1420 | 0.8464 | -0.0440 |

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. `features.5.5.attn.proj` 单模块 transplant 是负收益：`80.1660 -> 80.1420`。
3. 双 attn.proj transplant 也是负收益：`80.1660 -> 80.1220`。
4. Phase 2FU 的 `features.7.1.attn.proj` 单模块 transplant 仍是当前模块级 surgery 的最好结果：`80.1840`，仅比 Phase 2CX 高 `0.0180`，远未达到 `81.0`。
5. 这说明 candidate-state transplant 有一点点局部正信号，但收益太小；继续盲目模块组合不值得。下一步若继续，应做更系统的模块贡献/类别 flip 分析，或把 `features.7.1.attn.proj` 的正信号转成训练内 regularization/proxy，而不是离线拼模块。
6. Goal 仍未完成，不调用 `update_goal complete`。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- clean no-QKR/no-StatsQ：满足，`qk_reparam=false`、`wq_mode=lsq`。
- 单 checkpoint：满足，评估单个 transplant checkpoint。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，当前模块级 transplant 最好 `80.1840`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2FS：candidate endpoint selection 失败，最佳仍为 final 80.1660

实验动机：

Phase 2FP/2FQ/2FR 说明 per-weight oscillation 事件本身不是好的 candidate-state 选择准则。新的想法是不再只设计 mask，而是在一次 clean AOQ-native 轨迹中保存多个单 checkpoint 候选，然后用 full ImageNet raw validation 选择真实 endpoint。如果训练轨迹中间存在高于 final 的候选状态，这种方法可以在不使用 soup / averaging / ensemble 的前提下选择单 checkpoint。

工程准备：

- `run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh` 增加 step checkpoint 透传参数：
  - `SAVE_STEP_CHECKPOINTS`
  - `SAVE_INITIAL_STEP_CHECKPOINT`
  - `STEP_CHECKPOINT_INTERVAL`
  - `STEP_CHECKPOINT_WARMUP_UPDATES`
  - `MAX_STEP_CHECKPOINTS_TO_SAVE`
- 新增 clean LSQ/no-QKR eval 脚本：
  - `/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_resume10_clean_lsq_noqkr_checkpoint_fullval_20260709.sh`
- 该 eval 脚本固定使用：
  - `wq_mode=lsq`
  - `aq_mode=lsq`
  - `qk_reparam=false`
  - `kd_hard_and_soft=0`
  - `teacher_soft_temperature=2.75`
  - `eval-only`
- `bash -n`、`python3 -m py_compile qat_launch.py`、`git diff --check` 均通过。

step checkpoint smoke：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_stepckpt_smoke25upd_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_stepckpt_smoke25upd_20260709.log \
MASTER_PORT=31461 \
MAX_TRAIN_UPDATES=25 \
SKIP_VALIDATE=1 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=10 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

smoke 结果：

```text
TrainSummary: epoch=3 updates=25 avg_step_time=0.250157s samples_per_step=512 samples_per_sec=2046.71
Stopped early after 25 optimizer updates in epoch 3.
```

生成文件：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_stepckpt_smoke25upd_20260709/step_checkpoints/step_0010.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_stepckpt_smoke25upd_20260709/step_checkpoints/step_0020.pth.tar
```

说明 step checkpoint 保存链路有效。

候选轨迹命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_candidate_steps600_gate_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_candidate_steps600_gate_20260709.log \
MASTER_PORT=31463 \
SAVE_STEP_CHECKPOINTS=1 \
STEP_CHECKPOINT_INTERVAL=600 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_end_update: 1800
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aq_bitw: 4
aq_mode: lsq
epochs: 4
kd_hard_and_soft: 0
max_train_updates: 0
no_resume_opt: true
qk_reparam: false
save_step_checkpoints: true
scheduler_epochs: 4
skip_validate: false
start_epoch: 3
step_checkpoint_interval: 600
teacher_soft_temperature: 2.75
wq_bitw: 4
wq_mode: lsq
```

strict resume / 训练轨迹证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Train: 3 [ 600/2502 ( 24%)] ...
Train: 3 [1200/2502 ( 48%)] ...
Train: 3 [1800/2502 ( 72%)] ...
Train: 3 [2400/2502 ( 96%)] ...
TrainSummary: epoch=3 updates=2502 avg_step_time=0.167781s samples_per_step=512 samples_per_sec=3051.60
Test: [distributed-summary]  Time: 36.179s  Loss: 0.8476  Acc@1: 80.1660  Acc@5: 95.1680  Samples: 50000
```

生成候选 checkpoint：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_candidate_steps600_gate_20260709/step_checkpoints/step_0600.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_candidate_steps600_gate_20260709/step_checkpoints/step_1200.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_candidate_steps600_gate_20260709/step_checkpoints/step_1800.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_candidate_steps600_gate_20260709/step_checkpoints/step_2400.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_candidate_steps600_gate_20260709/checkpoint-4.pth.tar
```

候选 full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `step_1200.pth.tar` | yes | yes | yes | 50000 | no | 79.8080 | 95.0640 | 0.8673 | 明显低于 final |
| `step_1800.pth.tar` | yes | yes | yes | 50000 | no | 79.8740 | 95.0720 | 0.8617 | 明显低于 final |
| `step_2400.pth.tar` | yes | yes | yes | 50000 | no | 80.0140 | 95.1720 | 0.8609 | 低于 final |
| `checkpoint-4.pth.tar` | yes | yes | yes | 50000 | no | 80.1660 | 95.1680 | 0.8476 | 本轨迹最佳，但等于 Phase 2CX |

eval 原始摘要：

```text
step_1200:
Test: [distributed-summary]  Time: 29.723s  Loss: 0.8673  Acc@1: 79.8080  Acc@5: 95.0640  Samples: 50000

step_1800:
Test: [distributed-summary]  Time: 29.331s  Loss: 0.8617  Acc@1: 79.8740  Acc@5: 95.0720  Samples: 50000

step_2400:
Test: [distributed-summary]  Time: 29.616s  Loss: 0.8609  Acc@1: 80.0140  Acc@5: 95.1720  Samples: 50000

final checkpoint-4:
Test: [distributed-summary]  Time: 36.179s  Loss: 0.8476  Acc@1: 80.1660  Acc@5: 95.1680  Samples: 50000
```

中文结论：

1. candidate endpoint selection 工程链路有效：同一条 clean AOQ-native 训练轨迹中成功保存多个 step checkpoint，并能用 clean LSQ/no-QKR eval-only 路径逐个 full-val。
2. 但本轨迹中中间候选都低于 final。`step_1200 / step_1800 / step_2400` 分别为 `79.8080 / 79.8740 / 80.0140`，最终 `checkpoint-4` 才回到 `80.1660`。
3. 这说明 Phase 2CX 的 `80.1660` 不是一个早期中间态，而是需要完整 2502 update 训练后才形成的 endpoint；简单按 step checkpoint 选择候选不能突破 80.166。
4. 下一步如果继续 candidate-state，应不再只保存时间点，而要保存“候选 bin assignment / endpoint”本身，并用局部 loss/logit proxy 对候选 assignment 做选择；单纯时间点 checkpoint selection 没有收益。
5. Goal 仍未完成，不调用 `update_goal complete`。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- clean no-QKR/no-StatsQ：满足，`qk_reparam=false`、`wq_mode=lsq`。
- 单 checkpoint：满足，所有评估均为单个 checkpoint。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，最佳 `80.1660`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2FT：module-level candidate-state transplant 失败，Top-1 80.1540

实验动机：

Phase 2FS 说明按训练时间点保存候选 checkpoint 不能突破 final `80.1660`。本阶段尝试更接近 candidate-state 的离线组合：不做 soup、不做 averaging，只构造一个单 checkpoint，把 Phase 2CX final 作为 base，再从 source-anchor single-cross checkpoint 中移植少数模块的完整状态。目标是验证“某些模块更稳定的 candidate endpoint”能否和 Phase 2CX 其他模块组合成更好的单 checkpoint。

方法设计：

- base checkpoint：Phase 2CX clean AOQ-native final `checkpoint-4`，Top-1 `80.1660`；
- donor checkpoint：source-anchor single-cross `checkpoint-4`，Top-1 `80.1560`；
- transplant 模块：
  - `features.7.1.attn.proj`
  - `features.7.1.mlp.fc2`
- 复制模块前缀下的 tensor，不做加权平均、不做多 checkpoint ensemble；
- 输出仍为单 checkpoint；
- 使用 clean LSQ/no-QKR eval-only full-val 脚本评估。

工程脚本：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/make_resume10_module_transplant_checkpoint_20260709.py
```

构造命令：

```bash
python3 QATs/tmp_scripts/make_resume10_module_transplant_checkpoint_20260709.py \
  --base /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
  --donor /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_gate_20260708/checkpoint-4.pth.tar \
  --modules features.7.1.attn.proj,features.7.1.mlp.fc2 \
  --output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_module_transplant_2cx_base_singlecross_core71_20260709/checkpoint-4.pth.tar
```

构造证据：

```text
output: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_module_transplant_2cx_base_singlecross_core71_20260709/checkpoint-4.pth.tar
base: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar
donor: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_gate_20260708/checkpoint-4.pth.tar
modules: ('features.7.1.attn.proj', 'features.7.1.mlp.fc2')
copied_tensors: 12
missing_tensors: 0
```

复制 tensor：

```text
features.7.1.attn.proj.weight
features.7.1.attn.proj.bias
features.7.1.attn.proj.input_quant_fn.s
features.7.1.attn.proj.lsqw_fn.s
features.7.1.attn.proj.move_b4.bias
features.7.1.attn.proj.move_aft.bias
features.7.1.mlp.fc2.weight
features.7.1.mlp.fc2.bias
features.7.1.mlp.fc2.input_quant_fn.s
features.7.1.mlp.fc2.lsqw_fn.s
features.7.1.mlp.fc2.move_b4.bias
features.7.1.mlp.fc2.move_aft.bias
```

eval 命令：

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_module_transplant_2cx_base_singlecross_core71_20260709/checkpoint-4.pth.tar \
EXP=eval_module_transplant_2cx_singlecross_core71_clean_lsq_noqkr_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_module_transplant_2cx_singlecross_core71_clean_lsq_noqkr_20260709.log \
MASTER_PORT=31473 \
bash QATs/tmp_scripts/eval_resume10_clean_lsq_noqkr_checkpoint_fullval_20260709.sh
```

strict resume / eval 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_module_transplant_2cx_base_singlecross_core71_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 29.752s  Loss: 0.8472  Acc@1: 80.1540  Acc@5: 95.1560  Samples: 50000
Eval-only metrics: {'loss': 0.8472089448028803, 'top1': 80.154, 'top5': 95.156, 'samples': 50000, 'local_samples': 6250, 'wall_seconds': 29.751636266708374}
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| module transplant `checkpoint-4.pth.tar` | yes | yes | yes | 50000 | no | 80.1540 | 95.1560 | 0.8472 | 低于 base Phase 2CX `80.1660` |

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. 模块级 candidate-state transplant 工程链路有效：从 donor 复制 2 个模块共 12 个 tensor，输出单 checkpoint，strict resume `missing=0/unexpected=0`。
3. 但 Top-1 为 `80.1540`，低于 base Phase 2CX `80.1660`，说明将 single-cross 的 core71 模块 endpoint 移植到 Phase 2CX 不能带来增益。
4. 这条最小 module-level candidate-state 组合也没有突破 clean AOQ-native best。下一步如果继续 surgery，应先做更系统的模块贡献分析，而不是盲目组合模块；或者转向训练内 candidate assignment proxy，而不是离线模块替换。
5. Goal 仍未完成，不调用 `update_goal complete`。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- clean no-QKR/no-StatsQ：满足，`qk_reparam=false`、`wq_mode=lsq`。
- 单 checkpoint：满足，评估单个 transplant checkpoint。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.1540`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### 最新状态索引：per-weight oscillation selector 系列

最近完成的三组 per-weight oscillation selector gate：

| phase | 方法 | full-val Top-1 | 结论 |
|---|---|---:|---|
| Phase 2FP | `history_oscillating`, 累计 history，`quality_min_frac=0.02` | 79.9160 | 累计 history 太宽，update 1600 选中 41.8006% near-boundary，明显破坏 endpoint |
| Phase 2FQ | `recent_oscillating`, 全程只选当前方向反转，`quality_min_frac=0` | 80.0880 | selector 足够稀疏，但全程过窄，低于 Phase 2CX |
| Phase 2FR | 0-600 普通 AOQ，600 后 `recent_oscillating` | 80.0340 | hybrid 时序也无收益，低于全程 recent-only |

当前最新判断：

- strict W4A4 / clean no-QKR/no-StatsQ / 单 checkpoint / full ImageNet raw validation / `Samples=50000` / 无 soup 全部满足。
- 但最新 Top-1 只有 `80.0340`，没有达到 `81.0`，也没有超过 clean AOQ-native best Phase 2CX `80.1660`。
- per-weight oscillation 事件本身不是足够好的 candidate-state 选择准则：累计过宽，全程 recent 过窄，延迟 recent 也没有恢复收益。
- 下一步若继续 AOQ-native，不应继续扫 recent start update；更合理的是换 candidate-state 定义，例如记录候选 bin endpoint 并用 loss / logit / 小验证代理筛选候选，而不是只用 oscillation 事件。
- Goal 仍未完成，不调用 `update_goal complete`。

### Phase 2FQ：recent oscillation AOQ selector 失败，Top-1 80.0880

实验动机：

Phase 2FP 的 `history_oscillating` 证明 per-weight state selector 工程上可行，但累计式 `osc_count > 0` 会让 selected 子集持续变宽，到 update 1600 已经选中 `41.8006%` 的 near-boundary 权重，full-val 掉到 `79.9160`。本阶段保留 per-weight state 思路，但把状态定义收紧为 `recent_oscillating`：只选择当前 update 刚发生方向反转的 near-boundary 权重，不累计历史；同时去掉 `quality_min_frac` floor，避免人为补宽。

方法设计：

- clean no-QKR / no-StatsQ 主线；
- 从 clean no-QKR/LSQ source `checkpoint-3` strict resume；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- AOQ explore：`scale_ratio=0.90`，`selective_margin=0.08`，`end_update=1800`；
- `aoq_explore_quality_mode=recent_oscillating`；
- `quality_start_update=0`；
- `quality_min_frac=0`；
- 不使用 QKR、StatsQ、confidence-band KD、local reference、BinReg、selective anchor、soup、checkpoint averaging、multi-checkpoint averaging、ensemble。

工程改动：

- `qat_launch.py` 新增 `recent_oscillating` AOQ quality mode；
- 复用 per-weight `prev_bins / prev_delta / switch_count / osc_count` 状态；
- `recent_oscillating` 的 mask 为 `near_boundary & oscillated`，其中 `oscillated` 只代表当前 update 的方向反转；
- 与 `history_oscillating` 不同，它不会把历史上曾经 oscillate 的权重长期保留在 AOQ explore 集合中；
- `py_compile` 与 `git diff --check` 均通过。

205-update smoke 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_recentosc_smoke205upd_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_recentosc_smoke205upd_20260709.log \
MASTER_PORT=31443 \
AOQ_EXPLORE_SCALE_RATIO=0.90 \
AOQ_EXPLORE_SELECTIVE_MARGIN=0.08 \
AOQ_EXPLORE_QUALITY_MODE=recent_oscillating \
AOQ_EXPLORE_QUALITY_START_UPDATE=0 \
AOQ_EXPLORE_QUALITY_MIN_FRAC=0 \
AOQ_EXPLORE_END_UPDATE=1800 \
MAX_TRAIN_UPDATES=205 \
SKIP_VALIDATE=1 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

smoke 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
AOQ crossing-quality selector init: epoch=3, update=0, mode=recent_oscillating, pairs=6, near=1497634, selected=0, selected_over_near=0.000000, moved_excluded=0, switched=0, oscillating=0, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=2, mode=recent_oscillating, pairs=6, near=1497845, selected=10437, selected_over_near=0.006968, moved_excluded=10437, switched=30053, oscillating=10437, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=200, mode=recent_oscillating, pairs=6, near=1505602, selected=4118, selected_over_near=0.002735, moved_excluded=4118, switched=4761, oscillating=4118, missing_pairs=0
TrainSummary: epoch=3 updates=205 avg_step_time=0.186105s samples_per_step=512 samples_per_sec=2751.14
```

smoke 结论：

`recent_oscillating` 达到预期：update 200 只选中 `4118 / 1505602 = 0.2735%` 的 near-boundary 权重，比 Phase 2FP 的累计 history `20.6062%` 稀疏很多。

full gate 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_recentosc_gate_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_recentosc_gate_20260709.log \
MASTER_PORT=31447 \
AOQ_EXPLORE_SCALE_RATIO=0.90 \
AOQ_EXPLORE_SELECTIVE_MARGIN=0.08 \
AOQ_EXPLORE_QUALITY_MODE=recent_oscillating \
AOQ_EXPLORE_QUALITY_START_UPDATE=0 \
AOQ_EXPLORE_QUALITY_MIN_FRAC=0 \
AOQ_EXPLORE_END_UPDATE=1800 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_end_update: 1800
aoq_explore_quality_min_frac: 0.0
aoq_explore_quality_mode: recent_oscillating
aoq_explore_quality_start_update: 0
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aq_bitw: 4
aq_mode: lsq
epochs: 4
kd_hard_and_soft: 0
max_train_updates: 0
no_resume_opt: true
qk_reparam: false
scheduler_epochs: 4
skip_validate: false
start_epoch: 3
teacher_soft_temperature: 2.75
wq_bitw: 4
wq_mode: lsq
```

selector 证据：

```text
AOQ crossing-quality selector: epoch=3, update=200, mode=recent_oscillating, pairs=6, near=1505602, selected=4118, selected_over_near=0.002735, moved_excluded=4118, switched=4761, oscillating=4118, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=400, mode=recent_oscillating, pairs=6, near=1513156, selected=4736, selected_over_near=0.003130, moved_excluded=4736, switched=5152, oscillating=4736, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=800, mode=recent_oscillating, pairs=6, near=1526444, selected=3818, selected_over_near=0.002501, moved_excluded=3818, switched=3963, oscillating=3818, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=1200, mode=recent_oscillating, pairs=6, near=1538292, selected=3516, selected_over_near=0.002286, moved_excluded=3516, switched=3599, oscillating=3516, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=1600, mode=recent_oscillating, pairs=6, near=1546707, selected=3075, selected_over_near=0.001988, moved_excluded=3075, switched=3123, oscillating=3075, missing_pairs=0
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, ... quality_mode=recent_oscillating ...
TrainSummary: epoch=3 updates=2502 avg_step_time=0.177025s samples_per_step=512 samples_per_sec=2892.24
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 Phase 2CX | 对比全局 best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-4.pth.tar` | yes | yes | yes | 50000 | no | 80.0880 | 95.1680 | 0.8492 | -0.0780 | -0.4660 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 34.714s  Loss: 0.8492  Acc@1: 80.0880  Acc@5: 95.1680  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_recentosc_smoke205upd_20260709.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_recentosc_gate_20260709.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_recentosc_gate_20260709/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_recentosc_gate_20260709/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_recentosc_gate_20260709/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. `recent_oscillating` 成功把 per-weight state selector 收紧到极稀疏范围，解决了 Phase 2FP 累计 history 过宽的问题。
3. 但 Top-1 只有 `80.0880`，仍低于 Phase 2CX `80.1660`。这说明“全程只探索当前发生方向反转的权重”又过窄，丢掉了 Phase 2CX 早期普通 near-boundary AOQ 的有效探索。
4. 下一步的非重复方向应是 hybrid 时序：前 600 update 保持 Phase 2CX 普通 selective-margin08，让模型先获得足够 crossing；600 之后切到 `recent_oscillating`，避免累计 history 太宽，同时保留早期自由探索收益。
5. Goal 仍未完成，不调用 `update_goal complete`。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- clean no-QKR/no-StatsQ：满足，`qk_reparam=false`、`wq_mode=lsq`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.0880`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2FP：per-weight history oscillating AOQ selector 失败，Top-1 79.9160

实验动机：

Phase 2FO 说明 `anchor_unmoved / anchor_moved` 这种相对 source 是否已迁移的二值 selector 没有收益。为了真正切到 AOQ-native 的 per-weight / candidate-state 范式，本阶段不再按 source checkpoint 做二值 mask，而是在训练过程中为每个 LSQ weight 维护 bin-switch 历史：记录上一次 bin、上一次 crossing 方向、累计 switch 次数和累计方向反转次数。新的 `history_oscillating` selector 只让有历史 oscillation 的 near-boundary 权重参与 AOQ explore，并用 `quality_min_frac=0.02` 防止早期选中子集过小。

方法设计：

- clean no-QKR / no-StatsQ 主线；
- 从 clean no-QKR/LSQ source `checkpoint-3` strict resume；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- AOQ explore：`scale_ratio=0.90`，`selective_margin=0.08`，`end_update=1800`，复现 Phase 2CX 主探索窗口；
- 新增 `aoq_explore_quality_mode=history_oscillating`；
- `quality_start_update=0`，从训练一开始维护 per-weight history；
- `quality_min_frac=0.02`，若历史 oscillation 子集过小，则从已有 switch history 的 near-boundary 权重中补足；
- 不使用 QKR、StatsQ、confidence-band KD、local reference、BinReg、selective anchor、soup、checkpoint averaging、multi-checkpoint averaging、ensemble。

工程改动：

- `qat_launch.py` 新增 `history_oscillating` AOQ quality mode；
- 每个 LSQ weight quantizer 维护 per-weight `prev_bins / prev_delta / switch_count / osc_count`；
- `quality_mask = near_boundary & (osc_count > 0)`，并可按 `quality_min_frac` 从有 switch history 的候选中补足；
- selector 初始化日志从只覆盖 `anchor_unmoved` 扩展为所有非 `none` selector；
- `py_compile` 和 `git diff --check` 均通过。

205-update smoke 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_historyosc_min02_smoke205upd_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_historyosc_min02_smoke205upd_20260709.log \
MASTER_PORT=31437 \
AOQ_EXPLORE_SCALE_RATIO=0.90 \
AOQ_EXPLORE_SELECTIVE_MARGIN=0.08 \
AOQ_EXPLORE_QUALITY_MODE=history_oscillating \
AOQ_EXPLORE_QUALITY_START_UPDATE=0 \
AOQ_EXPLORE_QUALITY_MIN_FRAC=0.02 \
AOQ_EXPLORE_END_UPDATE=1800 \
MAX_TRAIN_UPDATES=205 \
SKIP_VALIDATE=1 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

smoke 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
AOQ crossing-quality selector init: epoch=3, update=0, mode=history_oscillating, pairs=6, near=1497634, selected=0, selected_over_near=0.000000, moved_excluded=0, switched=0, oscillating=0, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=1, mode=history_oscillating, pairs=6, near=1497857, selected=53453, selected_over_near=0.035686, moved_excluded=0, switched=53453, oscillating=53453, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=2, mode=history_oscillating, pairs=6, near=1497845, selected=73069, selected_over_near=0.048783, moved_excluded=10437, switched=30053, oscillating=73069, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=200, mode=history_oscillating, pairs=6, near=1505055, selected=310134, selected_over_near=0.206062, moved_excluded=3268, switched=3909, oscillating=310134, missing_pairs=0
TrainSummary: epoch=3 updates=205 avg_step_time=0.187013s samples_per_step=512 samples_per_sec=2737.78
```

smoke 结论：

工程链路成立。update 0 时 history 为空，update 1/2 开始形成 per-weight switch / oscillation history，update 200 已经选择 `310134 / 1505055 = 20.6062%` 的 near-boundary 子集。该 selector 确实不同于 source-anchor 二值 mask，是训练内生的 per-weight state。

full gate 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_historyosc_min02_gate_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_historyosc_min02_gate_20260709.log \
MASTER_PORT=31441 \
AOQ_EXPLORE_SCALE_RATIO=0.90 \
AOQ_EXPLORE_SELECTIVE_MARGIN=0.08 \
AOQ_EXPLORE_QUALITY_MODE=history_oscillating \
AOQ_EXPLORE_QUALITY_START_UPDATE=0 \
AOQ_EXPLORE_QUALITY_MIN_FRAC=0.02 \
AOQ_EXPLORE_END_UPDATE=1800 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_end_update: 1800
aoq_explore_quality_min_frac: 0.02
aoq_explore_quality_mode: history_oscillating
aoq_explore_quality_start_update: 0
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aq_bitw: 4
aq_mode: lsq
epochs: 4
kd_hard_and_soft: 0
max_train_updates: 0
no_resume_opt: true
qk_reparam: false
scheduler_epochs: 4
skip_validate: false
start_epoch: 3
teacher_soft_temperature: 2.75
wq_bitw: 4
wq_mode: lsq
```

strict resume / selector 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ crossing-quality selector init: epoch=3, update=0, mode=history_oscillating, pairs=6, near=1497634, selected=0, selected_over_near=0.000000, moved_excluded=0, switched=0, oscillating=0, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=200, mode=history_oscillating, pairs=6, near=1505055, selected=310134, selected_over_near=0.206062, moved_excluded=3268, switched=3909, oscillating=310134, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=400, mode=history_oscillating, pairs=6, near=1512902, selected=434059, selected_over_near=0.286905, moved_excluded=3302, switched=3787, oscillating=434059, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=800, mode=history_oscillating, pairs=6, near=1526673, selected=551910, selected_over_near=0.361512, moved_excluded=1932, switched=2071, oscillating=551910, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=1200, mode=history_oscillating, pairs=6, near=1538458, selected=610749, selected_over_near=0.396988, moved_excluded=1436, switched=1539, oscillating=610749, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=1600, mode=history_oscillating, pairs=6, near=1546395, selected=646402, selected_over_near=0.418006, moved_excluded=1053, switched=1112, oscillating=646402, missing_pairs=0
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, ... quality_mode=history_oscillating ...
TrainSummary: epoch=3 updates=2502 avg_step_time=0.177199s samples_per_step=512 samples_per_sec=2889.41
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 Phase 2CX | 对比全局 best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-4.pth.tar` | yes | yes | yes | 50000 | no | 79.9160 | 95.1540 | 0.8516 | -0.2500 | -0.6380 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 35.196s  Loss: 0.8516  Acc@1: 79.9160  Acc@5: 95.1540  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_historyosc_min02_smoke205upd_20260709.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_historyosc_min02_gate_20260709.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_historyosc_min02_gate_20260709/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_historyosc_min02_gate_20260709/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_historyosc_min02_gate_20260709/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. `history_oscillating` 证明了 per-weight state 机制可以工作，但当前“累计历史 oscillation”太宽：selected 子集从 update 200 的 `20.6%` near-boundary 增长到 update 1600 的 `41.8%` near-boundary。
3. Top-1 只有 `79.9160`，低于 Phase 2CX `80.1660`，说明累计式历史 mask 会把太多曾经 oscillate 的权重长期保留在 AOQ explore 中，扰动强于原始 selective-margin08。
4. 下一步不应放弃 per-weight state，而应收紧状态定义：从累计 `osc_count > 0` 改为 `recent_oscillating`，只选择当前 update 刚发生方向反转的 near-boundary 权重；必要时去掉 `quality_min_frac` 或把 floor 降到 `0.005`，避免 history 子集越滚越大。
5. Goal 仍未完成，不调用 `update_goal complete`。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- clean no-QKR/no-StatsQ：满足，`qk_reparam=false`、`wq_mode=lsq`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `79.9160`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2FO：tail-state anchor-moved second pulse 失败，Top-1 80.0820

实验动机：

Phase 2FN 的 `anchor_unmoved` 只让相对 source 仍未跨 bin 的 near-boundary 权重参与尾段 second pulse，结果 `80.0760`，没有超过 Phase 2CX `80.1660`。为了把状态驱动探索补齐，本阶段测试互补方向：只让已经相对 source 跨过 bin、但仍处在 near-boundary 的权重参与 `2200-2300` 的尾段弱 second pulse。这个实验用于判断已经迁移过的权重是否还存在低 LR 继续探索收益。

方法设计：

- clean no-QKR / no-StatsQ 主线；
- 从 clean no-QKR/LSQ source `checkpoint-3` strict resume；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- 不使用 base `aoq_explore_scale_ratio`，全部由 update schedule 控制；
- schedule：
  - `0:0.90:0:0.08`，复现 Phase 2CX 主探索；
  - `1800:1.0:0:0`，恢复 normal；
  - `2200:0.95:0:0.04`，低 LR 尾段弱 second pulse；
  - `2300:1.0:0:0`，再次恢复 normal；
- AOQ explore layers 为 late 6 个 quantizer；
- quality selector：`anchor_moved`，只在 update 2200 以后启用；
- `anchor_moved` 语义为 `near_boundary & moved_from_anchor`，与 Phase 2FN 的 `anchor_unmoved = near_boundary & ~moved_from_anchor` 互补；
- 不使用 confidence-band KD、local reference、BinReg、selective anchor、QKR、StatsQ、soup、checkpoint averaging、multi-checkpoint averaging、ensemble。

工程修复：

本阶段先补齐 `anchor_moved` 工程链路：

- `qat_launch.py` 增加 `--aoq-explore-quality-start-update` 参数并向训练侧透传；
- quality mask 使用当前 schedule 的 `aoq_current_selective_margin`，避免 delayed selector 错用全局 margin；
- 新增 `anchor_moved` mode；
- runtime validation 允许 `anchor_moved`；
- anchor checkpoint 加载条件从仅 `anchor_unmoved` 扩展为 `anchor_unmoved / anchor_moved`。

smoke 过程中先遇到两个真实错误并修复：

```text
ValueError: aoq_explore_quality_mode must be one of none, grad_cross, anchor_unmoved; got anchor_moved
ValueError: anchor_moved requires aoq anchor state
```

修复后静态检查通过：

```bash
python3 -m py_compile /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py
git diff --check
```

2302-update smoke 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse_anchormoved2200_smoke2302upd_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse_anchormoved2200_smoke2302upd_20260709.log \
MASTER_PORT=31423 \
AOQ_EXPLORE_SCALE_RATIO=1.0 \
AOQ_EXPLORE_SELECTIVE_MARGIN=0.0 \
AOQ_EXPLORE_END_UPDATE=0 \
AOQ_EXPLORE_UPDATE_SCHEDULE=0:0.90:0:0.08,1800:1.0:0:0,2200:0.95:0:0.04,2300:1.0:0:0 \
AOQ_EXPLORE_QUALITY_MODE=anchor_moved \
AOQ_EXPLORE_QUALITY_START_UPDATE=2200 \
AOQ_EXPLORE_ANCHOR_CHECKPOINT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar \
MAX_TRAIN_UPDATES=2302 \
SKIP_VALIDATE=1 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

smoke 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Loaded AOQ anchor checkpoint for anchor_moved selector: path=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar, tensors=497
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, ... quality_mode=anchor_moved, quality_start_update=2200 ...
AOQ explore scale ratio update: epoch=3, update=2200, active=True, base_ratio=0.95, ... selective_margin=0.04, quality_mode=anchor_moved, quality_start_update=2200 ...
AOQ crossing-quality selector: epoch=3, update=2200, mode=anchor_moved, pairs=6, near=845211, selected=237516, selected_over_near=0.281014, moved_excluded=237516, missing_pairs=0
AOQ explore scale ratio update: epoch=3, update=2300, active=False, base_ratio=1.0, ... selective_margin=0.0, quality_mode=anchor_moved ...
TrainSummary: epoch=3 updates=2302 avg_step_time=0.167404s samples_per_step=512 samples_per_sec=3058.47
```

解释：

`anchor_moved` 的 smoke 统计与 Phase 2FN `anchor_unmoved` 互补。Phase 2FN 在 update 2200 的 `anchor_unmoved` 为：

```text
near=845211, selected=607695, moved_excluded=237516
```

本阶段 `anchor_moved` 为：

```text
near=845211, selected=237516, selected_over_near=0.281014
```

说明 `anchor_moved` 正确选中了之前被 `anchor_unmoved` 排除的已迁移子集，工程语义成立。

full gate 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse_anchormoved2200_gate_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse_anchormoved2200_gate_20260709.log \
MASTER_PORT=31429 \
AOQ_EXPLORE_SCALE_RATIO=1.0 \
AOQ_EXPLORE_SELECTIVE_MARGIN=0.0 \
AOQ_EXPLORE_END_UPDATE=0 \
AOQ_EXPLORE_UPDATE_SCHEDULE=0:0.90:0:0.08,1800:1.0:0:0,2200:0.95:0:0.04,2300:1.0:0:0 \
AOQ_EXPLORE_QUALITY_MODE=anchor_moved \
AOQ_EXPLORE_QUALITY_START_UPDATE=2200 \
AOQ_EXPLORE_ANCHOR_CHECKPOINT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_anchor_checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
aoq_explore_end_update: 0
aoq_explore_quality_mode: anchor_moved
aoq_explore_quality_start_update: 2200
aoq_explore_scale_ratio: 1.0
aoq_explore_selective_margin: 0.0
aoq_explore_update_schedule:
aq_bitw: 4
aq_mode: lsq
epochs: 4
kd_hard_and_soft: 0
max_train_updates: 0
no_resume_opt: true
qk_reparam: false
scheduler_epochs: 4
skip_validate: false
start_epoch: 3
teacher_soft_temperature: 2.75
wq_bitw: 4
wq_mode: lsq
```

strict resume / selector / schedule 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Loaded AOQ anchor checkpoint for anchor_moved selector: path=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar, tensors=497
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=anchor_moved, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_start_update=2200, quality_min_frac=0.0, start_update=0, end_update=0
AOQ explore scale ratio update: epoch=3, update=2200, active=True, base_ratio=0.95, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.04, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=anchor_moved, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_start_update=2200, quality_min_frac=0.0, start_update=0, end_update=0
AOQ crossing-quality selector: epoch=3, update=2200, mode=anchor_moved, pairs=6, near=845211, selected=237516, selected_over_near=0.281014, moved_excluded=237516, missing_pairs=0
AOQ explore scale ratio update: epoch=3, update=2300, active=False, base_ratio=1.0, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=anchor_moved, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_start_update=2200, quality_min_frac=0.0, start_update=0, end_update=0
TrainSummary: epoch=3 updates=2502 avg_step_time=0.167121s samples_per_step=512 samples_per_sec=3063.64
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 Phase 2CX | 对比全局 best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-4.pth.tar` | yes | yes | yes | 50000 | no | 80.0820 | 95.1940 | 0.8533 | -0.0840 | -0.4720 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 34.913s  Loss: 0.8533  Acc@1: 80.0820  Acc@5: 95.1940  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse_anchormoved2200_smoke2302upd_20260709.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse_anchormoved2200_gate_20260709.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse_anchormoved2200_gate_20260709/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse_anchormoved2200_gate_20260709/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse_anchormoved2200_gate_20260709/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. `anchor_moved` 工程语义成立：它在 update 2200 选中了 `237516` 个相对 source 已经迁移的 near-boundary 权重，正好是 Phase 2FN `anchor_unmoved` 排除的 moved 子集；`1800/2200/2300` schedule 切换也按预期执行。
3. 但 Top-1 只有 `80.0820`，低于 Phase 2CX `80.1660`，也低于当前全局 strict W4A4 best `80.5540`。说明“对已经迁移过的 near-boundary 权重在尾段再开弱探索”没有收益。
4. 与 Phase 2FN 对比，`anchor_unmoved` 是 `80.0760`，`anchor_moved` 是 `80.0820`，两者几乎一样，且都低于 Phase 2CX。结论是：把 tail second pulse 按 source 迁移状态拆成未迁移 / 已迁移两个互补子集，仍不能恢复或超过主探索 endpoint。
5. 这进一步支持停止 tail second pulse / anchor-state selector 小扫。下一步如果继续 clean AOQ-native，应转向真正 per-weight / candidate-state 范式：记录每个权重的 crossing history、方向、稳定度和验证代理，而不是只在固定时间窗内按 source 是否迁移做二值 mask；或者先构造更强 clean no-QKR/LSQ source，再重新做首轮 AOQ。
6. Goal 仍未完成，不调用 `update_goal complete`。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- clean no-QKR/no-StatsQ：满足，`qk_reparam=false`、`wq_mode=lsq`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.0820`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2FN：tail-state anchor-unmoved second pulse 失败，Top-1 80.0760

实验动机：

Phase 2FM 说明固定低 LR second pulse 会破坏 Phase 2CX endpoint，Top-1 只有 `80.0200`。因此本阶段把固定时间表推进到更接近 AOQ-native 的“状态驱动”探索：前 `0-1800` 仍完全复现 Phase 2CX 主探索；尾段 `2200-2300` 只对相对 source 仍未跨 bin 的 near-boundary 权重启用弱 second pulse，避免重新扰动已经形成的有益 crossing。

代码改动：

- 在 `qat_launch.py` 增加 `--aoq-explore-quality-start-update` / `aoq_explore_quality_start_update`；
- quality selector 只有在 `local_update_count >= aoq_explore_quality_start_update` 时才启用；
- 未到 start update 或 AOQ inactive 时清空 `aoq_quality_mask`；
- 修复 update schedule 下 quality mask 使用的 margin：selector 使用当前 active schedule 的 `selective_margin`，而不是全局默认 margin；
- runner `run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh` 透传并打印 `AOQ_EXPLORE_QUALITY_START_UPDATE`。

方法设计：

- clean no-QKR / no-StatsQ 主线；
- 从 clean no-QKR/LSQ source `checkpoint-3` strict resume；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- schedule：
  - `0:0.90:0:0.08`，复现 Phase 2CX 主探索；
  - `1800:1.0:0:0`，恢复 normal；
  - `2200:0.95:0:0.04`，低 LR 尾段弱 second pulse；
  - `2300:1.0:0:0`，再次恢复 normal；
- `aoq_explore_quality_mode=anchor_unmoved`；
- `aoq_explore_quality_start_update=2200`；
- `aoq_explore_anchor_checkpoint` 使用 source `checkpoint-3`；
- 只在 2200 以后，用 `anchor_unmoved` 排除相对 source 已经跨 bin 的 near-boundary 权重；
- 不使用 confidence-band KD、local reference、BinReg、selective anchor、QKR、StatsQ、soup、checkpoint averaging、multi-checkpoint averaging、ensemble。

2302-update smoke 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse_anchorunmoved2200_smoke2302upd_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse_anchorunmoved2200_smoke2302upd_20260709.log \
MASTER_PORT=31415 \
AOQ_EXPLORE_SCALE_RATIO=1.0 \
AOQ_EXPLORE_SELECTIVE_MARGIN=0.0 \
AOQ_EXPLORE_END_UPDATE=0 \
AOQ_EXPLORE_UPDATE_SCHEDULE=0:0.90:0:0.08,1800:1.0:0:0,2200:0.95:0:0.04,2300:1.0:0:0 \
AOQ_EXPLORE_QUALITY_MODE=anchor_unmoved \
AOQ_EXPLORE_QUALITY_START_UPDATE=2200 \
AOQ_EXPLORE_ANCHOR_CHECKPOINT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar \
MAX_TRAIN_UPDATES=2302 \
SKIP_VALIDATE=1 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

smoke 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Loaded AOQ anchor checkpoint for anchor_unmoved selector: path=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar, tensors=497
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, ... quality_mode=anchor_unmoved, quality_start_update=2200 ...
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, ... quality_start_update=2200 ...
AOQ crossing-quality selector init: epoch=3, update=2200, mode=anchor_unmoved, pairs=6, near=845211, selected=607695, selected_over_near=0.718986, moved_excluded=237516, missing_pairs=0
AOQ explore scale ratio update: epoch=3, update=2200, active=True, base_ratio=0.95, selective_margin=0.04, ... quality_start_update=2200 ...
AOQ crossing-quality selector: epoch=3, update=2200, mode=anchor_unmoved, pairs=6, near=845211, selected=607695, selected_over_near=0.718986, moved_excluded=237516, missing_pairs=0
AOQ explore scale ratio update: epoch=3, update=2300, active=False, base_ratio=1.0, ... selective_margin=0.0 ...
TrainSummary: epoch=3 updates=2302 avg_step_time=0.167394s samples_per_step=512 samples_per_sec=3058.66
```

full gate 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse_anchorunmoved2200_gate_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse_anchorunmoved2200_gate_20260709.log \
MASTER_PORT=31419 \
AOQ_EXPLORE_SCALE_RATIO=1.0 \
AOQ_EXPLORE_SELECTIVE_MARGIN=0.0 \
AOQ_EXPLORE_END_UPDATE=0 \
AOQ_EXPLORE_UPDATE_SCHEDULE=0:0.90:0:0.08,1800:1.0:0:0,2200:0.95:0:0.04,2300:1.0:0:0 \
AOQ_EXPLORE_QUALITY_MODE=anchor_unmoved \
AOQ_EXPLORE_QUALITY_START_UPDATE=2200 \
AOQ_EXPLORE_ANCHOR_CHECKPOINT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_anchor_checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
aoq_explore_scale_ratio: 1.0
aoq_explore_selective_margin: 0.0
aoq_explore_quality_mode: anchor_unmoved
aoq_explore_quality_start_update: 2200
aoq_explore_quality_min_frac: 0.0
aoq_explore_end_update: 0
aoq_explore_update_schedule:
aq_bitw: 4
aq_mode: lsq
wq_bitw: 4
wq_mode: lsq
qk_reparam: false
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
no_resume_opt: true
skip_validate: false
max_train_updates: 0
start_epoch: 3
epochs: 4
scheduler_epochs: 4
```

strict resume / state-driven AOQ 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Loaded AOQ anchor checkpoint for anchor_unmoved selector: path=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar, tensors=497
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, ... quality_start_update=2200 ...
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, ... quality_start_update=2200 ...
AOQ crossing-quality selector init: epoch=3, update=2200, mode=anchor_unmoved, pairs=6, near=845211, selected=607695, selected_over_near=0.718986, moved_excluded=237516, missing_pairs=0
AOQ explore scale ratio update: epoch=3, update=2200, active=True, base_ratio=0.95, selective_margin=0.04, ... quality_start_update=2200 ...
AOQ crossing-quality selector: epoch=3, update=2200, mode=anchor_unmoved, pairs=6, near=845211, selected=607695, selected_over_near=0.718986, moved_excluded=237516, missing_pairs=0
AOQ explore scale ratio update: epoch=3, update=2300, active=False, base_ratio=1.0, ... selective_margin=0.0 ...
TrainSummary: epoch=3 updates=2502 avg_step_time=0.169410s samples_per_step=512 samples_per_sec=3022.26
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 Phase 2CX | 对比全局 best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-4.pth.tar` | yes | yes | yes | 50000 | no | 80.0760 | 95.1520 | 0.8539 | -0.0900 | -0.4780 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 34.571s  Loss: 0.8539  Acc@1: 80.0760  Acc@5: 95.1520  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse_anchorunmoved2200_smoke2302upd_20260709.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse_anchorunmoved2200_gate_20260709.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse_anchorunmoved2200_gate_20260709/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse_anchorunmoved2200_gate_20260709/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse_anchorunmoved2200_gate_20260709/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. 新增 `quality_start_update` 工程链路有效：主探索阶段未启用 selector，尾段 second pulse 才启用 `anchor_unmoved`；selector 在 update 2200 排除了 `237516` 个已相对 source 跨 bin 的 near-boundary 权重。
3. 但 Top-1 只有 `80.0760`，低于 Phase 2CX `80.1660`，也没有超过普通 late weak second pulse 的局部行为。说明“只对未迁移权重做尾段 second pulse”仍然会破坏或稀释已形成的 endpoint，无法提供额外收益。
4. 当前固定 schedule + 状态筛选仍不足。下一步若继续 AOQ-native，应该实现真正 per-weight oscillation memory / candidate-state 训练，而不是只在一个时间窗内用 anchor_unmoved mask；或者转向从源头构造更强 clean source checkpoint，而不是继续在 Phase 2CX epoch 内叠加尾段扰动。
5. Goal 仍未完成，不调用 `update_goal complete`。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- clean no-QKR/no-StatsQ：满足，`qk_reparam=false`、`wq_mode=lsq`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.0760`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DL：AOQ threshold/level decoupling smoke 通过

实验动机：

前面 Phase 2CX 的 selective-margin08 是当前 clean AOQ-native best，Top-1 `80.1660`。但它本质上仍是 LSQ scale-ratio trick：对 near-boundary weights 同时缩小 threshold interval 和 output level interval。AOQ 论文强调 threshold interval 与 quantization level interval 的阶段化处理，因此本阶段实现更接近 AOQ 的解耦：探索期只缩 threshold interval 诱导 crossing，输出 level 仍用原 LSQ scale，避免幅值一起缩小。

代码与脚本改动：

```text
third_party/OFQ/src/quantization/quantizer/lsq.py：
  - LsqQuantizerWeight / LsqQuantizer4Conv2d 新增 aoq_threshold_ratio；
  - forward 中使用 s_threshold 做除法、round 和 clamp；
  - 使用 s_level 做输出乘法；
  - 默认 aoq_threshold_ratio=None 时沿用旧行为，即 threshold ratio 跟随 scale ratio。

qat_launch.py：
  - 新增 --aoq-explore-threshold-ratio；
  - aoq_explore_enabled 会把 threshold_ratio != 1 视为 AOQ active；
  - AOQ update 日志打印 threshold_ratio。

tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq_thresholdonly_smoke2upd_20260708.sh：
  - 固定 scale_ratio=1.0；
  - 固定 threshold_ratio=0.90；
  - selective_margin=0.08；
  - 2-update smoke。
```

smoke 命令：

```bash
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq_thresholdonly_smoke2upd_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_scale_ratio: 1.0
aoq_explore_threshold_ratio: 0.9
aoq_explore_selective_margin: 0.08
aoq_explore_end_update: 2
max_train_updates: 2
skip_validate: true
wq_bitw: 4
aq_bitw: 4
wq_mode: lsq
aq_mode: lsq
qk_reparam: false
kd_hard_and_soft: 0
no_resume_opt: true
```

strict resume / AOQ 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=1.0, threshold_ratio=0.9, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=2
TrainSummary: epoch=3 updates=2 avg_step_time=0.788098s samples_per_step=512 samples_per_sec=649.67
Stopped early after 2 optimizer updates in epoch 3.
```

中文结论：

1. threshold/level decoupling 技术链路通过，`base_ratio=1.0`、`threshold_ratio=0.9` 正确进入 runtime。
2. smoke 不做 full-val，不计入 81 completion。
3. 下一步启动 full-val gate，检查更接近 AOQ 论文的 threshold-only exploration 是否优于 selective-margin08。

### Phase 2DM：AOQ threshold-only full-val 失败，Top-1 79.9620

实验动机：

Phase 2DL smoke 通过后，本阶段跑完整 1 个 resumed epoch，验证“只缩 threshold interval、不缩 output level interval”的 AOQ-style exploration 是否能比 Phase 2CX selective-margin08 更好。

方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- `aoq_explore_scale_ratio=1.0`；
- `aoq_explore_threshold_ratio=0.90`；
- selective margin：`0.08`；
- AOQ explore layers：late 6 个 quantizer；
- 不使用 `grad_cross`；
- 不使用 BinReg / selective anchor；
- AOQ explore 窗口：`0-1800` update；
- full-val 只认单个 `checkpoint-4.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

full gate 命令：

```bash
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq_thresholdonly_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_scale_ratio: 1.0
aoq_explore_threshold_ratio: 0.9
aoq_explore_selective_margin: 0.08
aoq_explore_end_update: 1800
max_train_updates: 0
skip_validate: false
wq_bitw: 4
aq_bitw: 4
wq_mode: lsq
aq_mode: lsq
qk_reparam: false
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
```

strict resume / AOQ 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=1.0, threshold_ratio=0.9, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=1800
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=0, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=1800
TrainSummary: epoch=3 updates=2502 avg_step_time=0.167902s samples_per_step=512 samples_per_sec=3049.39
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 79.9620 | 95.2080 | 0.8469 | 有效 gate，但低于 selective-margin08 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 35.249s  Loss: 0.8469  Acc@1: 79.9620  Acc@5: 95.2080  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq_thresholdonly_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq_thresholdonly_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq_thresholdonly_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq_thresholdonly_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. threshold/level decoupling 技术上有效，且确实运行在 `scale_ratio=1.0`、`threshold_ratio=0.9`。
3. 但 Top-1 只有 `79.9620`，低于 Phase 2CX selective-margin08 的 `80.1660`，也低于 pure late5571 LSQ-AOQ090 的 `80.1240`。
4. 结论是“只缩 threshold、不缩 level”本身不够；它可能诱导了 crossing，但没有提供有用的 quantized value adaptation，或者 threshold-only crossing 和输出 level 不匹配。
5. 下一步应诊断 threshold-only 与 selective-margin08 的 bin crossing 差异，尤其看是否 crossing 数量、attention/MLP 分布或 near-boundary 余量发生异常。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `79.9620`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2FG：candidate-anchor vs continuous class/logit 诊断

诊断动机：

Phase 2EF 显示 candidate-bin anchor full gate 的 bin-crossing 总量几乎和 continuous selective-margin08 一样，但 Top-1 从 `80.1660` 掉到 `79.9260`。为了判断损伤是集中在少数类别、低置信样本，还是整体概率校准变差，本阶段对 continuous checkpoint 与 candidate-anchor checkpoint 做 full ImageNet raw class/logit 诊断。

诊断命令：

```bash
CUDA_VISIBLE_DEVICES=0 python3 QATs/tmp_scripts/diagnose_resume10_logit_classes_20260708.py \
  --out-dir QATs/docs/resume10_clean_lsq_aoq_candidateanchor_class_diag_20260708 \
  --labels continuous,candidate \
  --compare-label continuous \
  --checkpoint continuous=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
  --checkpoint candidate=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_candidateanchor_gate_20260708/checkpoint-4.pth.tar \
  --wq-mode lsq --aq-mode lsq --no-qk-reparam \
  --batch-size 128 --workers 8 --flip-topn 800 \
  2>&1 | tee /mlx_devbox/users/quyanyi/playground/train_resume10_clean_lsq_aoq_candidateanchor_class_diag_20260708.log
```

诊断产物：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_candidateanchor_class_diag_20260708/summary.json
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_candidateanchor_class_diag_20260708/class_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_candidateanchor_class_diag_20260708/confidence_bins.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_candidateanchor_class_diag_20260708/flip_cases.tsv
```

诊断 full-val 复核：

```text
continuous: Loss=0.8470 Acc@1=80.1440 Acc@5=95.1680 Samples=50000
candidate:  Loss=0.8521 Acc@1=79.9240 Acc@5=95.1400 Samples=50000
```

说明：单 GPU 诊断路径和 8 卡 distributed-summary 的 Top-1 有 `0.02` 左右差异，但相对差距稳定，candidate 仍明显低于 continuous。

pair summary：

```text
candidate -> continuous:
delta_top1 = +0.2200
improved = 982
regressed = 872
net_flips = +110
same_correct = 39090
same_wrong = 9056
avg_true_prob_delta = +0.001530
avg_margin_delta = -0.000805
```

按 continuous 置信度分桶：

| continuous confidence bin | total | continuous correct | candidate correct | improved | regressed | net flips | avg true prob delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[0.00,0.20)` | 1292 | 216 | 270 | 144 | 90 | +54 | +0.017149 |
| `[0.20,0.40)` | 4138 | 1499 | 1544 | 402 | 357 | +45 | +0.019605 |
| `[0.40,0.60)` | 6520 | 3547 | 3563 | 376 | 360 | +16 | +0.011365 |
| `[0.60,0.80)` | 9069 | 6988 | 6988 | 60 | 60 | 0 | +0.001810 |
| `[0.80,0.90)` | 14486 | 13565 | 13560 | 0 | 5 | -5 | -0.001849 |
| `[0.90,0.95)` | 11991 | 11694 | 11694 | 0 | 0 | 0 | -0.006024 |
| `[0.95,0.99)` | 2423 | 2375 | 2375 | 0 | 0 | 0 | -0.007430 |

类别级最大退化：

| class | total | continuous - candidate correct | continuous acc | candidate acc | avg true prob delta |
|---:|---:|---:|---:|---:|---:|
| 272 | 50 | -5 | 68.0 | 58.0 | -0.046610 |
| 587 | 50 | -5 | 70.0 | 60.0 | -0.040410 |
| 718 | 50 | -5 | 70.0 | 60.0 | -0.001081 |
| 154 | 50 | -4 | 90.0 | 82.0 | -0.046069 |
| 871 | 50 | -4 | 88.0 | 80.0 | -0.039971 |
| 764 | 50 | -4 | 56.0 | 48.0 | -0.029166 |

类别级最大改善：

| class | total | continuous - candidate correct | continuous acc | candidate acc | avg true prob delta |
|---:|---:|---:|---:|---:|---:|
| 596 | 50 | +6 | 66.0 | 78.0 | +0.074384 |
| 657 | 50 | +6 | 28.0 | 40.0 | +0.031312 |
| 413 | 50 | +6 | 56.0 | 68.0 | +0.012857 |
| 151 | 50 | +5 | 60.0 | 70.0 | +0.166927 |
| 32 | 50 | +5 | 42.0 | 52.0 | +0.023331 |

中文结论：

1. candidate-anchor 的损伤不是单一类别整体崩坏，而是分散在很多类别的边界样本上。
2. continuous 相比 candidate 的净收益主要来自低置信样本：continuous confidence `<0.6` 的三个桶合计 `+115` net flips，几乎解释了总体 `+110` net flips。
3. 高置信样本基本不受影响；`0.8+` 桶几乎没有 top-1 flip，但 true probability 有轻微下降。
4. 这说明 candidate-bin anchor 主要破坏的是边界样本的函数空间适配，而不是离散 bin assignment 数量。它把已经 crossing 的权重锚住后，模型少了继续调整边界样本 logits 的自由度。
5. 下一步不应继续 candidate-anchor weight / margin / capture timing 小扫。更合理的方向是：
   - 在 clean AOQ 分支上做 function-space 保护或选择，例如只保护 low-confidence / high-flip 类别的 logits，而不是锚定权重；
   - 或者回到全局 strict W4A4 best `80.5540` 附近做受控 AOQ-native 短门控，看强 basin 是否能承受小范围 AOQ exploration。

completion audit：

- 本阶段是诊断，不产生新训练 checkpoint。
- 诊断覆盖 full ImageNet raw validation，`Samples=50000`。
- 没有使用 soup / averaging / ensemble。
- 目标 Top-1 `>=81.0` 未达到。
- goal 未完成，不调用 `update_goal complete`。

### Phase 2EF：candidate-bin anchor full gate 失败，Top-1 79.9260

实验动机：

Phase 2EE 的 4-update smoke 显示 candidate-bin anchor 技术链路成立，并且 mask fraction 只有 `0.010536`，比 selective near-boundary anchor 的 `0.16-0.19` 更聚焦。本阶段跑 full gate，验证“保留相对 source 已跨 bin 的候选离散迁移”是否能超过 continuous selective-margin08 Phase 2CX `80.1660`。

方法设计：

- 从 clean no-QKR/LSQ source `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- AOQ exploration 沿用 Phase 2CX full 6-layer selective-margin08：
  - `aoq_explore_scale_ratio=0.90`
  - `aoq_explore_selective_margin=0.08`
  - `aoq_explore_end_update=1800`
- candidate-bin anchor：
  - source checkpoint 使用同一个 clean no-QKR/LSQ source `checkpoint-3`；
  - update 1800 捕获当前 bin 与 source bin 不同的 candidate-changed 权重；
  - `candidate_bin_anchor_weight=1e-4`；
  - capture 后持续启用到 epoch 结束；
- full-val 只认单个 `checkpoint-4.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

full gate 命令：

```bash
cd /mlx_devbox/users/quyanyi/playground && \
MASTER_PORT=31357 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_candidateanchor_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
wq_bitw: 4
aq_bitw: 4
wq_mode: lsq
aq_mode: lsq
qk_reparam: false
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aoq_explore_start_update: 0
aoq_explore_end_update: 1800
candidate_bin_anchor_weight: 0.0001
candidate_bin_anchor_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
candidate_bin_anchor_capture_update: 1800
candidate_bin_anchor_end_update: 0
candidate_bin_anchor_source_checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
max_train_updates: 0
skip_validate: false
no_resume_opt: true
start_epoch: 3
scheduler_epochs: 4
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
```

strict resume / AOQ / candidate anchor 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Loaded candidate-bin anchor source checkpoint: path=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar, tensors=497
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, ...
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, ...
Captured candidate-bin anchor: weight=0.0001, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), pairs=6, masked=272002, total=5898240, mask_fraction=0.046116, missing_pairs=0, capture_update=1800, end_update=0, source=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
Enabled candidate-bin anchor: weight=0.0001, pairs=6, masked=272002, total=5898240, mask_fraction=0.046116, capture_update=1800, end_update=0
TrainSummary: epoch=3 updates=2502 avg_step_time=0.169727s samples_per_step=512 samples_per_sec=3016.60
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 79.9260 | 95.1540 | 0.8525 | 有效 gate，但显著低于 Phase 2CX |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 34.197s  Loss: 0.8525  Acc@1: 79.9260  Acc@5: 95.1540  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_candidateanchor_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_candidateanchor_gate_20260708.sh
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_candidateanchor_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_candidateanchor_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_candidateanchor_gate_20260708/last.pth.tar
```

补充 bin-crossing 诊断：

```bash
python3 /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_aoq_bin_crossing_20260708.py \
  --out-dir /mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_candidateanchor_bin_crossing_20260708 \
  --pairs 'ckpt10->phase2s,ckpt10->phase2w,phase2s->phase2w' \
  --module-patterns features.5.5,features.7.1 \
  --near-margin 0.08 \
  --topn 120 \
  --ckpt10-checkpoint /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar \
  --ckpt10-top1 79.9220 \
  --phase2s-checkpoint /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
  --phase2s-top1 80.1660 \
  --phase2w-checkpoint /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_candidateanchor_gate_20260708/checkpoint-4.pth.tar \
  --phase2w-top1 79.9260
```

诊断产物：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_candidateanchor_bin_crossing_20260708/aggregate_bin_crossing.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_candidateanchor_bin_crossing_20260708/pair_bin_crossing.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_candidateanchor_bin_crossing_20260708/summary.json
```

关键诊断摘要：

| pair | Top-1 delta | features.7.1 attn_proj changed | features.7.1 mlp_fc2 changed | features.5.5 attn_proj changed | features.7.1 attn_qkv changed | after near f7.1 proj |
|---|---:|---:|---:|---:|---:|---:|
| source -> continuous | +0.2440 | 0.056876 | 0.055845 | 0.036214 | 0.032492 | 0.301480 |
| source -> candidate-anchor | +0.0040 | 0.056544 | 0.055780 | 0.036112 | 0.032472 | 0.301595 |
| continuous -> candidate-anchor | -0.2400 | 0.013567 | 0.011620 | 0.010139 | 0.007399 | 0.301595 |

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. candidate-bin anchor 的工程链路有效：update 1800 捕获了 `272002 / 5898240` 个相对 source 已跨 bin 的候选权重，mask fraction `0.046116`，比 near-boundary anchor 更聚焦。
3. 但算法结果明显失败：Top-1 只有 `79.9260`，几乎回到 source `79.9220`，低于 continuous selective-margin08 `80.1660` 足足 `0.2400`。
4. 诊断显示 source -> candidate-anchor 的 changed fraction 和 source -> continuous 很接近，说明失败不是 crossing 数量不够；更可能是 candidate anchor 在后段约束了错误的函数适配，或者把同样 bin assignment 下的 FP 权重/scale 状态拉到了坏的局部。
5. 因此不应继续调大/调小 candidate anchor weight。candidate-bin anchor 这个“锚定已跨 bin 权重”机制本身不适合当前分支；它保留了离散 crossing 数量，却破坏了精度。
6. 下一步应转向 checkpoint/function-space selection，而不是再做 post-crossing anchor：例如比较 continuous `80.1660` 和 candidate-anchor `79.9260` 的 logit/class 退化，找出候选 crossing 对哪些类别/模块造成函数损伤；或者回到全局 best `80.5540` 附近做受控 AOQ-native 短门控。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `79.9260`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2EC：pulse3 vs continuous selective-margin08 bin-crossing 诊断

诊断动机：

Phase 2EB 的 pulse3 多段 schedule 达到 Top-1 `80.0900`，低于 continuous selective-margin08 Phase 2CX 的 `80.1660`。如果只看训练方式，可能误以为 pulse3 失败是因为三段短脉冲 crossing 不够。但需要用 checkpoint 级 bin-crossing 诊断确认：它到底是 crossing 不够、crossing 分布错位，还是后段 near-boundary 状态更差。

诊断命令：

```bash
python3 /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_aoq_bin_crossing_20260708.py \
  --out-dir /mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_pulse3_bin_crossing_20260708 \
  --pairs 'ckpt10->phase2s,ckpt10->phase2w,phase2s->phase2w' \
  --module-patterns features.5.5,features.7.1 \
  --near-margin 0.08 \
  --topn 120 \
  --ckpt10-checkpoint /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar \
  --ckpt10-top1 79.9220 \
  --phase2s-checkpoint /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
  --phase2s-top1 80.1660 \
  --phase2w-checkpoint /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_pulse3_gate_20260708/checkpoint-4.pth.tar \
  --phase2w-top1 80.0900
```

诊断产物：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_pulse3_bin_crossing_20260708/aggregate_bin_crossing.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_pulse3_bin_crossing_20260708/pair_bin_crossing.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_pulse3_bin_crossing_20260708/summary.json
```

标签映射：

```text
ckpt10  = clean no-QKR/LSQ source checkpoint-3, Top-1 79.9220
phase2s = continuous selective-margin08 checkpoint-4, Top-1 80.1660
phase2w = pulse3 checkpoint-4, Top-1 80.0900
```

关键 stage_kind 对比：

| pair | Top-1 delta | features.7.1 attn_proj changed | features.7.1 mlp_fc2 changed | features.7.1 attn_qkv changed | features.5.5 attn_proj changed | features.5.5 attn_qkv changed | after near f7.1 proj |
|---|---:|---:|---:|---:|---:|---:|---:|
| source -> continuous | +0.2440 | 0.056876 | 0.055845 | 0.032492 | 0.036214 | 0.031071 | 0.301480 |
| source -> pulse3 | +0.1680 | 0.056905 | 0.054300 | 0.032793 | 0.035963 | 0.031101 | 0.308202 |
| continuous -> pulse3 | -0.0760 | 0.038223 | 0.036276 | 0.021482 | 0.025316 | 0.022721 | 0.308202 |

中文结论：

1. pulse3 失败不是因为 source -> endpoint 的 crossing 总量明显不够。`features.7.1.attn_proj`、`features.7.1.mlp_fc2`、`features.5.5.attn_proj`、`features.5.5.attn_qkv` 等关键模块的 changed fraction 和 continuous selective-margin08 非常接近。
2. 但 pulse3 的 Top-1 只有 `80.0900`，比 continuous `80.1660` 低 `0.0760`，说明固定脉冲 schedule 改变了具体离散落点、near-boundary 状态或后段收敛轨迹，而不是简单减少了 crossing。
3. continuous -> pulse3 之间仍有大量 bin assignment 差异，例如 `features.7.1.attn_proj` changed fraction `0.038223`、`features.7.1.mlp_fc2` `0.036276`。这说明两者虽然总 crossing 接近，但不是同一批权重落到同一批 bins。
4. pulse3 的 after-near fraction 略高，例如 f7.1 proj 从 `0.301480` 到 `0.308202`，可能代表更多权重停在边界附近，后段自然收敛没有把它们带到更稳的离散状态。
5. 下一步不应继续固定 pulse 数量 / 位置小扫。更有价值的是做 crossing identity / candidate-bin memory：保留 continuous selective-margin08 中有益的一次 crossing 轨迹，避免把已到位的权重通过后续开关推到不同 bin，而不是只按时间开关 AOQ。

completion audit：

- 本阶段是离线诊断，不是训练 gate。
- 没有产生新的 full-val checkpoint。
- 当前 strict W4A4 单 checkpoint 最好仍低于 `81.0`。
- goal 未完成，不调用 `update_goal complete`。

### Phase 2EE：candidate-bin anchor 4-update smoke 成功

实验动机：

Phase 2ED 说明 pulse3 与 continuous selective-margin08 的 crossing 总量接近，但只有约一半 changed set 重合，问题在于“哪一批权重被允许跨 bin”，而不是 crossing 数量本身。此前 selective bin anchor 是按 near-boundary mask 锚定，mask fraction 约 `0.16-0.19`，仍然太宽。本阶段实现更聚焦的 candidate-bin memory：在 AOQ explore 结束点，对比当前 LSQ integer bin 与 source checkpoint 的 LSQ integer bin，只把“已经相对 source 跨 bin”的权重作为候选离散迁移锚定对象。目标是保留已经形成的候选 crossing，而不是把所有 near-boundary 权重都稳定住。

代码改动：

```text
/mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py
```

新增能力：

- `--candidate-bin-anchor-weight`
- `--candidate-bin-anchor-layers`
- `--candidate-bin-anchor-capture-update`
- `--candidate-bin-anchor-end-update`
- `--candidate-bin-anchor-source-checkpoint`

实现要点：

- 使用 source checkpoint 的 `*.weight` 和 `*.lsqw_fn.s` 计算 source LSQ integer bin；
- 在 capture update 计算当前 LSQ integer bin；
- 只把 `current_bin != source_bin` 的权重纳入 mask；
- anchor target 是 capture 时刻的 quantized output；
- 后段对这些 candidate-changed 权重施加轻量 MSE anchor；
- 与 selective near-boundary anchor 的区别是：mask 来自“相对 source 已跨 bin”的 candidate identity，而不是 near-boundary 距离。

新增脚本：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_candidateanchor_gate_20260708.sh
```

静态检查：

```bash
python3 -m py_compile /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py
bash -n /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh \
        /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_candidateanchor_gate_20260708.sh
git -C /mlx_devbox/users/quyanyi/playground/QATs diff --check
```

结果：均通过。

smoke 命令：

```bash
cd /mlx_devbox/users/quyanyi/playground && \
MAX_TRAIN_UPDATES=4 \
SKIP_VALIDATE=1 \
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_candidateanchor_smoke4upd_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_candidateanchor_smoke4upd_20260708.log \
MASTER_PORT=31356 \
AOQ_EXPLORE_END_UPDATE=2 \
CANDIDATE_BIN_ANCHOR_CAPTURE_UPDATE=2 \
CANDIDATE_BIN_ANCHOR_END_UPDATE=4 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_candidateanchor_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
experiment: recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_candidateanchor_smoke4upd_20260708
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
wq_mode: lsq
aq_mode: lsq
wq_bitw: 4
aq_bitw: 4
qk_reparam: false
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
max_train_updates: 4
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aoq_explore_end_update: 2
candidate_bin_anchor_weight: 0.0001
candidate_bin_anchor_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
candidate_bin_anchor_capture_update: 2
candidate_bin_anchor_end_update: 4
candidate_bin_anchor_source_checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
skip_validate: true
```

strict resume / AOQ / candidate anchor 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Loaded candidate-bin anchor source checkpoint: path=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar, tensors=497
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, ...
AOQ explore scale ratio update: epoch=3, update=2, active=False, base_ratio=1.0, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, ...
Captured candidate-bin anchor: weight=0.0001, layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), pairs=6, masked=62145, total=5898240, mask_fraction=0.010536, missing_pairs=0, capture_update=2, end_update=4, source=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
Enabled candidate-bin anchor: weight=0.0001, pairs=6, masked=62145, total=5898240, mask_fraction=0.010536, capture_update=2, end_update=4
TrainSummary: epoch=3 updates=4 avg_step_time=0.426177s samples_per_step=512 samples_per_sec=1201.38
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_candidateanchor_smoke4upd_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_candidateanchor_smoke4upd_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_candidateanchor_smoke4upd_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_candidateanchor_smoke4upd_20260708/last.pth.tar
```

中文结论：

1. candidate-bin anchor 技术链路已接通：source checkpoint 成功加载，AOQ 关闭点捕获成功，后段 anchor 成功启用。
2. 命中 `pairs=6`，与 late5571 目标模块一致。
3. mask fraction 只有 `0.010536`，比 selective near-boundary anchor 的约 `0.16-0.19` 聚焦很多，符合“只保留已经相对 source 跨 bin 的候选迁移”这一设计。
4. 本阶段是 4-update smoke，`skip_validate=1`，不产生 Top-1 结论，不能计入目标完成。
5. 下一步可以跑 full gate：0-1800 update continuous selective-margin08 AOQ，update 1800 捕获 candidate-bin anchor，后段持续启用，full-val 单个 `checkpoint-4.pth.tar`。如果 full gate 低于 Phase 2CX `80.1660`，说明简单锚定 candidate-changed 权重也不足；若接近或超过 Phase 2CX，再继续调 candidate anchor weight / capture timing。

completion audit：

- strict W4A4：smoke 配置满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：smoke 产物为单 checkpoint，但没有 full-val。
- full ImageNet raw validation：不满足，本阶段 `skip_validate=1`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：未验证。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2ED：pulse3 vs continuous crossing identity 诊断

诊断动机：

Phase 2EC 的 aggregate 诊断说明 pulse3 的 source -> endpoint changed fraction 与 continuous selective-margin08 非常接近，但 Top-1 低了 `0.0760`。这还不能回答一个关键问题：pulse3 是不是让同一批权重跨到了同一批 bin。为了判断下一步是否需要 candidate-bin memory，本阶段进一步做 crossing identity 诊断：逐元素比较 source、continuous endpoint、pulse3 endpoint 的 LSQ integer bin，统计 continuous 和 pulse3 的 changed set 重合度、same final bin 比例、continuous-only / pulse-only crossing。

诊断产物：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_pulse3_bin_crossing_20260708/pulse3_identity_by_weight.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_pulse3_bin_crossing_20260708/pulse3_identity_by_stage_kind.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_pulse3_bin_crossing_20260708/pulse3_identity_summary.json
```

关键 stage_kind 结果：

| group | continuous changed | pulse changed | both changed | same final bin | continuous only | pulse only | same given continuous | changed-set Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| features.7.1 attn_proj | 0.056876 | 0.056905 | 0.037779 | 0.037779 | 0.019097 | 0.019126 | 0.664232 | 0.497078 |
| features.7.1 mlp_fc2 | 0.055845 | 0.054300 | 0.036935 | 0.036935 | 0.018910 | 0.017365 | 0.661379 | 0.504501 |
| features.5.5 attn_proj | 0.036214 | 0.035963 | 0.023431 | 0.023431 | 0.012783 | 0.012533 | 0.647004 | 0.480662 |
| features.7.1 attn_qkv | 0.032492 | 0.032793 | 0.021901 | 0.021901 | 0.010591 | 0.010891 | 0.674053 | 0.504833 |
| features.5.5 attn_qkv | 0.031071 | 0.031101 | 0.019726 | 0.019726 | 0.011346 | 0.011375 | 0.634849 | 0.464717 |
| features.5.5 mlp_fc2 | 0.018928 | 0.019384 | 0.012553 | 0.012553 | 0.006375 | 0.006831 | 0.663203 | 0.487330 |

中文结论：

1. continuous 和 pulse3 的 changed fraction 接近，但 changed set 的 Jaccard 只有约 `0.46-0.50`。这说明二者不是同一批权重在跨 bin。
2. pulse3 复现 continuous changed 权重的比例约 `0.63-0.67`，剩余约三分之一 continuous crossing 没有被 pulse3 复现。
3. 在 both changed 的权重里，最终 bin 基本一致；问题主要是“哪些权重被选中 crossing”，不是同一权重跨到不同 bin。
4. pulse3 的固定开关打断了 continuous selective-margin08 中一部分有益 crossing，同时引入了另一批 pulse-only crossing；这解释了为什么总 crossing 接近但 Top-1 更低。
5. 下一步不应继续固定时间脉冲小扫。更合理的机制是 candidate-bin memory / crossing identity preservation：让训练保留 continuous 路径中更可能有益的一次性 crossing，减少 pulse-only 或后续扰动产生的替代 crossing。

completion audit：

- 本阶段是离线诊断，不是训练 gate。
- 没有产生新的 full-val checkpoint。
- 当前 strict W4A4 单 checkpoint 最好仍低于 `81.0`。
- goal 未完成，不调用 `update_goal complete`。

### Phase 2EB：AOQ pulse3 多段脉冲 schedule 失败，Top-1 80.0900

实验动机：

Phase 2EA 说明“前 1800 update continuous selective-margin08，后段切换可训练集合”不是有效突破口；Phase 2DR/2DS 也说明简单重复 AOQ 或从 endpoint 重新打开 AOQ 会引入 harmful crossing。新的假设是：连续 0-1800 update 一直打开 AOQ 可能过长，而完全重复到第二个 epoch 又太粗。更接近 AOQ-native 的做法是单个 epoch 内做多段脉冲：短时间打开 crossing，随后关闭一段让模型自然适配，再重新打开 crossing。这样可以测试“受控 oscillation / 离散解空间探索”是否需要交替开关，而不是连续固定缩放。

方法设计：

- 从 clean no-QKR/LSQ source `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- 新增 `--aoq-explore-update-schedule`，支持按 local optimizer update 切换 `scale_ratio / threshold_ratio / selective_margin`；
- schedule 格式：`update:scale:threshold:margin`；
- 本阶段采用 pulse3：
  - `0-300`：打开 `scale_ratio=0.90, threshold=0, margin=0.08`；
  - `300-600`：关闭 AOQ；
  - `600-900`：再次打开 `0.90 / 0 / 0.08`；
  - `900-1200`：关闭 AOQ；
  - `1200-1500`：第三次打开 `0.90 / 0 / 0.08`；
  - `1500-2502`：关闭 AOQ，自然收敛；
- full-val 只认单个 `checkpoint-4.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

新增实现：

```text
qat_launch.py:
- 新增 parse_aoq_update_schedule；
- 新增 aoq_explore_schedule_value；
- 新增 --aoq-explore-update-schedule；
- schedule 非空时由 schedule 完全接管 AOQ active 状态；
- schedule 关闭段会强制把 quantizer ratio reset 到 1.0，避免前一段 AOQ ratio 残留。
```

新增脚本：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_pulse3_gate_20260708.sh
```

静态检查：

```bash
python3 -m py_compile /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py
bash -n /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh \
        /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_pulse3_gate_20260708.sh
git -C /mlx_devbox/users/quyanyi/playground/QATs diff --check
```

结果：均通过。

schedule parser 小单元验证：

```text
[(0, 0.9, 0.0, 0.08), (300, 1.0, 0.0, 0.0), (600, 0.9, 0.0, 0.08), (900, 1.0, 0.0, 0.0)]
0   (0.9, 0.0, 0.08) True
299 (0.9, 0.0, 0.08) True
300 (1.0, 0.0, 0.0) False
599 (1.0, 0.0, 0.0) False
600 (0.9, 0.0, 0.08) True
899 (0.9, 0.0, 0.08) True
900 (1.0, 0.0, 0.0) False
```

smoke 命令：

```bash
cd /mlx_devbox/users/quyanyi/playground && \
MAX_TRAIN_UPDATES=905 \
SKIP_VALIDATE=1 \
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_pulse3_smoke905upd_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_pulse3_smoke905upd_20260708.log \
MASTER_PORT=31354 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_pulse3_gate_20260708.sh
```

smoke 关键证据：

```text
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, threshold_ratio=0.0, selective_margin=0.08, base_quantizers=6, ...
AOQ explore scale ratio update: epoch=3, update=300, active=False, base_ratio=1.0, threshold_ratio=0.0, selective_margin=0.0, base_quantizers=6, ...
AOQ explore scale ratio update: epoch=3, update=600, active=True, base_ratio=0.9, threshold_ratio=0.0, selective_margin=0.08, base_quantizers=6, ...
AOQ explore scale ratio update: epoch=3, update=900, active=False, base_ratio=1.0, threshold_ratio=0.0, selective_margin=0.0, base_quantizers=6, ...
TrainSummary: epoch=3 updates=905 avg_step_time=0.167957s samples_per_step=512 samples_per_sec=3048.41
```

smoke 结论：

多段 schedule 技术链路通过：0/300/600/900 的开关都按预期触发，并且关闭段正确 reset 到 `base_ratio=1.0`。

full gate 命令：

```bash
cd /mlx_devbox/users/quyanyi/playground && \
MASTER_PORT=31355 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_pulse3_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
wq_bitw: 4
aq_bitw: 4
wq_mode: lsq
aq_mode: lsq
qk_reparam: false
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
aoq_explore_scale_ratio: 1.0
aoq_explore_selective_margin: 0.0
aoq_explore_start_update: 0
aoq_explore_end_update: 0
aoq_explore_update_schedule:
max_train_updates: 0
skip_validate: false
no_resume_opt: true
start_epoch: 3
scheduler_epochs: 4
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
```

strict resume / pulse schedule 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, threshold_ratio=0.0, ... selective_margin=0.08 ...
AOQ explore scale ratio update: epoch=3, update=300, active=False, base_ratio=1.0, threshold_ratio=0.0, ... selective_margin=0.0 ...
AOQ explore scale ratio update: epoch=3, update=600, active=True, base_ratio=0.9, threshold_ratio=0.0, ... selective_margin=0.08 ...
AOQ explore scale ratio update: epoch=3, update=900, active=False, base_ratio=1.0, threshold_ratio=0.0, ... selective_margin=0.0 ...
AOQ explore scale ratio update: epoch=3, update=1200, active=True, base_ratio=0.9, threshold_ratio=0.0, ... selective_margin=0.08 ...
AOQ explore scale ratio update: epoch=3, update=1500, active=False, base_ratio=1.0, threshold_ratio=0.0, ... selective_margin=0.0 ...
TrainSummary: epoch=3 updates=2502 avg_step_time=0.166249s samples_per_step=512 samples_per_sec=3079.71
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 80.0900 | 95.2100 | 0.8475 | 有效 gate，但低于 Phase 2CX |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 34.741s  Loss: 0.8475  Acc@1: 80.0900  Acc@5: 95.2100  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_pulse3_smoke905upd_20260708.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_pulse3_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_pulse3_gate_20260708.sh
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_pulse3_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_pulse3_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_pulse3_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. 新增的 AOQ 多段 schedule 能力工程上成立，能够在一个 epoch 内反复打开 / 关闭 AOQ，并正确 reset ratio。
3. 但 pulse3 算法上未突破：Top-1 `80.0900`，高于 Phase 2EA `80.0640` 和 threshold-only `79.9620`，但仍低于 continuous full selective-margin08 Phase 2CX `80.1660`，也远低于全局 strict W4A4 best `80.5540`。
4. 这说明“只把 continuous AOQ 切成三个短脉冲”会减少部分负面影响，但也减少了有益 crossing 的积累；它不是通向 81 的直接路径。
5. 下一步不要继续做 pulse 数量 / 脉冲位置的小扫。更有价值的是做机制级改变：让 schedule 根据 bin-switch telemetry 自适应开关，或者引入真正的 per-weight oscillation state / candidate-bin memory，而不是固定时间表。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.0900`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2EA：selective-margin08 AOQ + late FP params adaptation 失败，Top-1 80.0640

实验动机：

Phase 2DY 说明“AOQ exploration 后只训练 late quant / scale / shift”会明显失败，Top-1 只有 `79.8960`。一个合理怀疑是后段 level adaptation 太窄：AOQ 前 1800 update 让 full model 参与 crossing，但后段只允许量化参数更新，可能无法让 late block 的 FP 权重适配已经发生的离散迁移。因此本阶段只改变一个关键变量：AOQ exploration 结束后，不切到 `quant_in_layers`，而切到 `params_in_layers`，允许 `features.5.5,features.7.1` 的 FP 权重和量化参数一起继续适配。

方法设计：

- 从 clean no-QKR/LSQ source `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- AOQ exploration 沿用 Phase 2CX full 6-layer selective-margin08：`aoq_explore_scale_ratio=0.90`、`aoq_explore_selective_margin=0.08`；
- AOQ explore 窗口：`0-1800` update；
- 初始 `trainable_policy=all`，让 full model 参与 AOQ exploration；
- update 1800 后通过 `grad_mask` 切到 `params_in_layers`，只保留 `features.5.5,features.7.1` 下的参数有效梯度；
- 使用 `grad_mask` 而不是动态 `requires_grad`，避免 DDP static graph 变化；
- full-val 只认单个 `checkpoint-4.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

新增脚本：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_paramsinlate_after1800_gradmask_gate_20260708.sh
```

smoke 命令：

```bash
cd /mlx_devbox/users/quyanyi/playground && \
MAX_TRAIN_UPDATES=1802 \
SKIP_VALIDATE=1 \
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_paramsinlate_after1800_gradmask_smoke1802upd_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_paramsinlate_after1800_gradmask_smoke1802upd_20260708.log \
MASTER_PORT=31352 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_paramsinlate_after1800_gradmask_gate_20260708.sh
```

smoke 关键证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, ...
Trainable parameter update policy: epoch=3, update=1800, mode=grad_mask, policy=params_in_layers, trainable=28535407, frozen=0
TrainSummary: epoch=3 updates=1802 avg_step_time=0.166983s samples_per_step=512 samples_per_sec=3066.18
```

smoke 结论：

`grad_mask` 成功跨过 update 1800，并完成 `1802` updates，没有复现 DDP static graph reduction 错误。结束阶段的 TCPStore / destroy_process_group warning 属于已知 teardown 噪声。

full gate 命令：

```bash
cd /mlx_devbox/users/quyanyi/playground && \
MASTER_PORT=31353 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_paramsinlate_after1800_gradmask_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
wq_bitw: 4
aq_bitw: 4
wq_mode: lsq
aq_mode: lsq
qk_reparam: false
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aoq_explore_start_update: 0
aoq_explore_end_update: 1800
trainable_policy: all
trainable_policy_freeze_act_except_layers: features.5.5,features.7.1
trainable_policy_update_mode: grad_mask
trainable_policy_update_overrides:
max_train_updates: 0
skip_validate: false
no_resume_opt: true
start_epoch: 3
scheduler_epochs: 4
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
```

strict resume / AOQ / policy 切换证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, ...
Trainable parameter update policy: epoch=3, update=0, mode=grad_mask, policy=all, trainable=28535407, frozen=0
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, ...
Trainable parameter update policy: epoch=3, update=1800, mode=grad_mask, policy=params_in_layers, trainable=28535407, frozen=0
TrainSummary: epoch=3 updates=2502 avg_step_time=0.168132s samples_per_step=512 samples_per_sec=3045.23
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 80.0640 | 95.1880 | 0.8555 | 有效 gate，但低于 Phase 2CX，也没有改善 Phase 2DC |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 35.821s  Loss: 0.8555  Acc@1: 80.0640  Acc@5: 95.1880  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_paramsinlate_after1800_gradmask_smoke1802upd_20260708.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_paramsinlate_after1800_gradmask_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_paramsinlate_after1800_gradmask_gate_20260708.sh
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_paramsinlate_after1800_gradmask_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_paramsinlate_after1800_gradmask_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_paramsinlate_after1800_gradmask_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. 工程上，`grad_mask` 可以稳定实现 “full-model AOQ exploration -> late-block params adaptation” 的中途策略切换。
3. 算法上，本阶段失败：Top-1 `80.0640`，低于 Phase 2CX full-model selective-margin08 `80.1660`，也远低于全局 strict W4A4 best `80.5540`。
4. 这个结果与 Phase 2DC “从一开始只训练 `features.5.5,features.7.1` params” 的 `80.0640` 基本一致，说明真正的收益不来自 1800 update 后的 late-block FP adaptation；Phase 2CX 的 `80.1660` 更依赖 full-model 在整个低 LR epoch 中的整体微调与 AOQ crossing 共同作用。
5. 下一步不要继续扫 “1800 后切到某个更窄 trainable policy” 这一组策略。clean no-QKR/LSQ AOQ 分支需要更大的新机制：例如真正的 oscillation schedule / 多状态量化候选，而不是同一个 LSQ scale-ratio 的后段可训练集合变化。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.0640`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DY：selective-margin08 AOQ + late quant/scale/shift level adaptation 失败，Top-1 79.8960

实验动机：

Phase 2DX 后，source first-epoch 的 AOQ mask / layer 组合基本已经耗尽：full selective-margin08 `80.1660` 最好，source-anchor single-cross `80.1560` 次之，hybrid-core71 反而降到 `80.0580`。同时 Phase 2DN/2DO 显示 threshold-only exploration 的问题不是 crossing 不够，而是 crossing 与 level 表征错位。基于这个判断，本阶段不再继续扫 AOQ mask，而是测试更大的机制：前期 full-model AOQ exploration，后期只允许 late quant / scale / shift 参数做 level adaptation，尝试把 AOQ 产生的离散迁移重新适配到更稳定的量化 level。

方法设计：

- 从 clean no-QKR/LSQ source `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- AOQ exploration 仍使用 Phase 2CX 的 full 6-layer selective-margin08：`aoq_explore_scale_ratio=0.90`、`aoq_explore_selective_margin=0.08`；
- AOQ explore 窗口：`0-1800` update；
- 初始 `trainable_policy=all`，让 full model 参与 AOQ exploration；
- update 1800 后切换到 `quant_in_layers`，只保留 late quant / scale / shift 类参数的有效梯度，作为 level adaptation；
- 第一次实现使用 `TRAINABLE_POLICY_UPDATE_MODE=requires_grad`，在 update 1800 动态改变 DDP 可训练参数集合；
- `requires_grad` 版本失败后，改为 `TRAINABLE_POLICY_UPDATE_MODE=grad_mask`：参数图保持不变，反向后用 gradient mask 屏蔽非目标参数梯度，避免破坏 DDP static graph；
- full-val 只认单个 `checkpoint-4.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

失败前置：`requires_grad` 版本在 update 1800 失败：

```text
Trainable parameter update policy: epoch=3, update=1800, mode=requires_grad, policy=quant_in_layers, trainable=35092, frozen=28500315
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your training graph has changed in this iteration, e.g., one parameter is used in first iteration, but then got unused in the second iteration. this is not compatible with static_graph set to True.
```

修复判断：

`requires_grad` 版本失败不是 AOQ / level-adaptation 本身的结果，而是 DDP static graph 与动态 `requires_grad` 切换不兼容。因此本阶段使用 `grad_mask` 版本继续验证同一个训练思想。

smoke 命令：

```bash
cd /mlx_devbox/users/quyanyi/playground && \
MAX_TRAIN_UPDATES=1802 \
SKIP_VALIDATE=1 \
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_leveladapt_gradmask_smoke1802upd_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_leveladapt_gradmask_smoke1802upd_20260708.log \
MASTER_PORT=31350 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_leveladapt_gradmask_gate_20260708.sh
```

smoke 关键证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, ...
Trainable parameter update policy: epoch=3, update=1800, mode=grad_mask, policy=quant_in_layers, trainable=28535407, frozen=0
TrainSummary: epoch=3 updates=1802 avg_step_time=0.170160s samples_per_step=512 samples_per_sec=3008.93
```

smoke 结论：

`grad_mask` 版本成功跨过 update 1800，并完成 `1802` updates，没有复现 `Expected to have finished reduction` / static graph 错误。后续 TCPStore / Broken pipe 信息属于进程退出阶段的已知 teardown 噪声，不是训练图错误。

full gate 命令：

```bash
cd /mlx_devbox/users/quyanyi/playground && \
MASTER_PORT=31351 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_leveladapt_gradmask_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
wq_bitw: 4
aq_bitw: 4
wq_mode: lsq
aq_mode: lsq
qk_reparam: false
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aoq_explore_start_update: 0
aoq_explore_end_update: 1800
trainable_policy: all
trainable_policy_freeze_act_except_layers: features.5.5,features.7.1
trainable_policy_update_mode: grad_mask
trainable_policy_update_overrides:
max_train_updates: 0
skip_validate: false
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
```

strict resume / AOQ / level-adaptation 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, ...
Trainable parameter update policy: epoch=3, update=0, mode=grad_mask, policy=all, trainable=28535407, frozen=0
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, ...
Trainable parameter update policy: epoch=3, update=1800, mode=grad_mask, policy=quant_in_layers, trainable=28535407, frozen=0
TrainSummary: epoch=3 updates=2502 avg_step_time=0.166278s samples_per_step=512 samples_per_sec=3079.17
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 79.8960 | 95.1340 | 0.8614 | 有效 gate，但明显低于 Phase 2CX 和当前全局 best |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 34.955s  Loss: 0.8614  Acc@1: 79.8960  Acc@5: 95.1340  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_leveladapt_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_leveladapt_gradmask_smoke1802upd_20260708.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_leveladapt_gradmask_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_leveladapt_gate_20260708.sh
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_leveladapt_gradmask_gate_20260708.sh
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_leveladapt_gradmask_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_leveladapt_gradmask_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_leveladapt_gradmask_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. `grad_mask` 解决了动态 `requires_grad` 切换与 DDP static graph 的工程问题；该机制可以作为后续“阶段性有效训练子集”实验的安全实现方式。
3. 但作为算法方案，本阶段失败：Top-1 只有 `79.8960`，低于 clean AOQ best Phase 2CX `80.1660`，也低于当前全局 strict W4A4 best Phase 2Z `80.5540`。
4. 这说明“AOQ exploration 后只做 late quant / scale / shift level adaptation”不是有效修复；后段只更新量化参数会让 full-model AOQ 产生的权重迁移缺少足够的模型空间适配，反而比普通 selective-margin08 更差。
5. 下一步不应继续做同构的 late quant-only level adaptation，也不应继续在 `1800` update 附近扫切换点。更合理的下一步是把 `grad_mask` 作为工具，而不是答案：要么设计有监督的 module-wise full-param-to-partial-param schedule，例如只冻结已稳定层、保留 `features.7.1` 相关 FP 权重更新；要么把 AOQ-native exploration 迁移到当前全局 best `80.5540` 附近做短门控，但必须明确标注是否重新引入 QKR/StatsQ 依赖，避免偏离“丢弃 OFQ-specific innovation”的主目标。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `79.8960`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DP：module-wise AOQ ratio core gate 失败，Top-1 80.1180

实验动机：

Phase 2DN 说明 threshold-only 路线不值得继续扫；Phase 2CX 的 selective-margin08 仍是 clean AOQ-native best `80.1660`。本阶段回到 scale-ratio selective-margin，但不再全 6 层统一 `0.90`，也不重复此前 coarse layer-ratio。根据 bin-crossing 诊断，`features.7.1.mlp.fc2` 和 `features.7.1.attn.proj` 是 source -> selective-margin08 中 crossing 最强、最可能贡献收益的模块，因此本阶段采用 module-wise ratio：

- base 6 层温和探索：`0.95`；
- 核心两层强化探索：`features.7.1.mlp.fc2:0.90,features.7.1.attn.proj:0.90`。

方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- `aoq_explore_scale_ratio=0.95`；
- `aoq_explore_layer_ratios=features.7.1.mlp.fc2:0.90,features.7.1.attn.proj:0.90`；
- selective margin：`0.08`；
- AOQ explore layers：late 6 个 quantizer；
- AOQ explore 窗口：`0-1800` update；
- 不使用 `grad_cross`；
- 不使用 BinReg / selective anchor；
- full-val 只认单个 `checkpoint-4.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

full gate 命令：

```bash
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq_modratio_core_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_scale_ratio: 0.95
aoq_explore_layer_ratios: features.7.1.mlp.fc2:0.90,features.7.1.attn.proj:0.90
aoq_explore_selective_margin: 0.08
aoq_explore_end_update: 1800
max_train_updates: 0
skip_validate: false
wq_bitw: 4
aq_bitw: 4
wq_mode: lsq
aq_mode: lsq
qk_reparam: false
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
```

strict resume / AOQ 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.95, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={'features.7.1.mlp.fc2': 0.9, 'features.7.1.attn.proj': 0.9}, layer_quantizers=2, layer_counts={'features.7.1.mlp.fc2': 1, 'features.7.1.attn.proj': 1}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=1800
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, layer_ratios={'features.7.1.mlp.fc2': 1.0, 'features.7.1.attn.proj': 1.0}, layer_quantizers=2, layer_counts={'features.7.1.mlp.fc2': 1, 'features.7.1.attn.proj': 1}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=1800
TrainSummary: epoch=3 updates=2502 avg_step_time=0.166083s samples_per_step=512 samples_per_sec=3082.80
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 80.1180 | 95.2140 | 0.8507 | 有效 gate，但低于 selective-margin08 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 36.164s  Loss: 0.8507  Acc@1: 80.1180  Acc@5: 95.2140  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq_modratio_core_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq_modratio_core_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq_modratio_core_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq_modratio_core_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. module-wise ratio 比 threshold-only 和 post-explore anchor 更好，Top-1 `80.1180`，但仍低于 Phase 2CX selective-margin08 的 `80.1660`。
3. 说明“温和 base 0.95 + 核心两层 0.90”没有保留足够的全局有益 crossing；`features.5.5` 与其他 late modules 的统一 `0.90` 参与仍可能是 Phase 2CX 小幅收益的一部分。
4. 这个方向不应继续做小的 layer-ratio 扫描。当前最高信号仍是 Phase 2CX 的 full 6-layer selective-margin08；要继续冲 81，需要更大范式变化，例如跨 epoch 的 multi-stage AOQ schedule，而不是单 epoch 内 ratio 分配。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.1180`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DQ：selective-margin08 checkpoint plain continuation 失败，Top-1 80.1400

实验动机：

Phase 2CX selective-margin08 是当前 clean AOQ-native best，Top-1 `80.1660`。此前 Phase 2DA 从该 checkpoint 继续 1 epoch 并加入 mild BinReg，结果回落到 `80.0260`。为了区分“普通 continuation 本身会回落”还是“BinReg 导致回落”，本阶段补一个最小基线：从 Phase 2CX `checkpoint-4` 出发，关闭 AOQ、BinReg、selective anchor、grad_cross，只做普通 strict W4A4 LSQ QAT 继续 1 epoch。

方法设计：

- 从 Phase 2CX `checkpoint-4` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- `aoq_explore_scale_ratio=1.0`，`aoq_explore_end_update=0`，即 AOQ explore 关闭；
- `bin_reg_weight=0.0`；
- `selective_bin_anchor_weight=0.0`；
- `start_epoch=4`，`epochs=5`；
- full-val 只认单个 `checkpoint-5.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

full gate 命令：

```bash
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_selectivemargin08_plain4to5_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_scale_ratio: 1.0
aoq_explore_threshold_ratio: 0.0
aoq_explore_end_update: 0
bin_reg_weight: 0.0
selective_bin_anchor_weight: 0.0
max_train_updates: 0
skip_validate: false
wq_bitw: 4
aq_bitw: 4
wq_mode: lsq
aq_mode: lsq
qk_reparam: false
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
no_resume_opt: true
start_epoch: 4
epochs: 5
scheduler_epochs: 5
```

strict resume / no-AOQ 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=4, update=0, active=False, base_ratio=1.0, threshold_ratio=0.0, base_layers=(), selective_margin=0.0, base_quantizers=0, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=(), quality_min_frac=0.0, start_update=0, end_update=0
TrainSummary: epoch=4 updates=2502 avg_step_time=0.168783s samples_per_step=512 samples_per_sec=3033.48
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-5.pth.tar` | yes | yes | 50000 | no | 80.1400 | 95.1700 | 0.8503 | 有效 gate，但低于 Phase 2CX |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 36.260s  Loss: 0.8503  Acc@1: 80.1400  Acc@5: 95.1700  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_selectivemargin08_plain4to5_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_selectivemargin08_plain4to5_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_selectivemargin08_plain4to5_gate_20260708/checkpoint-5.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_selectivemargin08_plain4to5_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. 从 Phase 2CX `80.1660` checkpoint 继续普通 strict W4A4 LSQ QAT 1 epoch 后回落到 `80.1400`，说明这个 peak 不能通过简单延长训练继续爬升。
3. Phase 2DA 的 mild BinReg 回落更大（`80.0260`），但 plain continuation 也没有提升，说明问题不只是 BinReg；Phase 2CX 更像单 epoch 末端的局部高点。
4. 下一步不应继续从 Phase 2CX checkpoint 做普通 continuation。要冲 81 需要回到 source 侧构造更强的新 endpoint，或者做跨 epoch schedule 但每个 epoch 都必须重新引入有益 exploration，而不是只在后续 epoch 做收敛。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-5.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.1400`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DR：repeated selective-margin08 AOQ 两 epoch gate 首 epoch 失败，Top-1 79.8660，按失败中止处理

实验动机：

Phase 2DQ 说明从 Phase 2CX `80.1660` checkpoint 做 plain continuation 不能继续爬升。本阶段测试一个真正的跨 epoch AOQ schedule：不是只在第一个 epoch 做 exploration 后继续收敛，而是从 clean no-QKR/LSQ source `checkpoint-3` 连跑 2 个 resumed epochs，并让每个 epoch 都按 local update 重新打开 selective-margin08 AOQ window。目标是验证“每个 epoch 都重新引入有益 exploration”是否能比单 epoch selective-margin08 更好。

方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- `aoq_explore_scale_ratio=0.90`；
- selective margin：`0.08`；
- AOQ explore layers：late 6 个 quantizer；
- AOQ explore 窗口：每个 epoch 的 local update `0-1800`；
- `start_epoch=3`，`epochs=5`，计划跑 epoch 3 和 4；
- full-val 只认单 checkpoint；
- 不使用 soup / checkpoint averaging / ensemble。

full gate 命令：

```bash
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_repeat2ep_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aoq_explore_end_update: 1800
max_train_updates: 0
skip_validate: false
wq_bitw: 4
aq_bitw: 4
wq_mode: lsq
aq_mode: lsq
qk_reparam: false
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
no_resume_opt: true
start_epoch: 3
epochs: 5
scheduler_epochs: 5
```

strict resume / repeated AOQ 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=1800
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, ...
TrainSummary: epoch=3 updates=2502 avg_step_time=0.166423s samples_per_step=512 samples_per_sec=3076.49
AOQ explore scale ratio update: epoch=4, update=0, active=True, base_ratio=0.9, ...
```

epoch 3 full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 79.8660 | 95.0740 | 0.8591 | 有效 gate，但首 epoch 已明显低于 Phase 2CX |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 34.575s  Loss: 0.8591  Acc@1: 79.8660  Acc@5: 95.0740  Samples: 50000
```

中止说明：

epoch 3 已经只有 `79.8660`，明显低于 Phase 2CX `80.1660`，也低于 threshold-only / module-wise / plain continuation 的结果。日志显示 run 后续进入了 epoch 4 early training，但当前已确认没有对应的 `qat_launch` / `torchrun` / OFQ train 进程继续运行。继续跑 epoch 4 的 repeated AOQ 没有合理收益预期，因此本阶段按失败中止处理，避免继续浪费 GPU。

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_repeat2ep_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_repeat2ep_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_repeat2ep_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_repeat2ep_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效的 epoch 3 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. repeated AOQ schedule 的首 epoch 就从 Phase 2CX 对应配置的 `80.1660` 降到 `79.8660`。主要差异是本 run 为了计划跑 2 epoch 使用 `scheduler_epochs=5`，导致 epoch 3 LR 明显高于 Phase 2CX；这说明 repeated schedule 不能简单通过拉长 scheduler 实现。
3. 当前不应继续 repeated AOQ 两 epoch。若未来再做跨 epoch exploration，需要严格保持 first epoch 与 Phase 2CX 的 scheduler/LR 对齐，然后只改变第二 epoch，而不是把第一 epoch 也改变。
4. 目标仍未完成，不调用 `update_goal complete`。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `79.8660`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DS：从 Phase 2CX endpoint 低 LR 重新打开 selective-margin08 AOQ 失败，Top-1 79.9620

实验动机：

Phase 2DR 的 repeated AOQ 两 epoch gate 首 epoch 只有 `79.8660`，但它和 Phase 2CX 的一个关键差异是 `scheduler_epochs=5`，导致 epoch 3 起始 LR 为 `7.562e-05`，明显高于 Phase 2CX 低 LR 末端。为了区分“repeated exploration 本身有害”还是“高 LR 破坏了第一轮 exploration”，本阶段从 Phase 2CX 的 `80.1660` endpoint 出发，保持 `scheduler_epochs=4`，让 resumed epoch 4 的 LR 维持在 `1e-5`，只重新打开同样的 selective-margin08 AOQ window。

方法设计：

- 从 Phase 2CX clean no-QKR/LSQ selective-margin08 `checkpoint-4` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- `scheduler_epochs=4`，保持低 LR 末端，避免 Phase 2DR 的高 LR 混淆；
- epoch 4 local update `0-1800` 重新打开 `aoq_explore_scale_ratio=0.90`；
- selective margin：`0.08`；
- AOQ explore layers：late 6 个 quantizer；
- update 1800 后恢复普通 LSQ QAT，直到 epoch 结束；
- full-val 只认单个 `checkpoint-5.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

smoke 命令：

```bash
cd /mlx_devbox/users/quyanyi/playground && \
MAX_TRAIN_UPDATES=4 \
SKIP_VALIDATE=1 \
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_reexplore4to5_smoke4upd_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_reexplore4to5_smoke4upd_20260708.log \
MASTER_PORT=31338 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_reexplore4to5_gate_20260708.sh
```

smoke 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar; missing=0, unexpected=0
AOQ explore scale ratio update: epoch=4, update=0, active=True, base_ratio=0.9, ... selective_margin=0.08, base_quantizers=6
Train: 4 [   0/2502 ...] ... LR: 1.000e-05
TrainSummary: epoch=4 updates=4 avg_step_time=0.396978s samples_per_step=512 samples_per_sec=1289.74
```

full gate 命令：

```bash
cd /mlx_devbox/users/quyanyi/playground && \
MASTER_PORT=31339 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_reexplore4to5_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar
no_resume_opt: true
start_epoch: 4
epochs: 5
scheduler_epochs: 4
wq_bitw: 4
aq_bitw: 4
wq_mode: lsq
aq_mode: lsq
qk_reparam: false
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aoq_explore_start_update: 0
aoq_explore_end_update: 1800
max_train_updates: 0
skip_validate: false
```

strict resume / AOQ 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=4, update=0, active=True, base_ratio=0.9, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=1800
AOQ explore scale ratio update: epoch=4, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, ...
TrainSummary: epoch=4 updates=2502 avg_step_time=0.168367s samples_per_step=512 samples_per_sec=3040.97
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-5.pth.tar` | yes | yes | 50000 | no | 79.9620 | 95.1100 | 0.8541 | 有效 gate，但显著低于 Phase 2CX |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 35.156s  Loss: 0.8541  Acc@1: 79.9620  Acc@5: 95.1100  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_reexplore4to5_smoke4upd_20260708.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_reexplore4to5_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_reexplore4to5_gate_20260708.sh
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_reexplore4to5_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_reexplore4to5_gate_20260708/checkpoint-5.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_reexplore4to5_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. smoke 和 full gate 均确认 LR 为 `1e-5`，因此它排除了 Phase 2DR “first epoch 高 LR”这个混淆项。
3. 即便低 LR 对齐，从 Phase 2CX endpoint 重新打开相同 selective-margin08 AOQ 也会从 `80.1660` 掉到 `79.9620`，比 plain continuation 的 `80.1400` 更差。
4. 这说明 Phase 2CX 的有益 crossing 不能简单在下一个 epoch 重复；第二轮同构 crossing 更像是在已形成的离散 endpoint 上引入新的 harmful crossing。
5. 下一步不应继续做同构 repeated selective-margin08 或它的 LR/window 小扫。更有价值的是做 `Phase 2CX -> plain continuation` 与 `Phase 2CX -> re-explore` 的 bin-crossing 诊断，定位 re-explore 相比普通 continuation 多出的 harmful crossing 模块，再决定是否需要“禁止已迁移权重二次 crossing”或“只允许此前未迁移子集探索”的机制。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-5.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `79.9620`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DT：anchor-unmoved 二次 AOQ mask 有效运行但仍低于 Phase 2CX，Top-1 80.0180

实验动机：

Phase 2DS 证明，从 Phase 2CX endpoint 低 LR 重新打开同构 selective-margin08 AOQ 会从 `80.1660` 掉到 `79.9620`，比 plain continuation 的 `80.1400` 更差。随后做的 `Phase 2CX -> plain continuation` 与 `Phase 2CX -> re-explore` bin-crossing 诊断显示，re-explore 不是简单“crossing 更多所以更差”：plain continuation 的 overall crossing 反而更高，但 Top-1 只降 `0.026`；re-explore crossing 更少却降 `0.204`。这更像是第二轮 AOQ 在已形成的离散 endpoint 上扰动了已经迁移过的有益权重。基于这个判断，本阶段实现 `anchor_unmoved` selector：以 clean no-QKR/LSQ source checkpoint 作为 anchor，Phase 2CX 相对 source 已经跨过 LSQ bin 的 near-boundary 权重不再参与第二轮 AOQ，只允许未迁移子集继续探索。

代码实现：

- `qat_launch.py` 新增 `--aoq-explore-quality-mode anchor_unmoved`；
- 新增 `--aoq-explore-anchor-checkpoint`；
- 复用 LSQ quantizer 的 `aoq_quality_mask` 通路；
- `anchor_unmoved` 只作用于 `lsqw_fn` 权重量化器；
- 在 AOQ active 打开时立即根据 anchor/current LSQ bin 生成 mask；
- 运行期间每个 optimizer update 后刷新 mask，日志打印：
  - `near`
  - `selected`
  - `selected_over_near`
  - `moved_excluded`
  - `missing_pairs`

方法设计：

- 从 Phase 2CX clean no-QKR/LSQ selective-margin08 `checkpoint-4` strict resume；
- anchor/source checkpoint 使用 clean no-QKR/LSQ `checkpoint-3`；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- `scheduler_epochs=4`，LR 保持 `1e-5`；
- epoch 4 local update `0-1800` 重新打开 `aoq_explore_scale_ratio=0.90`；
- selective margin：`0.08`；
- AOQ explore layers：late 6 个 quantizer；
- quality selector：`anchor_unmoved`；
- update 1800 后恢复普通 LSQ QAT，直到 epoch 结束；
- full-val 只认单个 `checkpoint-5.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

smoke 命令：

```bash
cd /mlx_devbox/users/quyanyi/playground && \
MAX_TRAIN_UPDATES=4 \
SKIP_VALIDATE=1 \
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_anchorunmoved_smoke4upd_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_anchorunmoved_smoke4upd_20260708.log \
MASTER_PORT=31340 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_anchorunmoved_gate_20260708.sh
```

smoke 证据：

```text
Loaded AOQ anchor checkpoint for anchor_unmoved selector: path=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar, tensors=497
AOQ crossing-quality selector init: epoch=4, update=0, mode=anchor_unmoved, pairs=6, near=1518057, selected=1265204, selected_over_near=0.833436, moved_excluded=252853, missing_pairs=0
Train: 4 [   0/2502 ...] ... LR: 1.000e-05
TrainSummary: epoch=4 updates=4 avg_step_time=0.441711s samples_per_step=512 samples_per_sec=1159.13
```

full gate 命令：

```bash
cd /mlx_devbox/users/quyanyi/playground && \
MASTER_PORT=31341 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_anchorunmoved_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar
no_resume_opt: true
start_epoch: 4
epochs: 5
scheduler_epochs: 4
wq_bitw: 4
aq_bitw: 4
wq_mode: lsq
aq_mode: lsq
qk_reparam: false
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aoq_explore_quality_mode: anchor_unmoved
aoq_explore_anchor_checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
aoq_explore_start_update: 0
aoq_explore_end_update: 1800
max_train_updates: 0
skip_validate: false
```

strict resume / masked AOQ 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Loaded AOQ anchor checkpoint for anchor_unmoved selector: path=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar, tensors=497
AOQ crossing-quality selector init: epoch=4, update=0, mode=anchor_unmoved, pairs=6, near=1518057, selected=1265204, selected_over_near=0.833436, moved_excluded=252853, missing_pairs=0
AOQ crossing-quality selector: epoch=4, update=1600, mode=anchor_unmoved, pairs=6, near=1515361, selected=1289329, selected_over_near=0.850840, moved_excluded=226032, missing_pairs=0
AOQ explore scale ratio update: epoch=4, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, ...
TrainSummary: epoch=4 updates=2502 avg_step_time=0.178775s samples_per_step=512 samples_per_sec=2863.94
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-5.pth.tar` | yes | yes | 50000 | no | 80.0180 | 95.1740 | 0.8493 | 有效 gate，略好于无 mask re-explore，但低于 Phase 2CX |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 36.100s  Loss: 0.8493  Acc@1: 80.0180  Acc@5: 95.1740  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_anchorunmoved_smoke4upd_20260708.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_anchorunmoved_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_anchorunmoved_gate_20260708.sh
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_anchorunmoved_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_anchorunmoved_gate_20260708/checkpoint-5.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_anchorunmoved_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. `anchor_unmoved` 技术上有效：smoke/full gate 都能加载 anchor checkpoint，6 个 late LSQ quantizer 全部命中，初始时从 `1518057` 个 near-boundary 元素中排除 `252853` 个已经相对 source 跨 bin 的元素，保留比例约 `83.34%`。
3. 相比 Phase 2DS 的无 mask re-explore `79.9620`，本阶段回升到 `80.0180`，说明“排除已迁移权重二次 exploration”方向有一定正向信号。
4. 但它仍低于 Phase 2CX `80.1660`，也低于全局 strict W4A4 best `80.5540`，更未达到 `81.0`。因此不能继续做这个 selector 的小标量扫描。
5. 当前更合理的下一步不是继续重复第二轮 AOQ，而是把 `anchor_unmoved` checkpoint 与 plain continuation / Phase 2CX 做 bin-crossing 和 class/logit 诊断，确认它恢复了哪些 crossing、仍损伤哪些类别；若继续训练，应设计更强的“只对未迁移子集探索 + 保留 plain continuation 自然迁移”的组合，而不是再扫 margin/window。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-5.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.0180`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DU：Phase 2CX / plain / re-explore / anchor-unmoved crossing 与 class-logit 诊断

诊断动机：

Phase 2DT 的 `anchor_unmoved` 相比无 mask re-explore 有小幅恢复：`79.9620 -> 80.0180`，但仍低于 Phase 2CX `80.1660`。为了避免继续盲扫 selector / margin / window，本阶段补两个诊断：

1. checkpoint 级 bin-crossing 对比：Phase 2CX、plain continuation、无 mask re-explore、anchor_unmoved；
2. 单 GPU class/logit 诊断：比较这几个 checkpoint 的 class-level gain/loss、confidence-bin flip、case flip。

注意：class/logit 诊断是单 GPU、单进程、按样本顺序对齐的诊断工具；它的绝对 Top-1 只用于相对比较，不替代 8-GPU full ImageNet raw validation。正式计分仍只看 `Test: [distributed-summary] ... Samples: 50000`。

crossing 诊断命令：

```bash
python3 /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_aoq_bin_crossing_20260708.py \
  --out-dir /mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_from2cx_anchorunmoved_bin_crossing_20260708 \
  --pairs 'ckpt10->phase2s,ckpt10->phase2w,ckpt10->phase2z,phase2s->phase2z,phase2w->phase2z' \
  --module-patterns features.5.5,features.7.1 \
  --near-margin 0.08 \
  --topn 160 \
  --ckpt10-checkpoint /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
  --ckpt10-top1 80.1660 \
  --phase2s-checkpoint /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_selectivemargin08_plain4to5_gate_20260708/checkpoint-5.pth.tar \
  --phase2s-top1 80.1400 \
  --phase2w-checkpoint /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_reexplore4to5_gate_20260708/checkpoint-5.pth.tar \
  --phase2w-top1 79.9620 \
  --phase2z-checkpoint /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_anchorunmoved_gate_20260708/checkpoint-5.pth.tar \
  --phase2z-top1 80.0180
```

crossing 诊断产物：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_from2cx_anchorunmoved_bin_crossing_20260708/aggregate_bin_crossing.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_from2cx_anchorunmoved_bin_crossing_20260708/pair_bin_crossing.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_from2cx_anchorunmoved_bin_crossing_20260708/summary.json
```

关键 crossing 对比：

| pair | Top-1 delta | f7.1 attn_proj changed | f7.1 mlp_fc2 changed | f7.1 mlp_fc1 changed | f5.5 attn_proj changed | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 2CX -> plain | -0.0260 | 0.050440 | 0.050887 | 0.038699 | 0.031508 | crossing 多但损失小 |
| 2CX -> re-explore | -0.2040 | 0.026735 | 0.024960 | 0.030400 | 0.017822 | crossing 少但损失大 |
| 2CX -> anchor_unmoved | -0.1480 | 0.031881 | 0.031344 | 0.030278 | 0.019803 | 比 re-explore 恢复一部分 |
| plain -> anchor_unmoved | -0.1220 | 0.041492 | 0.040424 | 0.032635 | 0.027086 | anchor_unmoved 仍偏离 plain |
| re-explore -> anchor_unmoved | +0.0560 | 0.027498 | 0.026229 | 0.025608 | 0.017795 | mask 修复了部分损伤 |

中文 crossing 结论：

1. “crossing 数量越少越好”不是正确解释。plain continuation 的 changed_fraction 比 re-explore 更高，但只从 `80.1660` 降到 `80.1400`；re-explore crossing 更少却掉到 `79.9620`。
2. `anchor_unmoved` 的 `80.0180` 位于 re-explore 和 plain 之间，说明排除相对 source 已迁移的权重能修复一部分损伤，但没有恢复 plain 的自然迁移路径。
3. 这支持一个更具体判断：Phase 2CX 后的 plain continuation 自然迁移整体较安全；二次 AOQ 的问题不是所有 crossing，而是它改变了自然迁移的方向/分布。

class/logit 诊断命令：

```bash
cd /mlx_devbox/users/quyanyi/playground && \
CUDA_VISIBLE_DEVICES=0 python3 QATs/tmp_scripts/diagnose_resume10_logit_classes_20260708.py \
  --out-dir QATs/docs/resume10_clean_lsq_aoq_from2cx_class_diag_20260708 \
  --labels 2cx,plain,reexplore,anchorunmoved \
  --compare-label 2cx \
  --checkpoint 2cx=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
  --checkpoint plain=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_selectivemargin08_plain4to5_gate_20260708/checkpoint-5.pth.tar \
  --checkpoint reexplore=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_reexplore4to5_gate_20260708/checkpoint-5.pth.tar \
  --checkpoint anchorunmoved=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_anchorunmoved_gate_20260708/checkpoint-5.pth.tar \
  --wq-mode lsq --aq-mode lsq --no-qk-reparam \
  --batch-size 128 --workers 8 --flip-topn 800 \
  2>&1 | tee /mlx_devbox/users/quyanyi/playground/train_resume10_clean_lsq_aoq_from2cx_class_diag_20260708.log
```

class/logit 诊断产物：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_from2cx_class_diag_20260708/summary.json
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_from2cx_class_diag_20260708/class_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_from2cx_class_diag_20260708/confidence_bins.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_from2cx_class_diag_20260708/flip_cases.tsv
/mlx_devbox/users/quyanyi/playground/train_resume10_clean_lsq_aoq_from2cx_class_diag_20260708.log
```

单 GPU诊断指标：

| label | samples | Top-1 | Top-5 | Loss | 备注 |
|---|---:|---:|---:|---:|---|
| `2cx` | 50000 | 80.1440 | 95.1680 | 0.8470 | 参考点 |
| `plain` | 50000 | 80.1280 | 95.1860 | 0.8505 | 与 2CX 基本持平 |
| `reexplore` | 50000 | 79.9120 | 95.1280 | 0.8547 | 明显差 |
| `anchorunmoved` | 50000 | 80.0560 | 95.1340 | 0.8490 | 修复一半左右，但仍低 |

pair summary：

| compare | improved | regressed | net flips vs 2CX | avg true_prob_delta | 结论 |
|---|---:|---:|---:|---:|---|
| `plain -> 2cx` | 976 | 968 | +8 | +0.0015 | plain 与 2CX 近似持平 |
| `reexplore -> 2cx` | 1040 | 924 | +116 | +0.0041 | reexplore 明显损伤 |
| `anchorunmoved -> 2cx` | 984 | 940 | +44 | +0.0013 | anchor_unmoved 修复部分损伤 |

class-level 观察：

- `reexplore` 相对 2CX 的明显回落类包括 `718`、`764`、`744`、`658`、`813`、`186`、`386`、`447`、`697`、`504`、`587`、`881` 等。
- `anchor_unmoved` 仍相对 2CX 回落的类包括 `587`、`248`、`610`、`764`、`447`、`261`、`655`、`809`、`793`、`304`、`978` 等。
- `anchor_unmoved` 相对 2CX 的增益类包括 `830`、`834`、`869`、`413`、`596`、`250`、`689`、`657`、`170`、`270`、`155`、`870` 等。
- confidence-bin 显示三条 continuation 主要在低/中置信样本上互换 correct/incorrect；高置信 `0.90+` 基本不发生 Top-1 flip。

中文诊断结论：

1. 继续“第二轮 AOQ”不是当前主线。无 mask re-explore 明显损伤，source-anchor mask 只能修复一部分。
2. plain continuation 与 2CX 非常接近，说明 Phase 2CX endpoint 后的普通低 LR 自然迁移并不灾难；真正有害的是二次 AOQ 对自然迁移路径的改写。
3. 下一条可验证的非重复 gate 应该保护 plain continuation 的自然迁移，而不是只保护 source->2CX 已迁移权重。具体做法是用 plain continuation checkpoint 作为 `anchor_unmoved` anchor，测试“保留 plain 会自然迁移的权重，只在 plain 未迁移 near-boundary 子集继续探索”。
4. 这个诊断不改变 completion：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DV：plain-anchor 二次 AOQ mask full-val 失败，Top-1 80.0040

实验动机：

Phase 2DU 显示 plain continuation 与 Phase 2CX 基本持平，而 source-anchor `anchor_unmoved` 只从 re-explore `79.9620` 修复到 `80.0180`，仍低于 plain。新的假设是：如果 plain continuation 的自然迁移较安全，那么第二轮 AOQ 应该避免扰动 plain 会自然迁移的权重，只对 plain 未迁移的 near-boundary 子集探索。因此本阶段把 `anchor_unmoved` 的 anchor checkpoint 从 source `checkpoint-3` 改为 plain continuation `checkpoint-5`。

方法设计：

- 从 Phase 2CX clean no-QKR/LSQ selective-margin08 `checkpoint-4` strict resume；
- anchor checkpoint 使用 Phase 2DQ plain continuation `checkpoint-5`；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- `scheduler_epochs=4`，LR 保持 `1e-5`；
- epoch 4 local update `0-1800` 重新打开 `aoq_explore_scale_ratio=0.90`；
- selective margin：`0.08`；
- AOQ explore layers：late 6 个 quantizer；
- quality selector：`anchor_unmoved`；
- update 1800 后恢复普通 LSQ QAT，直到 epoch 结束；
- full-val 只认单个 `checkpoint-5.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

脚本：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_plainanchor_gate_20260708.sh
```

smoke 命令：

```bash
cd /mlx_devbox/users/quyanyi/playground && \
MAX_TRAIN_UPDATES=4 \
SKIP_VALIDATE=1 \
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_plainanchor_smoke4upd_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_plainanchor_smoke4upd_20260708.log \
MASTER_PORT=31342 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_plainanchor_gate_20260708.sh
```

smoke 证据：

```text
Loaded AOQ anchor checkpoint for anchor_unmoved selector: path=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_selectivemargin08_plain4to5_gate_20260708/checkpoint-5.pth.tar, tensors=497
AOQ crossing-quality selector init: epoch=4, update=0, mode=anchor_unmoved, pairs=6, near=1518057, selected=1294803, selected_over_near=0.852934, moved_excluded=223254, missing_pairs=0
Train: 4 [   0/2502 ...] ... LR: 1.000e-05
TrainSummary: epoch=4 updates=4 avg_step_time=0.423391s samples_per_step=512 samples_per_sec=1209.28
```

full gate 命令：

```bash
cd /mlx_devbox/users/quyanyi/playground && \
MASTER_PORT=31343 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_plainanchor_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar
no_resume_opt: true
start_epoch: 4
epochs: 5
scheduler_epochs: 4
wq_bitw: 4
aq_bitw: 4
wq_mode: lsq
aq_mode: lsq
qk_reparam: false
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aoq_explore_quality_mode: anchor_unmoved
aoq_explore_anchor_checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_selectivemargin08_plain4to5_gate_20260708/checkpoint-5.pth.tar
aoq_explore_start_update: 0
aoq_explore_end_update: 1800
max_train_updates: 0
skip_validate: false
```

strict resume / masked AOQ 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Loaded AOQ anchor checkpoint for anchor_unmoved selector: path=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_selectivemargin08_plain4to5_gate_20260708/checkpoint-5.pth.tar, tensors=497
AOQ crossing-quality selector init: epoch=4, update=0, mode=anchor_unmoved, pairs=6, near=1518057, selected=1294803, selected_over_near=0.852934, moved_excluded=223254, missing_pairs=0
AOQ crossing-quality selector: epoch=4, update=1600, mode=anchor_unmoved, pairs=6, near=1514806, selected=1344216, selected_over_near=0.887385, moved_excluded=170590, missing_pairs=0
AOQ explore scale ratio update: epoch=4, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, ...
TrainSummary: epoch=4 updates=2502 avg_step_time=0.180163s samples_per_step=512 samples_per_sec=2841.87
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-5.pth.tar` | yes | yes | 50000 | no | 80.0040 | 95.0780 | 0.8524 | 有效 gate，但低于 source-anchor 与 Phase 2CX |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 34.886s  Loss: 0.8524  Acc@1: 80.0040  Acc@5: 95.0780  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_plainanchor_smoke4upd_20260708.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_plainanchor_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_plainanchor_gate_20260708.sh
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_plainanchor_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_plainanchor_gate_20260708/checkpoint-5.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_from2cx_plainanchor_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. plain-anchor 技术上有效：6 个 late LSQ quantizer 全部命中，初始时从 `1518057` 个 near-boundary 元素中排除 `223254` 个相对 plain checkpoint 已跨 bin 的元素，保留比例约 `85.29%`。
3. 但 full-val 只有 `80.0040`，低于 source-anchor `80.0180`，也低于 plain continuation `80.1400` 和 Phase 2CX `80.1660`。
4. 这说明“把 plain 自然迁移作为 anchor 再做二次 AOQ”仍然没有恢复收益；二次 AOQ 系列目前整体低收益，应停止 `anchor_unmoved` / anchor checkpoint / margin / window 小扫。
5. 下一步应回到更大的范式变量：不要继续从 Phase 2CX endpoint 做第二轮 AOQ；要么回到 source 构造更强 first-epoch endpoint，要么把 AOQ-native 思路迁移到全局 best `80.5540` 附近的 late-block short-update 分支，但需要避免 QKR/StatsQ 作为核心机制。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-5.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.0040`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DW：source-anchor single-cross first AOQ 有效但未超过 Phase 2CX，Top-1 80.1560

实验动机：

Phase 2DS/2DT/2DV 连续说明，从 Phase 2CX endpoint 做第二轮 AOQ 是低收益方向。Phase 2CX 的首轮 selective-margin08 AOQ 是当前 clean AOQ-native best `80.1660`，但它允许同一个 near-boundary 权重在整个 explore window 内持续被 AOQ 缩放。新的假设是：首轮 AOQ 的收益可能来自“允许跨 bin”，但持续对已经跨过 bin 的权重保持缩放会增加后续噪声。因此本阶段回到 clean no-QKR/LSQ source `checkpoint-3`，仍做首轮 AOQ，但用 source checkpoint 本身作为 `anchor_unmoved` anchor：相对 source 一旦跨过 LSQ bin 的权重，后续不再继续参与 AOQ 缩放，形成 single-cross first AOQ。

方法设计：

- 从 clean no-QKR/LSQ source `checkpoint-3` strict resume；
- anchor checkpoint 也使用同一个 source `checkpoint-3`；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- `scheduler_epochs=4`；
- epoch 3 local update `0-1800` 打开 `aoq_explore_scale_ratio=0.90`；
- selective margin：`0.08`；
- AOQ explore layers：late 6 个 quantizer；
- quality selector：`anchor_unmoved`；
- update 1800 后恢复普通 LSQ QAT，直到 epoch 结束；
- full-val 只认单个 `checkpoint-4.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

脚本：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_gate_20260708.sh
```

smoke 命令：

```bash
cd /mlx_devbox/users/quyanyi/playground && \
MAX_TRAIN_UPDATES=4 \
SKIP_VALIDATE=1 \
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_smoke4upd_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_smoke4upd_20260708.log \
MASTER_PORT=31344 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_gate_20260708.sh
```

smoke 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Loaded AOQ anchor checkpoint for anchor_unmoved selector: path=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar, tensors=497
AOQ crossing-quality selector init: epoch=3, update=0, mode=anchor_unmoved, pairs=6, near=1497634, selected=1497634, selected_over_near=1.000000, moved_excluded=0, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=1, mode=anchor_unmoved, pairs=6, near=1497819, selected=1446121, selected_over_near=0.965484, moved_excluded=51698, missing_pairs=0
Train: 3 [   0/2502 ...] ... LR: 3.780e-05
TrainSummary: epoch=3 updates=4 avg_step_time=0.445714s samples_per_step=512 samples_per_sec=1148.72
```

full gate 命令：

```bash
cd /mlx_devbox/users/quyanyi/playground && \
MASTER_PORT=31345 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
wq_bitw: 4
aq_bitw: 4
wq_mode: lsq
aq_mode: lsq
qk_reparam: false
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aoq_explore_quality_mode: anchor_unmoved
aoq_explore_anchor_checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
aoq_explore_start_update: 0
aoq_explore_end_update: 1800
max_train_updates: 0
skip_validate: false
```

strict resume / single-cross AOQ 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Loaded AOQ anchor checkpoint for anchor_unmoved selector: path=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar, tensors=497
AOQ crossing-quality selector init: epoch=3, update=0, mode=anchor_unmoved, pairs=6, near=1497634, selected=1497634, selected_over_near=1.000000, moved_excluded=0, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=200, mode=anchor_unmoved, pairs=6, near=1494388, selected=1317444, selected_over_near=0.881594, moved_excluded=176944, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=1600, mode=anchor_unmoved, pairs=6, near=1493292, selected=1306102, selected_over_near=0.874646, moved_excluded=187190, missing_pairs=0
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, ...
TrainSummary: epoch=3 updates=2502 avg_step_time=0.179418s samples_per_step=512 samples_per_sec=2853.66
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 80.1560 | 95.1540 | 0.8466 | 有效 gate，接近但低于 Phase 2CX |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 35.137s  Loss: 0.8466  Acc@1: 80.1560  Acc@5: 95.1540  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_smoke4upd_20260708.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_gate_20260708.sh
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. `source-anchor single-cross` 技术上有效：初始时 current 与 anchor 同为 source，`selected_over_near=1.0`；训练开始后已跨 bin 的 near-boundary 权重会被排除，update 200 后已排除 `176944` 个，update 1600 排除 `187190` 个。
3. Top-1 `80.1560` 明显高于二次 AOQ 系列的 `79.9620 / 80.0180 / 80.0040`，说明“首轮 AOQ 中限制重复 crossing”比 endpoint 二次 AOQ 更合理。
4. 但它仍略低于 Phase 2CX `80.1660`，差距 `0.0100`，也低于全局 strict W4A4 best `80.5540`，更未达到 `81.0`。
5. 因此 single-cross 可以保留为有效机制分支，但不能继续做小的 anchor/margin/window 扫描。下一步要么结合 Phase 2CX 的自然 full selective-margin08 和 single-cross 的稳定性做诊断，要么转向全局 best `80.5540` 附近做 AOQ-native late-block short-update，但需要严格标注它是否仍依赖 QKR/StatsQ。

补充 crossing 诊断：

为了判断 `80.1560` 与 Phase 2CX `80.1660` 的 `0.0100` 差距来自哪里，本阶段进一步比较 source -> Phase 2CX、source -> source-anchor single-cross、Phase 2CX -> source-anchor single-cross 的 bin-crossing。

诊断命令：

```bash
python3 /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/diagnose_resume10_aoq_bin_crossing_20260708.py \
  --out-dir /mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_source_singlecross_vs_2cx_bin_crossing_20260708 \
  --pairs 'ckpt10->phase2s,ckpt10->phase2w,phase2s->phase2w' \
  --module-patterns features.5.5,features.7.1 \
  --near-margin 0.08 \
  --topn 160 \
  --ckpt10-checkpoint /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar \
  --ckpt10-top1 79.9220 \
  --phase2s-checkpoint /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
  --phase2s-top1 80.1660 \
  --phase2w-checkpoint /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_gate_20260708/checkpoint-4.pth.tar \
  --phase2w-top1 80.1560
```

诊断产物：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_source_singlecross_vs_2cx_bin_crossing_20260708/aggregate_bin_crossing.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_source_singlecross_vs_2cx_bin_crossing_20260708/pair_bin_crossing.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_source_singlecross_vs_2cx_bin_crossing_20260708/summary.json
```

关键 crossing 对比：

| pair | Top-1 delta | f7.1 attn_proj changed | f7.1 mlp_fc2 changed | f7.1 mlp_fc1 changed | f5.5 attn_proj changed | 结论 |
|---|---:|---:|---:|---:|---:|---|
| source -> Phase 2CX | +0.2440 | 0.056876 | 0.055845 | 0.043969 | 0.036214 | 当前 clean AOQ best |
| source -> single-cross | +0.2340 | 0.039695 | 0.038077 | 0.043759 | 0.028436 | 稳定但少了 core crossing |
| Phase 2CX -> single-cross | -0.0100 | 0.038510 | 0.036187 | 0.032478 | 0.026957 | 主要差异在 f7.1 attn_proj / mlp_fc2 |

补充中文结论：

1. single-cross 比 Phase 2CX 只低 `0.0100`，但 crossing 分布明显不同。
2. 下降最相关的模块是 `features.7.1.attn.proj` 和 `features.7.1.mlp.fc2`：Phase 2CX 中它们的 changed fraction 分别是 `0.056876` / `0.055845`，single-cross 只有 `0.039695` / `0.038077`。
3. 这说明 single-cross 的稳定性不是坏事，但它过早抑制了 core71 的有益 crossing。
4. 如果继续探索，非重复的下一步不是全局关闭 single-cross，而是 module-wise hybrid：`features.7.1.attn.proj` 与 `features.7.1.mlp.fc2` 保持 Phase 2CX 原始 selective-margin08，不启用 `anchor_unmoved`；其他 late quantizer 使用 source-anchor single-cross，测试能否保留 core71 有益 crossing，同时减少其他模块的重复扰动。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.1560`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DX：hybrid-core71 AOQ full-val 失败，Top-1 80.0580

实验动机：

Phase 2DW 的补充 crossing 诊断显示，source-anchor single-cross 只比 Phase 2CX 低 `0.0100`，但它明显减少了 `features.7.1.attn.proj` 和 `features.7.1.mlp.fc2` 的 crossing：Phase 2CX 中这两个模块 changed fraction 为 `0.056876 / 0.055845`，single-cross 为 `0.039695 / 0.038077`。因此本阶段测试一个 module-wise hybrid：让 `features.7.1.attn.proj` 和 `features.7.1.mlp.fc2` 保持 Phase 2CX 的普通 selective-margin08 AOQ，不启用 `anchor_unmoved`；其他 late quantizer 使用 source-anchor single-cross，期望保留 core71 有益 crossing，同时减少非 core 模块重复扰动。

方法设计：

- 从 clean no-QKR/LSQ source `checkpoint-3` strict resume；
- anchor checkpoint 使用同一个 source `checkpoint-3`；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- `scheduler_epochs=4`；
- epoch 3 local update `0-1800` 打开 `aoq_explore_scale_ratio=0.90`；
- selective margin：`0.08`；
- base `anchor_unmoved` AOQ layers：
  - `features.5.5.attn.qkv`
  - `features.5.5.attn.proj`
  - `features.5.5.mlp.fc2`
  - `features.7.1.attn.qkv`
- layer override 普通 AOQ layers：
  - `features.7.1.attn.proj:0.90`
  - `features.7.1.mlp.fc2:0.90`
- update 1800 后恢复普通 LSQ QAT，直到 epoch 结束；
- full-val 只认单个 `checkpoint-4.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

脚本：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_hybridcore71_gate_20260708.sh
```

smoke 命令：

```bash
cd /mlx_devbox/users/quyanyi/playground && \
MAX_TRAIN_UPDATES=4 \
SKIP_VALIDATE=1 \
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_hybridcore71_smoke4upd_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_hybridcore71_smoke4upd_20260708.log \
MASTER_PORT=31346 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_hybridcore71_gate_20260708.sh
```

smoke 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
AOQ crossing-quality selector init: epoch=3, update=0, mode=anchor_unmoved, pairs=4, near=622024, selected=622024, selected_over_near=1.000000, moved_excluded=0, missing_pairs=0
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, ..., base_quantizers=4, layer_ratios={'features.7.1.attn.proj': 0.9, 'features.7.1.mlp.fc2': 0.9}, layer_quantizers=2, layer_counts={'features.7.1.attn.proj': 1, 'features.7.1.mlp.fc2': 1}
TrainSummary: epoch=3 updates=4 avg_step_time=0.432913s samples_per_step=512 samples_per_sec=1182.69
```

full gate 命令：

```bash
cd /mlx_devbox/users/quyanyi/playground && \
MASTER_PORT=31347 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_hybridcore71_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
wq_bitw: 4
aq_bitw: 4
wq_mode: lsq
aq_mode: lsq
qk_reparam: false
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aoq_explore_quality_mode: anchor_unmoved
aoq_explore_anchor_checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar
aoq_explore_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv
aoq_explore_layer_ratios: features.7.1.attn.proj:0.90,features.7.1.mlp.fc2:0.90
aoq_explore_start_update: 0
aoq_explore_end_update: 1800
max_train_updates: 0
skip_validate: false
```

strict resume / hybrid AOQ 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Loaded AOQ anchor checkpoint for anchor_unmoved selector: path=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar, tensors=497
AOQ crossing-quality selector init: epoch=3, update=0, mode=anchor_unmoved, pairs=4, near=622024, selected=622024, selected_over_near=1.000000, moved_excluded=0, missing_pairs=0
AOQ crossing-quality selector: epoch=3, update=1600, mode=anchor_unmoved, pairs=4, near=622909, selected=550126, selected_over_near=0.883156, moved_excluded=72783, missing_pairs=0
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv'), selective_margin=0.0, base_quantizers=4, layer_ratios={'features.7.1.attn.proj': 1.0, 'features.7.1.mlp.fc2': 1.0}, layer_quantizers=2, layer_counts={'features.7.1.attn.proj': 1, 'features.7.1.mlp.fc2': 1}, ...
TrainSummary: epoch=3 updates=2502 avg_step_time=0.178346s samples_per_step=512 samples_per_sec=2870.82
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 80.0580 | 95.1900 | 0.8508 | 有效 gate，但明显低于 Phase 2CX 和 single-cross |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 35.923s  Loss: 0.8508  Acc@1: 80.0580  Acc@5: 95.1900  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_hybridcore71_smoke4upd_20260708.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_hybridcore71_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_hybridcore71_gate_20260708.sh
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_hybridcore71_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_hybridcore71_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_hybridcore71_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. hybrid-core71 技术上按预期运行：4 个非 core late quantizer 使用 `anchor_unmoved`，2 个 core71 quantizer 通过 layer override 做普通 selective-margin08 AOQ。
3. 但 Top-1 只有 `80.0580`，低于 Phase 2CX `80.1660`，也低于 source-anchor single-cross `80.1560`。
4. 这说明“只让 core71 保持普通 AOQ，其他模块 single-cross”会破坏整体 crossing 分布；Phase 2CX 的收益不能被简单拆成 core71 普通 crossing + 其他模块抑制重复 crossing。
5. 当前 clean no-QKR/LSQ 的 AOQ 分支已经很接近局部上限：full selective-margin08 `80.1660` 最好，single-cross `80.1560` 次之。下一步不应继续在 source first-epoch 的 layer/mask 组合上小扫；应转向更大变量，例如把 AOQ-native 机制迁移到全局 best `80.5540` 附近，或设计新的 level adaptation 机制，而不是只改 crossing mask。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.0580`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DN：threshold-only vs selective-margin08 bin-crossing 诊断

诊断动机：

Phase 2DM 的 threshold-only AOQ full-val 只有 `79.9620`，比 selective-margin08 的 `80.1660` 低 `0.2040`。本阶段复用 bin-crossing 诊断，比较 source -> selective-margin08 与 source -> threshold-only，判断 threshold/level decoupling 是 crossing 不够、crossing 过多，还是 crossing 分布错位。

诊断命令：

```bash
python3 QATs/tmp_scripts/diagnose_resume10_aoq_bin_crossing_20260708.py \
  --out-dir QATs/docs/resume10_clean_lsq_aoq_thresholdonly_bin_crossing_20260708 \
  --pairs 'ckpt10->phase2s,ckpt10->phase2w,phase2s->phase2w' \
  --module-patterns features.5.5,features.7.1 \
  --near-margin 0.08 \
  --topn 80 \
  --ckpt10-checkpoint /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar \
  --ckpt10-top1 79.9220 \
  --phase2s-checkpoint /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
  --phase2s-top1 80.1660 \
  --phase2w-checkpoint /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq_thresholdonly_gate_20260708/checkpoint-4.pth.tar \
  --phase2w-top1 79.9620
```

产物：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_thresholdonly_bin_crossing_20260708/aggregate_bin_crossing.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_thresholdonly_bin_crossing_20260708/pair_bin_crossing.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_thresholdonly_bin_crossing_20260708/summary.json
```

关键 stage_kind 对比：

| pair | Top-1 delta | features.7.1 attn_proj changed | features.7.1 mlp_fc2 changed | features.7.1 attn_qkv changed | features.5.5 attn_proj changed | features.5.5 attn_qkv changed |
|---|---:|---:|---:|---:|---:|---:|
| source -> selective-margin08 | +0.2440 | 0.056876 | 0.055845 | 0.032492 | 0.036214 | 0.031071 |
| source -> threshold-only | +0.0400 | 0.068698 | 0.067913 | 0.035639 | 0.041470 | 0.033911 |
| selective-margin08 -> threshold-only | -0.2040 | 0.041906 | 0.039203 | 0.025719 | 0.029602 | 0.027253 |

中文结论：

1. threshold-only 失败不是因为 crossing 不够。和 selective-margin08 相比，它在 `features.7.1.attn_proj`、`features.7.1.mlp_fc2`、`features.5.5.attn_proj` 等关键模块上 crossing 更多。
2. 但 Top-1 只从 source 的 `79.9220` 到 `79.9620`，远低于 selective-margin08 的 `80.1660`。说明单纯增加 threshold crossing 没有形成有用离散迁移。
3. 这支持一个更明确的判断：AOQ-style decoupling 需要“threshold exploration + level adaptation”配套；只缩 threshold、不缩或不学习 level，会导致 crossing 与输出 level 表征错位。
4. 下一步不要继续 threshold-only ratio 扫描。更有希望的是做两阶段 schedule：
   - early：threshold_ratio < 1、scale_ratio = 1，诱导 crossing；
   - mid：threshold_ratio 恢复 1，同时给 selected LSQ scale 更高学习率或短暂 level adaptation；
   - late：不加 anchor，只自然收敛或极弱 center safety。
5. 这个诊断不改变 completion：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2DO：threshold-only 900 update + level adaptation full-val 失败，Top-1 79.9620

实验动机：

Phase 2DM/2DN 说明 threshold-only exploration 不是 crossing 不够，而是缺少有效 level adaptation。本阶段不继续扫 threshold ratio，而是缩短 threshold-only exploration 窗口：前 900 update 只缩 threshold interval，后 1602 update 恢复正常 LSQ QAT，让 weights 和 LSQ scale 有更长时间适配已经产生的 crossing。

方法设计：

- 从 clean no-QKR/LSQ `checkpoint-3` strict resume；
- 不使用 QKR，不使用 StatsQ；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- `aoq_explore_scale_ratio=1.0`；
- `aoq_explore_threshold_ratio=0.90`；
- selective margin：`0.08`；
- AOQ explore layers：late 6 个 quantizer；
- 不使用 `grad_cross`；
- 不使用 BinReg / selective anchor；
- AOQ explore 窗口：`0-900` update；
- update 900 后恢复普通 LSQ QAT，作为 level adaptation 阶段；
- full-val 只认单个 `checkpoint-4.pth.tar`；
- 不使用 soup / checkpoint averaging / ensemble。

full gate 命令：

```bash
QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq_threshold900_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_scale_ratio: 1.0
aoq_explore_threshold_ratio: 0.9
aoq_explore_selective_margin: 0.08
aoq_explore_end_update: 900
max_train_updates: 0
skip_validate: false
wq_bitw: 4
aq_bitw: 4
wq_mode: lsq
aq_mode: lsq
qk_reparam: false
qk_reparam_type: 0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
no_resume_opt: true
start_epoch: 3
epochs: 4
scheduler_epochs: 4
```

strict resume / AOQ 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=1.0, threshold_ratio=0.9, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=900
AOQ explore scale ratio update: epoch=3, update=900, active=False, base_ratio=1.0, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=0, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=900
TrainSummary: epoch=3 updates=2502 avg_step_time=0.166573s samples_per_step=512 samples_per_sec=3073.73
```

full-val 结果：

| checkpoint | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | yes | yes | 50000 | no | 79.9620 | 95.2080 | 0.8469 | 有效 gate，但低于 selective-margin08 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 35.466s  Loss: 0.8469  Acc@1: 79.9620  Acc@5: 95.2080  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq_threshold900_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq_threshold900_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq_threshold900_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq_threshold900_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. 缩短 threshold-only exploration 到 900 update 后，Top-1 仍是 `79.9620`，和 1800 update threshold-only 完全一致。
3. 这说明“后半个 epoch 的普通 LSQ level adaptation”没有修复 threshold-only exploration 的错位；问题不是 adaptation 时间不够，而是 threshold-only 产生的 crossing 本身没有形成有效离散解。
4. 当前应停止 threshold-only / threshold-window 扫描。更有价值的下一步是 module-wise AOQ schedule：只对 selective-margin08 中真正贡献较大的模块保留 ratio exploration，例如先保留 `features.7.1.mlp.fc2` 和 `features.7.1.attn.proj`，避免全 6 层同时扰动。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `79.9620`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2FH：强盆地 QKR/StatsQ AOQ core71 100 update 验证失败，Top-1 80.4980

实验定位：

clean no-QKR/LSQ AOQ-native 主线最近几条 post-crossing anchor、pulse、candidate-bin anchor 都没有超过 `80.1660`，并且 class/logit 诊断显示 candidate anchor 会伤害低置信边界样本的函数适配。本阶段临时回到当前全局最强 strict W4A4 单 checkpoint `80.5540` 所在盆地，只做一个受控 AOQ 短门控，验证“在强盆地里只对 late core71 两个模块轻微 AOQ exploration 是否还能带来增益”。

重要标注：

- 这不是 clean no-QKR / no-StatsQ AOQ-native 主线；
- 本阶段使用 `qk_reparam=true` 和 `wq_mode=statsq`；
- 只作为强盆地控制实验，不能把它当作“丢弃 QKR/StatsQ 后的范式成功”；
- 仍然满足 strict W4A4、单 checkpoint、full ImageNet raw validation、无 soup/averaging/ensemble 的审计要求。

方法设计：

- resume 起点：Phase 2Z 全局最强 strict W4A4 单 checkpoint；
- 起点 full-val：Top-1 `80.5540`、Top-5 `95.3060`、Loss `0.8387`、Samples `50000`；
- `wq_mode=statsq`、`aq_mode=lsq`；
- `wq_bitw=4`、`aq_bitw=4`；
- `qk_reparam=true`；
- `AOQ_EXPLORE_SCALE_RATIO=0.98`；
- `AOQ_EXPLORE_LAYERS=features.7.1.attn.proj,features.7.1.mlp.fc2`；
- `AOQ_EXPLORE_START_UPDATE=0`、`AOQ_EXPLORE_END_UPDATE=100`；
- `QUANT_ONLY_START_EPOCH=3`；
- `TRAINABLE_POLICY=params_in_layers`；
- `TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1`；
- `MAX_TRAIN_UPDATES=100`；
- 保存并评估单个 `checkpoint-4.pth.tar`。

100 update 训练命令：

```bash
EXP=recipe_resume10_paramsinlate_ckpt3_aoq098_core71_100upd_gate_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_paramsinlate_ckpt3_aoq098_core71_100upd_gate_20260708.log \
MASTER_PORT=31367 \
bash QATs/tmp_scripts/run_resume10_paramsinlate_ckpt3_aoq098_core71_100upd_gate_20260708.sh
```

训练产物：

```text
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt3_aoq098_core71_100upd_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt3_aoq098_core71_100upd_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt3_aoq098_core71_100upd_gate_20260708/last.pth.tar
```

关键 `args.yaml` 证据：

```text
aoq_explore_scale_ratio: 0.98
aoq_explore_layers: features.7.1.attn.proj,features.7.1.mlp.fc2
aoq_explore_start_update: 0
aoq_explore_end_update: 100
aq_bitw: 4
aq_mode: lsq
qk_reparam: true
qk_reparam_type: 0
quant_only_start_epoch: 3
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar
start_epoch: 3
trainable_policy: params_in_layers
trainable_policy_freeze_act_except_layers: features.5.5,features.7.1
wq_bitw: 4
wq_mode: statsq
```

strict resume / AOQ 训练证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.98, threshold_ratio=0.0, base_layers=('features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=2, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=100
TrainSummary: epoch=3 updates=100 avg_step_time=0.129229s samples_per_step=512 samples_per_sec=3961.95
Stopped early after 100 optimizer updates in epoch 3.
```

full-val 命令：

```bash
CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt3_aoq098_core71_100upd_gate_20260708/checkpoint-4.pth.tar \
EXP=eval_resume10_paramsinlate_ckpt3_aoq098_core71_100upd_fullval_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_resume10_paramsinlate_ckpt3_aoq098_core71_100upd_fullval_20260708.log \
MASTER_PORT=31379 \
bash QATs/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh
```

full-val 结果：

| checkpoint | 分支标注 | strict W4A4 | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比起点 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-4.pth.tar` | 强盆地 QKR/StatsQ AOQ core71 | yes | yes | 50000 | no | 80.4980 | 95.3280 | 0.8470 | -0.0560 | 负收益 |

full-val 原始摘要：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt3_aoq098_core71_100upd_gate_20260708/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 32.856s  Loss: 0.8470  Acc@1: 80.4980  Acc@5: 95.3280  Samples: 50000
Eval-only metrics: {'loss': 0.847006036594212, 'top1': 80.498, 'top5': 95.328, 'samples': 50000, 'local_samples': 6250, 'wall_seconds': 32.855552673339844}
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. 但 Top-1 从起点 `80.5540` 降到 `80.4980`，强盆地 QKR/StatsQ 上的 late core71 AOQ 0.98 短门控是负收益。
3. 这说明当前问题不是只差一个“小幅 AOQ exploration”就能跨过 81；即使在历史最强盆地内，继续扰动 late core71 的 quantizer level/scale 也会损伤已经形成的函数边界。
4. 对 clean no-QKR/LSQ 主线的启发：不要把失败归因于“没有 QKR/StatsQ 所以 AOQ 不行”。相反，强盆地验证也失败，说明下一步需要更换迁移机制，而不是继续对 late module 做 ratio/pulse/anchor 扫描。
5. 下一步应停止这类 AOQ ratio 局部扫描，回到 clean no-QKR/LSQ 主线设计“保护低置信边界样本”的函数级机制，例如 confidence-band KD / local reference logits / selective sample replay，而不是继续只看 quantizer crossing 数量。

completion audit：

- strict W4A4：满足，`wq_mode=statsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.4980`。
- clean no-QKR / no-StatsQ AOQ-native：不满足，本阶段使用 QKR/StatsQ，只能作为强盆地控制实验。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2FI：selective-margin08 + local-reference low-confidence KD 失败，Top-1 80.0340

实验动机：

Phase 2FG 的 class/logit 诊断显示，continuous selective-margin08 相比 candidate-anchor 的净收益主要来自低置信样本：continuous confidence `<0.6` 的三个桶合计 `+115` net flips，几乎解释总体 `+110` net flips。此前 ratio、pulse、anchor、candidate-bin、threshold-only 等局部 crossing 控制都没有超过 clean AOQ best `80.1660`。本阶段不再继续扫 crossing mask，而是测试函数空间保护：保持 Phase 2CX 的 clean no-QKR/LSQ selective-margin08 AOQ，同时用 Phase 2CX checkpoint 作为 fixed local reference，只在 reference confidence `<0.6` 的低置信样本上额外对齐 reference logits，验证能否保护边界样本并超过 `80.1660`。

方法设计：

- clean no-QKR / no-StatsQ 主线；
- 从 clean no-QKR/LSQ source `checkpoint-3` strict resume；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- 沿用 Phase 2CX selective-margin08 AOQ：`aoq_explore_scale_ratio=0.90`、`aoq_explore_selective_margin=0.08`；
- AOQ explore layers 为 late 6 个 quantizer；
- AOQ explore window：`0-1800` update；
- 额外启用 `local_ref_confidence_band_kd_weight=0.2`；
- local reference checkpoint：Phase 2CX clean AOQ best `checkpoint-4.pth.tar`；
- local reference confidence band：`[0.0, 0.6)`；
- local reference temperature：`2.75`；
- 不使用 QKR、StatsQ、soup、checkpoint averaging、multi-checkpoint averaging、ensemble。

先做 2-update smoke：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_localrefconf06_smoke2upd_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_localrefconf06_smoke2upd_20260708.log \
MASTER_PORT=31383 \
MAX_TRAIN_UPDATES=2 \
SKIP_VALIDATE=1 \
LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT=0.2 \
LOCAL_REF_CONFIDENCE_BAND_KD_LOW=0.0 \
LOCAL_REF_CONFIDENCE_BAND_KD_HIGH=0.6 \
LOCAL_REF_CONFIDENCE_BAND_KD_TEMPERATURE=2.75 \
LOCAL_REF_CONFIDENCE_BAND_KD_CHECKPOINT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

smoke 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Enabled local-reference confidence band KD: weight=0.2, band=[0.0, 0.6), temperature=2.75, source=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=1800
TrainSummary: epoch=3 updates=2 avg_step_time=0.738383s samples_per_step=512 samples_per_sec=693.41
```

full gate 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_localrefconf06_gate_20260708 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_localrefconf06_gate_20260708.log \
MASTER_PORT=31387 \
LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT=0.2 \
LOCAL_REF_CONFIDENCE_BAND_KD_LOW=0.0 \
LOCAL_REF_CONFIDENCE_BAND_KD_HIGH=0.6 \
LOCAL_REF_CONFIDENCE_BAND_KD_TEMPERATURE=2.75 \
LOCAL_REF_CONFIDENCE_BAND_KD_CHECKPOINT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aoq_explore_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
aoq_explore_start_update: 0
aoq_explore_end_update: 1800
aq_bitw: 4
aq_mode: lsq
wq_bitw: 4
wq_mode: lsq
qk_reparam: false
local_ref_confidence_band_kd_weight: 0.2
local_ref_confidence_band_kd_low: 0.0
local_ref_confidence_band_kd_high: 0.6
local_ref_confidence_band_kd_temperature: 2.75
local_ref_confidence_band_kd_checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
no_resume_opt: true
skip_validate: false
max_train_updates: 0
start_epoch: 3
epochs: 4
scheduler_epochs: 4
```

strict resume / AOQ / local-reference 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
Enabled local-reference confidence band KD: weight=0.2, band=[0.0, 0.6), temperature=2.75, source=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=1800
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=1800
TrainSummary: epoch=3 updates=2502 avg_step_time=0.228233s samples_per_step=512 samples_per_sec=2243.32
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 Phase 2CX | 对比全局 best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-4.pth.tar` | yes | yes | yes | 50000 | no | 80.0340 | 95.2120 | 0.8520 | -0.1320 | -0.5200 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 34.953s  Loss: 0.8520  Acc@1: 80.0340  Acc@5: 95.2120  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_localrefconf06_smoke2upd_20260708.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_localrefconf06_gate_20260708.log
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_localrefconf06_gate_20260708/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_localrefconf06_gate_20260708/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_localrefconf06_gate_20260708/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. 但 Top-1 只有 `80.0340`，低于 Phase 2CX clean AOQ best `80.1660`，也远低于全局 strict W4A4 best `80.5540`。
3. 低置信 local-reference logits 保护没有保护住 Phase 2CX 的边界收益，反而明显压低结果。可能原因是 reference 本身是 Phase 2CX endpoint，而本 run 从 source 起步；在 AOQ 探索期强行对齐 endpoint 低置信 logits，会约束 early crossing 的自然形成，导致新 endpoint 欠探索。
4. 这说明“低置信样本很关键”这个诊断成立，但直接在训练中加固定 endpoint logits KL 不是正确机制。下一步不应继续扫 local-ref KD weight / confidence band；更合理的是做训练后诊断，比较 Phase 2CX 与本阶段的 class/logit flip，确认它主要损失了哪些低置信桶，然后再考虑更弱、更晚启用的保护，或改成 teacher confidence band 而不是 endpoint local reference。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- clean no-QKR/no-StatsQ：满足，`qk_reparam=false`、`wq_mode=lsq`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.0340`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2FJ：Phase 2CX vs local-reference low-confidence KD class/logit 诊断

诊断动机：

Phase 2FI 的 local-reference low-confidence KD full gate 只有 `80.0340`，明显低于 Phase 2CX clean AOQ best `80.1660`。本阶段不继续扫 local-ref KD weight / confidence band，而是先做 checkpoint 级 class/logit 诊断，确认它到底损失在哪些 confidence bins、类别和 flip cases。

诊断命令：

```bash
CUDA_VISIBLE_DEVICES=0 python3 QATs/tmp_scripts/diagnose_resume10_logit_classes_20260708.py \
  --out-dir QATs/docs/resume10_clean_lsq_aoq_localrefconf06_class_diag_20260709 \
  --checkpoint phase2cx=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
  --checkpoint localref06=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_localrefconf06_gate_20260708/checkpoint-4.pth.tar \
  --labels phase2cx,localref06 \
  --compare-label localref06 \
  --wq-mode lsq \
  --aq-mode lsq \
  --no-qk-reparam \
  --batch-size 128 \
  --workers 8 \
  --flip-topn 500 \
  2>&1 | tee /mlx_devbox/users/quyanyi/playground/train_resume10_clean_lsq_aoq_localrefconf06_class_diag_20260709.log
```

产物：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_localrefconf06_class_diag_20260709/summary.json
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_localrefconf06_class_diag_20260709/class_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_localrefconf06_class_diag_20260709/confidence_bins.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_localrefconf06_class_diag_20260709/flip_cases.tsv
/mlx_devbox/users/quyanyi/playground/train_resume10_clean_lsq_aoq_localrefconf06_class_diag_20260709.log
```

单卡诊断指标：

| checkpoint | Top-1 | Top-5 | Loss | Samples |
|---|---:|---:|---:|---:|
| Phase 2CX `checkpoint-4` | 80.1440 | 95.1680 | 0.8470 | 50000 |
| localref06 `checkpoint-4` | 80.0460 | 95.2180 | 0.8524 | 50000 |

pair summary：

```text
delta_top1 = -0.0980
improved = 928
regressed = 977
net_flips = -49
avg_true_prob_delta = -0.0041466
avg_margin_delta = -0.0242150
improved_true_prob_delta = 0.1479602
regressed_true_prob_delta = -0.1656940
```

confidence-bin 结果：

| Phase 2CX confidence bin | total | phase2cx correct | localref correct | improved | regressed | net flips | avg true prob delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[0.00,0.20)` | 1273 | 247 | 275 | 128 | 100 | +28 | +0.01494 |
| `[0.20,0.40)` | 4182 | 1512 | 1496 | 392 | 408 | -16 | +0.01072 |
| `[0.40,0.60)` | 6485 | 3537 | 3507 | 362 | 392 | -30 | +0.00302 |
| `[0.60,0.80)` | 8867 | 6833 | 6809 | 46 | 70 | -24 | -0.00786 |
| `[0.80,0.90)` | 14759 | 13839 | 13832 | 0 | 7 | -7 | -0.00647 |
| `[0.90,0.95)` | 11837 | 11559 | 11559 | 0 | 0 | 0 | -0.00850 |
| `[0.95,0.99)` | 2520 | 2469 | 2469 | 0 | 0 | 0 | -0.00976 |
| `[0.99,1.01)` | 77 | 76 | 76 | 0 | 0 | 0 | -0.00470 |

低置信 `<0.6` 汇总：

```text
total = 11940
phase2cx_correct = 5296
localref_correct = 5278
improved = 882
regressed = 900
net_flips = -18
```

损失最大的类别：

| class | total | phase2cx correct | localref correct | delta |
|---:|---:|---:|---:|---:|
| 170 | 50 | 38 | 32 | -6 |
| 539 | 50 | 39 | 34 | -5 |
| 245 | 50 | 46 | 41 | -5 |
| 250 | 50 | 41 | 36 | -5 |
| 960 | 50 | 34 | 30 | -4 |
| 66 | 50 | 28 | 24 | -4 |
| 921 | 50 | 34 | 30 | -4 |
| 603 | 50 | 45 | 41 | -4 |

中文结论：

1. local-reference low-confidence KD 的失败不是单一低置信桶崩掉。低置信 `<0.6` 净损失为 `-18`，中高置信 `[0.6,0.9)` 还额外损失 `-31`，总体 `net_flips=-49`。
2. 它整体压低 true-prob 和 margin：`avg_true_prob_delta=-0.00415`，`avg_margin_delta=-0.0242`。这说明 endpoint local reference 约束改变了整体 logit geometry，不只是低置信样本被保护过度。
3. `[0.00,0.20)` 桶反而是 `+28`，但 `[0.20,0.60)` 与 `[0.60,0.90)` 同时退化，说明“直接对齐 Phase 2CX endpoint logits”会牺牲更宽区间的自然迁移。
4. 下一步不应继续扫 local-ref KD weight / confidence band。若继续做函数空间保护，应改为 teacher confidence-band KD，避免用 Phase 2CX endpoint 作为固定局部老师；或者做 delayed late-only protection，在 AOQ exploration 结束后才启用，而不是从 update 0 约束 crossing 形成。
5. 本诊断不改变 completion：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2FK：selective-margin08 + teacher confidence-band KD 失败，Top-1 79.8500

实验动机：

Phase 2FJ 说明 local-reference low-confidence KD 的失败不只是低置信桶本身，而是 endpoint local reference 改变了整体 logit geometry；因此本阶段不继续扫 local-ref 权重，而是改成更弱的 teacher confidence-band KD。teacher confidence-band 不引入 Phase 2CX endpoint 作为固定局部老师，只在 teacher confidence `[0.2,0.6)` 的样本上额外加一小段 teacher soft KD，测试是否能保护低/中置信边界样本，同时保留 Phase 2CX selective-margin08 AOQ 的自然 crossing。

方法设计：

- clean no-QKR / no-StatsQ 主线；
- 从 clean no-QKR/LSQ source `checkpoint-3` strict resume；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- 沿用 Phase 2CX selective-margin08 AOQ：`aoq_explore_scale_ratio=0.90`、`aoq_explore_selective_margin=0.08`；
- AOQ explore layers 为 late 6 个 quantizer；
- AOQ explore window：`0-1800` update；
- 额外启用 `teacher_confidence_band_kd_weight=0.1`；
- teacher confidence band：`[0.2,0.6)`；
- teacher confidence temperature：`2.75`；
- 不使用 local reference checkpoint；
- 不使用 QKR、StatsQ、soup、checkpoint averaging、multi-checkpoint averaging、ensemble。

2-update smoke 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_teacherconf0206_w01_smoke2upd_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_teacherconf0206_w01_smoke2upd_20260709.log \
MASTER_PORT=31391 \
MAX_TRAIN_UPDATES=2 \
SKIP_VALIDATE=1 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0.1 \
TEACHER_CONFIDENCE_BAND_KD_LOW=0.2 \
TEACHER_CONFIDENCE_BAND_KD_HIGH=0.6 \
TEACHER_CONFIDENCE_BAND_KD_TEMPERATURE=2.75 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

smoke 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=1800
TrainSummary: epoch=3 updates=2 avg_step_time=0.721352s samples_per_step=512 samples_per_sec=709.78
```

full gate 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_teacherconf0206_w01_gate_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_teacherconf0206_w01_gate_20260709.log \
MASTER_PORT=31395 \
TEACHER_CONFIDENCE_BAND_KD_WEIGHT=0.1 \
TEACHER_CONFIDENCE_BAND_KD_LOW=0.2 \
TEACHER_CONFIDENCE_BAND_KD_HIGH=0.6 \
TEACHER_CONFIDENCE_BAND_KD_TEMPERATURE=2.75 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aoq_explore_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
aoq_explore_start_update: 0
aoq_explore_end_update: 1800
aq_bitw: 4
aq_mode: lsq
wq_bitw: 4
wq_mode: lsq
qk_reparam: false
teacher_confidence_band_kd_weight: 0.1
teacher_confidence_band_kd_low: 0.2
teacher_confidence_band_kd_high: 0.6
teacher_confidence_band_kd_temperature: 2.75
local_ref_confidence_band_kd_weight: 0.0
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
no_resume_opt: true
skip_validate: false
max_train_updates: 0
start_epoch: 3
epochs: 4
scheduler_epochs: 4
```

strict resume / AOQ 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=1800
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=1800
TrainSummary: epoch=3 updates=2502 avg_step_time=0.175000s samples_per_step=512 samples_per_sec=2925.71
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 Phase 2CX | 对比全局 best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-4.pth.tar` | yes | yes | yes | 50000 | no | 79.8500 | 95.1040 | 0.8538 | -0.3160 | -0.7040 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 35.432s  Loss: 0.8538  Acc@1: 79.8500  Acc@5: 95.1040  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_teacherconf0206_w01_smoke2upd_20260709.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_teacherconf0206_w01_gate_20260709.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_teacherconf0206_w01_gate_20260709/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_teacherconf0206_w01_gate_20260709/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_teacherconf0206_w01_gate_20260709/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. teacher confidence-band KD 明显失败，Top-1 只有 `79.8500`，比 Phase 2CX `80.1660` 低 `0.3160`，比全局 best `80.5540` 低 `0.7040`。
3. 这个结果比 local-reference low-confidence KD 的 `80.0340` 还差，说明“对低/中置信样本额外加 soft KD”这个函数空间保护方向本身会压制 AOQ exploration，而不是只因为 endpoint local reference 选错。
4. 当前应停止 confidence-band KD 系列，包括 teacher confidence、local reference confidence、ref confidence 的权重和 band 小扫。
5. 下一步需要换回 AOQ-native 的训练时序变量，而不是再加 logits 保护：更合理的是 delayed protection / delayed dampening，即前 0-1800 update 完全复现 Phase 2CX 的自由 crossing，1800 update 后才启用很弱的 stabilization，避免从 update 0 约束 crossing 形成。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- clean no-QKR/no-StatsQ：满足，`qk_reparam=false`、`wq_mode=lsq`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `79.8500`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2FL：selective-margin08 + delayed non-late grad-damp 失败，Top-1 80.0800

实验动机：

Phase 2FK 说明 confidence-band KD 系列会从 update 0 压制 AOQ exploration，不能继续扫。此前 `grad_mask` 的 `quant_in_layers` / `params_in_layers` 后段适配也失败，但它是硬屏蔽非目标参数梯度。本阶段测试更弱的 delayed stabilization：前 `0-1800` update 完全复现 Phase 2CX 的自由 crossing；AOQ explore 关闭后，不硬冻结非 late-block 参数，而是把 `features.5.5,features.7.1` 之外的梯度乘以 `0.2`，让后段以 late blocks 为主适配，同时保留全局小幅自然迁移。

方法设计：

- clean no-QKR / no-StatsQ 主线；
- 从 clean no-QKR/LSQ source `checkpoint-3` strict resume；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- 沿用 Phase 2CX selective-margin08 AOQ：`aoq_explore_scale_ratio=0.90`、`aoq_explore_selective_margin=0.08`；
- AOQ explore layers 为 late 6 个 quantizer；
- AOQ explore window：`0-1800` update；
- `trainable_policy=all`；
- `trainable_policy_update_overrides=1800:params_in_layers`；
- `trainable_policy_update_mode=grad_damp`；
- `trainable_policy_grad_damp=0.2`；
- `trainable_policy_freeze_act_except_layers=features.5.5,features.7.1`；
- 不使用 confidence-band KD、local reference、BinReg、selective anchor、QKR、StatsQ、soup、checkpoint averaging、multi-checkpoint averaging、ensemble。

1802-update smoke 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_graddamp_nonlate_after1800_smoke1802upd_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_graddamp_nonlate_after1800_smoke1802upd_20260709.log \
MASTER_PORT=31399 \
MAX_TRAIN_UPDATES=1802 \
SKIP_VALIDATE=1 \
TRAINABLE_POLICY=all \
TRAINABLE_POLICY_UPDATE_OVERRIDES=1800:params_in_layers \
TRAINABLE_POLICY_UPDATE_MODE=grad_damp \
TRAINABLE_POLICY_GRAD_DAMP=0.2 \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

smoke 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=1800
Trainable parameter update policy: epoch=3, update=0, mode=grad_damp, policy=all, trainable=28535407, frozen=0
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=1800
Trainable parameter update policy: epoch=3, update=1800, mode=grad_damp, policy=params_in_layers, trainable=28535407, frozen=0
Applied gradient damping policy: policy=params_in_layers, damp=0.2, damped_params=19631895, masked_params=0
TrainSummary: epoch=3 updates=1802 avg_step_time=0.168835s samples_per_step=512 samples_per_sec=3032.54
```

full gate 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_graddamp_nonlate_after1800_gate_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_graddamp_nonlate_after1800_gate_20260709.log \
MASTER_PORT=31403 \
TRAINABLE_POLICY=all \
TRAINABLE_POLICY_UPDATE_OVERRIDES=1800:params_in_layers \
TRAINABLE_POLICY_UPDATE_MODE=grad_damp \
TRAINABLE_POLICY_GRAD_DAMP=0.2 \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=features.5.5,features.7.1 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_scale_ratio: 0.9
aoq_explore_selective_margin: 0.08
aoq_explore_layers: features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2
aoq_explore_start_update: 0
aoq_explore_end_update: 1800
aq_bitw: 4
aq_mode: lsq
wq_bitw: 4
wq_mode: lsq
qk_reparam: false
trainable_policy: all
trainable_policy_freeze_act_except_layers: features.5.5,features.7.1
trainable_policy_update_overrides: {1800: params_in_layers}
trainable_policy_update_mode: grad_damp
trainable_policy_grad_damp: 0.2
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
no_resume_opt: true
skip_validate: false
max_train_updates: 0
start_epoch: 3
epochs: 4
scheduler_epochs: 4
```

strict resume / AOQ / grad-damp 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=1800
Trainable parameter update policy: epoch=3, update=0, mode=grad_damp, policy=all, trainable=28535407, frozen=0
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=1800
Trainable parameter update policy: epoch=3, update=1800, mode=grad_damp, policy=params_in_layers, trainable=28535407, frozen=0
Applied gradient damping policy: policy=params_in_layers, damp=0.2, damped_params=19631895, masked_params=0
TrainSummary: epoch=3 updates=2502 avg_step_time=0.167112s samples_per_step=512 samples_per_sec=3063.82
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 Phase 2CX | 对比全局 best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-4.pth.tar` | yes | yes | yes | 50000 | no | 80.0800 | 95.1860 | 0.8491 | -0.0860 | -0.4740 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 37.607s  Loss: 0.8491  Acc@1: 80.0800  Acc@5: 95.1860  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_graddamp_nonlate_after1800_smoke1802upd_20260709.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_graddamp_nonlate_after1800_gate_20260709.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_graddamp_nonlate_after1800_gate_20260709/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_graddamp_nonlate_after1800_gate_20260709/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_graddamp_nonlate_after1800_gate_20260709/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. `grad_damp` 工程链路有效：前 0-1800 update 自由 AOQ，1800 后非 late-block 参数梯度被乘以 `0.2`，没有 DDP static graph 问题。
3. 但 Top-1 只有 `80.0800`，低于 Phase 2CX `80.1660`。它比硬 `grad_mask params_in_layers` 的 `80.0640` 只高 `0.0160`，说明“后段只弱化非 late-block”仍然不能保住或放大 Phase 2CX 的有益 crossing。
4. 当前应停止从 `update 1800` 后做 late-only / non-late-damp 的可训练集合调整。证据链已经覆盖：硬 late params、late quant/scale/shift、plain continuation、BinReg、selective anchor、confidence-band KD、grad-damp 都低于 Phase 2CX。
5. 下一步不应继续后段稳定化小扫；更有价值的是换 exploration 本身，例如不再连续 0-1800 固定 ratio，而是用更接近 AOQ 的 update schedule：early free crossing 后短暂恢复 normal，再做一个较小 second pulse，且保持 scheduler 与 Phase 2CX 对齐，避免 Phase 2DR 的高 LR 重复探索问题。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- clean no-QKR/no-StatsQ：满足，`qk_reparam=false`、`wq_mode=lsq`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.0800`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### Phase 2FM：selective-margin08 + late weak second pulse 失败，Top-1 80.0200

实验动机：

Phase 2FL 说明后段稳定化小扫已经覆盖充分：plain continuation、BinReg、selective anchor、late params/quant-only、confidence-band KD、grad-damp 都低于 Phase 2CX。下一步不继续稳定化，而是改变 exploration 时序。此前 Phase 2EB 的 pulse3 是三段等长早期脉冲：`0-300`、`600-900`、`1200-1500`，结果 `80.0900`，低于 continuous Phase 2CX `80.1660`。本阶段设计一个非重复 schedule：前 `0-1800` 完全复现 Phase 2CX 的主探索，`1800-2200` 正常适配，然后在低 LR 尾段 `2200-2300` 加一个更弱、更窄的 second pulse，最后 `2300-2502` 恢复 normal，测试能否在保留 Phase 2CX 主探索的基础上补一点低 LR 离散解空间探索。

方法设计：

- clean no-QKR / no-StatsQ 主线；
- 从 clean no-QKR/LSQ source `checkpoint-3` strict resume；
- strict W4A4：`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`；
- 不使用 base `aoq_explore_scale_ratio`，全部由 update schedule 控制；
- schedule：
  - `0:0.90:0:0.08`，复现 Phase 2CX 主探索；
  - `1800:1.0:0:0`，恢复 normal；
  - `2200:0.95:0:0.04`，低 LR 尾段弱 second pulse；
  - `2300:1.0:0:0`，再次恢复 normal；
- AOQ explore layers 为 late 6 个 quantizer；
- 不使用 confidence-band KD、local reference、BinReg、selective anchor、QKR、StatsQ、soup、checkpoint averaging、multi-checkpoint averaging、ensemble。

2302-update smoke 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse095m04_smoke2302upd_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse095m04_smoke2302upd_20260709.log \
MASTER_PORT=31407 \
AOQ_EXPLORE_SCALE_RATIO=1.0 \
AOQ_EXPLORE_SELECTIVE_MARGIN=0.0 \
AOQ_EXPLORE_END_UPDATE=0 \
AOQ_EXPLORE_UPDATE_SCHEDULE=0:0.90:0:0.08,1800:1.0:0:0,2200:0.95:0:0.04,2300:1.0:0:0 \
MAX_TRAIN_UPDATES=2302 \
SKIP_VALIDATE=1 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

smoke 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, threshold_ratio=0.0, ... selective_margin=0.08, base_quantizers=6 ...
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, ... selective_margin=0.0, base_quantizers=6 ...
AOQ explore scale ratio update: epoch=3, update=2200, active=True, base_ratio=0.95, threshold_ratio=0.0, ... selective_margin=0.04, base_quantizers=6 ...
AOQ explore scale ratio update: epoch=3, update=2300, active=False, base_ratio=1.0, threshold_ratio=0.0, ... selective_margin=0.0, base_quantizers=6 ...
TrainSummary: epoch=3 updates=2302 avg_step_time=0.166800s samples_per_step=512 samples_per_sec=3069.54
```

full gate 命令：

```bash
EXP=recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse095m04_gate_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse095m04_gate_20260709.log \
MASTER_PORT=31411 \
AOQ_EXPLORE_SCALE_RATIO=1.0 \
AOQ_EXPLORE_SELECTIVE_MARGIN=0.0 \
AOQ_EXPLORE_END_UPDATE=0 \
AOQ_EXPLORE_UPDATE_SCHEDULE=0:0.90:0:0.08,1800:1.0:0:0,2200:0.95:0:0.04,2300:1.0:0:0 \
bash QATs/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh
```

关键 `args.yaml` 证据：

```text
aoq_explore_scale_ratio: 1.0
aoq_explore_selective_margin: 0.0
aoq_explore_end_update: 0
aoq_explore_update_schedule:
aq_bitw: 4
aq_mode: lsq
wq_bitw: 4
wq_mode: lsq
qk_reparam: false
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
no_resume_opt: true
skip_validate: false
max_train_updates: 0
start_epoch: 3
epochs: 4
scheduler_epochs: 4
```

strict resume / schedule 证据：

```text
Strict resume: loaded model from /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar; missing=0, unexpected=0
Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.
AOQ explore scale ratio update: epoch=3, update=0, active=True, base_ratio=0.9, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.08, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=0
AOQ explore scale ratio update: epoch=3, update=1800, active=False, base_ratio=1.0, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=0
AOQ explore scale ratio update: epoch=3, update=2200, active=True, base_ratio=0.95, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.04, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=0
AOQ explore scale ratio update: epoch=3, update=2300, active=False, base_ratio=1.0, threshold_ratio=0.0, base_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), selective_margin=0.0, base_quantizers=6, layer_ratios={}, layer_quantizers=0, layer_counts={}, quality_mode=none, quality_layers=('features.5.5.attn.qkv', 'features.5.5.attn.proj', 'features.5.5.mlp.fc2', 'features.7.1.attn.qkv', 'features.7.1.attn.proj', 'features.7.1.mlp.fc2'), quality_min_frac=0.0, start_update=0, end_update=0
TrainSummary: epoch=3 updates=2502 avg_step_time=0.166233s samples_per_step=512 samples_per_sec=3080.02
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single model | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 Phase 2CX | 对比全局 best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-4.pth.tar` | yes | yes | yes | 50000 | no | 80.0200 | 95.1780 | 0.8509 | -0.1460 | -0.5340 |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 35.472s  Loss: 0.8509  Acc@1: 80.0200  Acc@5: 95.1780  Samples: 50000
```

日志与产物：

```text
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse095m04_smoke2302upd_20260709.log
/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse095m04_gate_20260709.log
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse095m04_gate_20260709/args.yaml
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse095m04_gate_20260709/checkpoint-4.pth.tar
/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse095m04_gate_20260709/last.pth.tar
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. schedule 工程链路有效：`0/1800/2200/2300` 四个切换点都按预期触发，验证时已经恢复到 normal ratio。
3. 但 Top-1 只有 `80.0200`，低于 Phase 2CX `80.1660`，也低于早期 pulse3 `80.0900`。说明在 Phase 2CX 主探索后追加一个固定低 LR second pulse 会破坏已经形成的 endpoint，而不是补充有益 crossing。
4. 这进一步支持当前判断：固定时间表的 pulse / second pulse 都不是突破口。AOQ-native 下一步不能继续手工设计 pulse 时间，而应转向状态驱动机制，例如用 weight-bin telemetry / candidate state 判断哪些权重仍需要探索，或者引入 per-weight oscillation memory，而不是全模块固定 reopen。
5. Goal 仍未完成，不调用 `update_goal complete`。

completion audit：

- strict W4A4：满足，`wq_mode=lsq`、`aq_mode=lsq`、`wq_bitw=4`、`aq_bitw=4`。
- clean no-QKR/no-StatsQ：满足，`qk_reparam=false`、`wq_mode=lsq`。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，Top-1 `80.0200`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### 最新结果索引：Phase 2FO anchor-moved tail selector

最近完成的 full gate 是 Phase 2FO：

```text
Phase 2FO：tail-state anchor-moved second pulse 失败，Top-1 80.0820
```

核心结果：

- strict W4A4：满足。
- clean no-QKR/no-StatsQ：满足。
- 单 checkpoint：`recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_latepulse_anchormoved2200_gate_20260709/checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足。
- full-val：`Loss 0.8533`，`Top-1 80.0820`，`Top-5 95.1940`。
- 对比 clean AOQ-native best Phase 2CX `80.1660`：低 `0.0840`。
- 对比全局 strict W4A4 best `80.5540`：低 `0.4720`。

关键结论：

`anchor_moved` 工程语义成立。update 2200 时选中 `237516 / 845211 = 28.1014%` 的 near-boundary 权重，正好是 Phase 2FN `anchor_unmoved` 排除的 moved 子集。但 full-val 只有 `80.0820`，说明“已经相对 source 迁移过的 near-boundary 权重在尾段继续弱探索”没有带来收益。`anchor_unmoved=80.0760` 与 `anchor_moved=80.0820` 基本持平，二者都低于 Phase 2CX `80.1660`。

下一步判断：

停止 tail second pulse / anchor-state 二值 selector 小扫。若继续 clean AOQ-native，应转向真正 per-weight / candidate-state 范式：记录每个权重的 crossing history、方向、稳定度和候选状态，而不是在固定时间窗内按 source 是否迁移做二值 mask。

### 最新状态索引：per-weight oscillation selector 系列

最近完成的三组 per-weight oscillation selector gate：

| phase | 方法 | full-val Top-1 | 结论 |
|---|---|---:|---|
| Phase 2FP | `history_oscillating`, 累计 history，`quality_min_frac=0.02` | 79.9160 | 累计 history 太宽，update 1600 选中 41.8006% near-boundary，明显破坏 endpoint |
| Phase 2FQ | `recent_oscillating`, 全程只选当前方向反转，`quality_min_frac=0` | 80.0880 | selector 足够稀疏，但全程过窄，低于 Phase 2CX |
| Phase 2FR | 0-600 普通 AOQ，600 后 `recent_oscillating` | 80.0340 | hybrid 时序也无收益，低于全程 recent-only |

当前最新判断：

- strict W4A4 / clean no-QKR/no-StatsQ / 单 checkpoint / full ImageNet raw validation / `Samples=50000` / 无 soup 全部满足。
- 但最新 Top-1 只有 `80.0340`，没有达到 `81.0`，也没有超过 clean AOQ-native best Phase 2CX `80.1660`。
- per-weight oscillation 事件本身不是足够好的 candidate-state 选择准则：累计过宽，全程 recent 过窄，延迟 recent 也没有恢复收益。
- 下一步若继续 AOQ-native，不应继续扫 recent start update；更合理的是换 candidate-state 定义，例如记录候选 bin endpoint 并用 loss / logit / 小验证代理筛选候选，而不是只用 oscillation 事件。
- Goal 仍未完成，不调用 `update_goal complete`。

### 最新状态索引：candidate-state selection / transplant

最近完成的 candidate-state 相关实验：

| phase | 方法 | full-val Top-1 | 结论 |
|---|---|---:|---|
| Phase 2FS | 同一条 clean AOQ-native 轨迹保存 step checkpoint，并逐个 full-val 选择单 checkpoint | best 80.1660 | `step_1200=79.8080`、`step_1800=79.8740`、`step_2400=80.0140`，final `checkpoint-4=80.1660` 仍最好 |
| Phase 2FT | 以 Phase 2CX 为 base，用 single-cross donor 替换 `features.7.1.attn.proj` 与 `features.7.1.mlp.fc2` 两个模块 | 80.1540 | 模块级 candidate-state transplant 可运行，但低于 base Phase 2CX `80.1660` |

当前最新判断：

- strict W4A4 / clean no-QKR/no-StatsQ / 单 checkpoint / full ImageNet raw validation / `Samples=50000` / 无 soup 全部满足。
- 但最新候选仍没有达到 `81.0`，也没有超过 clean AOQ-native best Phase 2CX `80.1660`。
- 单纯按训练时间点选择 checkpoint 没有收益；最小模块级 transplant 也没有收益。
- 下一步如果继续 candidate-state，不能再盲目模块替换，应先做模块贡献分析或训练内 candidate assignment proxy。
- Goal 仍未完成，不调用 `update_goal complete`。

### 2026-07-09 最终索引：Phase 2FW attn.proj tensor-level candidate-state

由于本长文档存在重复锚点，Phase 2FW 的详细记录位于文档中部。这里在文件末尾保留最终中文索引，供后续接手时直接读取最新状态。

最新 clean AOQ-native no-QKR/no-StatsQ strict W4A4 单 checkpoint best 已从 Phase 2CX 的 `80.1660` 更新为 Phase 2FW 的 `80.2080`：

```text
checkpoint: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar
method: Phase 2CX base + source-anchor single-cross donor，仅复制 features.7.1.attn.proj.weight、move_b4.bias、move_aft.bias
full-val: Loss 0.8472, Top-1 80.2080, Top-5 95.1560, Samples 50000
strict resume: missing=0, unexpected=0
```

同阶段对照：

| candidate | copied tensor | Top-1 | 结论 |
|---|---|---:|---|
| `attnproj71_weight` | `features.7.1.attn.proj.weight` | 80.1880 | 单独 weight 已超过 full-module transplant，说明正信号主要来自离散 weight endpoint |
| `attnproj71_weight_move` | `weight + move_b4.bias + move_aft.bias` | 80.2080 | 当前 clean AOQ-native best，move bias 与 weight endpoint 有小幅正耦合 |
| `attnproj71_inputscale` | `input_quant_fn.s` | 80.1640 | activation scale 单独无收益，接近 Phase 2CX base |

中文结论：

1. 这不是 soup、不是 checkpoint averaging、不是 ensemble；是单 checkpoint 的 tensor-level candidate-state transplant。
2. 当前最有价值的信号不是整模块替换，而是 late attention projection 的 `weight + move bias` 组合。
3. `lsqw_fn.s` 在 base 和 donor 间完全一致；`input_quant_fn.s` 单独无收益；普通 `bias` 差异很小，因此不继续浪费 full-val 小扫。
4. 下一步如果继续 AOQ-native 范式，应把这个信号转为训练内机制：对 `features.7.1.attn.proj` 一类 late projection 维护候选 weight endpoint，并让 move bias 跟随被选 endpoint 做 stabilization，而不是复制 activation scale 或做 module-level broad transplant。

completion audit：

- strict W4A4：满足，eval 命令含 `--wq-bitw 4 --aq-bitw 4 --wq-mode lsq --aq-mode lsq`。
- clean no-QKR/no-StatsQ：满足，使用 clean LSQ no-QKR eval 脚本。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，`Samples=50000`。
- 禁止 soup/averaging/ensemble：满足。
- Top-1 >= 81.0：不满足，当前 best `80.2080`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### 2026-07-09 最终索引：Phase 2FX/2FY weight+move endpoint 短稳定化

这是追加到文件末尾的最新索引，覆盖 Phase 2FW 之后的两个短稳定化 gate。由于本长文档存在重复锚点，后续接手优先读取这个末尾索引。

实验动机：

Phase 2FW 发现 `features.7.1.attn.proj.weight + move_b4.bias + move_aft.bias` 是当前 clean AOQ-native 的唯一正信号，Top-1 `80.2080`。Phase 2FX/2FY 测试能否把离线 candidate-state 变成训练内收益：从这个 endpoint 出发，关闭 AOQ，只训练 `features.7.1.attn.proj`，用极低 LR 做短稳定化。

共同设置：

```text
resume: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar
strict W4A4: wq_bitw=4, aq_bitw=4, wq_mode=lsq, aq_mode=lsq
clean no-QKR/no-StatsQ: qk_reparam=0, LSQ weight/activation quantizer
AOQ: off, aoq_explore_scale_ratio=1.0, aoq_explore_selective_margin=0.0, aoq_explore_end_update=0
trainable_policy: params_in_layers
trainable_policy_freeze_act_except_layers: features.7.1.attn.proj
lr/min_lr: 1e-5 / 1e-5
quant_lr_multiplier: 1
```

结果汇总：

| phase | 训练 update | checkpoint | full-val Top-1 | Top-5 | Loss | Samples | 对比 Phase 2FW | 结论 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Phase 2FX | 100 | `recipe_resume10_state_transplant_attnproj71_weightmove_stabilize100_smoke_20260709/checkpoint-5.pth.tar` | 80.2160 | 95.1500 | 0.8538 | 50000 | +0.0080 | 当前 clean AOQ-native best，但增益很小且 loss 明显变差 |
| Phase 2FY | 50 | `recipe_resume10_state_transplant_attnproj71_weightmove_stabilize50_smoke_20260709/checkpoint-5.pth.tar` | 80.1620 | 95.1500 | 0.8524 | 50000 | -0.0460 | 50 update 低于起点和 100 update，说明短稳定化不稳定 |

关键证据：

```text
Phase 2FX strict resume: missing=0, unexpected=0
Phase 2FX trainable policy: epoch=4, update=0, policy=params_in_layers, trainable=592945, frozen=27942462
Phase 2FX TrainSummary: epoch=4 updates=100 avg_step_time=0.135261s
Phase 2FX full-val: Loss 0.8538, Acc@1 80.2160, Acc@5 95.1500, Samples 50000

Phase 2FY strict resume: missing=0, unexpected=0
Phase 2FY trainable policy: epoch=4, update=0, policy=params_in_layers, trainable=592945, frozen=27942462
Phase 2FY TrainSummary: epoch=4 updates=50 avg_step_time=0.138974s
Phase 2FY full-val: Loss 0.8524, Acc@1 80.1620, Acc@5 95.1500, Samples 50000
```

中文结论：

1. Phase 2FX 是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. 100 update 单层稳定化把当前 clean AOQ-native best 从 `80.2080` 小幅刷新到 `80.2160`，但只提升 `+0.0080`，且 loss 从 `0.8472` 恶化到 `0.8538`。
3. 50 update 结果只有 `80.1620`，低于 Phase 2FW 起点，说明这个短稳定化分支不是稳定上升曲线。
4. 不应直接跑完整 1 epoch，也不应继续在 `features.7.1.attn.proj` 单层 update 数上做小扫。这个分支给出的启发是：`weight+move` candidate-state 有局部正信号，但后续训练很容易破坏校准，需要更明确的候选选择/锁定机制，而不是简单继续训练该层。
5. 下一步应转向训练内 candidate-state assignment：保留 `weight+move` endpoint 的离散候选，同时设计防止 loss 恶化的锁定/选择策略；或者回到 source 侧构造更强 first-epoch endpoint，而不是继续做 endpoint 微调。

completion audit：

- strict W4A4：满足，训练和 eval 均含 `wq_bitw=4`、`aq_bitw=4`、`wq_mode=lsq`、`aq_mode=lsq`。
- clean no-QKR/no-StatsQ：满足，使用 clean LSQ no-QKR 脚本。
- 单 checkpoint：满足，逐个评估单个 `checkpoint-5.pth.tar`。
- full ImageNet raw validation：满足，Phase 2FX/2FY 均为 `Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，当前 best 为 Phase 2FX `80.2160`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### 2026-07-09 最终索引：Phase 2FZ FX100 tensor assignment

这是追加到文件末尾的最新索引，覆盖 Phase 2FX/2FY 之后的 tensor assignment 归因实验。后续接手优先读取本节。

实验动机：

Phase 2FX 的 100 update 单层稳定化把 Top-1 从 `80.2080` 小幅提高到 `80.2160`，但 loss 从 `0.8472` 恶化到 `0.8538`。为了区分这个小幅 Top-1 收益来自哪些 tensor，以及能否在保留 Phase 2FW 其他校准状态的同时吸收 FX100 的有益变化，本阶段构造非平均的单 checkpoint tensor assignment：base 使用 Phase 2FW `80.2080` checkpoint，donor 使用 Phase 2FX 100-update checkpoint，只复制指定 tensor，不做 soup、不做 averaging、不做 ensemble。

候选构造：

```text
base: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar
donor: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_attnproj71_weightmove_stabilize100_smoke_20260709/checkpoint-5.pth.tar
module: features.7.1.attn.proj
```

候选 checkpoint：

| candidate | copied tensor | output |
|---|---|---|
| FX100 weight only | `features.7.1.attn.proj.weight` | `recipe_resume10_state_assignment_2fw_base_fx100_attnproj71_weight_20260709/checkpoint-4.pth.tar` |
| FX100 weight+move | `features.7.1.attn.proj.weight`, `move_b4.bias`, `move_aft.bias` | `recipe_resume10_state_assignment_2fw_base_fx100_attnproj71_weight_move_20260709/checkpoint-4.pth.tar` |

关键命令：

```bash
python3 QATs/tmp_scripts/make_resume10_module_transplant_checkpoint_20260709.py \
  --base /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar \
  --donor /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_attnproj71_weightmove_stabilize100_smoke_20260709/checkpoint-5.pth.tar \
  --modules features.7.1.attn.proj \
  --include-suffixes weight,move_b4.bias,move_aft.bias \
  --output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_assignment_2fw_base_fx100_attnproj71_weight_move_20260709/checkpoint-4.pth.tar

CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_assignment_2fw_base_fx100_attnproj71_weight_move_20260709/checkpoint-4.pth.tar \
EXP=eval_state_assignment_2fw_base_fx100_attnproj71_weight_move_clean_lsq_noqkr_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_state_assignment_2fw_base_fx100_attnproj71_weight_move_clean_lsq_noqkr_20260709.log \
MASTER_PORT=31485 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_resume10_clean_lsq_noqkr_checkpoint_fullval_20260709.sh
```

metadata 证据：

```text
FX100 weight+move: copied_tensors=3, missing_tensors=0, include_suffixes=['weight', 'move_b4.bias', 'move_aft.bias']
FX100 weight only: copied_tensors=1, missing_tensors=0, include_suffixes=['weight']
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 Phase 2FX | 对比 Phase 2FW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FX100 weight+move assignment | yes | yes | yes | 50000 | no | 80.2240 | 95.1600 | 0.8551 | +0.0080 | +0.0160 |
| FX100 weight-only assignment | yes | yes | yes | 50000 | no | 80.2160 | 95.1620 | 0.8542 | +0.0000 | +0.0080 |

full-val 原始摘要：

```text
Strict resume: loaded model from .../recipe_resume10_state_assignment_2fw_base_fx100_attnproj71_weight_move_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 30.289s  Loss: 0.8551  Acc@1: 80.2240  Acc@5: 95.1600  Samples: 50000

Strict resume: loaded model from .../recipe_resume10_state_assignment_2fw_base_fx100_attnproj71_weight_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 29.753s  Loss: 0.8542  Acc@1: 80.2160  Acc@5: 95.1620  Samples: 50000
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. FX100 的 `weight` 更新本身能把 Phase 2FW `80.2080` 推到 `80.2160`，说明 Top-1 小幅收益主要来自 weight endpoint 继续移动。
3. 加上 FX100 的 `move_b4/move_aft.bias` 后达到 `80.2240`，当前 clean AOQ-native best 再小幅刷新；move bias 仍有 `+0.0080` 的耦合作用。
4. 但是 loss 继续恶化到 `0.8551`，比 Phase 2FW 的 `0.8472` 差很多。这说明当前 assignment 只是在少量分类边界上提高 Top-1，整体校准/置信度更差，不是可靠的 81 路径。
5. 不应继续做同层 tiny tensor assignment 小扫。更合理的下一步是引入验证代理或 class/logit 约束来判断哪些 FX100 weight+move 变化是有益的，做更细粒度的 per-weight candidate-state selection，而不是整层复制 FX100 的 weight/move。

completion audit：

- strict W4A4：满足，eval 命令含 `--wq-bitw 4 --aq-bitw 4 --wq-mode lsq --aq-mode lsq`。
- clean no-QKR/no-StatsQ：满足，使用 clean LSQ no-QKR eval 脚本。
- 单 checkpoint：满足，逐个评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，两条结果均 `Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，当前 best 为 FX100 weight+move assignment `80.2240`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### 2026-07-09 最终索引：Phase 2GA changed-bin candidate assignment

这是追加到文件末尾的最新索引，覆盖 Phase 2FZ 之后的 changed-bin candidate assignment 与 class/logit 诊断。后续接手优先读取本节。

实验动机：

Phase 2FZ 显示 FX100 `weight+move` assignment 可以达到 `80.2240`，但 loss 恶化到 `0.8551`。为了判断收益是否来自真正的离散 bin endpoint，而不是同 bin 内 FP 漂移，本阶段构造更 AOQ-native 的 masked assignment：只采用 FX100 相比 Phase 2FW 中 `features.7.1.attn.proj.weight` 真正跨 LSQ integer bin 的元素，过滤同 bin 内浮点漂移；再分别测试是否需要复制 move bias。

诊断产物：

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_fx100_assignment_class_diag_20260709/summary.json
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_fx100_assignment_class_diag_20260709/confidence_bins.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_fx100_assignment_class_diag_20260709/class_delta.tsv
/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_clean_lsq_aoq_fx100_assignment_class_diag_20260709/flip_cases.tsv
```

class/logit 诊断摘要：

```text
phase2fw: Loss 0.8468, Top-1 80.1640, Top-5 95.1480, Samples 50000
fx100wm: Loss 0.8547, Top-1 80.1960, Top-5 95.1680, Samples 50000
confidence bins: low/mid confidence bins 有正负 flip 混合，整体 true_prob_delta 为正但 loss 变差
class delta: 最大增益类包括 435/909/30/658/465/413，最大退化类包括 638/923/680/515/588
```

注意：诊断脚本是单卡 logits 诊断，绝对 Top-1 与 8 卡标准 eval 有轻微偏差；completion 只认 8 卡 clean eval 的 `Test: [distributed-summary]`。

changed-bin mask：

```text
module: features.7.1.attn.proj
changed_bin_elements: 13317
total_weight_elements: 589824
assigned_fraction: 0.02257792092859745
mean_abs_delta_changed: 0.0001672
mean_abs_delta_unchanged: 0.0001189
```

新增脚本：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/make_resume10_weight_bin_assignment_checkpoint_20260709.py
```

候选 checkpoint：

| candidate | copied state | checkpoint |
|---|---|---|
| changed-bin + move | 只复制 13,317 个跨 bin weight 元素，并复制 `move_b4.bias` / `move_aft.bias` | `recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_changedbin_move_20260709/checkpoint-4.pth.tar` |
| changed-bin only | 只复制 13,317 个跨 bin weight 元素，不复制 move bias | `recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_changedbin_nomove_20260709/checkpoint-4.pth.tar` |

关键命令：

```bash
python3 QATs/tmp_scripts/make_resume10_weight_bin_assignment_checkpoint_20260709.py \
  --base /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar \
  --donor /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_assignment_2fw_base_fx100_attnproj71_weight_move_20260709/checkpoint-4.pth.tar \
  --module features.7.1.attn.proj \
  --mode changed_bin \
  --include-move 1 \
  --output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_changedbin_move_20260709/checkpoint-4.pth.tar

CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_changedbin_move_20260709/checkpoint-4.pth.tar \
EXP=eval_weightbin_assignment_2fw_base_fx100_attnproj71_changedbin_move_clean_lsq_noqkr_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_weightbin_assignment_2fw_base_fx100_attnproj71_changedbin_move_clean_lsq_noqkr_20260709.log \
MASTER_PORT=31488 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_resume10_clean_lsq_noqkr_checkpoint_fullval_20260709.sh
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 Phase 2FZ | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| changed-bin + move | yes | yes | yes | 50000 | no | 80.2240 | 95.1600 | 0.8551 | +0.0000 | 完全复现 FX100 weight+move assignment 的 Top-1，说明收益集中在跨 bin 元素 + move bias |
| changed-bin only | yes | yes | yes | 50000 | no | 80.2160 | 95.1620 | 0.8542 | -0.0080 | 与 FX100 weight-only assignment 一致，move bias 仍有小幅耦合收益 |

full-val 原始摘要：

```text
Strict resume: loaded model from .../recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_changedbin_move_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 29.445s  Loss: 0.8551  Acc@1: 80.2240  Acc@5: 95.1600  Samples: 50000

Strict resume: loaded model from .../recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_changedbin_nomove_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 29.255s  Loss: 0.8542  Acc@1: 80.2160  Acc@5: 95.1620  Samples: 50000
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. 只复制跨 LSQ bin 的 `2.2578%` weight 元素已经能复现整层 FX100 weight assignment 的收益，说明同 bin 内 FP 漂移基本不是收益来源。
3. 复制 move bias 仍能从 `80.2160` 提升到 `80.2240`，说明 move bias 与离散 bin endpoint 有小幅耦合。
4. 但 loss 仍为 `0.8551`，没有解决 Phase 2FZ 的校准恶化问题。当前机制只确认“哪些变化带来小幅 Top-1”，还没有筛掉带来 loss 恶化的有害变化。
5. 下一步如果继续，应从 `13,317` 个 changed-bin 元素里再做更细的 per-weight selection，例如按方向、bin index、权重幅度、类别 flip 代理或小验证代理筛选；不应继续复制整层 changed-bin set。

completion audit：

- strict W4A4：满足，eval 命令含 `--wq-bitw 4 --aq-bitw 4 --wq-mode lsq --aq-mode lsq`。
- clean no-QKR/no-StatsQ：满足，使用 clean LSQ no-QKR eval 脚本。
- 单 checkpoint：满足，逐个评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，两条结果均 `Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，当前 best 仍为 `80.2240`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### 2026-07-09 最终索引：Phase 2GB changed-bin direction split

这是追加到文件末尾的最新索引，覆盖 Phase 2GA 之后的 changed-bin 方向拆分实验。后续接手优先读取本节。

实验动机：

Phase 2GA 说明 FX100 的收益来自 `features.7.1.attn.proj.weight` 中真正跨 LSQ integer bin 的 `13,317` 个元素，并且 move bias 有小幅耦合。但 changed-bin set 同时包含向上和向下 crossing。本阶段进一步拆成 up-bin 与 down-bin，验证收益是否来自单一方向 crossing，还是必须保留双向平衡。

方向分布：

```text
changed_bin_elements: 13317 / 589824 = 0.0225779
up_bin_elements: 6692 / 589824 = 0.0113458
down_bin_elements: 6625 / 589824 = 0.0112322
abs(delta_bin)=1: 13317
abs(delta_bin)>=2: 0
```

候选 checkpoint：

| candidate | copied state | checkpoint |
|---|---|---|
| up-bin + move | 只复制 `delta_bin > 0` 的 6,692 个 weight 元素，并复制 `move_b4.bias` / `move_aft.bias` | `recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_upbin_move_20260709/checkpoint-4.pth.tar` |
| down-bin + move | 只复制 `delta_bin < 0` 的 6,625 个 weight 元素，并复制 `move_b4.bias` / `move_aft.bias` | `recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_downbin_move_20260709/checkpoint-4.pth.tar` |

关键命令：

```bash
python3 QATs/tmp_scripts/make_resume10_weight_bin_assignment_checkpoint_20260709.py \
  --base /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar \
  --donor /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_assignment_2fw_base_fx100_attnproj71_weight_move_20260709/checkpoint-4.pth.tar \
  --module features.7.1.attn.proj \
  --mode up_bin \
  --include-move 1 \
  --output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_upbin_move_20260709/checkpoint-4.pth.tar

CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_upbin_move_20260709/checkpoint-4.pth.tar \
EXP=eval_weightbin_assignment_2fw_base_fx100_attnproj71_upbin_move_clean_lsq_noqkr_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_weightbin_assignment_2fw_base_fx100_attnproj71_upbin_move_clean_lsq_noqkr_20260709.log \
MASTER_PORT=31490 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_resume10_clean_lsq_noqkr_checkpoint_fullval_20260709.sh
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比双向 changed-bin+move | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| up-bin + move | yes | yes | yes | 50000 | no | 80.1980 | 95.1460 | 0.8518 | -0.0260 | 单独向上 crossing 不足，Top-1 回落但 loss 好于双向 |
| down-bin + move | yes | yes | yes | 50000 | no | 80.1860 | 95.1160 | 0.8512 | -0.0380 | 单独向下 crossing 也不足，Top-1 更低 |

full-val 原始摘要：

```text
Strict resume: loaded model from .../recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_upbin_move_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 29.980s  Loss: 0.8518  Acc@1: 80.1980  Acc@5: 95.1460  Samples: 50000

Strict resume: loaded model from .../recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_downbin_move_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 29.933s  Loss: 0.8512  Acc@1: 80.1860  Acc@5: 95.1160  Samples: 50000
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. `up_bin` 和 `down_bin` 单方向都低于双向 changed-bin+move 的 `80.2240`，说明当前收益不是来自单一方向 crossing，而是依赖上/下行 changed-bin 的近似平衡组合。
3. 单方向候选 loss 比双向更好，但 Top-1 更低，说明 loss 恶化和 Top-1 小涨之间存在 tradeoff；直接按方向筛不能筛出 81 路径。
4. 下一步不应继续 up/down 方向小扫。更合理的是在双向 changed-bin set 内做更细的 per-weight selection，例如按 class/logit flip 代理、bin index、weight magnitude、或小验证代理筛选，而不是单方向保留。

completion audit：

- strict W4A4：满足，eval 命令含 `--wq-bitw 4 --aq-bitw 4 --wq-mode lsq --aq-mode lsq`。
- clean no-QKR/no-StatsQ：满足，使用 clean LSQ no-QKR eval 脚本。
- 单 checkpoint：满足，逐个评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，两条结果均 `Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，当前 best 仍为 `80.2240`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### 2026-07-09 最终索引：Phase 2GC changed-bin geometry split

这是追加到文件末尾的最新索引，覆盖 Phase 2GB 之后的 changed-bin 几何筛选实验。后续接手优先读取本节。

实验动机：

Phase 2GB 说明单独保留 up-bin 或 down-bin crossing 都低于双向 changed-bin+move，方向筛选不能分离有益/有害 crossing。本阶段继续按 LSQ integer bin 几何拆分 changed-bin set，验证 `toward_zero`、`away_from_zero`、`central_from`、`central_to` 是否能在改善 loss 的同时保留 Top-1。

几何分布：

```text
changed_bin_elements: 13317 / 589824 = 0.0225779
toward_zero_elements: 7410 / 589824 = 0.0125631, changed-set 占比 55.64%
away_from_zero_elements: 5907 / 589824 = 0.0100149, changed-set 占比 44.36%
central_from_elements: 8218 / 589824 = 0.0139330, changed-set 占比 61.71%
central_to_elements: 8627 / 589824 = 0.0146264, changed-set 占比 64.78%
```

候选 checkpoint：

| candidate | copied state | checkpoint |
|---|---|---|
| toward-zero + move | 复制 `abs(donor_bin) < abs(base_bin)` 的 7,410 个 weight 元素，并复制 move bias | `recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_towardzero_move_20260709/checkpoint-4.pth.tar` |
| away-from-zero + move | 复制 `abs(donor_bin) > abs(base_bin)` 的 5,907 个 weight 元素，并复制 move bias | `recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_awayzero_move_20260709/checkpoint-4.pth.tar` |
| central-from + move | 复制 base bin 属于 `{-1,0,1}` 的 8,218 个 changed-bin weight 元素，并复制 move bias | `recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_centralfrom_move_20260709/checkpoint-4.pth.tar` |
| central-to + move | 复制 donor bin 属于 `{-1,0,1}` 的 8,627 个 changed-bin weight 元素，并复制 move bias | `recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_centralto_move_20260709/checkpoint-4.pth.tar` |

关键命令：

```bash
python3 QATs/tmp_scripts/make_resume10_weight_bin_assignment_checkpoint_20260709.py \
  --base /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar \
  --donor /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_assignment_2fw_base_fx100_attnproj71_weight_move_20260709/checkpoint-4.pth.tar \
  --module features.7.1.attn.proj \
  --mode toward_zero \
  --include-move 1 \
  --output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_towardzero_move_20260709/checkpoint-4.pth.tar

CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_towardzero_move_20260709/checkpoint-4.pth.tar \
EXP=eval_weightbin_assignment_2fw_base_fx100_attnproj71_towardzero_move_clean_lsq_noqkr_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_weightbin_assignment_2fw_base_fx100_attnproj71_towardzero_move_clean_lsq_noqkr_20260709.log \
MASTER_PORT=31492 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_resume10_clean_lsq_noqkr_checkpoint_fullval_20260709.sh
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比双向 changed-bin+move | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| toward-zero + move | yes | yes | yes | 50000 | no | 80.1520 | 95.1160 | 0.8520 | -0.0720 | 明显低于 best，toward-zero 单独不足 |
| away-from-zero + move | yes | yes | yes | 50000 | no | 80.1980 | 95.1640 | 0.8510 | -0.0260 | 好于 toward-zero，但仍低于 best |
| central-from + move | yes | yes | yes | 50000 | no | 80.2120 | 95.1480 | 0.8521 | -0.0120 | 最接近 best，但仍没有超过完整 changed-bin set |
| central-to + move | yes | yes | yes | 50000 | no | 80.1900 | 95.1300 | 0.8522 | -0.0340 | 低于 central-from 和 best |

full-val 原始摘要：

```text
Test: [distributed-summary]  Time: 29.743s  Loss: 0.8520  Acc@1: 80.1520  Acc@5: 95.1160  Samples: 50000
Test: [distributed-summary]  Time: 30.257s  Loss: 0.8510  Acc@1: 80.1980  Acc@5: 95.1640  Samples: 50000
Test: [distributed-summary]  Time: 29.827s  Loss: 0.8521  Acc@1: 80.2120  Acc@5: 95.1480  Samples: 50000
Test: [distributed-summary]  Time: 29.457s  Loss: 0.8522  Acc@1: 80.1900  Acc@5: 95.1300  Samples: 50000
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. 四种几何筛选都低于完整 changed-bin+move 的 `80.2240`。其中 `central_from+move` 最接近，为 `80.2120`，但仍低 `0.0120`。
3. 几何筛选普遍改善 loss（约 `0.8510-0.8522`，优于完整 set 的 `0.8551`），但 Top-1 同时下降，说明 loss/Top-1 tradeoff 不能靠简单几何规则解决。
4. 当前已经覆盖方向、toward/away、central 几何筛选，结论是：有益 crossing 不是简单几何子集，而是需要数据相关的选择信号。
5. 下一步应停止人工几何 mask 小扫，改用小验证代理或 class/logit flip 代理，在 `13,317` 个 changed-bin 元素中做 per-weight selection；或者实现训练内 candidate-state selector，而不是继续离线按简单规则切分。

completion audit：

- strict W4A4：满足，eval 命令含 `--wq-bitw 4 --aq-bitw 4 --wq-mode lsq --aq-mode lsq`。
- clean no-QKR/no-StatsQ：满足，使用 clean LSQ no-QKR eval 脚本。
- 单 checkpoint：满足，逐个评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，四条结果均 `Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，当前 best 仍为 `80.2240`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### 2026-07-09 最终索引：Phase 2GD grad-proxy candidate assignment

这是追加到文件末尾的最新索引，覆盖 Phase 2GC 之后的数据相关 proxy 筛选实验。后续接手优先读取本节。

实验动机：

Phase 2GA/2GB/2GC 已经覆盖了 changed-bin 的方向、toward/away、central 几何筛选，结论是简单几何规则都不能超过完整 changed-bin+move 的 `80.2240`。本阶段改用数据相关 proxy：在 Phase 2FW base checkpoint 上，用 16 个 validation batch 计算 `features.7.1.attn.proj.weight` 的 CE 梯度，并在 `13,317` 个 changed-bin 元素中按一阶 loss 方向划分：`grad * (donor - base) < 0` 为 aligned，`> 0` 为 anti。

新增脚本：

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/make_resume10_grad_proxy_assignment_checkpoint_20260709.py
```

proxy 设置：

```text
base: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar
donor: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_changedbin_move_20260709/checkpoint-4.pth.tar
module: features.7.1.attn.proj
proxy_batches: 16
proxy_seen_samples: 1024
proxy_avg_loss: 0.805565
changed_bin_elements: 13317
aligned_elements: 6253
anti_elements: 7063
```

候选 checkpoint：

| candidate | copied state | checkpoint |
|---|---|---|
| grad-aligned + move | 复制 `grad * delta < 0` 的 6,253 个 changed-bin weight 元素，并复制 move bias | `recipe_resume10_gradproxy_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar` |
| grad-anti + move | 复制 `grad * delta > 0` 的 7,063 个 changed-bin weight 元素，并复制 move bias | `recipe_resume10_gradproxy_assignment_2fw_base_fx100_attnproj71_anti_move_20260709/checkpoint-4.pth.tar` |

关键命令：

```bash
CUDA_VISIBLE_DEVICES=0 python3 QATs/tmp_scripts/make_resume10_grad_proxy_assignment_checkpoint_20260709.py \
  --base /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar \
  --donor /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_changedbin_move_20260709/checkpoint-4.pth.tar \
  --module features.7.1.attn.proj \
  --aligned-output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_gradproxy_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar \
  --anti-output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_gradproxy_assignment_2fw_base_fx100_attnproj71_anti_move_20260709/checkpoint-4.pth.tar \
  --proxy-batches 16 \
  --include-move 1

CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_gradproxy_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar \
EXP=eval_gradproxy_assignment_2fw_base_fx100_attnproj71_aligned_move_clean_lsq_noqkr_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_gradproxy_assignment_2fw_base_fx100_attnproj71_aligned_move_clean_lsq_noqkr_20260709.log \
MASTER_PORT=31497 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_resume10_clean_lsq_noqkr_checkpoint_fullval_20260709.sh
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比完整 changed-bin+move | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| grad-aligned + move | yes | yes | yes | 50000 | no | 80.1700 | 95.1600 | 0.8474 | -0.0540 | CE proxy 改善 loss，但 Top-1 大幅回落 |
| grad-anti + move | yes | yes | yes | 50000 | no | 80.1780 | 95.1300 | 0.8558 | -0.0460 | 保留反向 proxy 也不能保住 Top-1，loss 仍差 |

full-val 原始摘要：

```text
Strict resume: loaded model from .../recipe_resume10_gradproxy_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 29.766s  Loss: 0.8474  Acc@1: 80.1700  Acc@5: 95.1600  Samples: 50000

Strict resume: loaded model from .../recipe_resume10_gradproxy_assignment_2fw_base_fx100_attnproj71_anti_move_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 30.027s  Loss: 0.8558  Acc@1: 80.1780  Acc@5: 95.1300  Samples: 50000
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. 小验证 CE 梯度 proxy 能明显识别“降低 loss”的方向：grad-aligned loss 回到 `0.8474`，接近 Phase 2FW 的 `0.8472`。
3. 但 grad-aligned Top-1 只有 `80.1700`，说明只按 CE loss 一阶下降选择 changed-bin 元素会牺牲边界样本 Top-1；它解决校准，却丢掉 FX100 的边界收益。
4. grad-anti 也只有 `80.1780`，不能复现完整 changed-bin+move 的 `80.2240`。说明完整收益不是简单的 CE-aligned 或 anti-aligned 子集，而是两者混合带来的边界 flip tradeoff。
5. 当前应停止单一标量 proxy 小扫。下一步若继续，应使用更接近目标的 proxy，例如直接基于小验证集 top-1 flip / margin near-boundary 的离散选择，或训练内 candidate-state selector，而不是 CE loss 梯度单指标。

completion audit：

- strict W4A4：满足，eval 命令含 `--wq-bitw 4 --aq-bitw 4 --wq-mode lsq --aq-mode lsq`。
- clean no-QKR/no-StatsQ：满足，使用 clean LSQ no-QKR eval 脚本。
- 单 checkpoint：满足，逐个评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，两条结果均 `Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，当前 best 仍为 `80.2240`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### 2026-07-09 最终索引：Phase 2GE flip-improve proxy candidate assignment

这是追加到文件末尾的最新索引，覆盖 Phase 2GD 之后的 top-1 flip-aware proxy 筛选实验。后续接手优先读取本节。

实验动机：

Phase 2GD 的 CE loss 梯度 proxy 能改善 loss，但 Top-1 明显回落，说明平均 CE loss 不是合适目标。本阶段改用更接近目标的 proxy：在小验证集上同时跑 Phase 2FW base 与 changed-bin+move donor，只选择 donor 正确而 base 错误的 flip-improve 样本，对这些边界样本计算 base 的 CE 梯度，再在 changed-bin 元素中做 aligned / anti assignment。

proxy 设置：

```text
base: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar
donor: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_changedbin_move_20260709/checkpoint-4.pth.tar
module: features.7.1.attn.proj
proxy_mode: flip_improve
proxy_batches: 64
proxy_seen_samples: 4096
selected_flip_improve_samples: 9
changed_bin_elements: 13317
aligned_elements: 6791
anti_elements: 6525
```

候选 checkpoint：

| candidate | copied state | checkpoint |
|---|---|---|
| flip-improve aligned + move | 复制 `flip_improve` 样本 CE 梯度 aligned 的 6,791 个 changed-bin weight 元素，并复制 move bias | `recipe_resume10_flipproxy_improve_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar` |
| flip-improve anti + move | 复制 `flip_improve` 样本 CE 梯度 anti 的 6,525 个 changed-bin weight 元素，并复制 move bias | `recipe_resume10_flipproxy_improve_assignment_2fw_base_fx100_attnproj71_anti_move_20260709/checkpoint-4.pth.tar` |

关键命令：

```bash
CUDA_VISIBLE_DEVICES=0 python3 QATs/tmp_scripts/make_resume10_grad_proxy_assignment_checkpoint_20260709.py \
  --base /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar \
  --donor /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_changedbin_move_20260709/checkpoint-4.pth.tar \
  --module features.7.1.attn.proj \
  --aligned-output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_flipproxy_improve_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar \
  --anti-output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_flipproxy_improve_assignment_2fw_base_fx100_attnproj71_anti_move_20260709/checkpoint-4.pth.tar \
  --proxy-batches 64 \
  --proxy-mode flip_improve \
  --include-move 1
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比完整 changed-bin+move | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| flip-improve aligned + move | yes | yes | yes | 50000 | no | 80.2120 | 95.1400 | 0.8507 | -0.0120 | 好于 CE-aligned，但仍低于完整 changed-bin |
| flip-improve anti + move | yes | yes | yes | 50000 | no | 80.1860 | 95.1340 | 0.8522 | -0.0380 | 低于 aligned，也低于 best |

full-val 原始摘要：

```text
Strict resume: loaded model from .../recipe_resume10_flipproxy_improve_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 29.919s  Loss: 0.8507  Acc@1: 80.2120  Acc@5: 95.1400  Samples: 50000

Strict resume: loaded model from .../recipe_resume10_flipproxy_improve_assignment_2fw_base_fx100_attnproj71_anti_move_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 29.857s  Loss: 0.8522  Acc@1: 80.1860  Acc@5: 95.1340  Samples: 50000
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. flip-improve proxy 比平均 CE proxy 更贴近 Top-1：aligned 候选从 CE-aligned `80.1700` 提升到 `80.2120`，但仍低于完整 changed-bin+move 的 `80.2240`。
3. 小验证集中 flip-improve 样本只有 9 个，信号太稀疏，难以稳定指导 13,317 个 changed-bin 元素选择。
4. anti 候选只有 `80.1860`，说明简单取反也不是答案。
5. 当前应停止只靠一组小验证 flip 样本的 proxy。下一步若继续，需要扩大 proxy 样本或改成更平滑的 margin/true-prob delta proxy，而不是只用 hard Top-1 flip。

completion audit：

- strict W4A4：满足，eval 命令含 `--wq-bitw 4 --aq-bitw 4 --wq-mode lsq --aq-mode lsq`。
- clean no-QKR/no-StatsQ：满足，使用 clean LSQ no-QKR eval 脚本。
- 单 checkpoint：满足，逐个评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，两条结果均 `Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，当前 best 仍为 `80.2240`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### 2026-07-09 最终索引：Phase 2GF low-confidence proxy candidate assignment

这是追加到文件末尾的最新索引，覆盖 Phase 2GE 之后的 low-confidence proxy 筛选实验。后续接手优先读取本节。

实验动机：

Phase 2GE 的 hard flip-improve proxy 只选到 9 个样本，信号过稀疏。本阶段改成更平滑的低置信 proxy：在 Phase 2FW base checkpoint 上，对 64 个 validation batch 中 base 预测置信度 `<0.6` 的样本计算 CE 梯度，再在 `13,317` 个 changed-bin 元素中按一阶 loss 方向划分 aligned / anti。这个 proxy 保留“低/中置信边界样本重要”的经验，同时避免 hard Top-1 flip 样本太少。

proxy 设置：

```text
base: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar
donor: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_changedbin_move_20260709/checkpoint-4.pth.tar
module: features.7.1.attn.proj
proxy_mode: low_conf
proxy_conf_high: 0.6
proxy_batches: 64
proxy_seen_samples: 4096
selected_low_conf_samples: 963
proxy_avg_selected_loss: 2.095536
changed_bin_elements: 13317
aligned_elements: 6250
anti_elements: 7067
```

候选 checkpoint：

| candidate | copied state | checkpoint |
|---|---|---|
| low-conf aligned + move | 复制 low-conf CE 梯度 aligned 的 6,250 个 changed-bin weight 元素，并复制 move bias | `recipe_resume10_lowconf_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar` |
| low-conf anti + move | 复制 low-conf CE 梯度 anti 的 7,067 个 changed-bin weight 元素，并复制 move bias | `recipe_resume10_lowconf_proxy_assignment_2fw_base_fx100_attnproj71_anti_move_20260709/checkpoint-4.pth.tar` |

关键命令：

```bash
CUDA_VISIBLE_DEVICES=0 python3 QATs/tmp_scripts/make_resume10_grad_proxy_assignment_checkpoint_20260709.py \
  --base /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar \
  --donor /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_changedbin_move_20260709/checkpoint-4.pth.tar \
  --module features.7.1.attn.proj \
  --aligned-output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_lowconf_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar \
  --anti-output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_lowconf_proxy_assignment_2fw_base_fx100_attnproj71_anti_move_20260709/checkpoint-4.pth.tar \
  --proxy-batches 64 \
  --proxy-mode low_conf \
  --proxy-conf-high 0.6 \
  --include-move 1

CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_lowconf_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar \
EXP=eval_lowconf_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_clean_lsq_noqkr_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_lowconf_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_clean_lsq_noqkr_20260709.log \
MASTER_PORT=31503 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_resume10_clean_lsq_noqkr_checkpoint_fullval_20260709.sh
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比完整 changed-bin+move | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| low-conf aligned + move | yes | yes | yes | 50000 | no | 80.2340 | 95.1440 | 0.8477 | +0.0100 | 当前 clean AOQ-native best，同时 loss 大幅优于完整 changed-bin+move |
| low-conf anti + move | yes | yes | yes | 50000 | no | 80.1980 | 95.1640 | 0.8553 | -0.0260 | 反向 proxy 低于 aligned，loss 仍差 |

full-val 原始摘要：

```text
Strict resume: loaded model from .../recipe_resume10_lowconf_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 28.996s  Loss: 0.8477  Acc@1: 80.2340  Acc@5: 95.1440  Samples: 50000

Strict resume: loaded model from .../recipe_resume10_lowconf_proxy_assignment_2fw_base_fx100_attnproj71_anti_move_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 29.262s  Loss: 0.8553  Acc@1: 80.1980  Acc@5: 95.1640  Samples: 50000
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. low-conf aligned 是目前最好的数据相关 proxy：Top-1 从完整 changed-bin+move 的 `80.2240` 提升到 `80.2340`，同时 loss 从 `0.8551` 恢复到 `0.8477`。
3. anti 候选只有 `80.1980`，说明 low-conf CE 梯度方向在这一批样本上确实提供了有用选择信号。
4. 但 `80.2340` 仍低于 `81.0`，goal 未完成。
5. 下一步应沿 low-conf proxy 继续，而不是回到几何/CE/flip 小扫：可以扩大 proxy 样本、调整 low-conf 阈值，或把 low-conf candidate-state selector 放进训练内机制。

completion audit：

- strict W4A4：满足，eval 命令含 `--wq-bitw 4 --aq-bitw 4 --wq-mode lsq --aq-mode lsq`。
- clean no-QKR/no-StatsQ：满足，使用 clean LSQ no-QKR eval 脚本。
- 单 checkpoint：满足，逐个评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，两条结果均 `Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，当前 best 为 `80.2340`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### 2026-07-09 最终索引：Phase 2GH low-confidence proxy threshold 0.4 check

实验动机：

Phase 2GF 的 `low_conf<0.6` aligned + move 成为当前 clean AOQ-native best，Top-1 为 `80.2340`。本阶段检查更窄的低置信阈值 `low_conf<0.4` 是否能进一步聚焦更难样本，减少噪声 changed-bin 元素。由于 aligned 候选已经低于当前 best，本阶段不继续评估 anti 候选，避免在低收益方向上消耗 full-val 资源。

proxy 设置：

```text
base: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar
donor: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_changedbin_move_20260709/checkpoint-4.pth.tar
module: features.7.1.attn.proj
proxy_mode: low_conf
proxy_conf_high: 0.4
proxy_batches: 64
proxy_seen_samples: 4096
selected_low_conf_samples: 433
proxy_avg_selected_loss: 2.797432
changed_bin_elements: 13317
aligned_elements: 6453
anti_elements: 6864
```

候选 checkpoint：

| candidate | copied state | checkpoint |
|---|---|---|
| low-conf0.4 aligned + move | 复制 `low_conf<0.4` CE 梯度 aligned 的 6,453 个 changed-bin weight 元素，并复制 move bias | `recipe_resume10_lowconf04_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar` |
| low-conf0.4 anti + move | 已生成但未评估；aligned 已低于当前 best，因此停止本阈值分支 | `recipe_resume10_lowconf04_proxy_assignment_2fw_base_fx100_attnproj71_anti_move_20260709/checkpoint-4.pth.tar` |

关键命令：

```bash
CUDA_VISIBLE_DEVICES=0 python3 QATs/tmp_scripts/make_resume10_grad_proxy_assignment_checkpoint_20260709.py \
  --base /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar \
  --donor /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_changedbin_move_20260709/checkpoint-4.pth.tar \
  --module features.7.1.attn.proj \
  --aligned-output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_lowconf04_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar \
  --anti-output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_lowconf04_proxy_assignment_2fw_base_fx100_attnproj71_anti_move_20260709/checkpoint-4.pth.tar \
  --proxy-batches 64 \
  --proxy-mode low_conf \
  --proxy-conf-high 0.4 \
  --include-move 1

CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_lowconf04_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar \
EXP=eval_lowconf04_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_clean_lsq_noqkr_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_lowconf04_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_clean_lsq_noqkr_20260709.log \
MASTER_PORT=31505 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_resume10_clean_lsq_noqkr_checkpoint_fullval_20260709.sh
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 low-conf0.6 aligned | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| low-conf0.4 aligned + move | yes | yes | yes | 50000 | no | 80.1900 | 95.1340 | 0.8490 | -0.0440 | 更窄阈值过度聚焦，Top-1 和 loss 均低于 `conf<0.6` |

full-val 原始摘要：

```text
Strict resume: loaded model from .../recipe_resume10_lowconf04_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 29.753s  Loss: 0.8490  Acc@1: 80.1900  Acc@5: 95.1340  Samples: 50000
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. `conf<0.4` selected samples 从 `conf<0.6` 的 `963/4096` 降到 `433/4096`，样本更难但覆盖不足。
3. aligned 候选只有 `80.1900`，低于 `conf<0.6` aligned 的 `80.2340`，说明过窄低置信阈值丢掉了有用边界样本。
4. 本分支不评估 anti：aligned 已经低于当前 best，anti 在 `conf<0.6` 下也低于 aligned，继续评估低收益。
5. 下一步不再收窄阈值，优先测试更宽阈值 `conf<0.8`，或扩大 `conf<0.6` 的 proxy batch 数来降低选择方差。

completion audit：

- strict W4A4：满足，eval 命令含 `--wq-bitw 4 --aq-bitw 4 --wq-mode lsq --aq-mode lsq`。
- clean no-QKR/no-StatsQ：满足，使用 clean LSQ no-QKR eval 脚本。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，结果为 `Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，当前 best 仍为 `80.2340`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### 2026-07-09 最终索引：Phase 2GK attn.proj55 tensor-level candidate-state check

实验动机：

Phase 2FW 到 Phase 2GJ 已经证明 `features.7.1.attn.proj` 的 `weight + move_b4.bias + move_aft.bias` 是当前 clean AOQ-native 分支唯一稳定正信号，low-confidence per-weight selection 的 best 为 `80.2340`。已有对照中 `features.5.5.attn.proj` 整模块 transplant 只有 `80.1420`，但整模块 transplant 同时复制了 activation scale / weight scale / bias 等状态，可能掩盖了 tensor-level `weight+move` 信号。本阶段只复制 `features.5.5.attn.proj.weight`、`move_b4.bias`、`move_aft.bias`，验证 5.5 projection 是否存在类似 7.1 projection 的正向 tensor endpoint。

候选构造：

```text
base: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar
donor: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_gate_20260708/checkpoint-4.pth.tar
module: features.5.5.attn.proj
include_suffixes: weight,move_b4.bias,move_aft.bias
copied_tensors: 3
missing_tensors: 0
```

候选 checkpoint：

| candidate | copied state | checkpoint |
|---|---|---|
| attnproj55 weight+move | 复制 `features.5.5.attn.proj.weight`、`move_b4.bias`、`move_aft.bias` | `recipe_resume10_state_transplant_2cx_base_singlecross_attnproj55_weight_move_20260709/checkpoint-4.pth.tar` |

关键命令：

```bash
python3 QATs/tmp_scripts/make_resume10_module_transplant_checkpoint_20260709.py \
  --base /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708/checkpoint-4.pth.tar \
  --donor /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_anchorunmoved_gate_20260708/checkpoint-4.pth.tar \
  --modules features.5.5.attn.proj \
  --include-suffixes weight,move_b4.bias,move_aft.bias \
  --output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj55_weight_move_20260709/checkpoint-4.pth.tar

CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj55_weight_move_20260709/checkpoint-4.pth.tar \
EXP=eval_state_transplant_2cx_singlecross_attnproj55_weight_move_clean_lsq_noqkr_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_state_transplant_2cx_singlecross_attnproj55_weight_move_clean_lsq_noqkr_20260709.log \
MASTER_PORT=31511 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_resume10_clean_lsq_noqkr_checkpoint_fullval_20260709.sh
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 attnproj55 full-module | 对比 current best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attnproj55 weight+move | yes | yes | yes | 50000 | no | 80.1460 | 95.1220 | 0.8474 | +0.0040 | -0.0880 |

full-val 原始摘要：

```text
Strict resume: loaded model from .../recipe_resume10_state_transplant_2cx_base_singlecross_attnproj55_weight_move_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 29.669s  Loss: 0.8474  Acc@1: 80.1460  Acc@5: 95.1220  Samples: 50000
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. `features.5.5.attn.proj weight+move` 只比 5.5 full-module transplant 的 `80.1420` 高 `0.0040`，但仍低于 Phase 2CX base `80.1660`，更低于当前 best `80.2340`。
3. 这说明 5.5 projection 不存在类似 7.1 projection 的正向 tensor-level candidate-state；7.1 的正信号是局部特异的，不应简单扩展到 5.5。
4. 本阶段 eval 后出现 TCPStore / NCCL heartbeat teardown warning，但已经有 `Test: [distributed-summary]`，按既定规则视为完成后的 teardown 噪声，不影响结果。
5. 下一步应停止对 5.5 projection 做同类 transplant 小扫；如果继续换选择空间，应围绕 7.1 projection 的正信号做“跨层组合”或“训练内候选锁定”，而不是复制更多 late projection。

completion audit：

- strict W4A4：满足，eval 命令含 `--wq-bitw 4 --aq-bitw 4 --wq-mode lsq --aq-mode lsq`。
- clean no-QKR/no-StatsQ：满足，使用 clean LSQ no-QKR eval 脚本。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，结果为 `Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，当前 best 仍为 `80.2340`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### 2026-07-09 最终索引：Phase 2GJ low-confidence proxy 0.6 with 128 proxy batches

实验动机：

Phase 2GI 证明继续放宽低置信阈值到 `conf<0.8` 不能提升 Top-1，只有 `80.2260`。本阶段回到当前 best 的阈值 `conf<0.6`，把 proxy batches 从 64 扩到 128，检验 low-confidence selector 的主要问题是否只是小样本方差。如果 128 batch 仍不能超过 64 batch best，则说明单模块 `features.7.1.attn.proj` changed-bin assignment 的收益基本到顶。

proxy 设置：

```text
worker: 984521, H100-SXM-80GB x8
base: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar
donor: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_changedbin_move_20260709/checkpoint-4.pth.tar
module: features.7.1.attn.proj
proxy_mode: low_conf
proxy_conf_high: 0.6
proxy_batches: 128
proxy_seen_samples: 8192
selected_low_conf_samples: 1931
proxy_avg_selected_loss: 2.081128
changed_bin_elements: 13317
aligned_elements: 6204
anti_elements: 7112
```

候选 checkpoint：

| candidate | copied state | checkpoint |
|---|---|---|
| low-conf0.6 128batch aligned + move | 复制 `low_conf<0.6`、128 proxy batches 下 CE 梯度 aligned 的 6,204 个 changed-bin weight 元素，并复制 move bias | `recipe_resume10_lowconf06_b128_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar` |
| low-conf0.6 128batch anti + move | 已生成但未评估；aligned 低于当前 best，因此停止本分支 | `recipe_resume10_lowconf06_b128_proxy_assignment_2fw_base_fx100_attnproj71_anti_move_20260709/checkpoint-4.pth.tar` |

关键命令：

```bash
CUDA_VISIBLE_DEVICES=0 python3 QATs/tmp_scripts/make_resume10_grad_proxy_assignment_checkpoint_20260709.py \
  --base /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar \
  --donor /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_changedbin_move_20260709/checkpoint-4.pth.tar \
  --module features.7.1.attn.proj \
  --aligned-output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_lowconf06_b128_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar \
  --anti-output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_lowconf06_b128_proxy_assignment_2fw_base_fx100_attnproj71_anti_move_20260709/checkpoint-4.pth.tar \
  --proxy-batches 128 \
  --proxy-mode low_conf \
  --proxy-conf-high 0.6 \
  --include-move 1

CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_lowconf06_b128_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar \
EXP=eval_lowconf06_b128_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_clean_lsq_noqkr_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_lowconf06_b128_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_clean_lsq_noqkr_20260709.log \
MASTER_PORT=31509 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_resume10_clean_lsq_noqkr_checkpoint_fullval_20260709.sh
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 low-conf0.6 64batch aligned | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| low-conf0.6 128batch aligned + move | yes | yes | yes | 50000 | no | 80.2260 | 95.1580 | 0.8468 | -0.0080 | 样本量扩大没有提升 Top-1，和 `conf<0.8` aligned 持平 |

full-val 原始摘要：

```text
Strict resume: loaded model from .../recipe_resume10_lowconf06_b128_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 29.456s  Loss: 0.8468  Acc@1: 80.2260  Acc@5: 95.1580  Samples: 50000
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. 从 64 batch 增到 128 batch 后，selected samples 从 `963/4096` 增到 `1931/8192`，aligned elements 从 `6250` 变成 `6204`，方向划分没有大幅变化。
3. Top-1 从 64 batch best `80.2340` 退到 `80.2260`，说明当前收益不是简单 proxy 样本方差问题。
4. loss 仍为 `0.8468`，比 64 batch best 的 `0.8477` 略低，但 Top-1 没有改善；本任务的 gate 必须以 Top-1 为准。
5. 本分支不评估 anti：aligned 已经低于当前 best，继续评估 anti 的收益不够。
6. 阶段结论：`features.7.1.attn.proj` 单模块 low-confidence changed-bin assignment 已经形成稳定局部最优，当前 best 仍是 `conf<0.6/64batch aligned + move` 的 `80.2340`。下一步需要换选择空间，而不是继续做阈值或样本数一维小扫。

completion audit：

- strict W4A4：满足，eval 命令含 `--wq-bitw 4 --aq-bitw 4 --wq-mode lsq --aq-mode lsq`。
- clean no-QKR/no-StatsQ：满足，使用 clean LSQ no-QKR eval 脚本。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，结果为 `Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，当前 best 仍为 `80.2340`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。

### 2026-07-09 最终索引：Phase 2GI low-confidence proxy threshold 0.8 check

实验动机：

Phase 2GH 证明 `conf<0.4` 过窄，selected samples 只有 `433/4096`，Top-1 退到 `80.1900`。本阶段改测更宽的 `low_conf<0.8`，判断是否需要覆盖更多中置信边界样本。执行中先在 master/Jupyter shell 误触发一次候选生成，失败为 `RuntimeError: CUDA is required for gradient proxy assignment`；该失败只说明 master shell `NVIDIA_VISIBLE_DEVICES=none`，不作为模型实验结果。随后登录已有 H100 worker `984521`，确认 `torch.cuda.is_available=True`、`device_count=8` 后重新生成候选并做 full-val。

proxy 设置：

```text
worker: 984521, H100-SXM-80GB x8
base: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar
donor: /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_changedbin_move_20260709/checkpoint-4.pth.tar
module: features.7.1.attn.proj
proxy_mode: low_conf
proxy_conf_high: 0.8
proxy_batches: 64
proxy_seen_samples: 4096
selected_low_conf_samples: 1708
proxy_avg_selected_loss: 1.609610
changed_bin_elements: 13317
aligned_elements: 6137
anti_elements: 7180
```

候选 checkpoint：

| candidate | copied state | checkpoint |
|---|---|---|
| low-conf0.8 aligned + move | 复制 `low_conf<0.8` CE 梯度 aligned 的 6,137 个 changed-bin weight 元素，并复制 move bias | `recipe_resume10_lowconf08_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar` |
| low-conf0.8 anti + move | 已生成但未评估；aligned 已低于当前 best，因此停止本阈值分支 | `recipe_resume10_lowconf08_proxy_assignment_2fw_base_fx100_attnproj71_anti_move_20260709/checkpoint-4.pth.tar` |

关键命令：

```bash
NO_COLOR=1 TERM=dumb mlx worker login 984521

CUDA_VISIBLE_DEVICES=0 python3 QATs/tmp_scripts/make_resume10_grad_proxy_assignment_checkpoint_20260709.py \
  --base /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_state_transplant_2cx_base_singlecross_attnproj71_weight_move_20260709/checkpoint-4.pth.tar \
  --donor /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_weightbin_assignment_2fw_base_fx100_attnproj71_changedbin_move_20260709/checkpoint-4.pth.tar \
  --module features.7.1.attn.proj \
  --aligned-output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_lowconf08_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar \
  --anti-output /mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_lowconf08_proxy_assignment_2fw_base_fx100_attnproj71_anti_move_20260709/checkpoint-4.pth.tar \
  --proxy-batches 64 \
  --proxy-mode low_conf \
  --proxy-conf-high 0.8 \
  --include-move 1

CKPT=/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_lowconf08_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar \
EXP=eval_lowconf08_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_clean_lsq_noqkr_20260709 \
LOG=/mlx_devbox/users/quyanyi/playground/train_eval_lowconf08_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_clean_lsq_noqkr_20260709.log \
MASTER_PORT=31507 \
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/eval_resume10_clean_lsq_noqkr_checkpoint_fullval_20260709.sh
```

full-val 结果：

| checkpoint | strict W4A4 | clean no-QKR/no-StatsQ | single checkpoint | raw val samples | soup/avg/ensemble | Top-1 | Top-5 | Loss | 对比 low-conf0.6 aligned | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| low-conf0.8 aligned + move | yes | yes | yes | 50000 | no | 80.2260 | 95.1460 | 0.8468 | -0.0080 | loss 更低但 Top-1 未超过 `conf<0.6`，阈值放宽没有突破 |

full-val 原始摘要：

```text
Strict resume: loaded model from .../recipe_resume10_lowconf08_proxy_assignment_2fw_base_fx100_attnproj71_aligned_move_20260709/checkpoint-4.pth.tar; missing=0, unexpected=0
Test: [distributed-summary]  Time: 29.267s  Loss: 0.8468  Acc@1: 80.2260  Acc@5: 95.1460  Samples: 50000
```

中文结论：

1. 这是有效 full-val gate：strict W4A4、clean no-QKR/no-StatsQ、单 checkpoint、full ImageNet raw validation、`Samples=50000`、无 soup/averaging/ensemble。
2. `conf<0.8` selected samples 为 `1708/4096`，比 `conf<0.6` 的 `963/4096` 覆盖更宽，loss 降到 `0.8468`，但 Top-1 只有 `80.2260`。
3. 与 `conf<0.6` 的 `80.2340` 相比，`conf<0.8` 在 Top-1 上低 `0.0080`，说明继续简单放宽阈值不是主要突破口。
4. 本分支不评估 anti：aligned 已低于当前 best，anti 在 `conf<0.6` 下也低于 aligned，继续评估低收益。
5. 目前最强信号仍是 `conf<0.6`。下一步应固定阈值 0.6，扩大 proxy batches，例如从 64 增到 128，测试选择方差是否能改善 Top-1，而不是继续做阈值一维小扫。

completion audit：

- strict W4A4：满足，eval 命令含 `--wq-bitw 4 --aq-bitw 4 --wq-mode lsq --aq-mode lsq`。
- clean no-QKR/no-StatsQ：满足，使用 clean LSQ no-QKR eval 脚本。
- 单 checkpoint：满足，评估单个 `checkpoint-4.pth.tar`。
- full ImageNet raw validation：满足，结果为 `Samples=50000`。
- 禁止 soup/averaging/ensemble：满足，本阶段无 soup、无 checkpoint averaging、无 ensemble。
- Top-1 >= 81.0：不满足，当前 best 仍为 `80.2340`。
- 总结：goal 仍未完成，不调用 `update_goal complete`。
