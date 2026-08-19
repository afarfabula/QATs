# Stage2 50→51 Algorithm Progress (2026-07-01)

## Objective

Redesign stage2 QAT algorithm from the fixed stage1 checkpoint-50 and require a normal single stage2 epoch (50→51) to reach raw full-val Top-1 >= 79.0%.

## Fixed baseline and gate

- Resume checkpoint: `/mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_continue10_to50_sysdisk_20260630/checkpoint-50.pth.tar`
- Baseline raw full-val: Top-1 78.116 / Top-5 94.200
- Main gate: checkpoint-51 raw full ImageNet distributed validation Top-1 >= 79.0
- Secondary reporting: EMA checkpoint full-val if model EMA is enabled
- Runtime gate: one stage2 epoch <= 10 minutes
- Prohibited: checkpoint soup, multi-checkpoint averaging, using >1 epoch weights as checkpoint-51

## Cleanup

- Stopped failed long stage2-to-100 run at epoch 88 startup; best observed raw Top-1 was 78.234 @ epoch58, insufficient for the 50→51 gate.
- Removed failed long-run checkpoints under worker `/tmp/qats_stage2_outputs/.../*to100*`; retained logs and scripts.
- Worker `/tmp/qats_stage2_outputs` reduced from ~30G to ~6G.

## Candidate matrix

| ID | Algorithm idea | Script | Resume | Raw Top-1 | Raw Top-5 | EMA Top-1 | EMA Top-5 | Delta vs 78.116 | Runtime | Gate |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| C1 | KD + prev-step selected-head attn KL + fixed checkpoint-50 anchor selected-head attn KL + student EMA | `tmp_scripts/run_stage2_anchor_prev_custom10_from50_to51_bsz128_kd_ema999_20260701.sh` | checkpoint-50 | N/A | N/A | N/A | N/A | N/A | stopped at 250/1251; avg step 0.543s, estimated >11min/epoch | Fail: runtime >10min |
| C2 | KD + EMA-ref selected-head attn KL (momentum 0.99) + student EMA | `tmp_scripts/run_stage2_emaref_custom10_from50_to51_bsz128_kd_ema999_20260701.sh` | checkpoint-50 | 78.028 | 94.124 | 78.058 | 94.128 | -0.088 raw / -0.058 EMA | train avg 0.428s/step; ~8.9min train + 31.9s raw val | Fail: below baseline |
| C3 | KD + prev-step ref-logit KL + weak selected-head attn KL + clean hard-label CE + student EMA | `tmp_scripts/run_stage2_reflogit_weakattn_cleance_from50_to51_bsz128_kd_ema999_20260701.sh` | checkpoint-50 | 78.016 | 94.120 | 78.110 | 94.116 | -0.100 raw / -0.006 EMA | train avg 0.429s/step; ~8.9min train + 31.4s raw val | Fail: raw below baseline |
| C4 | Partial-unfreeze head_norm_attn_quant + weak prev-step selected-head attn KL + KD + student EMA | `tmp_scripts/run_stage2_partial_attnquant_weakref_from50_to51_bsz128_kd_ema999_20260701.sh` | checkpoint-50 | 78.114 | 94.160 | 78.064 | 94.158 | -0.002 raw / -0.052 EMA | train avg 0.439s/step; ~9.1min train + 31.6s raw val | Fail: below 79, but best stability |
| C5 | Partial-unfreeze head_norm_attn_quant + weak prev-step selected-head attn KL + KD, no EMA, LR 3e-5 | `tmp_scripts/run_stage2_partial_attnquant_weakref_lr3e5_noema_from50_to51_bsz128_kd_20260701.sh` | checkpoint-50 | 78.076 | 94.142 | N/A | N/A | -0.040 raw | train avg 0.431s/step; ~9.0min train + 31.8s raw val | Fail: higher LR hurts |
| C6 | Narrow head_norm_quant + weak prev-step selected-head attn KL + KD, no EMA, LR 1e-5 | `tmp_scripts/run_stage2_headnormquant_weakref_lr1e5_noema_from50_to51_bsz128_kd_20260701.sh` | checkpoint-50 | 78.104 | 94.142 | N/A | N/A | -0.012 raw | train avg 0.423s/step; ~8.8min train + 31.8s raw val | Fail: below baseline |
| C7 | Partial-unfreeze fixed checkpoint-50 anchor-ref selected-head attn KL + KD, no EMA | `tmp_scripts/run_stage2_partial_anchoronly_from50_to51_bsz128_kd_noema_20260701.sh` | checkpoint-50 | 78.062 | 94.196 | N/A | N/A | -0.054 raw | train avg 0.428s/step; ~8.9min train + 31.8s raw val | Fail: anchor too strong |
| C8 | Partial-unfreeze hard-label CE + weak prev-step selected-head attn KL, no KD/EMA | `tmp_scripts/run_stage2_partial_ce_weakref_from50_to51_bsz128_noema_20260701.sh` | checkpoint-50 | 76.324 | 92.750 | N/A | N/A | -1.792 raw | train avg 0.403s/step; ~8.4min train + 31.3s raw val | Fail: KD is necessary |
| B0 | Eval-only checkpoint-50 baseline audit | `tmp_scripts/eval_stage2_start_ckpt50_baseline_20260701.sh` | checkpoint-50 | 78.116 | 94.200 | N/A | N/A | 0.000 | eval-only 32.5s | Baseline confirmed |
| C9 | Partial-unfreeze KD + prev-step ref-logit KL only, no attention KL/EMA | `tmp_scripts/run_stage2_partial_reflogit_only_from50_to51_bsz128_kd_noema_20260701.sh` | checkpoint-50 | 78.050 | 94.226 | N/A | N/A | -0.066 raw | train avg 0.408s/step; ~8.5min train + 31.6s raw val | Fail: below baseline |
| C10 | Partial-unfreeze KD + weak prev-step selected-head attn KL, no EMA, LR 5e-6 | `tmp_scripts/run_stage2_partial_weakref_lr5e6_noema_from50_to51_bsz128_kd_20260701.sh` | checkpoint-50 | 78.124 | 94.186 | N/A | N/A | +0.008 raw | train avg 0.428s/step; ~8.9min train + 31.6s raw val | Best so far but fail 79 gate |
| C11 | Pure quant-only KD + weak prev-step selected-head attn KL, no EMA, LR 5e-6 | `tmp_scripts/run_stage2_quantonly_weakref_lr5e6_noema_from50_to51_bsz128_kd_20260701.sh` | checkpoint-50 | 78.032 | 94.148 | N/A | N/A | -0.084 raw | train avg 0.422s/step; ~8.8min train + 31.6s raw val | Fail: too narrow |
| C12 | Partial-unfreeze KD + dynamic custom-top5 prev-step attn KL, no EMA, LR 5e-6 | `tmp_scripts/run_stage2_partial_dynamic_customtop5_lr5e6_noema_from50_to51_bsz128_kd_20260701.sh` | checkpoint-50 | 78.184 | 94.170 | N/A | N/A | +0.068 raw | train avg 0.429s/step; ~8.9min train + 31.8s raw val | Best so far but fail 79 gate |
| C13 | Partial-unfreeze KD + dynamic all-head top5 prev-step attn KL, no EMA, LR 5e-6 | `tmp_scripts/run_stage2_partial_dynamic_alltop5_lr5e6_noema_from50_to51_bsz128_kd_20260701.sh` | checkpoint-50 | 77.990 | 94.162 | N/A | N/A | -0.126 raw | train avg 0.450s/step; ~9.4min train + 31.9s raw val | Fail: all-head top-k too strong |
| C14 | Partial-unfreeze KD + dynamic custom-top3 prev-step attn KL, no EMA, LR 5e-6 | `tmp_scripts/run_stage2_partial_dynamic_customtop3_lr5e6_noema_from50_to51_bsz128_kd_20260701.sh` | checkpoint-50 | 78.106 | 94.212 | N/A | N/A | -0.010 raw | train avg 0.434s/step; ~9.0min train + 31.8s raw val | Fail: top3 too weak/worse than C12 |
| C15 | Partial-unfreeze KD + dynamic custom-top5 JS attention loss, no EMA, LR 5e-6 | `tmp_scripts/run_stage2_partial_dynamic_customtop5_js_lr5e6_noema_from50_to51_bsz128_kd_20260701.sh` | checkpoint-50 | 78.046 | 94.178 | N/A | N/A | -0.070 raw | train avg 0.433s/step; ~9.0min train + 32.0s raw val | Fail: JS worse than KL |
| C15b | Partial-unfreeze KD + dynamic custom-top5 KL, lower ref weight 5e-5, no EMA, LR 5e-6 | `tmp_scripts/run_stage2_partial_dynamic_customtop5_w5e5_lr5e6_noema_from50_to51_bsz128_kd_20260701.sh` | checkpoint-50 | 78.126 | 94.176 | N/A | N/A | +0.010 raw | train avg 0.428s/step; ~8.9min train + 31.7s raw val | Fail: lower weight worse than C12 |
| C16 | Full-param KD + dynamic custom-top5 KL, no EMA, LR 5e-6 | `tmp_scripts/run_stage2_full_dynamic_customtop5_lr5e6_noema_from50_to51_bsz128_kd_20260701.sh` | checkpoint-50 | 78.198 | 94.204 | N/A | N/A | +0.082 raw | train avg 0.417s/step; ~8.7min train + 31.4s raw val | Best so far but fail 79 gate |
| C17 | Full-param KD + dynamic custom-top5 KL, ref warmup 200 updates, no EMA, LR 5e-6 | `tmp_scripts/run_stage2_full_dynamic_customtop5_refwarm200_lr5e6_noema_from50_to51_bsz128_kd_20260701.sh` | checkpoint-50 | 78.136 | 94.176 | N/A | N/A | +0.020 raw | train avg 0.396s/step; ~8.3min train + 31.3s raw val | Fail: warmup hurts |
| C18 | Full-param KD + dynamic custom-top5 KL clipped at 70, no EMA, LR 5e-6 | `tmp_scripts/run_stage2_full_dynamic_customtop5_clip70_lr5e6_noema_from50_to51_bsz128_kd_20260701.sh` | checkpoint-50 | 78.008 | 94.148 | N/A | N/A | -0.108 raw | train avg 0.416s/step; ~8.7min train + 31.5s raw val | Fail: clipping hurts |
| C19 | Full-param KD + layer-wise dynamic custom top1 KL, no EMA, LR 5e-6 | `tmp_scripts/run_stage2_full_dynamic_custom_layertop1_lr5e6_noema_from50_to51_bsz128_kd_20260701.sh` | checkpoint-50 | 78.082 | 94.168 | N/A | N/A | -0.034 raw | train avg 0.418s/step; ~8.7min train + 31.6s raw val | Fail: layer-wise weaker than C16 |
| C20 | Full-param KD + dynamic custom-top5 KL with EMA-ref momentum 0.99, no student EMA, LR 5e-6 | `tmp_scripts/run_stage2_full_dynamic_customtop5_emaref99_lr5e6_noema_from50_to51_bsz128_kd_20260701.sh` | checkpoint-50 | 78.050 | 94.172 | N/A | N/A | -0.066 raw | train avg 0.418s/step; ~8.7min train + 31.4s raw val | Fail: EMA-ref worse than prev-step |
| C21 | Full-param KD + auxiliary hard-label CE 0.05 + dynamic custom-top5 KL, no EMA, LR 5e-6 | `tmp_scripts/run_stage2_full_dynamic_customtop5_auxce005_lr5e6_noema_from50_to51_bsz128_kd_20260701.sh` | checkpoint-50 | 78.112 | 94.216 | N/A | N/A | -0.004 raw | train avg 0.415s/step; ~8.7min train + 31.7s raw val | Fail: aux CE hurts |
| C22 | Full-param KD + prev-step dynamic custom-top5 KL + teacher-attn dynamic custom-top5 KL, no EMA, LR 5e-6 | `tmp_scripts/run_stage2_full_dynamic_customtop5_teacherattn_lr5e6_noema_from50_to51_bsz128_kd_20260701.sh` | checkpoint-50 | 78.030 | 94.190 | N/A | N/A | -0.086 raw | train avg 0.417s/step; ~8.7min train + 31.4s raw val | Fail: teacher attn too strong/mismatched |
| C23 | Full-param KD + post-resume setup-alpha calibration 16 batches + C16 dynamic custom-top5 KL | `tmp_scripts/run_stage2_full_dynamic_customtop5_postcalib16_lr5e6_noema_from50_to51_bsz128_kd_20260701.sh` | checkpoint-50 | 78.086 | 94.198 | N/A | N/A | -0.030 raw | train avg 0.417s/step; ~8.7min train + 31.9s raw val | Fail: post-resume calibration hurts |
| C19 | Full-param KD + layer-wise dynamic custom top1 KL, no EMA, LR 5e-6 | `tmp_scripts/run_stage2_full_dynamic_custom_layertop1_lr5e6_noema_from50_to51_bsz128_kd_20260701.sh` | checkpoint-50 | 78.082 | 94.168 | N/A | N/A | -0.034 raw | train avg 0.418s/step; ~8.7min train + 31.6s raw val | Fail: layer-wise weaker than C16 |
| C15b | Partial-unfreeze KD + dynamic custom-top5 KL, lower ref weight 5e-5, no EMA, LR 5e-6 | `tmp_scripts/run_stage2_partial_dynamic_customtop5_w5e5_lr5e6_noema_from50_to51_bsz128_kd_20260701.sh` | checkpoint-50 | 78.126 | 94.176 | N/A | N/A | +0.010 raw | train avg 0.428s/step; ~8.9min train + 31.7s raw val | Fail: lower weight worse than C12 |

## Candidate details

### C1: fixed-anchor + prev-step dual-ref

Design rationale: prev-step ref constrains local update-to-update attention drift, while a fixed anchor ref at checkpoint-50 constrains whole-epoch drift from the high-accuracy starting point. This is algorithmically distinct from previous prev-step-only recipe.

Outcome: stopped early because the dual-ref extra forward path made the 100-250 step average about 0.543s/step, exceeding the 10min/epoch runtime gate.

### C2: EMA-ref selected-head KL

Design rationale: replace previous-step ref with a low-pass temporal reference model. This keeps the refmodel idea but changes it from local step memory to multi-step smoothed memory, aiming to suppress high-frequency attention/quantization oscillation with only one ref forward path.

Current command/script: `tmp_scripts/run_stage2_emaref_custom10_from50_to51_bsz128_kd_ema999_20260701.sh`.

Outcome: raw checkpoint-51 reached Top-1 78.028 / Top-5 94.124. EMA checkpoint-51 reached Top-1 78.058 / Top-5 94.128. This is below the checkpoint-50 baseline and therefore failed the 79.0 gate.

### C3: ref-logit + weak selected-head KL + clean hard-label CE

Design rationale: C2 suggests strong attention KL alone pulls 50→51 below the starting checkpoint. C3 keeps a refmodel but shifts the main stabilizer to logit-space consistency and adds hard-label CE to keep the student aligned with the ground-truth decision boundary during the single adaptation epoch. Attention KL is kept weak and selected-head only, so it remains a stability regularizer rather than the dominant update.

Outcome: raw checkpoint-51 reached Top-1 78.016 / Top-5 94.120. EMA checkpoint-51 reached Top-1 78.110 / Top-5 94.116. EMA nearly recovers the starting point but raw is still below baseline and far below 79.0.

### C4: quant-only/partial-unfreeze refmodel stage2

Design rationale: C2/C3 both show that full-parameter stage2 updates drift below the checkpoint-50 starting point. C4 changes the trainable parameter policy so stage2 primarily recalibrates quantization and local attention/head parameters instead of moving the whole FP backbone. It keeps a weak prev-step refmodel so the update remains a refmodel-based stabilization method.

Outcome: raw checkpoint-51 reached Top-1 78.114 / Top-5 94.160, nearly preserving the 78.116 starting point. EMA checkpoint-51 reached Top-1 78.064 / Top-5 94.158, so EMA is not useful for this setting.

### C5: partial-unfreeze higher-LR no-EMA calibration

Design rationale: C4 identified partial-unfreeze as the first stage2 direction that does not substantially damage the starting checkpoint. C5 removes student EMA overhead/state and increases the calibration LR while keeping weak prev-step refmodel attention stabilization, aiming to turn the stable partial update into a positive one-epoch gain.

Outcome: raw checkpoint-51 reached Top-1 78.076 / Top-5 94.142. Higher LR without EMA hurts relative to C4, so C4 remains the best stability result.

### C6: narrower quant/head-norm-only refmodel stage2

Design rationale: C4 was nearly neutral but included attention projection updates. C6 narrows the trainable set to quantization and normalization/head parameters only, testing whether attention projection movement is the source of the remaining small drift.

Outcome: raw checkpoint-51 reached Top-1 78.104 / Top-5 94.142. Narrowing to head_norm_quant does not beat C4, so the slight C4 drift is not caused solely by attention projection updates.

### C7: partial-unfreeze fixed-anchor refmodel stage2

Design rationale: C1 showed dual prev-step+anchor ref is too slow. C7 uses only the fixed checkpoint-50 anchor ref with partial trainable parameters, preserving the refmodel idea while constraining whole-epoch drift to the known good starting point without the extra prev-step ref cost.

Outcome: raw checkpoint-51 reached Top-1 78.062 / Top-5 94.196. Fixed attention anchor loss is harmful at this weight; it increases the auxiliary KL term without improving classification.

## Interim conclusions after C1-C7

1. The target remains unmet: best raw checkpoint-51 result is C4 Top-1 78.114, still below checkpoint-50 baseline 78.116 and far below 79.0.
2. Full-parameter one-epoch stage2 with selected-head attention KL consistently drifts below baseline (C2/C3).
3. Partial-unfreeze is the only direction that nearly preserves checkpoint-50 (C4), but it does not create gain.
4. Student EMA does not help in this 50→51 gate; C4 raw is better than its EMA.
5. Strong fixed attention anchor is harmful (C7), and dual anchor+prev is too slow (C1).
6. Higher LR in partial calibration hurts (C5). Narrower head_norm_quant is also worse than C4 (C6).
7. Hard-label CE-only partial calibration is destructive (C8), so KD remains necessary for stability.
8. Next candidates should avoid simple ref-weight tweaks and instead test baseline/no-update reproducibility, KD-only partial variants, or online data/calibration policy changes.

### C8: partial CE-only refmodel stage2

Design rationale: remove teacher soft-target KD and use hard-label CE for one-epoch calibration, while keeping weak prev-step refmodel attention stabilization.

Outcome: raw checkpoint-51 reached Top-1 76.324 / Top-5 92.750. This is a large regression; KD is necessary for stable stage2.

### B0: checkpoint-50 baseline eval audit

Outcome: eval-only full ImageNet validation of the fixed start checkpoint produced Top-1 78.116 / Top-5 94.200, exactly matching the baseline. Current deltas are therefore real training effects, not eval-path drift.

### C9: partial ref-logit-only stage2

Design rationale: remove attention KL entirely to test whether attention regularization is the source of degradation, while preserving refmodel via logit consistency.

Outcome: raw checkpoint-51 reached Top-1 78.050 / Top-5 94.226. Removing attention KL does not recover the baseline.

### C10: partial weak-ref lower-LR no-EMA stage2

Design rationale: C4 nearly preserved the checkpoint-50 baseline but used student EMA and LR 1e-5. C10 keeps the same partial trainable policy and weak prev-step selected-head attention KL, removes student EMA, and lowers LR to 5e-6 to test whether the small negative drift can become a positive gain.

Outcome: raw checkpoint-51 reached Top-1 78.124 / Top-5 94.186. This is the best 50→51 raw result so far and is +0.008 over the checkpoint-50 baseline, but it is far below the 79.0 gate.

## Objective audit after ten attempts

- Fixed checkpoint-50 start: satisfied for all C1-C10 and B0.
- Normal 50→51 single epoch: satisfied for completed candidates C2-C10; C1 was intentionally stopped by runtime gate.
- No soup / no averaging: satisfied; every reported raw result is a single checkpoint-51 or eval-only checkpoint-50.
- Full ImageNet distributed validation: satisfied for reported raw/EMA metrics; samples=50000 in logs.
- Raw checkpoint as main gate: satisfied.
- Refmodel idea retained: satisfied for C1-C10 via prev-step, EMA-ref, anchor-ref, or ref-logit.
- Epoch runtime <=10min: satisfied for completed training candidates C2-C10; C1 failed runtime and was stopped.
- Record command/script/resume/recipe/raw/EMA/delta/gate: satisfied in this document for completed candidates.
- Success gate Top-1 >=79.0: not achieved. Best raw is C10 Top-1 78.124.

Conclusion: goal is not complete. The most useful current finding is that full-param stage2 hurts; partial-unfreeze + KD + weak prev-step attention KL is the only direction that preserves/improves baseline slightly, with best C10 at +0.008.

### C11: pure quant-only weak-ref stage2

Design rationale: update only fake-quant/shift parameters with KD and weak prev-step attention KL, testing whether pure quantizer calibration is more stable than partial-unfreeze.

Outcome: raw checkpoint-51 reached Top-1 78.032 / Top-5 94.148. Pure quant-only is too narrow and does not improve over C10.

### C12: dynamic custom-top5 selected-head refmodel stage2

Design rationale: replace fixed custom10 averaging with online top-k selection. For each batch, the loss is computed over the custom10 head pool and only the five heads with the largest current student-vs-ref attention KL are optimized. This implements dynamic abnormal-head selection while keeping the memory/runtime bounded.

Outcome: raw checkpoint-51 reached Top-1 78.184 / Top-5 94.170. This is the best result so far (+0.068 over checkpoint-50) but still below 79.0.

### C13: dynamic all-head top5 refmodel stage2

Design rationale: extend dynamic top-k selection beyond the custom10 pool to all collected heads.

Outcome: raw checkpoint-51 reached Top-1 77.990 / Top-5 94.162. Full all-head dynamic top-k selects very large-KL heads and over-regularizes; dynamic selection should stay within a curated suspicious-head pool.

### C14: dynamic custom-top3 selected-head refmodel stage2

Design rationale: C12 showed dynamic custom-top5 is beneficial. C14 reduces the active heads from five to three to test whether less KL pressure is better.

Outcome: raw checkpoint-51 reached Top-1 78.106 / Top-5 94.212. This is worse than C12, so top5 within the custom suspicious-head pool is currently the best dynamic selection setting.

## Updated audit after C14

- Best raw checkpoint-51: C12 Top-1 78.184 / Top-5 94.170, delta +0.068.
- Success gate Top-1 >=79.0: still not achieved.
- Runtime gate <=10min: satisfied for C12/C13/C14; C13 is near the limit but still under 10min train.
- Strongest algorithmic finding: online dynamic selection within a curated suspicious-head pool is materially better than fixed custom10, all-head dynamic top-k, pure quant-only, ref-logit-only, anchor-ref, CE-only, or EMA variants.
- Next rational mechanism: C12 plus per-layer/per-head weight normalization or top-k loss clipping, because C13/C14 show the selected KL magnitude matters.

### C15: dynamic custom-top5 JS refmodel stage2

Design rationale: keep C12 dynamic custom-top5 selection but replace KL with JS divergence to soften the attention consistency loss.

Outcome: raw checkpoint-51 reached Top-1 78.046 / Top-5 94.178. JS is worse than directional KL, so C12 remains best.

### C15b: dynamic custom-top5 lower KL weight

Design rationale: keep C12 but reduce ref-attention weight from 1e-4 to 5e-5 to test whether less KL pressure helps.

Outcome: raw checkpoint-51 reached Top-1 78.126 / Top-5 94.176. Lower weight is worse than C12, so the C12 weight 1e-4 remains best.

### C16: full-parameter dynamic custom-top5 refmodel stage2

Design rationale: C12 established dynamic custom-top5 as the best mechanism under partial-unfreeze. C16 tests whether full-parameter updates can benefit once the attention ref loss is dynamic rather than fixed.

Outcome: raw checkpoint-51 reached Top-1 78.198 / Top-5 94.204. This is the best result so far (+0.082 over checkpoint-50), but it still does not reach the 79.0 gate.

## Current final audit

Objective criteria:

- Fixed start checkpoint: all reported candidates use the required checkpoint-50.
- Normal 50→51 single epoch: satisfied for completed candidates; C1 was stopped only because runtime exceeded the explicit gate.
- No soup / no averaging / no >1epoch ckpt: satisfied.
- Full ImageNet distributed validation: satisfied for reported metrics, with `Samples: 50000` in logs.
- Raw Top-1 main gate: satisfied in reporting.
- EMA reporting: reported where EMA was enabled; later best candidates intentionally disable EMA after it underperformed.
- Refmodel retained: satisfied; C16 uses prev-step refmodel with dynamic custom-top5 attention KL.
- Runtime <=10min: satisfied for current best C16; train avg 0.417s/step, about 8.7min train plus ~31s validation.
- Progress document: this file records command/script, resume, recipe, metrics, delta, and gate status.
- Success gate Top-1 >=79.0: not achieved. Current best is C16 Top-1 78.198.

Goal status: not complete. Do not extend to 100epoch because the 50→51 79.0 gate is not met.

Most defensible next step if continuing: C16 plus a dynamic loss schedule, e.g. start without ref KL for early updates then enable dynamic custom-top5, or layer-normalized/clipped dynamic KL to keep the useful dynamic-head mechanism without over-regularizing.

### C17: full-param dynamic custom-top5 with update-level ref warmup

Design rationale: delay ref KL for the first 200 optimizer updates so the model first adapts with KD, then applies dynamic attention stabilization.

Outcome: raw checkpoint-51 reached Top-1 78.136 / Top-5 94.176. This is worse than C16, so delayed ref activation is not useful.

### C18: full-param dynamic custom-top5 with KL clipping

Design rationale: C16 is best, while C13 showed over-large all-head top-k KL can over-regularize. C18 clips per-head KL at 70 to reduce extreme dynamic-head influence.

Outcome: raw checkpoint-51 reached Top-1 78.008 / Top-5 94.148. Clipping hurts; C16 remains best.

## Final audit after C18

- Fixed start checkpoint: all reported candidates use required checkpoint-50.
- Normal 50→51 single epoch: satisfied for completed candidates.
- No soup / no averaging / no >1epoch ckpt: satisfied.
- Full ImageNet distributed validation: satisfied for reported metrics, samples=50000.
- Raw Top-1 main gate: satisfied.
- EMA reporting: reported for EMA-enabled candidates; best candidates do not use EMA because EMA underperformed.
- Refmodel retained: satisfied; current best C16 uses prev-step refmodel with dynamic custom-top5 attention KL.
- Runtime <=10min: satisfied for best C16 and subsequent candidates.
- Progress document: satisfied; this document records scripts, recipes, metrics, deltas, and gates.
- Success gate Top-1 >=79.0: not achieved. Best is C16 Top-1 78.198, delta +0.082.

Conclusion: the goal remains active/incomplete. Do not launch 100epoch training. The strongest algorithmic result is dynamic custom-top5 head selection with full-parameter KD training.

Next non-redundant direction: layer-wise dynamic top-k or per-layer normalized dynamic KL, because global all-head top-k and clipped KL both underperform while custom-pool dynamic top5 works.

### C19: full-param layer-wise dynamic custom top1

Design rationale: avoid one layer/head dominating global dynamic top-k by selecting top1 within each layer from the custom suspicious-head pool.

Outcome: raw checkpoint-51 reached Top-1 78.082 / Top-5 94.168. Layer-wise balancing weakens the useful signal; C16 remains best.

### C20: full-param dynamic custom-top5 with EMA-ref

Design rationale: replace C16 prev-step refmodel with a slower EMA refmodel (momentum 0.99), testing whether multi-step temporal smoothing is better than immediate previous-step stabilization.

Outcome: raw checkpoint-51 reached Top-1 78.050 / Top-5 94.172. EMA-ref is much worse than prev-step ref for this 50→51 gate.

## Audit after C20

- Best result remains C16: raw Top-1 78.198 / Top-5 94.204, delta +0.082 over checkpoint-50.
- Target Top-1 >=79.0 is still not achieved.
- Do not run to 100epoch.
- Mechanisms tested after C16: update warmup, KL clipping, layer-wise dynamic selection, EMA-ref. All underperform C16.
- Strong conclusion: for this checkpoint and one-epoch gate, the best discovered stage2 mechanism is full-parameter KD with prev-step refmodel and dynamic custom-top5 directional attention KL.
- Next non-redundant work would require a new mechanism beyond these axes, e.g. teacher-ref attention/logit hybrid with dynamic custom-top5, or changing the data/calibration path; continuing local tweaks around C16 is unlikely to bridge 0.8 Top-1.

### C21: full-param dynamic custom-top5 with KD+auxiliary CE

Design rationale: C8 showed CE-only is destructive, but C16 uses KD only. C21 adds a small hard-label CE auxiliary loss (0.05) to C16 to test whether real-label correction improves the one-epoch gate.

Outcome: raw checkpoint-51 reached Top-1 78.112 / Top-5 94.216. Auxiliary CE does not help; C16 remains best.

## Audit after C21

- Best result remains C16: raw Top-1 78.198 / Top-5 94.204.
- Target Top-1 >=79.0 is not achieved.
- Tested post-C16 mechanisms: update warmup, KL clipping, layer-wise top-k, EMA-ref, auxiliary CE. All underperform C16.
- C16 is the current stable recipe and should be the base for any future nonlocal mechanism.

### C22: full-param dynamic custom-top5 with teacher-attention ref

Design rationale: add FP teacher attention as a second structural reference on top of C16 prev-step ref, reusing the KD teacher forward to avoid an extra forward pass.

Outcome: raw checkpoint-51 reached Top-1 78.030 / Top-5 94.190. Teacher attention KL is very large and hurts; C16 remains best.

## Final audit after C22

Concrete success criteria and evidence:

- Fixed start checkpoint: satisfied. Scripts use `/mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_continue10_to50_sysdisk_20260630/checkpoint-50.pth.tar`.
- Normal single 50→51 epoch: satisfied for completed candidates; no candidate result uses >1 epoch weights.
- No soup / no averaging: satisfied; all reported metrics are single checkpoint or eval-only baseline.
- Full ImageNet validation: satisfied for reported raw/EMA metrics; logs report 50000 samples.
- Raw Top-1 gate: satisfied in reporting; raw is primary.
- Refmodel retained: satisfied; best C16 uses prev-step refmodel and dynamic custom-top5 attention KL.
- Runtime <=10min: satisfied; best C16 is ~8.7min train + ~31s val.
- Experiment recording: satisfied in this document.
- Goal Top-1 >=79.0: not achieved. Best is C16 Top-1 78.198 / Top-5 94.204.

Goal status: incomplete. Do not run to 100epoch.

Best discovered recipe: C16 = full-param + KD + prev-step refmodel + dynamic custom-top5 directional attention KL, lr=min_lr=5e-6, no EMA.

Why C16 is strongest: dynamic selection in a curated head pool works; all-head dynamic, layer-wise balancing, JS, clipping, warmup, EMA-ref, auxiliary CE, teacher-attn ref, pure quant-only, partial-only all underperform.

### C23: post-resume quant calibration before C16 training

Design rationale: after loading checkpoint-50, run 16 no-grad setup-alpha/calibration batches before training, testing whether adapting quantizer/alpha/stat state before stage2 improves the one-epoch gate.

Outcome: raw checkpoint-51 reached Top-1 78.086 / Top-5 94.198. Post-resume calibration hurts; checkpoint-50 already contains useful trained quant state and should not be overwritten before stage2.

## Final audit after C23

- Best result remains C16: raw Top-1 78.198 / Top-5 94.204, delta +0.082 over checkpoint-50.
- Target Top-1 >=79.0 is not achieved.
- Stage2 should not be extended to 100epoch under the current goal.
- Mechanisms tested and rejected after C16: update warmup, KL clipping, layer-wise balancing, EMA-ref, auxiliary CE, teacher-attn hybrid, post-resume calibration.
- Best current recipe: full-param + KD + prev-step refmodel + dynamic custom-top5 directional attention KL, lr=min_lr=5e-6, no EMA.

## C16 50→60 diagnostic run

Purpose: user requested a diagnostic extension of the current best C16 recipe from checkpoint-50 to epoch60 to test whether the weak 50→51 gain is a warmup/too-few-epochs issue. This is an explicit exception to the earlier no-long-run rule; it is not counted as satisfying the 50→51 >=79 gate.

Recipe: C16 full-parameter training, KD, prev-step refmodel, dynamic custom-top5 directional attention KL, `ref_attn_kl_weight=1e-4`, `lr=min_lr=5e-6`, no EMA, no post-resume calibration.

Script: `tmp_scripts/run_stage2_c16_from50_to60_dynamic_customtop5_20260701.sh`.

Start checkpoint: `checkpoint-50.pth.tar`, raw Top-1 78.116 / Top-5 94.200.

Validation curve:

| checkpoint | trained epoch just completed | raw Top-1 | raw Top-5 | delta Top-1 vs ckpt50 |
|---:|---:|---:|---:|---:|
| 50 | start | 78.116 | 94.200 | +0.000 |
| 51 | 50 | 78.198 | 94.204 | +0.082 |
| 52 | 51 | 78.088 | 94.166 | -0.028 |
| 53 | 52 | 78.064 | 94.102 | -0.052 |
| 54 | 53 | 78.110 | 94.210 | -0.006 |
| 55 | 54 | 78.122 | 94.224 | +0.006 |
| 56 | 55 | 78.042 | 94.252 | -0.074 |
| 57 | 56 | 78.078 | 94.164 | -0.038 |
| 58 | 57 | 78.070 | 94.140 | -0.046 |
| 59 | 58 | 78.082 | 94.148 | -0.034 |
| 60 | 59 | 78.104 | 94.142 | -0.012 |

Runtime: training stayed stable at about 0.414-0.416s/step, roughly 8.6-8.7min train per epoch. Distributed validation took about 7s after the first validation warmup.

Conclusion: the 10-epoch extension does not support the hypothesis that C16 only needs more warmup. The best point remains checkpoint-51 at Top-1 78.198. Later epochs fluctuate around or below the checkpoint-50 baseline and do not trend upward. The next stage2 mechanism should reduce continuous KL pressure rather than simply run C16 longer.

Next mechanisms to test:

1. KL dropout / stochastic KL gate: compute the C16 dynamic custom-top5 KL only with probability `p` per update or per selected head, using inverted scaling (`loss *= mask / p`) so expected KL weight is unchanged. This keeps the refmodel idea but prevents the prev-step constraint from being applied every update.
2. Alternating stage1/stage2 schedule: run normal stage1/KD epochs and stage2 KL epochs in blocks, e.g. 9:1 or 10:1. This treats KL as a periodic stabilization step instead of a permanent training objective.

Preferred next experiment order: first implement KL dropout because it is local, cheap, and directly tests whether continuous KL pressure is suppressing improvement. If it improves one-epoch behavior, then test block alternation.

### C24: C16 with KL dropout p=0.5, no scaling

Design rationale: apply dynamic custom-top5 prev-step attention KL stochastically on about half of updates. This tests whether continuous KL pressure is suppressing improvement; no inverted scaling means the average KL strength is also reduced.

Implementation: added `--ref-attn-kl-drop-prob` and `--ref-attn-kl-drop-scale` controls. C24 uses `--ref-attn-kl-drop-prob 0.5` without scale.

Outcome: raw checkpoint-51 reached Top-1 78.114 / Top-5 94.102. This is below C16 78.198 and essentially returns to the checkpoint-50 baseline. Reducing average KL strength is not helpful.

### C25: C16 with KL dropout p=0.5 and inverted scaling

Design rationale: keep the same expected KL strength as C16 but make the KL pressure intermittent. This isolates "continuous pressure" from "total KL strength".

Implementation: `--ref-attn-kl-drop-prob 0.5 --ref-attn-kl-drop-scale`.

Outcome: raw checkpoint-51 reached Top-1 78.048 / Top-5 94.174. This is worse than both C16 and C24. Intermittent high-magnitude KL is harmful.

## Audit after KL-dropout tests

- C16 remains best: Top-1 78.198 / Top-5 94.204.
- C24 no-scale dropout: Top-1 78.114, indicates lowering average KL removes the small C16 gain.
- C25 scaled dropout: Top-1 78.048, indicates high intermittent KL is destabilizing.
- KL dropout at p=0.5 is not a promising direct improvement. If revisiting dropout, only mild dropout such as p=0.75 no-scale is worth a cheap check, but the stronger next idea is stage1/stage2 alternation.

Next preferred direction: epoch/block alternation, e.g. stage1/KD normal training for several epochs and a C16-style stage2 stabilization epoch periodically. This preserves stage1's normal improvement path while using refmodel KL as an occasional stabilizer rather than a persistent objective.

### C26: no-KL stage1 continuation from checkpoint-50

Design rationale: before investing in stage1/stage2 block alternation, test whether the normal stage1 continuation path from checkpoint-50 improves the 50→51 raw checkpoint by itself. This uses the same resume, KD, augmentation-disabled, LR=min_lr=5e-6 setup but disables the refmodel/KL scheme.

Script: `tmp_scripts/run_stage2_c26_stage1_nokl_from50_to51_20260701.sh`.

Outcome: raw checkpoint-51 reached Top-1 78.110 / Top-5 94.228. Train avg step time was 0.291s/step, much faster because no refmodel forward is used. This is below the checkpoint-50 baseline and below C16, so simple stage1/no-KL continuation is not a useful 50→51 stage2 candidate.

## Audit after C26

- C16 remains best: Top-1 78.198 / Top-5 94.204.
- Normal no-KL stage1 continuation does not improve from checkpoint-50: Top-1 78.110.
- Stage1/stage2 alternation is still potentially useful as a longer-term training narrative, but it is not a direct solution to the strict 50→51 >=79 gate because the stage1 component itself does not provide immediate one-epoch gain.
- Next non-redundant 50→51 algorithm should change the supervision/reference target, not just the frequency of C16 KL or the presence/absence of KL.

Recommended next candidate: teacher-logit-only or teacher-consistency focused stabilization with refmodel only on quantizer-sensitive blocks, e.g. freeze most FP weights and update quant/normalization/attention projection with KD plus a fixed checkpoint-50 anchor only for quantized attention logits. The evidence so far shows full-param dynamics can preserve accuracy, but attention-prob KL is too weak/misaligned to bridge to 79.

### C27: partial teacher Q/K relation stabilization

Design rationale: move away from attention-prob KL and test a teacher structural signal closer to Q/K relation geometry. To avoid destabilizing the whole model, only `head_norm_attn_quant` parameters receive gradients via the update-level trainable policy. No prev-step attention KL is used.

Script: `tmp_scripts/run_stage2_c27_partial_teacher_qkrel_1e3_from50_to51_20260701.sh`.

Recipe: KD + teacher Q/K relation loss weight 1e-3, no ref attention KL, trainable-policy `head_norm_attn_quant`, lr=min_lr=5e-6.

Outcome: raw checkpoint-51 reached Top-1 78.132 / Top-5 94.234. Train avg step time was 0.354s/step. TeacherQKRel was about 7e-6, so weight 1e-3 makes this auxiliary loss effectively negligible. It slightly beats no-KL C26 but is still far below C16 and the 79 gate.

## Audit after C27

- Best remains C16: Top-1 78.198 / Top-5 94.204.
- C27 teacher Q/K relation at weight 1e-3 is too weak to matter; the measured auxiliary term is ~7e-6.
- If using Q/K relation again, the loss must be rescaled or weighted orders of magnitude higher. However, because this direction currently behaves like a weak no-KL partial update, it is unlikely to bridge to 79 by itself.
- The 50→51 gate remains unmet.

### C28: partial teacher Q/K relation stabilization with large scale

Design rationale: C27 showed the teacher Q/K relation loss has the right runtime profile but its raw value is only ~7e-6, making weight 1e-3 negligible. C28 scales the same loss to weight 1000 so its contribution is around 0.006-0.007, comparable to C16's attention KL contribution.

Script: `tmp_scripts/run_stage2_c28_partial_teacher_qkrel_1e3scaled_from50_to51_20260701.sh`.

Recipe: KD + teacher Q/K relation loss weight 1000, no ref attention KL, trainable-policy `head_norm_attn_quant`, lr=min_lr=5e-6.

Outcome: raw checkpoint-51 reached Top-1 78.074 / Top-5 94.180. Train avg step time was 0.354s/step. Large-scale Q/K relation hurts relative to C27 and no-KL, so this teacher-geometry loss is not aligned with the one-epoch accuracy gate.

## Audit after C28

- Best remains C16: Top-1 78.198 / Top-5 94.204.
- Teacher Q/K relation was tested both near-zero scale (C27) and C16-comparable scale (C28); neither improves.
- The 50→51 Top-1 >=79.0 gate remains unmet.

### C29: no-KL stage1 continuation with LR 1e-5

Design rationale: C26 used the low C16-style LR 5e-6. C29 tests whether no-KL continuation failed simply because LR was too low by raising LR/min-LR to 1e-5, closer to the stage1 schedule floor.

Script: `tmp_scripts/run_stage2_c29_stage1_lr1e5_from50_to51_20260701.sh`.

Recipe: KD, no refmodel/KL, full-param training, lr=min_lr=1e-5, no EMA.

Outcome: raw checkpoint-51 reached Top-1 77.986 / Top-5 94.118. Train avg step time was 0.291s/step. This is worse than C26 and baseline, so no-KL continuation is not limited by too-small LR.

## Audit after C29

- Best remains C16: Top-1 78.198 / Top-5 94.204.
- No-KL continuation at LR 5e-6 and 1e-5 both fails to improve over checkpoint-50.
- Stage1 continuation is not a direct path to the 50→51 >=79 gate.

## Stage1 resume diagnostics after C30

Motivation: the strict 50→51 stage2 gate may be confounded by whether checkpoint-50 can be resumed productively at all. User requested strict resume with optimizer state restored and no per-epoch checkpoint spam.

### S1: stage1 50→60, no optimizer resume, tail schedule

Script: `tmp_scripts/run_stage1_continue50_to60_resume_diagnostic_20260701.sh`.

Recipe: no stage2 KL, KD/noaug stage1 continuation, `--no-resume-opt`, global batch 2048, scheduler tail from epoch50 to 60. Checkpoints saved every epoch to system disk for diagnosis.

Curve:

| checkpoint | raw Top-1 | raw Top-5 |
|---:|---:|---:|
| 50 | 78.116 | 94.200 |
| 51 | 77.980 | 94.028 |
| 52 | 78.040 | 94.160 |
| 53 | 77.982 | 94.146 |
| 54 | 77.988 | 94.204 |
| 55 | 78.086 | 94.190 |
| 56 | 77.950 | 94.184 |
| 57 | 78.260 | 94.222 |
| 58 | 78.112 | 94.156 |
| 59 | 78.114 | 94.218 |
| 60 | 78.168 | 94.210 |

Conclusion: resume is not completely broken; checkpoint-57 briefly exceeds the checkpoint-50 baseline and C16. However, the curve is noisy and not a monotonic stage1 lift.

### S2: stage1 60→100, no optimizer resume, schedule reset high LR

Script: `tmp_scripts/run_stage1_continue60_to100_tmp_20260701.sh`.

Recipe: resumed from S1 checkpoint-60, no optimizer resume, scheduler reset to 100-epoch trajectory. This accidentally raised LR near epoch60 to about 7.5e-5.

Observed curve before stop:

| checkpoint | raw Top-1 | raw Top-5 |
|---:|---:|---:|
| 61 | 77.604 | 93.960 |
| 62 | 77.514 | 93.994 |

Conclusion: high-LR schedule reset is destructive for this late checkpoint. Stopped instead of continuing to 100.

### S3: stage1 strict optimizer resume from checkpoint-50, high LR 100-epoch schedule

Script: `tmp_scripts/run_stage1_strict_resume50_to100_tmp_20260701.sh`.

Recipe: strict resume from checkpoint-50, optimizer state restored (`Restoring optimizer state from checkpoint...` observed in logs), 100-epoch scheduler. LR at epoch51 was about 1e-4.

Observed curve before stop:

| checkpoint | raw Top-1 | raw Top-5 |
|---:|---:|---:|
| 52 | 77.140 | 93.700 |

Conclusion: strict optimizer resume works technically, but the restored optimizer plus high LR 100-epoch schedule is too aggressive and destroys accuracy. Stopped early.

### S4: stage1 strict optimizer resume from checkpoint-50, fixed low LR 1e-5

Script: `tmp_scripts/run_stage1_strict_resume50_to100_lr1e5_tmp_20260701.sh`.

Recipe: strict resume from checkpoint-50, optimizer state restored, fixed `lr=min_lr=1e-5`, checkpoint every 5 epochs, hist 4.

Observed curve before stop:

| checkpoint | raw Top-1 | raw Top-5 |
|---:|---:|---:|
| 52 | 78.034 | 94.120 |
| 53 | 78.076 | 94.192 |
| 54 | 78.002 | 94.222 |
| 55 | 78.046 | 94.178 |

Conclusion: strict low-LR resume is stable but does not lift the model beyond checkpoint-50. Stopped because it was below baseline through checkpoint-55.

## Stage1 resume audit conclusion

- Restoring optimizer state is now verified and should be used for long stage1 continuation.
- The 50-epoch checkpoint is near a plateau under the current KD/noaug stage1 recipe; simply resuming to 100 does not reliably lift it.
- High LR after checkpoint-50 is harmful; low LR is stable but does not improve enough.
- Stage2 failures are not only caused by the KL term; the base checkpoint itself has weak one-epoch improvement dynamics.
- Next rational path is either:
  1. start stage1 continuation from an earlier checkpoint before this plateau and train strictly with optimizer state; or
  2. redesign stage2 as a true teacher/ref stabilization objective that changes the optimization target, not just a weak attention-prob KL around a plateaued checkpoint.

### S5: stage1 strict resume from checkpoint-50, fixed LR 1e-5

Script: `tmp_scripts/run_stage1_strict_resume50_to100_lr1e5_tmp_20260701.sh`.

Recipe: strict optimizer resume from checkpoint-50, fixed `lr=min_lr=1e-5`, no stage2 KL, checkpoint interval 5. Optimizer restoration was confirmed by log line `Restoring optimizer state from checkpoint...`.

Observed curve before stop:

| checkpoint | raw Top-1 | raw Top-5 |
|---:|---:|---:|
| 52 | 78.034 | 94.120 |
| 53 | 78.076 | 94.192 |
| 54 | 78.002 | 94.222 |
| 55 | 78.046 | 94.178 |

Conclusion: strict optimizer resume with low LR is stable but still does not improve over checkpoint-50. Stopped.

### S6: stage1 strict resume from checkpoint-40, 100-epoch high-LR schedule

Script: `tmp_scripts/run_stage1_strict_resume40_to100_tmp_20260701.sh`.

Recipe: strict optimizer resume from checkpoint-40, `lr=2e-4`, `scheduler_epochs=100`, no stage2 KL. Optimizer restoration was confirmed by log line `Restoring optimizer state from checkpoint...`.

Observed first validation before stop:

| checkpoint | raw Top-1 | raw Top-5 |
|---:|---:|---:|
| 42 | 76.400 | 93.306 |

Conclusion: resuming an intermediate QAT checkpoint into a 100-epoch high-LR schedule is destructive; LR around 1.3e-4 at epoch41 is too high for this already-quantized model. Stopped immediately.

## Updated stage1/stage2 strategy conclusion

- Strict optimizer resume is technically working; failures are not due to missing optimizer state after the correction.
- For late or mid QAT checkpoints, high LR schedule reset is destructive.
- Fixed low LR strict resume is stable but does not produce a strong one-epoch or short-horizon lift from checkpoint-50.
- Therefore, simply pushing stage1 from existing mid/late checkpoints does not solve the 50→51 >=79 stage2 gate.
- A credible next stage2 algorithm should not rely on ordinary continuation dynamics. It needs a stronger stabilization/supervision target, likely teacher-ref driven, and should be evaluated against checkpoint-50 as originally required.

### C31: partial dynamic attention KL plus prev-step ref-logit KL

Design rationale: C16's attention-prob KL is the best so far but weak. C31 adds a temporal logit consistency term from the prev-step refmodel while restricting gradients to quant/norm/attention parameters, testing whether output-level ref stabilization helps without full-model drift.

Script: `tmp_scripts/run_stage2_c31_partial_dynamic_attn_reflogit_teacherref_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, KD, prev-step refmodel, dynamic custom-top5 attention KL weight 1e-4, ref-logit KL weight 0.01 temperature 2.0, trainable-policy `head_norm_attn_quant`, lr=min_lr=5e-6, no EMA.

Outcome: raw checkpoint-51 reached Top-1 78.174 / Top-5 94.180. Delta vs checkpoint-50 is +0.058. This is below C16 Top-1 78.198 and far below the 79.0 gate.

Conclusion: adding prev-step logit KL to partial dynamic attention KL does not improve the best result. Continue with a cleaner teacher-ref attention experiment rather than temporal logit KL.

### C32: low-weight FP teacher attention KL, no prev-step attention KL

Design rationale: user noted that learning FP teacher attention scores is also a KL-style ref objective. C22 used teacher attention together with prev-step dynamic attention and was too strong. C32 isolates a low-weight FP-teacher attention KL without prev-step attention KL, while restricting gradients to quant/norm/attention parameters.

Script: `tmp_scripts/run_stage2_c32_teacher_attn_loww_dynamic_customtop5_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, KD, teacher attention KL weight 5e-6, dynamic custom-top5 head pool, no prev-step attention KL, trainable-policy `head_norm_attn_quant`, lr=min_lr=5e-6, no EMA.

Runtime: train avg step time 0.296s/step, much faster than prev-step-ref variants because no student-vs-ref extra forward is needed. Full validation was distributed over 50000 samples.

Outcome: raw checkpoint-51 reached Top-1 78.040 / Top-5 94.126. Delta vs checkpoint-50 is -0.076. TeacherAttnKL raw value was about 415, so even weight 5e-6 contributed about 0.002 loss. The result is below baseline and far below the 79.0 gate.

Conclusion: directly matching FP teacher attention probabilities is not a useful stage2 objective for this checkpoint, even at low weight. Future teacher-ref work should focus on logits/calibration/quantizer states rather than attention-prob KL.

### C33: teacher Q/K direction matching KD with partial trainable policy

Design rationale: direct FP teacher attention probability KL failed in C32. C33 tests OFQ's existing teacher-ref Q/K direction matching distillation (`kd_hard_and_soft=2`), which constrains Q/K geometry rather than final attention probability. Gradients are restricted to quant/norm/attention parameters.

Script: `tmp_scripts/run_stage2_c33_kd_qk_teacherref_partial_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=2`, no stage2 attention KL, trainable-policy `head_norm_attn_quant`, lr=min_lr=5e-6, no EMA.

Runtime: train avg step time 0.352s/step. Full validation was distributed over 50000 samples.

Outcome: raw checkpoint-51 reached Top-1 77.702 / Top-5 93.834. Delta vs checkpoint-50 is -0.414. Training loss was very large (~11.7 down to ~9), showing the default Q/K direction matching term is far too strong.

Conclusion: unscaled OFQ Q/K direction matching is destructive for this late checkpoint. If Q/K geometry is revisited, it needs an explicit tunable weight; the built-in `kd_hard_and_soft=2` path is not suitable as-is.

### C34: teacher soft-logit-only calibration with partial trainable policy

Design rationale: attention-prob KL and Q/K direction matching were either weak or destructive. C34 tests a cleaner teacher-ref stage2: use only teacher soft logits as the reference target, with no hard CE and no attention KL, while updating only quant/norm/attention parameters.

Script: `tmp_scripts/run_stage2_c34_teacher_softonly_partial_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=0` teacher soft logits only, trainable-policy `head_norm_attn_quant`, lr=min_lr=5e-6, no EMA.

Runtime: train avg step time 0.285s/step; full validation was distributed over 50000 samples.

Outcome: raw checkpoint-51 reached Top-1 78.198 / Top-5 94.222. Delta vs checkpoint-50 is +0.082. This matches C16's best Top-1 while improving Top-5 slightly, without any attention KL.

Conclusion: clean teacher-logit calibration on a restricted parameter subset is as strong as the best attention-ref recipe and simpler. This is the best non-attention-KL direction so far. Next test: add student EMA or a weak prev-step/dynamic attention regularizer on top only if it does not hurt raw.

### C35: C34 plus student EMA

Design rationale: C34 matched the best raw Top-1 without attention KL. C35 adds a normal single-run student EMA branch, testing whether EMA improves evaluation while preserving the raw checkpoint gate.

Script: `tmp_scripts/run_stage2_c35_teacher_softonly_partial_ema999_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=0` teacher soft logits only, trainable-policy `head_norm_attn_quant`, lr=min_lr=5e-6, student EMA decay 0.999.

Outcome: raw checkpoint-51 reached Top-1 78.198 / Top-5 94.222, matching C34 and C16. Delta vs checkpoint-50 is +0.082. This does not reach the 79.0 gate.

EMA note: `checkpoint-51.ema.pth.tar` was saved, but direct eval through the normal resume/eval path produced an invalid Top-1 0.258 / Top-5 1.054 despite the checkpoint having a normal-looking state_dict. Treat this EMA metric as invalid/untrusted until the EMA save/load path is fixed. Do not use it for model selection.

Conclusion: EMA does not currently provide a usable improvement signal. Teacher soft-logit calibration remains the cleanest best-equivalent stage2 direction, but raw Top-1 is still 78.198.

### C36: teacher soft-logit calibration plus dynamic prev-step attention KL

Design rationale: C34 teacher-soft-only calibration and C16 dynamic prev-step attention KL are the two strongest individual mechanisms. C36 tests whether they are complementary when applied together under the same restricted parameter policy.

Script: `tmp_scripts/run_stage2_c36_teacher_softonly_plus_dynamic_attn_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=0`, prev-step refmodel dynamic custom-top5 attention KL weight 1e-4, trainable-policy `head_norm_attn_quant`, lr=min_lr=5e-6, no EMA.

Runtime: train avg step time 0.408s/step; full validation was distributed over 50000 samples.

Outcome: raw checkpoint-51 reached Top-1 77.974 / Top-5 94.166. Delta vs checkpoint-50 is -0.142.

Conclusion: teacher soft-logit calibration and dynamic attention KL are not complementary in this form; adding attention KL to C34 hurts. The clean teacher-soft-only recipe C34/C35 raw remains the best teacher-ref direction and ties C16 at Top-1 78.198.

## Current best audit after C36

- Best raw Top-1 remains 78.198, achieved by both:
  - C16: full-param KD + prev-step dynamic custom-top5 attention KL.
  - C34/C35 raw: teacher soft-logit-only calibration with `head_norm_attn_quant` trainable policy.
- Neither reaches the required Top-1 >=79.0 gate.
- Attention-prob KL, teacher-attention KL, unscaled Q/K direction matching, EMA-ref, KL dropout, update warmup, clipping, post-resume calibration, and stage1 continuation diagnostics have all failed to bridge the gap.
- Most promising conceptual direction is no longer attention matching; it is teacher-logit/quantizer calibration with a stronger or more targeted quantization-error objective.

### C37: teacher soft-logit-only calibration with quant-only trainable policy

Design rationale: C34 showed teacher soft-logit calibration with `head_norm_attn_quant` ties the best Top-1. C37 narrows the trainable policy to quantizer/shift parameters only, testing whether the benefit comes mainly from quantizer calibration.

Script: `tmp_scripts/run_stage2_c37_teacher_softonly_quantonly_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=0` teacher soft logits only, trainable-policy `quant`, lr=min_lr=5e-6, no EMA.

Runtime: train avg step time 0.279s/step; full validation was distributed over 50000 samples.

Outcome: raw checkpoint-51 reached Top-1 78.148 / Top-5 94.120. Delta vs checkpoint-50 is +0.032. This is below C34/C35 raw Top-1 78.198.

Conclusion: quantizer-only calibration helps slightly but is not enough. C34's benefit requires updating a broader local subset (`head_norm_attn_quant`), not just quantizer/shift parameters.

### C38: teacher soft-logit-only calibration with head/norm/quant trainable policy

Design rationale: C37 showed quant-only calibration is weaker than C34. C38 tests the intermediate policy `head_norm_quant`, identifying whether the additional C34 gain comes from head/norm parameters or from attention projection parameters.

Script: `tmp_scripts/run_stage2_c38_teacher_softonly_headnormquant_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=0` teacher soft logits only, trainable-policy `head_norm_quant`, lr=min_lr=5e-6, no EMA.

Runtime: train avg step time 0.282s/step; full validation was distributed over 50000 samples.

Outcome: raw checkpoint-51 reached Top-1 78.074 / Top-5 94.190. Delta vs checkpoint-50 is -0.042. This is worse than C37 quant-only and C34 head_norm_attn_quant.

Conclusion: head/norm updates without attention projections do not help. The C34 gain requires attention-projection parameters together with quant/norm/head; this supports a local teacher-logit calibration view focused on quantized attention blocks rather than pure quantizer-only tuning.

### C39: teacher soft-logit-only calibration with attention-projection/quant trainable policy

Design rationale: C38 showed head/norm/quant without attention projections is worse than quant-only. C39 tests the complementary slice, `attn_quant`, to determine whether attention projection parameters plus quantizer/shift parameters reproduce C34's `head_norm_attn_quant` gain.

Script: `tmp_scripts/run_stage2_c39_teacher_softonly_attnquant_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=0` teacher soft logits only, trainable-policy `attn_quant`, lr=min_lr=5e-6, no EMA.

Runtime: full validation was distributed over 50000 samples.

Outcome: raw checkpoint-51 reached Top-1 78.074 / Top-5 94.238. Delta vs checkpoint-50 is -0.042. This is below C34/C35 raw Top-1 78.198 and far below the 79.0 gate.

Conclusion: attention projections without head/norm parameters also do not reproduce C34. The current best teacher-soft result requires the full local block (`head_norm_attn_quant`) but still only reaches 78.198. Next useful direction should stop slicing trainable policies and instead add a stronger quantization-error / activation-output teacher-ref objective.

### Infrastructure note: strict resume checkpointing added before further Stage2 experiments

After resume-state concerns, `qat_launch.py` was updated so newly saved main and step checkpoints carry full training state: raw model, optimizer, AMP loss scaler when present, LR scheduler, Python/Torch/CUDA RNG state, args, and EMA state when enabled. Checkpoint writes are now atomic via temporary file plus `os.replace`; checkpoint history pruning avoids deleting `.ema` files from raw pruning. Normal `--resume` now uses an in-repo strict resume path that restores optimizer/scheduler/scaler/RNG/EMA; `--no-resume-opt` explicitly restores weights only and logs that optimizer/scheduler/scaler/RNG are intentionally skipped.

Verification: `python -m py_compile QATs/qat_launch.py` passed. A CPU smoke test created a toy checkpoint and strict-resumed model/optimizer/scheduler/RNG/EMA successfully, printing restored optimizer state entries, scheduler state, EMA state, and RNG state. This is infra validation only; it does not satisfy the Stage2 Top-1 gate.


### C40: teacher soft-logit calibration plus FP-teacher attention-output MSE

Design rationale: C37-C39 showed that simply slicing trainable parameter policies cannot reach the gate. C40 adds a stronger teacher-ref stabilization target at the activation/output level: capture each Swin attention module output from the quantized student and FP teacher with forward hooks, and add MSE on those local outputs. This targets quantization-induced local block error more directly than attention-prob KL.

Script: `tmp_scripts/run_stage2_c40_teacher_softonly_attnout_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=0` teacher soft logits, trainable-policy `head_norm_attn_quant`, lr=min_lr=5e-6, teacher attention-output MSE on all attention layers with weight 1e-3, no EMA. Strict resume restored optimizer state entries from the start checkpoint; the old checkpoint did not contain scheduler/RNG, which is expected for legacy checkpoints.

Runtime: train avg step time 0.289336s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed under the 10 minute limit.

Outcome: raw checkpoint-51 reached Top-1 78.134 / Top-5 94.214. Delta vs checkpoint-50 is +0.018. This is below C34/C35 raw Top-1 78.198 and far below the 79.0 gate. TeacherAttnOut loss averaged about 7.7e-2, so weight 1e-3 only contributed about 7.7e-5 to total loss.

Conclusion: teacher attention-output MSE is technically stable and cheap enough, but weight 1e-3 is too weak and does not improve over teacher-soft-only calibration. One stronger-weight check is justified; if it still fails to beat C34, stop this activation-output MSE direction.

### C41: C40 with stronger teacher attention-output MSE weight

Design rationale: C40's attention-output MSE contribution was only about 7.7e-5, so C41 increases the same teacher-ref local-output target from weight 1e-3 to 1e-2 to test whether C40 was simply underweighted.

Script: `tmp_scripts/run_stage2_c41_teacher_softonly_attnout_w1e2_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=0` teacher soft logits, trainable-policy `head_norm_attn_quant`, lr=min_lr=5e-6, teacher attention-output MSE on all attention layers with weight 1e-2, no EMA.

Runtime: train avg step time 0.288608s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed under the 10 minute limit.

Outcome: raw checkpoint-51 reached Top-1 78.072 / Top-5 94.204. Delta vs checkpoint-50 is -0.044. This is worse than C40, C34, and the fixed checkpoint-50 baseline.

Conclusion: teacher attention-output MSE does not help in this form. Increasing its weight hurts, so stop the activation-output MSE branch and switch to a different stage2 algorithmic direction.

### C42: teacher hard+soft KD with partial trainable policy

Design rationale: C34 teacher soft-logit-only calibration tied the best result but may lack hard-label correction. C42 tests OFQ's hard+soft KD path with the same local trainable policy, keeping teacher-ref supervision while adding target-label signal.

Script: `tmp_scripts/run_stage2_c42_teacher_hardsoft_partial_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=1`, trainable-policy `head_norm_attn_quant`, lr=min_lr=5e-6, no attention KL, no EMA.

Runtime: train avg step time 0.285211s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed under the 10 minute limit.

Outcome: raw checkpoint-51 reached Top-1 78.086 / Top-5 94.170. Delta vs checkpoint-50 is -0.030. This is below C34/C35 raw Top-1 78.198 and below the 79.0 gate.

Conclusion: hard+soft KD does not improve the teacher-soft-only recipe. The next direction should add a real stabilization ref rather than only changing KD target composition.

### C43: teacher soft KD plus fixed anchor-ref logit KL

Design rationale: C42 showed KD target composition alone is not enough. C43 uses checkpoint-50 itself as a fixed anchor refmodel and adds logit KL from the current student to that anchor while teacher soft KD still provides the improvement direction. The intent is to prevent one-epoch local QAT calibration from drifting too far from the validated checkpoint-50 decision surface without using attention-prob KL.

Script: `tmp_scripts/run_stage2_c43_teacher_soft_anchorlogit_partial_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, teacher soft logits (`kd_hard_and_soft=0`), trainable-policy `head_norm_attn_quant`, fixed refmodel logit KL weight 0.05 temperature 2.0, lr=min_lr=5e-6, no EMA.

Runtime: train avg step time 0.385570s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed under the 10 minute limit. RefLogitKL averaged about 2.1e-2, so the added loss contribution was about 1e-3.

Outcome: raw checkpoint-51 reached Top-1 78.080 / Top-5 94.164. Delta vs checkpoint-50 is -0.036. This is below C34/C35 raw Top-1 78.198 and below the 79.0 gate.

Conclusion: fixed anchor-ref logit KL over-constrains or conflicts with teacher-soft calibration in this one-epoch setting. Do not continue this logit-anchor direction by just sweeping weights.

### C44: one-epoch two-stage trainable policy, quant-first then local-block

Design rationale: C34's full local-block calibration may drift too early, while C37 quant-only is stable but weak. C44 tests an intra-epoch staged update schedule: first half of the epoch masks gradients to quantizer/shift parameters only, then second half opens the broader `head_norm_attn_quant` local block. This is a training-dynamics change rather than a loss-weight sweep.

Script: `tmp_scripts/run_stage2_c44_quantfirst_then_partial_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, teacher soft logits (`kd_hard_and_soft=0`), lr=min_lr=5e-6, first 624 optimizer updates with grad-mask policy `quant`, then `head_norm_attn_quant`, no attention KL, no EMA. `grad_mask` mode was used because update-level `requires_grad` switching is incompatible with DDP `--static-graph`.

Runtime: train avg step time 0.290777s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed under the 10 minute limit.

Outcome: raw checkpoint-51 reached Top-1 78.072 / Top-5 94.214. Delta vs checkpoint-50 is -0.044. This is below C34/C35 raw Top-1 78.198 and below the 79.0 gate.

Conclusion: quant-first staging does not improve one-epoch stage2. The loss landscape likely needs better sample/objective selection, not only a staged parameter mask.

### C45: teacher-confidence weighted soft KD

Design rationale: C44 showed that staged parameter masking does not help. C45 tests sample selection inside the teacher-ref objective: use teacher top-1 confidence to reweight per-sample soft KD, normalized to mean weight 1. The intended effect is to let high-confidence teacher samples dominate the one-epoch local QAT update and reduce noisy teacher targets.

Script: `tmp_scripts/run_stage2_c45_teacher_confweighted_partial_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, teacher soft KD with confidence weighting power 2.0, trainable-policy `head_norm_attn_quant`, lr=min_lr=5e-6, no attention KL, no EMA.

Runtime: train avg step time 0.285455s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed under the 10 minute limit.

Outcome: raw checkpoint-51 reached Top-1 78.086 / Top-5 94.170. Delta vs checkpoint-50 is -0.030. This is below C34/C35 raw Top-1 78.198 and below the 79.0 gate.

Conclusion: teacher confidence reweighting does not improve the one-epoch stage2 result. Because recent strict-resume/code-path changes may affect comparability with the older C34 baseline, rerun the C34 recipe under the current code before judging additional variants.

### C46: current-code rerun of C34 teacher-soft-only baseline

Design rationale: Several recent variants under the strict-resume code path clustered around Top-1 78.08. C46 reruns the historical C34 recipe under the current code to check whether the old C34 result (78.198) remains reproducible after strict resume and other infrastructure changes.

Script: `tmp_scripts/run_stage2_c46_c34_rerun_strict_current_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=0` teacher soft logits only, trainable-policy `head_norm_attn_quant`, lr=min_lr=5e-6, no attention KL, no EMA. This is the same nominal recipe as C34, but using the current strict-resume implementation.

Runtime: train avg step time 0.285290s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed under the 10 minute limit.

Outcome: raw checkpoint-51 reached Top-1 78.086 / Top-5 94.170. Delta vs checkpoint-50 is -0.030. This is much lower than the historical C34 record of 78.198.

Conclusion: the current strict-resume path changes the effective baseline for this recipe, most likely because optimizer state is now actually restored from checkpoint-50 instead of being reset. This makes optimizer-state handling an explicit stage-transition design variable rather than a hidden implementation detail. Next test: weight-strict resume with optimizer reset as a deliberate stage2 recipe.

### C47: C34 recipe with deliberate stage2 optimizer reset

Design rationale: C46 showed that the current strict-resume rerun of C34 drops to 78.086, below the historical C34 record. C47 tests whether restored stage1 optimizer moments are suppressing stage2 adaptation by deliberately treating stage2 as a new optimization phase: weights are strictly loaded from checkpoint-50, but optimizer/scheduler/scaler/RNG are intentionally not restored via `--no-resume-opt`.

Script: `tmp_scripts/run_stage2_c47_c34_noresumeopt_current_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=0` teacher soft logits only, trainable-policy `head_norm_attn_quant`, lr=min_lr=5e-6, no attention KL, no EMA, explicit stage2 optimizer reset.

Runtime: train avg step time 0.285505s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed under the 10 minute limit. Log confirmed `--no-resume-opt` and printed that optimizer/scheduler/scaler/RNG were intentionally not restored.

Outcome: raw checkpoint-51 reached Top-1 78.086 / Top-5 94.188. Delta vs checkpoint-50 is -0.030. This is below the 79.0 gate and essentially the same as C46.

Conclusion: optimizer reset alone does not recover the historical C34 result and does not solve stage2. Continue only after checking whether newly added objective switches, especially teacher-confidence KD, are actually active in runtime.

### C48: historical-equivalent local policy without QKV, optimizer reset

Design rationale: Current C46/C47 reruns of C34 dropped to about 78.086. One suspected cause was that the newer `head_norm_attn_quant` policy included `.attn.qkv.` parameters, while the historical C34 policy may only have updated attention projection (`attn.proj`) plus head/norm/quant parameters. C48 adds and tests `head_norm_proj_quant`, which excludes QKV and updates only quant/shift, head/norm, and attention projection parameters.

Script: `tmp_scripts/run_stage2_c48_teacher_soft_projonly_noresumeopt_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=0` teacher soft logits only, trainable-policy `head_norm_proj_quant`, lr=min_lr=5e-6, no attention KL, no EMA, explicit stage2 optimizer reset. The run logged 3.159M trainable parameters, confirming QKV was excluded.

Runtime: train avg step time 0.283256s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed under the 10 minute limit.

Outcome: raw checkpoint-51 reached Top-1 78.040 / Top-5 94.216. Delta vs checkpoint-50 is -0.076. This is below C46/C47 and below the 79.0 gate.

Conclusion: excluding QKV does not recover the historical C34 result and worsens raw Top-1. Continue with broader algorithm changes rather than attempting to recreate C34 through minor policy archaeology.

### C49: full-parameter teacher-soft stage2 with optimizer reset

Design rationale: Local trainable policies consistently failed under the current code path. C49 tests whether the one-epoch stage2 bottleneck is caused by over-restricting updates by allowing all parameters to train under teacher soft KD, while still using the explicit stage2 optimizer reset.

Script: `tmp_scripts/run_stage2_c49_teacher_soft_fullparam_noresumeopt_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=0` teacher soft logits only, trainable-policy `all`, lr=min_lr=5e-6, no attention KL, no EMA, explicit optimizer reset.

Runtime: train avg step time 0.291351s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed under the 10 minute limit.

Outcome: raw checkpoint-51 reached Top-1 78.110 / Top-5 94.228. Delta vs checkpoint-50 is -0.006. This is the best result among the current strict/current-code reruns C46-C49 but still does not beat the 78.116 start and is far below the 79.0 gate.

Conclusion: full-parameter teacher-soft KD is not enough, but it is less bad than current local-policy variants. The next current-code test should combine full-parameter updates with the historical strongest stabilization mechanism: prev-step dynamic attention KL.

### C50: full-parameter hard+soft KD plus prev-step dynamic custom-top5 attention KL

Design rationale: C49 showed full-parameter teacher-soft is the strongest current-code non-attention run but still below the start checkpoint. C50 reruns the historically strongest stabilization idea under the current code path: full-parameter update, hard+soft KD, and prev-step refmodel attention KL on dynamically selected top-5 heads from the custom 10-head pool.

Script: `tmp_scripts/run_stage2_c50_full_dynamic_customtop5_current_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=1`, trainable-policy `all`, prev-step refmodel attention KL weight 1e-4, `ref-head-mode dynamic_custom_top5:custom:5:2,10:14,5:1,4:1,9:10,6:1,8:4,8:9,11:18,11:4`, lr=min_lr=5e-6, explicit stage2 optimizer reset, no EMA.

Runtime: train avg step time 0.418874s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed approximately within the 10 minute limit. RefAttnKL averaged about 8.15e1, so the added loss contribution was about 8e-3.

Outcome: raw checkpoint-51 reached Top-1 78.198 / Top-5 94.204. Delta vs checkpoint-50 is +0.082. This matches the historical best Top-1 seen from C16/C34 but still does not reach the 79.0 gate.

Conclusion: among current-code runs, the only mechanism that recovers a meaningful positive delta is prev-step dynamic attention KL with full-parameter updates. Continue by combining this stabilization with the stronger non-attention branch: full-parameter teacher soft-only KD, instead of hard+soft KD.

### C51: full-parameter teacher soft-only KD plus prev-step dynamic custom-top5 attention KL

Design rationale: C50 recovered the current best 78.198 using full-parameter hard+soft KD plus prev-step dynamic attention KL. C49 showed full-parameter teacher soft-only KD is the strongest non-attention current-code baseline. C51 combines soft-only teacher KD with the C50 dynamic attention-KL stabilization to test whether removing hard-label CE helps the refmodel-stabilized recipe.

Script: `tmp_scripts/run_stage2_c51_full_soft_dynamic_customtop5_current_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=0`, trainable-policy `all`, prev-step refmodel attention KL weight 1e-4, same dynamic custom-top5 head pool as C50, lr=min_lr=5e-6, explicit stage2 optimizer reset, no EMA.

Runtime: train avg step time 0.414122s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed approximately within the 10 minute limit.

Outcome: raw checkpoint-51 reached Top-1 78.198 / Top-5 94.204. Delta vs checkpoint-50 is +0.082. This exactly matches C50 and remains below the 79.0 gate.

Conclusion: changing hard+soft KD to soft-only KD does not improve the full dynamic attention-KL recipe. The current best remains 78.198; additional progress requires a larger algorithmic change than KD composition around this KL recipe.

### C52: C50 plus grouped LR with 4x quant/shift LR

Design rationale: C50/C51 show the full-parameter prev-step dynamic attention-KL skeleton is the only current-code branch with a positive delta, but it may under-adapt quantizer/shift parameters at a uniform lr=5e-6. C52 adds grouped LR: normal parameters keep base lr=5e-6, while quant/shift parameters use 4x lr. This is a QAT-specific optimizer design change rather than a scalar global LR sweep.

Script: `tmp_scripts/run_stage2_c52_full_dynamic_customtop5_quantlr4_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=1`, trainable-policy `all`, prev-step dynamic custom-top5 attention KL weight 1e-4, lr=min_lr=5e-6, `quant_lr_multiplier=4.0`, explicit stage2 optimizer reset, no EMA.

Runtime: train avg step time 0.418147s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed approximately within the 10 minute limit. Log confirmed grouped LR with ~28.29M base params and ~0.205M quant params.

Outcome: raw checkpoint-51 reached Top-1 78.070 / Top-5 94.144. Delta vs checkpoint-50 is -0.046. This is far below C50/C51 Top-1 78.198.

Conclusion: increasing quant/shift LR by 4x is harmful. Do not continue this high quant-LR direction unless using a much more controlled schedule. Current best remains C50/C51 at Top-1 78.198.

### C53: C51 with teacher soft-KD temperature 2.0

Design rationale: C50/C51 show that prev-step dynamic attention KL is necessary but still capped at 78.198. C53 tests whether smoothing the teacher soft target with temperature 2.0 gives a better one-epoch gradient under the same refmodel stabilization.

Script: `tmp_scripts/run_stage2_c53_full_softT2_dynamic_customtop5_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=0`, teacher soft temperature 2.0, trainable-policy `all`, prev-step dynamic custom-top5 attention KL weight 1e-4, lr=min_lr=5e-6, explicit stage2 optimizer reset, no EMA.

Runtime: train avg step time 0.414045s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed approximately within the 10 minute limit.

Outcome: raw checkpoint-51 reached Top-1 78.198 / Top-5 94.204. Delta vs checkpoint-50 is +0.082. This exactly matches C50/C51 and remains below the 79.0 gate.

Conclusion: teacher soft-KD temperature 2.0 does not improve the current best dynamic attention-KL skeleton. The identical training/validation trajectory suggests this axis is not useful enough to keep sweeping without more instrumentation.

### C54: C50 plus prev-step ref-logit KL

Design rationale: C50's refmodel already runs a forward pass for attention KL. C54 adds output-level temporal consistency from the same prev-step refmodel (`ref-logit KL`) to test a multi-level ref stabilization: attention structure plus logits.

Script: `tmp_scripts/run_stage2_c54_full_dynamic_customtop5_reflogit_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=1`, trainable-policy `all`, prev-step dynamic custom-top5 attention KL weight 1e-4, ref-logit KL weight 0.01 temperature 2.0, lr=min_lr=5e-6, explicit optimizer reset, no EMA.

Runtime: train avg step time 0.413830s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed approximately within the 10 minute limit. RefLogitKL averaged about 1.8e-2, adding about 1.8e-4 to the loss.

Outcome: raw checkpoint-51 reached Top-1 78.052 / Top-5 94.134. Delta vs checkpoint-50 is -0.064. This is much worse than C50/C51.

Conclusion: adding output-level temporal consistency conflicts with useful attention-KL stabilization. Do not continue this ref-logit branch by weight sweeping.

### C55: centered-cosine attention relation loss instead of attention KL

Design rationale: C50's probability KL may be magnitude-sensitive and dominated by high-KL heads. C55 replaces attention KL with a centered cosine relation loss on selected attention maps, aiming to preserve structural direction rather than probability magnitude. A 100-step sanity run showed raw centered-cosine loss around 0.25, so full C55 used weight 0.03 to match C50's loss contribution scale.

Script: `tmp_scripts/run_stage2_c55_full_dynamic_centeredcos_w003_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=1`, trainable-policy `all`, prev-step dynamic custom-top5 centered-cosine attention loss weight 0.03, lr=min_lr=5e-6, explicit optimizer reset, no EMA.

Runtime: train avg step time 0.419056s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed approximately within the 10 minute limit.

Outcome: raw checkpoint-51 reached Top-1 78.088 / Top-5 94.136. Delta vs checkpoint-50 is -0.028. This is below C50/C51 and below the 79.0 gate.

Conclusion: centered-cosine relation stabilization is inferior to directional probability KL for this checkpoint. The best remains C50/C51 at 78.198.

### C56: symmetric attention KL in C50 skeleton

Design rationale: C50 uses directional `kl_ref` attention loss. C56 tests symmetric KL, aiming to avoid one-sided probability matching and stabilize both directions of attention distribution drift.

Script: `tmp_scripts/run_stage2_c56_full_dynamic_symkl_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=1`, trainable-policy `all`, prev-step dynamic custom-top5 symmetric attention KL weight 1e-4, lr=min_lr=5e-6, explicit optimizer reset, no EMA.

Runtime: train avg step time 0.418490s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed approximately within the 10 minute limit.

Outcome: raw checkpoint-51 reached Top-1 77.960 / Top-5 94.168. Delta vs checkpoint-50 is -0.156. This is much worse than C50/C51.

Conclusion: symmetric KL is harmful. Directional KL to the prev-step ref distribution remains the better attention stabilization loss.

### C57: multi-step stale prev-ref, update interval 8

Design rationale: C50 updates the prev-step refmodel every optimizer step. C57 tests a multi-time-scale refmodel by updating the prev-step ref only every 8 optimizer updates. This is intended to make the ref less trivially close than immediate prev-step while avoiding the over-constraint of a fully fixed anchor.

Script: `tmp_scripts/run_stage2_c57_full_dynamic_refinterval8_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=1`, trainable-policy `all`, prev-step dynamic custom-top5 attention KL weight 1e-4, `ref_update_interval=8`, lr=min_lr=5e-6, explicit optimizer reset, no EMA.

Runtime: train avg step time 0.405734s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed approximately within the 10 minute limit. Log confirmed `ref_update_interval=8`.

Outcome: raw checkpoint-51 reached Top-1 78.060 / Top-5 94.140. Delta vs checkpoint-50 is -0.056. This is much worse than C50/C51.

Conclusion: stale multi-step prev-ref is harmful. Immediate prev-step remains the best temporal refmodel update rule among tested options.

### C58: teacher-gated dynamic custom-top5 head selection

Design rationale: C50 selects heads by largest student-vs-prev-step KL within a custom suspicious-head pool, without checking whether the selected head's movement is useful relative to the FP teacher. C58 implements `dynamic_teacher_agree_top5`: shortlist heads with high prev-step KL, then select those with lower student-vs-teacher attention disagreement. This keeps the prev-step refmodel core but gates head choice using teacher agreement.

Script: `tmp_scripts/run_stage2_c58_teacher_agree_top5_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=1`, trainable-policy `all`, prev-step attention KL weight 1e-4, `ref-head-mode dynamic_teacher_agree_top5:custom:5:2,10:14,5:1,4:1,9:10,6:1,8:4,8:9,11:18,11:4`, lr=min_lr=5e-6, explicit optimizer reset, no EMA.

Runtime: train avg step time 0.416176s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed approximately within the 10 minute limit. The teacher-gated selector reduced RefAttnKL average to about 4.66e1, roughly half of C50.

Outcome: raw checkpoint-51 reached Top-1 78.006 / Top-5 94.136. Delta vs checkpoint-50 is -0.110. This is much worse than C50/C51.

Conclusion: teacher-agreement gating of selected heads is harmful in this simple form. The useful C50 signal appears to come from stabilizing the largest prev-step changes, even if those heads are not closest to the teacher. Do not continue this head-gating branch without a fundamentally different selection criterion.

## Completion audit snapshot after C58

Objective gate: fixed checkpoint-50 -> normal single epoch 50→51 -> full ImageNet distributed raw Top-1 >=79.0.

Evidence checked:
- Fixed start checkpoint path was used in all C40-C58 scripts.
- Full validation logs report 50000 samples for completed full candidates.
- No soup / averaging / >1 epoch checkpoint was used for reported C40-C58 candidate metrics.
- Best current raw result remains C50/C51: Top-1 78.198 / Top-5 94.204, delta +0.082 vs 78.116.
- Runtime for best C50/C51 is about 0.414-0.419s/step plus ~32s validation, approximately within the 10 minute limit.
- EMA is not enabled for current best; earlier EMA branches underperformed or had invalid EMA eval.

Missing gate:
- No candidate has reached raw Top-1 >=79.0. The goal is not complete.

Current technical conclusion:
- Directional prev-step attention KL on dynamic custom-top5 heads is the only repeatedly useful stage2 stabilization mechanism.
- Tested alternatives that failed include EMA-ref, fixed anchor, ref-logit, teacher-attention KL, teacher Q/K relation, attention-output MSE, centered-cosine relation loss, symmetric KL, stale interval ref, teacher-gated head selection, quant-only / partial policies, full-param teacher-soft without KL, hard+soft KD alone, optimizer reset alone, grouped quant LR, and KD temperature.
- The remaining gap is about +0.802 Top-1, too large for small variants around C50.

### C59: half-epoch ref KL then release to KD-only

Design rationale: C50's prev-step KL helps, but applying it through the entire epoch may constrain late adaptation. C59 uses a schedule: first 624 optimizer updates use C50's ref KL, then `ref_stop_updates=624` turns ref consistency off for the second half of the epoch, leaving only KD.

Script: `tmp_scripts/run_stage2_c59_full_dynamic_stop624_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=1`, trainable-policy `all`, prev-step dynamic custom-top5 attention KL weight 1e-4 for the first 624 updates only, lr=min_lr=5e-6, explicit optimizer reset, no EMA.

Runtime: train avg step time 0.355822s/step because the second half skips refmodel forward; full validation was distributed over 50000 samples. Stage2 epoch stayed within the 10 minute limit. Logs confirmed `RefW` switched from 1e-4 to 0 after the midpoint.

Outcome: raw checkpoint-51 reached Top-1 78.010 / Top-5 94.190. Delta vs checkpoint-50 is -0.106. This is far below C50/C51.

Conclusion: releasing the ref constraint halfway through the epoch is harmful. Continuous immediate prev-step KL is necessary for the small C50 gain.

### C60: teacher-gated head selection with matched KL strength

Design rationale: C58's teacher-gated head selector reduced average RefAttnKL to about half of C50, so its failure could be due to weaker regularization rather than bad head selection. C60 doubles the ref attention weight to 2e-4 to make the effective loss contribution comparable to C50.

Script: `tmp_scripts/run_stage2_c60_teacher_agree_top5_w2e4_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=1`, trainable-policy `all`, teacher-gated dynamic custom-top5 prev-step attention KL weight 2e-4, lr=min_lr=5e-6, explicit optimizer reset, no EMA.

Runtime: train avg step time 0.416216s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed approximately within the 10 minute limit.

Outcome: raw checkpoint-51 reached Top-1 78.010 / Top-5 94.186. Delta vs checkpoint-50 is -0.106. This remains far below C50/C51.

Conclusion: teacher-gated head selection is harmful even when its loss contribution is matched to C50. The best head-selection rule remains selecting largest prev-step KL from the custom suspicious-head pool.

### C61: KD first, then enable prev-step attention KL in second half

Design rationale: C59 showed that turning KL off in the second half hurts. C61 tests the reverse schedule: let KD adapt freely in the first half, then enable C50's prev-step dynamic attention KL after 624 updates to stabilize the final checkpoint.

Script: `tmp_scripts/run_stage2_c61_full_dynamic_warm624_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=1`, trainable-policy `all`, prev-step dynamic custom-top5 attention KL weight 1e-4 enabled only after `ref_warmup_updates=624`, lr=min_lr=5e-6, explicit optimizer reset, no EMA.

Runtime: train avg step time 0.356746s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed within the 10 minute limit. Logs confirmed RefAttnKL was 0 in the first half and enabled in the second half.

Outcome: raw checkpoint-51 reached Top-1 78.104 / Top-5 94.172. Delta vs checkpoint-50 is -0.012. This is below C50/C51.

Conclusion: delaying KL until the second half loses the C50 gain. Continuous immediate prev-step KL remains better than front-half or back-half-only schedules.

### C62: full-param fixed oscillating-top5 prev-step attention KL

Design rationale: C50's best recipe uses dynamic top5 selection within the custom 10-head pool. C62 tests whether the gain comes simply from the user's fixed abnormal oscillating top5 heads, by using `ref-head-mode oscillating_top5` under the same full-parameter prev-step KL skeleton.

Script: `tmp_scripts/run_stage2_c62_full_fixed_oscillating_top5_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=1`, trainable-policy `all`, fixed oscillating-top5 prev-step attention KL weight 1e-4, lr=min_lr=5e-6, explicit optimizer reset, no EMA.

Runtime: train avg step time 0.411635s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed approximately within the 10 minute limit. RefAttnKL averaged about 5.0e1, lower than C50's dynamic custom-top5 average.

Outcome: raw checkpoint-51 reached Top-1 78.152 / Top-5 94.212. Delta vs checkpoint-50 is +0.036. This is below C50/C51 Top-1 78.198 and below the 79.0 gate.

Conclusion: fixed oscillating-top5 is useful but weaker than dynamic custom-top5. Dynamic selection within the custom pool contributes additional benefit beyond the fixed abnormal heads.

### C63: full-param dynamic custom-top7 prev-step attention KL

Design rationale: C50's dynamic custom-top5 is best; C14 showed top3 is too weak, while fixed custom10 and teacher-gated variants are worse. C63 tests whether expanding the dynamic selection from top5 to top7 within the same custom suspicious-head pool captures more useful instability without over-constraining as much as custom10.

Script: `tmp_scripts/run_stage2_c63_full_dynamic_customtop7_20260701.sh`.

Recipe: checkpoint-50, one epoch 50→51, `kd_hard_and_soft=1`, trainable-policy `all`, prev-step dynamic custom-top7 attention KL weight 1e-4, lr=min_lr=5e-6, explicit optimizer reset, no EMA.

Runtime: train avg step time 0.415496s/step; full validation was distributed over 50000 samples. Stage2 epoch stayed approximately within the 10 minute limit. RefAttnKL averaged about 7.6e1, close to but slightly below C50.

Outcome: raw checkpoint-51 reached Top-1 78.062 / Top-5 94.144. Delta vs checkpoint-50 is -0.054. This is much worse than C50/C51.

Conclusion: widening dynamic selection to top7 over-regularizes or selects harmful heads. Dynamic custom-top5 remains the best head-selection width.

## Final audit snapshot after C63

Concrete success criteria:
1. Start exactly from checkpoint-50 at `/mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_continue10_to50_sysdisk_20260630/checkpoint-50.pth.tar`.
2. Train a normal single epoch 50→51 stage2 run.
3. Validate on full ImageNet distributed validation, raw checkpoint is the primary gate.
4. Achieve raw Top-1 >=79.0.
5. Keep stage2 based on a refmodel/stabilization idea.
6. Keep each stage2 epoch under 10 minutes.
7. Record scripts, recipes, raw metrics, deltas, and gate status.

Evidence:
- All C40-C63 full candidates use the fixed checkpoint-50 path.
- Reported full candidates log distributed validation over 50000 samples.
- No reported full candidate uses checkpoint soup, averaging, or >1 epoch weights.
- Progress documentation is maintained in this file with scripts, recipes, outcomes, deltas, and conclusions.
- Best raw result remains C50/C51: Top-1 78.198 / Top-5 94.204, delta +0.082 vs checkpoint-50.
- Best runtime remains within the approximate 10 minute budget: train avg about 0.414-0.419s/step plus ~32s validation.

Gate status:
- Not achieved. Best raw Top-1 is 78.198, below required 79.0 by about 0.802.
- Do not extend any recipe to 100 epochs under the stated rules.

Best current recipe:
- C50/C51: full-parameter training, KD, immediate prev-step refmodel, directional KL on dynamic custom-top5 heads from the custom suspicious-head pool, lr=min_lr=5e-6, explicit stage2 optimizer reset, no EMA.

Rejected mechanisms include:
- EMA-ref, fixed anchor-ref, ref-logit KL, teacher-attention KL, teacher Q/K relation, attention-output MSE, centered-cosine relation loss, symmetric KL, stale interval ref, teacher-gated head selection, quant-only/partial policies, full-param teacher-soft without attention KL, hard+soft KD alone, optimizer reset alone, grouped quant LR, KD temperature, half-epoch KL schedules, fixed top5, top3, top7.

Practical next direction:
- Under the fixed checkpoint-50 and one-epoch constraint, the current evidence suggests 79.0 is not reachable by local ref/KL variations around this stage2 formulation.
- A materially different route is required: either raise the stage1 starting checkpoint before stage2, or redesign QAT around explicit quantization-error correction rather than attention probability regularization.

### C65: C50 prev-step dynamic custom-top5 + FP teacher block-output MSE

Design rationale: C40/C41 attention-output MSE and C50 attention KL variants did not bridge the 79.0 gate. C65 tests explicit quantization-error correction by adding FP-teacher intermediate block-output MSE on the last block of each Swin stage, while retaining C50's prev-step refmodel attention stabilization. This keeps the stage2 refmodel idea but moves part of the stabilization target from attention probabilities to actual quantized representations.

Implementation notes: Initial `features.1,features.3,features.5,features.7` hooks did not fire because OFQ Swin manually iterates `self.features[1:]` and calls sub-blocks directly instead of calling each stage `Sequential`. The working hook points are `features.1.1,features.3.1,features.5.5,features.7.1`. Sanity confirmed nonzero feature loss: pair MSE about 9.19e-2, 2.05e-1, 1.93e1, 3.65e0; average TeacherFeatOut about 5.73.

Script: `tmp_scripts/run_stage2_c65_full_blockout_w1e3_20260701.sh`.

Recipe: fixed checkpoint-50, one normal epoch 50→51, C50 skeleton (`kd_hard_and_soft=1`, trainable-policy all, prev-step dynamic custom-top5 attention KL weight 1e-4, lr=min_lr=5e-6, explicit optimizer reset, no EMA) plus `teacher_feature_output_weight=1e-3` on `features.1.1,features.3.1,features.5.5,features.7.1`.

Runtime: train avg step time 0.416540s/step; full distributed validation over 50000 samples took 31.879s. Stage2 epoch stayed within the 10 minute target.

Outcome: raw checkpoint-51 reached Top-1 78.064 / Top-5 94.156. Delta vs checkpoint-50 Top-1 78.116 is -0.052. Gate not achieved.

Conclusion: direct unnormalized block-output MSE is harmful despite being a real, nonzero loss. The deep-stage feature scale dominates (`features.5.5` MSE around 19.3), so the next candidate should use scale-normalized feature consistency rather than raw MSE.

### C66: C50 prev-step dynamic custom-top5 + energy-normalized FP teacher block-output loss

Design rationale: C65 proved raw block-output MSE is a real loss but harmful because deep-stage feature scale dominates. C66 changes the feature consistency objective to energy-normalized MSE, i.e. per-layer MSE divided by teacher feature energy, so each stage contributes on a comparable relative-error scale. This is a QAT stabilization redesign rather than just a weight tweak.

Script: `tmp_scripts/run_stage2_c66_full_blockout_norm_w2e2_20260701.sh`.

Recipe: fixed checkpoint-50, one normal epoch 50→51, C50 skeleton plus `teacher_feature_output_loss=norm_mse`, `teacher_feature_output_weight=0.02` on `features.1.1,features.3.1,features.5.5,features.7.1`. The weight was chosen to make the normalized feature loss contribution about equal to C65 raw MSE contribution: sanity showed TeacherFeatOut about 0.284, so 0.02 gives about 0.0057 loss contribution.

Runtime: train avg step time 0.417586s/step; full distributed validation over 50000 samples took 31.244s. Stage2 epoch stayed within the 10 minute target.

Outcome: raw checkpoint-51 reached Top-1 78.176 / Top-5 94.174. Delta vs checkpoint-50 Top-1 78.116 is +0.060. This is better than C65 but still below C50/C51 Top-1 78.198 and below the 79.0 gate.

Conclusion: scale-normalized teacher feature ref is useful and avoids C65's degradation, but it still does not solve the one-epoch 79.0 requirement. Next test should isolate interaction with attention KL by running teacher normalized block-output ref without prev-step attention KL.

### C67: normalized FP teacher block-output ref without prev-step attention KL

Design rationale: C66 improved over C65 but still did not beat C50. C67 isolates whether C50's prev-step attention KL conflicts with the normalized teacher feature ref by disabling attention KL and keeping KD + normalized block-output teacher ref.

Script: `tmp_scripts/run_stage2_c67_full_blockout_norm_noattnkl_20260701.sh`.

Recipe: fixed checkpoint-50, one normal epoch 50→51, KD hard+soft, trainable-policy all, `teacher_feature_output_loss=norm_mse`, `teacher_feature_output_weight=0.02` on `features.1.1,features.3.1,features.5.5,features.7.1`, `ref_attn_kl_weight=0`, lr=min_lr=5e-6, explicit optimizer reset, no EMA.

Runtime: train avg step time 0.300230s/step because attention KL is disabled; full distributed validation over 50000 samples took 31.784s. Runtime is well under the 10 minute target.

Outcome: raw checkpoint-51 reached Top-1 78.108 / Top-5 94.084. Delta vs checkpoint-50 Top-1 78.116 is -0.008. Gate not achieved.

Conclusion: teacher feature ref alone is insufficient and does not beat the start checkpoint. C50's prev-step attention KL remains the main positive component; feature ref is at best a weak auxiliary. The next useful axis is optimizer-state continuity rather than more local teacher-feature weighting.

### C68: C50 prev-step dynamic custom-top5 with optimizer-momentum continuity

Design rationale: C50/C51 use `--no-resume-opt`, which intentionally resets optimizer/scheduler/scaler/RNG for stage2. C68 tests whether preserving optimizer momentum from checkpoint-50 helps one-epoch stage2 stability. The checkpoint contains optimizer state but not lr scheduler, loss scaler, or RNG, so this is explicitly an optimizer-continuity test, not a full strict-resume test. The optimizer checkpoint lr is 1e-5, so the run restores optimizer state and then forces param-group lr back to the C50 recipe lr=5e-6 to isolate momentum continuity from LR changes.

Script: `tmp_scripts/run_stage2_c68_full_dynamic_customtop5_resumeopt_forcelr_20260701.sh`.

Recipe: fixed checkpoint-50, one normal epoch 50→51, C50 skeleton (`kd_hard_and_soft=1`, trainable-policy all, prev-step dynamic custom-top5 attention KL weight 1e-4, lr=min_lr=5e-6, no EMA), but without `--no-resume-opt`; optimizer state restored from checkpoint-50 and `--resume-opt-force-lr` forces lr back to 5e-6. Logs confirmed optimizer state entries=433, scheduler missing, RNG missing, forced restored optimizer lr=5e-6.

Runtime: train avg step time 0.414442s/step; full distributed validation over 50000 samples took 31.948s. Stage2 epoch stayed within the 10 minute target.

Outcome: raw checkpoint-51 reached Top-1 78.096 / Top-5 94.134. Delta vs checkpoint-50 Top-1 78.116 is -0.020. Gate not achieved.

Conclusion: optimizer momentum continuity from the stage1 checkpoint does not help and is worse than C50/C51. Resetting optimizer for stage2 remains better under current evidence. Next useful axis is quantizer/alpha update stability while keeping C50's prev-step attention KL.

### C69: C50 prev-step dynamic custom-top5 with quant/shift frozen (`non_quant` policy)

Design rationale: C65-C67 showed teacher feature ref is weak, and C68 showed optimizer momentum continuity is not helpful. C69 tests whether stage2 degradation comes from quantizer/alpha/shift parameter drift. It freezes all quant/shift parameters while keeping normal weights trainable, and retains C50's prev-step attention KL.

Implementation notes: Added `trainable-policy non_quant`, where all parameters matching quant/shift tokens (`input_quant_fn`, `lsqw_fn`, `statsq_fn`, `qk_quant`, `v_quant`, `quan_a_`, `move_`) are frozen and all other parameters remain trainable. Logs confirmed trainable=28,289,698 and frozen=204,672.

Script: `tmp_scripts/run_stage2_c69_full_dynamic_customtop5_nonquant_20260701.sh`.

Recipe: fixed checkpoint-50, one normal epoch 50→51, C50 skeleton (`kd_hard_and_soft=1`, prev-step dynamic custom-top5 attention KL weight 1e-4, lr=min_lr=5e-6, explicit optimizer reset, no EMA), plus `trainable_policy=non_quant`.

Runtime: train avg step time 0.349462s/step; full distributed validation over 50000 samples. Stage2 epoch stayed well within the 10 minute target.

Outcome: raw checkpoint-51 reached Top-1 78.082 / Top-5 94.162. Delta vs checkpoint-50 Top-1 78.116 is -0.034. Gate not achieved.

Conclusion: freezing all quant/shift parameters is harmful, so the issue is not simply quantizer drift. Stage2 still needs some quantizer adaptation. The next useful test is finer-grained quant stabilization: freeze activation quantizer and shift/move bias while leaving weight quantizer and ordinary weights trainable.

### C70: C50 prev-step dynamic custom-top5 with activation-quant/shift frozen (`freeze_act_quant` policy)

Design rationale: C69 froze all quant/shift parameters and was harmful. C70 tests a finer-grained quantizer stabilization policy: freeze activation quantizers and move/shift bias (`input_quant_fn`, `quan_a_`, `move_`) while leaving weight quantizers and ordinary weights trainable. The goal is to reduce activation-scale drift without fully disabling quantizer adaptation.

Implementation notes: Added `trainable-policy freeze_act_quant`. Logs confirmed trainable=28,290,845 and frozen=203,525, i.e. slightly fewer frozen parameters than C69 because weight quantizer parameters remain trainable.

Script: `tmp_scripts/run_stage2_c70_full_dynamic_customtop5_freezeactquant_20260701.sh`.

Recipe: fixed checkpoint-50, one normal epoch 50→51, C50 skeleton (`kd_hard_and_soft=1`, prev-step dynamic custom-top5 attention KL weight 1e-4, lr=min_lr=5e-6, explicit optimizer reset, no EMA), plus `trainable_policy=freeze_act_quant`.

Runtime: train avg step time 0.351096s/step; full distributed validation over 50000 samples took 31.767s. Stage2 epoch stayed well within the 10 minute target.

Outcome: raw checkpoint-51 reached Top-1 77.984 / Top-5 94.146. Delta vs checkpoint-50 Top-1 78.116 is -0.132. Gate not achieved.

Conclusion: freezing activation quantizers and shift/move parameters is worse than freezing all quant/shift parameters and much worse than C50. Quantizer adaptation is needed; this path is rejected. Next direction is not more quant freezing, but a dual-time-scale reference model: immediate prev-step ref for high-frequency attention stability plus a weak fixed checkpoint-50 anchor on early/mid layers to reduce slow drift without blocking late-layer adaptation.

### C71: dual-time-scale ref, prev-step dynamic custom-top5 plus weak early/mid fixed anchor

Design rationale: Previous attempts showed C50's prev-step attention KL is the main positive component, but it may only suppress high-frequency step-to-step attention oscillation. C71 adds a second, slower fixed checkpoint-50 anchor ref on a small early/mid head subset to reduce low-frequency drift while avoiding late-layer over-constraint.

Implementation notes: Added `--anchor-ref-head-mode` and `custom_subset:` head mode so anchor ref can use a different head subset from the prev-step ref. Initial run failed because `custom:` required all oscillating top5 heads; `custom_subset:` fixes this for anchor-only subsets. C71 used prev-step `dynamic_custom_top5` over the full custom suspicious pool and anchor `custom_subset:5:2,5:1,4:1,6:1,8:4`. Logs confirmed anchor head map and nonzero AnchorRefAttnKL around 5.3e1.

Script: `tmp_scripts/run_stage2_c71_full_dualref_anchor_earlymid_20260702.sh`.

Recipe: fixed checkpoint-50, one normal epoch 50→51, C50 skeleton plus `anchor_ref_attn_kl_weight=2e-5`, `anchor_ref_warmup_epochs=50`, and `anchor_ref_head_mode=custom_subset:5:2,5:1,4:1,6:1,8:4`.

Runtime: train avg step time 0.518437s/step; full distributed validation over 50000 samples took 31.627s. Runtime exceeds the 10 minute stage2 budget.

Outcome: raw checkpoint-51 reached Top-1 78.106 / Top-5 94.182. Delta vs checkpoint-50 Top-1 78.116 is -0.010. Gate not achieved.

Conclusion: this dual-ref anchor implementation is both slower and less accurate than C50. The anchor-ref direction is rejected unless a future design can compute anchor much more sparsely and show a clear accuracy gain. Next direction: test whether stage2 should skip setup-alpha calibration entirely and trust checkpoint-50's quantizer state.

### C72: skip pre-resume setup-alpha and trust checkpoint quantizer state only -- invalid startup

Design rationale: Most stage2 runs execute `setup_alpha` before loading the checkpoint. C72 tested whether stage2 can skip this initialization and rely entirely on checkpoint-50 quantizer/alpha state.

Script: `tmp_scripts/run_stage2_c72_full_dynamic_customtop5_nosetupalpha_20260702.sh`.

Recipe: C50 skeleton plus `setup_alpha_batches=0`.

Outcome: invalid candidate, stopped early before full epoch. Logs showed `setup alpha skipped`, then checkpoint resume reported `unexpected=103`, model param count changed from the normal 28,494,370 to 28,485,333, and the first-batch loss jumped from the normal about 2.13 to about 4.00. This indicates `setup_alpha` is required to instantiate or align quantization state before loading the checkpoint; skipping it makes checkpoint restore incompatible.

Conclusion: C72 is not a valid 50→51 stage2 candidate and was intentionally interrupted to avoid wasting compute. Pre-resume setup-alpha must remain enabled. A valid follow-up is post-resume setup-alpha: load checkpoint correctly, then run a small calibration pass after resume.

### C73: C50 prev-step dynamic custom-top5 with post-resume setup-alpha calibration

Design rationale: C72 showed pre-resume setup-alpha is required for checkpoint compatibility. C73 keeps the required pre-resume setup-alpha and adds a post-resume calibration pass to test whether recalibrating quantizer state after loading checkpoint-50 improves stage2 adaptation.

Script: `tmp_scripts/run_stage2_c73_full_dynamic_customtop5_postresume_alpha1_20260702.sh`.

Recipe: fixed checkpoint-50, one normal epoch 50→51, C50 skeleton plus `post_resume_setup_alpha_batches=1`.

Runtime: train avg step time 0.416061s/step; full distributed validation over 50000 samples. Stage2 epoch stayed within the 10 minute target.

Outcome: raw checkpoint-51 reached Top-1 78.042 / Top-5 94.166. Delta vs checkpoint-50 Top-1 78.116 is -0.074. Gate not achieved.

Conclusion: post-resume setup-alpha calibration hurts accuracy. The quantizer state loaded from checkpoint-50 should not be recalibrated after resume. Next direction: stabilize the online dynamic head selection itself using EMA-smoothed head scores instead of instantaneous top-k selection.

## Stage1 from-scratch 100-epoch strict checkpoint run

User redirected current work from further Stage2 candidates to a stage1-style from-scratch run to 100 epochs, with the explicit requirement that checkpoint-100 must support strict resume from epoch 100.

Script: `tmp_scripts/run_stage1_fromscratch_to100_strictckpt_20260702.sh`.

Recipe: Swin-T W4A4 OFQ QAT from scratch, KD hard+soft, no augmentation, batch size 256 per GPU on 8 GPUs (global batch 2048), lr=2e-4, min_lr=1e-5, scheduler_epochs=100, weight_decay=0, bf16 AMP, setup-alpha default 1, full validation each epoch. Output is on system disk, not `/tmp`:
`/mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage1_fromscratch_bs256_kd_noaug_strictckpt_100ep_20260702`.

Checkpoint policy: epoch checkpoint interval 10, checkpoint_hist 12. `save_epoch_checkpoint()` currently saves model state, optimizer state, lr_scheduler state, rng_state, args, arch, epoch, and EMA state if enabled. This run does not enable model EMA. Helper scripts prepared for after checkpoint-100 exists:
- `tmp_scripts/verify_stage1_ckpt100_strict_state_20260702.sh`
- `tmp_scripts/run_stage1_resume100_smoke_1update_20260702.sh`

Initial evidence: run launched successfully on worker 975345. Epoch 0 train avg step time about 0.550s over 624 updates. Epoch 0 full distributed validation reached Top-1 71.890 / Top-5 90.832 over 50000 samples. Training continued into epoch 1 normally.

## Stage1 from-scratch 100-epoch strict checkpoint completion

Date: 2026-07-02.

Goal context: Stage2 experiments were paused. The active goal was to complete the stage1-style from-scratch 100 epoch Swin-T W4A4 OFQ QAT run and verify that `checkpoint-100.pth.tar` can be used for strict resume from epoch 100.

Script: `tmp_scripts/run_stage1_fromscratch_to100_strictckpt_20260702.sh`.

Recipe: Swin-T W4A4 OFQ QAT from scratch, KD hard+soft, no augmentation, batch size 256 per GPU on 8 GPUs (global batch 2048), lr=2e-4, min_lr=1e-5, scheduler_epochs=100, weight_decay=0, bf16 AMP, full validation every epoch, checkpoint every 10 epochs, checkpoint_hist=12. No Stage2 KL was enabled.

Output directory:
`/mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage1_fromscratch_bs256_kd_noaug_strictckpt_100ep_20260702`.

Checkpoint save evidence: checkpoints were saved on system disk at every requested 10-epoch interval: `checkpoint-10.pth.tar`, `checkpoint-20.pth.tar`, `checkpoint-30.pth.tar`, `checkpoint-40.pth.tar`, `checkpoint-50.pth.tar`, `checkpoint-60.pth.tar`, `checkpoint-70.pth.tar`, `checkpoint-80.pth.tar`, `checkpoint-90.pth.tar`, `checkpoint-100.pth.tar`, plus `last.pth.tar` hard-linked to the latest checkpoint.

Final full ImageNet validation at epoch 99 / checkpoint-100:
- raw Top-1: 78.4760
- raw Top-5: 94.3420
- loss: 0.9084
- samples: 50000
- rank samples: `[6250, 6250, 6250, 6250, 6250, 6250, 6250, 6250]`

Late-stage validation trajectory:

| epoch | raw Top-1 | raw Top-5 |
|---:|---:|---:|
| 89 | 78.4100 | 94.2880 |
| 90 | 78.3820 | 94.2500 |
| 91 | 78.3500 | 94.3500 |
| 92 | 78.3780 | 94.3180 |
| 93 | 78.4060 | 94.3040 |
| 94 | 78.4040 | 94.3340 |
| 95 | 78.3880 | 94.4020 |
| 96 | 78.4340 | 94.3800 |
| 97 | 78.4380 | 94.3720 |
| 98 | 78.4020 | 94.3180 |
| 99 | 78.4760 | 94.3420 |

Checkpoint-100 strict-state inspection:
- path: `/mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage1_fromscratch_bs256_kd_noaug_strictckpt_100ep_20260702/checkpoint-100.pth.tar`
- file size: 342760506 bytes
- top-level keys: `arch`, `args`, `epoch`, `lr_scheduler`, `optimizer`, `rng_state`, `state_dict`, `version`
- missing required strict-resume keys: none
- epoch: 100
- arch: `swin_t`
- version: 2
- state_dict entries: 497
- optimizer keys: `param_groups`, `state`
- optimizer state entries: 433
- optimizer param groups: 1
- lr_scheduler state: `{'base_lr': 0.0002, 'min_lr': 1e-05, 'warmup_updates': 0, 'total_updates': 62500, 'last_lr': [1e-05]}`
- rng_state keys: `cuda`, `python`, `torch`
- model EMA state: not present because this stage1 run did not enable EMA.

Strict-resume smoke test:
- Script: `tmp_scripts/run_stage1_resume100_smoke_1update_20260702.sh`.
- Source checkpoint: checkpoint-100 from the completed stage1 run.
- Smoke output directory: `/mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage1_resume100_strict_smoke_1update_20260702`.
- Smoke behavior: strict-resumed at epoch 100, restored model, optimizer, LR scheduler, and RNG state, ran exactly one optimizer update with `--max_train_updates 1 --skip_validate`, saved smoke `checkpoint-101.pth.tar` in the separate smoke output directory, and exited with code 0.
- Key smoke log evidence:
  - `Strict resume: loaded model ... checkpoint-100.pth.tar; missing=0, unexpected=0`
  - `Strict resume: restored optimizer state entries=433`
  - `Strict resume: restored lr scheduler state={'base_lr': 0.0002, 'min_lr': 1e-05, 'warmup_updates': 0, 'total_updates': 62500, 'last_lr': [1e-05]}`
  - `Strict resume: restored RNG state=True`
  - `TrainSummary: epoch=100 updates=1 avg_step_time=2.593673s samples_per_step=2048 samples_per_sec=789.61`
  - `Stopped early after 1 optimizer updates in epoch 100.`

Conclusion: the stage1 0->100 epoch run is complete. `checkpoint-100.pth.tar` is saved on system disk, reached raw full ImageNet Top-1 78.4760 / Top-5 94.3420, contains the required strict-resume training state, and passed the one-update strict-resume smoke test. It is valid as a future Stage2 starting point.
