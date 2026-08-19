# Recipe1 progress log

Goal: Swin-T W4A4 fake-quant/QAT Recipe1 reaches raw full ImageNet Top-1 >=80.0 within 5 epochs and <=30 minutes. Candidate checkpoints default to `/tmp`; only best artifacts may be copied to system disk.

## Baseline from existing logs

Existing stage1 recipe reached Top-1 73.994 after completed epochs 0-4, Top-1 74.156 at epoch 5 validation, Top-1 78.476 at checkpoint-100, and about 78.65 after continuing to ~150 epochs. This establishes that current stage1 is stable but insufficiently fast for 5-epoch 80%.

## Storage policy

Candidate output root: `/tmp/qat_recipe1_runs`. System disk retains only docs/scripts/best summaries and, if achieved, best checkpoint.

## Literature

Detailed survey is in `docs/recipe1_vit_qat_literature_20260702.md`.

## Candidate status

Pending.

## B0 baseline alignment attempt

Script: `tmp_scripts/run_recipe1_b0_baseline_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_b0_baseline_stage1_kd_noaug_5ep_20260702`.

The fresh B0 run used the current stage1 baseline recipe and wrote only to `/tmp`. It completed:

| epoch | raw Top-1 | raw Top-5 |
|---:|---:|---:|
| 0 | 72.0480 | 90.8520 |
| 1 | 73.6920 | 91.8160 |

The run later stopped producing new log lines while GPU utilization remained high, so it was terminated to avoid wasting compute. For the baseline 5-epoch target, the existing completed 0->100 stage1 run remains the authoritative baseline evidence: epoch 0=71.890, epoch 1=72.686, epoch 2=73.322, epoch 3=73.690, epoch 4=73.994, epoch 5=74.156. The fresh B0 partial run is consistent with that trend but was not used as a completed 5-epoch baseline.

## Recipe1-A: setup-alpha16 + head_norm_quant

Script: `tmp_scripts/run_recipe1_a_headnormquant_setupalpha16_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_a_headnormquant_setupalpha16_5ep_20260702`.
Literature motivation: GPLQ / RepQ-ViT / ADFQ-ViT motivate stronger activation calibration and focusing early optimization on quantizer/norm parameters.

Outcome: rejected before full epoch. The run reached `setup alpha batches=16` but did not enter normal training promptly; GPU utilization was unhealthy and only partial memory was occupied. It was terminated to avoid wasting compute. No accuracy result. Next candidates avoid large setup-alpha and use stable full-parameter startup with additional teacher structural loss.

## Recipe1-C: teacher feature-output normalized MSE

Script: `tmp_scripts/run_recipe1_c_featureout_norm_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_c_featureout_norm_w02_5ep_20260702`.
Literature motivation: Q-ViT and APHQ-ViT motivate structural/feature reconstruction for quantized ViTs.

Recipe: full-parameter W4A4 QAT, KD hard+soft, no augmentation, teacher feature-output `norm_mse` on `features.1.1,features.3.1,features.5.5,features.7.1`, weight 0.02.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 71.9680 | 90.8180 | slightly below B0 epoch0 72.0480 |
| 1 | 73.6620 | 91.7220 | slightly below B0 epoch1 73.6920 |

Decision: stopped early after epoch1 because it showed no early gain and added feature-hook overhead. This candidate is not on track for 5-epoch 80%.

## Recipe1-B: full-param grouped quant LR multiplier 4

Script: `tmp_scripts/run_recipe1_b_quantlr4_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_b_full_quantlr4_5ep_20260702`.
Literature motivation: Q-ViT/GPLQ/RepQ-ViT emphasize rapid quantizer/scale adaptation. Recipe1-B keeps the stable baseline path and gives quant/shift parameters 4x LR while all parameters remain trainable.

Initial result:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.6880 | 91.3820 | better than B0 fresh epoch0 72.0480 and existing baseline epoch0 71.8900 |

Decision: continue to 5 epochs because epoch0 shows a clear early gain.

Recipe1-B continued result:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 1 | 74.3160 | 92.1480 | better than B0 fresh epoch1 73.6920 and already above old baseline epoch5 74.1560 |

Decision: continue Recipe1-B to 5 epochs. This is currently the strongest Recipe1 direction.

Recipe1-B final 5-epoch result:

| epoch | raw Top-1 | raw Top-5 | delta vs existing baseline Top-1 |
|---:|---:|---:|---:|
| 0 | 72.6880 | 91.3820 | +0.7980 |
| 1 | 74.3160 | 92.1480 | +1.6300 |
| 2 | 75.4120 | 92.6360 | +2.0900 |
| 3 | 76.3460 | 92.9960 | +2.6560 |
| 4 | 76.6840 | 93.1860 | +2.6900 |

Wall time: 1919 seconds, slightly above the 30 minute target (1800s) but close. Checkpoint stayed under `/tmp/qat_recipe1_runs/recipe1_b_full_quantlr4_5ep_20260702` and was not copied to system disk because the 80% gate was not reached.

Conclusion: grouped quant LR multiplier 4 is the strongest direction so far and gives a large early convergence gain, but still misses the 80% gate by 3.316 Top-1 at 5 epochs. Next tests should continue around quantizer fast adaptation, especially multiplier 8 and multiplier schedules.

## Recipe1-B8: full-param grouped quant LR multiplier 8

Script: `tmp_scripts/run_recipe1_b_quantlr8_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_b_full_quantlr8_5ep_20260702`.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.6620 | 91.1620 | slightly below Recipe1-B x4 epoch0 72.6880 / 91.3820 |

Decision: stopped after epoch0 because x8 did not improve over x4 and had lower Top-5. Next test x2 to see whether a gentler quantizer LR gives better later-epoch stability.

## Recipe1-B2: full-param grouped quant LR multiplier 2

Script: `tmp_scripts/run_recipe1_b_quantlr2_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_b_full_quantlr2_5ep_20260702`.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.2920 | 91.1540 | below x4 epoch0 72.6880 and below x8 epoch0 72.6620 |

Decision: stopped after epoch0. x2 is too weak. x4 remains the best quant-LR multiplier. Next test keeps quant multiplier 4 and increases the base learning rate to accelerate all parameters.

## Recipe1-B x2 and x4-lr4e4 status update

Recipe1-B2 (`quant_lr_multiplier=2`) observed epoch0 Top-1 72.2920 / Top-5 91.1540, below x4 and x8, so x2 was rejected.

A first launch of `recipe1_b_full_quantlr4_lr4e4_5ep_20260702` was accidentally terminated before epoch0 validation while cleaning up underperforming runs. It had only reached early epoch0 training and produced no valid candidate metric. It must be rerun before judging the base-lr 4e-4 direction.

## Recipe1-B lr4e4: full-param grouped quant LR multiplier 4 + base LR 4e-4

Script: `tmp_scripts/run_recipe1_b_quantlr4_lr4e4_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_b_full_quantlr4_lr4e4_5ep_20260702`.
Literature motivation: fast quantizer adaptation plus faster full-model tracking; tests whether the x4 quantizer LR recipe is limited by the base parameter LR.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 70.8080 | 90.3320 | far below x4 epoch0 72.6880 / 91.3820 and below B0 |

Decision: stopped after epoch0. Increasing base LR to 4e-4 destabilizes early W4A4 adaptation; this direction is rejected. No checkpoint was copied to system disk.

## Recipe1-F: pure soft KD + grouped quant LR multiplier 4

Script: `tmp_scripts/run_recipe1_f_puresoft_quantlr4_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_f_puresoft_quantlr4_5ep_20260702`.
Literature motivation: teacher-student fast convergence; remove hard-label CE pressure during early quantized adaptation and fit the FP teacher distribution directly.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.6880 | 91.3820 | identical to Recipe1-B x4 epoch0 |
| 1 | 74.3160 | 92.1480 | identical to Recipe1-B x4 epoch1 |

Decision: stopped after epoch1. In the current OFQ train.py path, pure soft KD does not change the observed validation trajectory versus hard+soft KD for this no-aug recipe. Future candidates should change the actual quantization/training schedule rather than only flipping this KD mode.

## Recipe1-G: progressive fake quant W8A8 epoch0 -> W4A4 epoch1+

Script: `tmp_scripts/run_recipe1_g_prog8to4_quantlr4_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_g_prog8to4_quantlr4_5ep_20260702`.
Infra change: added `--progressive-bit-schedule` support to `qat_launch.py` runtime path so fake-quant modules can switch bit-width by epoch. The effective markers were observed in log:

- `Applied progressive fake-quant bits: epoch=0 wbits=8 abits=8 weight_modules=118 act_modules=53`
- `Applied progressive fake-quant bits: epoch=1 wbits=4 abits=4 weight_modules=118 act_modules=53`

Literature motivation: progressive quantization / PTQ-to-QAT warm start. Use a higher-bit fake-quant stage to reduce initial quantization noise and improve early teacher matching, then switch to the target W4A4 regime.

Observed metrics:

| epoch | bit setting | raw Top-1 | raw Top-5 | note |
|---:|---|---:|---:|---|
| 0 | W8A8 | 74.7720 | 92.3280 | +2.084 Top-1 over x4 epoch0 72.6880 |
| 1 | W4A4 | 73.3660 | 91.4620 | collapse after direct 8->4 switch; below x4 epoch1 74.3160 |

Decision: stopped after epoch1. High-bit warmup is effective, but direct W8A8->W4A4 switching causes a quantizer/scale shock. Next candidate should use a smoother bit schedule such as W8A8->W6A6->W4A4, or recalibrate alpha at the switch point.

## Recipe1-G2: progressive fake quant W8A8 epoch0 -> W6A6 epoch1 -> W4A4 epoch2+

Script: `tmp_scripts/run_recipe1_g2_prog8to6to4_quantlr4_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_g2_prog8to6to4_quantlr4_5ep_20260702`.
Literature motivation: smoother progressive quantization to reduce the W8A8->W4A4 shock observed in Recipe1-G.

Observed metrics:

| epoch | bit setting | raw Top-1 | raw Top-5 | note |
|---:|---|---:|---:|---|
| 0 | W8A8 | 74.7720 | 92.3280 | same high-bit warmup gain as G |
| 1 | W6A6 | 75.6860 | 92.5940 | better than x4 epoch1 74.3160; transition shock reduced |
| 2 | W4A4 | 74.0640 | 91.7440 | drops after target-bit switch; below x4 epoch2 75.4120 |

Decision: stopped after epoch2. Smooth bit transition helps at W6A6 but still drops when entering target W4A4. The likely issue is LSQ scale mismatch: changing `bit/thd_pos` without rescaling learned `s` violates the LSQ initialization relation `s ~ 1/sqrt(thd_pos)`. Next candidate should rescale LSQ `s` on bit switch.

## Recipe1-G3: progressive fake quant W8A8 -> W6A6 -> W4A4 with LSQ scale rescale

Script: `tmp_scripts/run_recipe1_g3_prog8to6to4_rescale_quantlr4_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_g3_prog8to6to4_rescale_quantlr4_5ep_20260702`.
Infra change: added `--progressive-bit-rescale-lsq`. On LSQ bit switch, learned scale `s` is rescaled by `sqrt(old_thd_pos / new_thd_pos)` to preserve the LSQ initialization relation.

Observed metrics:

| epoch | bit setting | raw Top-1 | raw Top-5 | note |
|---:|---|---:|---:|---|
| 0 | W8A8 | 74.9680 | 92.5120 | strong high-bit warmup |
| 1 | W6A6 | 75.8360 | 92.8500 | best progressive intermediate result so far |
| 2 | W4A4 | 74.3240 | 91.9840 | still drops after target W4A4 switch; below x4 epoch2 75.4120 |

Decision: stopped after epoch2. LSQ rescale strongly reduced training-loss shock at bit switch, but full validation still drops when entering target W4A4. Next candidates should keep weight bit fixed at W4 and only progressively tighten activation quantization, or do explicit W4A4 alpha recalibration at the switch.

## Recipe1-G4: fixed W4 with progressive activation A8 -> A6 -> A4

Script: `tmp_scripts/run_recipe1_g4_w4_a8to6to4_rescale_quantlr4_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_g4_w4_a8to6to4_rescale_quantlr4_5ep_20260702`.
Motivation: keep target W4 weight quantization from the beginning while only reducing activation quantization noise progressively.

Observed metrics:

| epoch | bit setting | raw Top-1 | raw Top-5 | note |
|---:|---|---:|---:|---|
| 0 | W4A8 | 73.9520 | 91.9100 | weaker than high-bit weight warmup; not promising |

Decision: stopped after epoch0. Activation-only progression does not reproduce the early 74.9/75.8 gains; high-bit weight warmup appears to be the main contributor. Next candidate should keep G3 but raise LR when entering W4A4 to allow target-bit adaptation.

## Recipe1-G5: G3 plus target-W4A4 LR bump

Script: `tmp_scripts/run_recipe1_g5_prog8to6to4_rescale_lrbump_quantlr4_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_g5_prog8to6to4_rescale_lrbump_quantlr4_5ep_20260702`.
Motivation: after G3 showed strong W8A8/W6A6 results but weak W4A4 recovery, test whether the target W4A4 phase simply needs higher LR. Schedule: W8A8 epoch0, W6A6 epoch1, W4A4 epoch2+, LSQ rescale, with `epoch-lr-overrides=2:2e-4,3:2e-4`.

Observed metrics:

| epoch | bit / LR setting | raw Top-1 | raw Top-5 | note |
|---:|---|---:|---:|---|
| 0 | W8A8, normal LR | 74.9680 | 92.5120 | same as G3 |
| 1 | W6A6, normal LR | 75.8360 | 92.8500 | same as G3 |
| 2 | W4A4, LR bumped to 2e-4 | 72.5320 | 91.1000 | much worse than G3 epoch2 74.3240 |

Decision: stopped after epoch2. Target-W4A4 LR bump makes the transition worse. The issue is not simply insufficient LR; high-bit weight states appear incompatible with direct W4A4 target quantization. Current best valid 5-epoch W4A4 result remains Recipe1-B quant_lr_multiplier=4 at 76.684 Top-1.

## Recipe1-G6: G3 plus W4A4 alpha recalibration at target-bit switch

Script: `tmp_scripts/run_recipe1_g6_prog8to6to4_rescale_recalib4_quantlr4_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_g6_prog8to6to4_rescale_recalib4_quantlr4_5ep_20260702`.
Infra change: added `--progressive-bit-recalibrate-epochs` and `--progressive-bit-recalibrate-batches`; recalibration preserves optimizer parameter objects by copying newly calibrated LSQ `s.data` back into the old Parameter.

Observed markers:

- `progressive bit recalibrate alpha batches=4 quantizers=55`
- `Applied progressive bit alpha recalibration: epoch=2 batches=4 quantizers=55`

Observed metrics:

| epoch | bit / extra setting | raw Top-1 | raw Top-5 | note |
|---:|---|---:|---:|---|
| 0 | W8A8 | 74.9680 | 92.5120 | same as G3 |
| 1 | W6A6 | 75.8360 | 92.8500 | same as G3 |
| 2 | W4A4 + 4-batch alpha recalibration | 73.0500 | 91.1440 | worse than G3 epoch2 74.3240 |

Decision: stopped after epoch2. Explicit short alpha recalibration worsens W4A4 validation, likely because few-batch calibration shifts LSQ scales away from the trained high-bit basin. The progressive-bit family is informative but currently not a valid path to 5-epoch W4A4 80%. Current best valid W4A4 5-epoch result remains Recipe1-B quant_lr_multiplier=4 at 76.684 Top-1.

## Recipe1-H: full W4A4 quant_lr4 with constant LR 2e-4

Script: `tmp_scripts/run_recipe1_h_quantlr4_constlr2e4_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_h_quantlr4_constlr2e4_5ep_20260702`.
Motivation: test whether Recipe1-B is limited by cosine LR decay over only 5 epochs.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.3820 | 91.0600 | below Recipe1-B x4 epoch0 72.6880 |

Decision: stopped after epoch0. Constant LR 2e-4 hurts from the first epoch and is not the explanation for the x4 gap to 80%.

## Recipe1-I: full W4A4 quant_lr4 with setup-alpha-batches=4

Script: `tmp_scripts/run_recipe1_i_quantlr4_setupalpha4_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_i_quantlr4_setupalpha4_5ep_20260702`.
Motivation: after progressive-bit results showed scale sensitivity, test whether a slightly stronger initial W4A4 LSQ calibration improves the stable x4 recipe.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.4880 | 91.0440 | below Recipe1-B x4 epoch0 72.6880 |

Decision: stopped after epoch0. More setup-alpha batches do not improve the x4 full-W4A4 baseline; current best valid 5-epoch W4A4 result remains Recipe1-B quant_lr_multiplier=4 at 76.684 Top-1.

## Recipe1-J: full W4A4 quant_lr4 + teacher attention-output MSE

Script: `tmp_scripts/run_recipe1_j_quantlr4_attnout_w1e3_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_j_quantlr4_attnout_w1e3_5ep_20260702`.
Motivation: all-target-W4A4 structural distillation. Add lightweight FP teacher attention module output reconstruction on attention layers 6-11 with weight 1e-3, while keeping Recipe1-B quant_lr4 KD/no-aug backbone.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.7560 | 91.3300 | slightly above x4 epoch0 72.6880 |
| 1 | 74.3500 | 92.1520 | slightly above x4 epoch1 74.3160 |
| 2 | 75.6420 | 92.6980 | above x4 epoch2 75.4120 |
| 3 | 76.2380 | 93.0220 | below x4 epoch3 76.3460 |
| 4 | 76.6600 | 93.1140 | slightly below x4 final 76.6840 |

Wall time: 1916 seconds. Decision: completed full 5 epochs but did not beat current best and remains far below the 80% gate. Attention-output MSE gives early gains but over-regularizes late; next variant should reduce the weight, e.g. 5e-4.

## Recipe1-K: full W4A4 quant_lr4 + weaker teacher attention-output MSE

Script: `tmp_scripts/run_recipe1_k_quantlr4_attnout_w5e4_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_k_quantlr4_attnout_w5e4_5ep_20260702`.
Motivation: reduce J's attention-output regularization strength from 1e-3 to 5e-4 to avoid late over-regularization.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.5260 | 91.3400 | below Recipe1-B x4 epoch0 72.6880 and below J epoch0 72.7560 |

Decision: stopped after epoch0. Weaker attention-output loss is worse; J remains the best attention-output variant but does not beat Recipe1-B final.

## Recipe1-L: full W4A4 quant_lr4 + teacher Q/K relation loss

Script: `tmp_scripts/run_recipe1_l_quantlr4_qkrel_w1e4_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_l_quantlr4_qkrel_w1e4_5ep_20260702`.
Motivation: structural Q/K relation distillation as an alternative to attention-output MSE.

Outcome: runtime failure before first valid training metric. The run initialized grouped LR and entered epoch0 policy setup, then repeatedly emitted `terminate called without an active exception` and exited without validation. No accuracy result; no checkpoint retained. This direction is not usable without debugging qqkkvv/QK relation runtime stability.

## Recipe1-M: Recipe1-J with attention-output loss disabled after epoch2

Script: `tmp_scripts/run_recipe1_m_quantlr4_attnout_w1e3_off3_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_m_quantlr4_attnout_w1e3_off3_5ep_20260702`.
Infra change: added `--teacher-attn-output-weight-epoch-overrides`; M uses weight 1e-3 for epochs 0-2 and overrides epochs 3-4 to 0.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.7560 | 91.3300 | same as J |
| 1 | 74.3500 | 92.1520 | same as J |
| 2 | 75.6420 | 92.6980 | same as J |
| 3 | 76.4420 | 93.0260 | attention-output disabled; above J epoch3 but only slightly above x4 epoch3 |
| 4 | 76.6500 | 93.1980 | below J 76.660 and below x4 best 76.684 |

Wall time: 1917 seconds. Decision: did not beat current best. Disabling attention-output after epoch2 does not recover enough late accuracy.

## Recipe1-N: full W4A4 quant_lr4 + confidence-weighted pure soft KD

Script: `tmp_scripts/run_recipe1_n_quantlr4_confkd_p1_5ep_20260702.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_n_quantlr4_confkd_p1_5ep_20260702`.
Motivation: teacher-student fast convergence by weighting samples according to teacher confidence while using pure soft KD.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.6880 | 91.3820 | identical to Recipe1-B x4 epoch0 |
| 1 | 74.3160 | 92.1480 | identical to Recipe1-B x4 epoch1 |

Decision: stopped after epoch1. In the current implementation this confidence-weighted KD path does not change the observed trajectory versus Recipe1-B / pure-soft F. It is not a productive axis.

## Recipe1-O: full W4A4 quant_lr4 + late weak feature-output normalized MSE

Script: `tmp_scripts/run_recipe1_o_quantlr4_featout_late_w005_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_o_quantlr4_featout_late_w005_5ep_20260703`.
Motivation: unlike Recipe1-C, apply weaker feature-output reconstruction only on late Swin stages (`features.5.5,features.7.1`), starting from epoch1, to avoid disrupting initial quantizer adaptation.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.6880 | 91.3820 | same as x4; feature loss warmup not active |
| 1 | 74.4920 | 92.2220 | above x4 epoch1 74.3160 and above J epoch1 74.3500 |
| 2 | 75.5020 | 92.5880 | above x4 epoch2 75.4120 but below J epoch2 75.6420 |
| 3 | 76.3760 | 92.9700 | slightly above x4 epoch3 76.3460 |
| 4 | 76.6000 | 93.2080 | below x4 final 76.6840 |

Wall time: 1914 seconds. Decision: did not beat current best. Late weak feature-output helps mid-epoch but final accuracy remains lower than Recipe1-B.

## Recipe1-P: full W4A4 quant_lr4 + clean-start hard-label CE auxiliary

Script: `tmp_scripts/run_recipe1_p_quantlr4_cleance005_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_p_quantlr4_cleance005_5ep_20260703`.
Motivation: add a small hard-label CE auxiliary (`clean_start_target_loss_weight=0.05`) to the quant_lr4 KD path to improve fast Top-1 convergence.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.7520 | 91.3040 | slightly above x4 epoch0 but lower Top-5 |
| 1 | 74.2400 | 92.1760 | below x4 epoch1 74.3160 |

Decision: stopped after epoch1. The hard-label auxiliary raises training loss and does not improve fast validation accuracy. Not a productive axis.

## Recipe1-Q: full W4A4 quant_lr4 with quantizer-only warmup for first 100 updates

Script: `tmp_scripts/run_recipe1_q_quantlr4_quantwarm100_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_q_quantlr4_quantwarm100_5ep_20260703`.
Motivation: target-W4A4 training-stage design. Since quant_lr4 is the strongest axis, test whether explicitly warming only quant/shift parameters for the first 100 optimizer updates improves initial target-bit adaptation.

Observed markers:

- `Trainable parameter update policy: epoch=0, update=0, mode=grad_mask, policy=quant`
- `Trainable parameter update policy: epoch=0, update=100, mode=grad_mask, policy=all`

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.2160 | 90.8840 | below Recipe1-B x4 epoch0 72.6880 |

Decision: stopped after epoch0. Quantizer-only warmup harms early validation and is not a productive axis.

## Recipe1-R: full W4A4 quant_lr4 -> quant_lr8 after epoch2

Script: `tmp_scripts/run_recipe1_r_quantlr4_to8_epoch3_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_r_quantlr4_to8_epoch3_5ep_20260703`.
Infra change: added `--quant-lr-multiplier-epoch-overrides`, which updates quant/shift optimizer group `lr_scale` by epoch while preserving the base LR schedule.
Motivation: fixed x8 was worse early, fixed x4 was best; test whether x4 for early stable adaptation and x8 for late quantizer/scale refinement improves final W4A4 accuracy.

Observed markers:

- `Applied quant LR multiplier override: epoch=3, multiplier=8.0, groups=1`
- `Applied quant LR multiplier override: epoch=4, multiplier=8.0, groups=1`

Observed metrics:

| epoch | quant LR multiplier | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---:|---|
| 0 | 4 | 72.6880 | 91.3820 | same as x4 |
| 1 | 4 | 74.3160 | 92.1480 | same as x4 |
| 2 | 4 | 75.4120 | 92.6360 | same as x4 |
| 3 | 8 | 76.2380 | 92.9520 | below x4 epoch3 76.3460 |
| 4 | 8 | 76.6900 | 93.2740 | slightly above prior x4 best 76.6840 |

Wall time: 1914 seconds. Decision: this is the current best valid full-W4A4 5-epoch raw result, but the gain is only +0.006 Top-1 and remains far below the 80% gate. Do not mark goal complete.

## Recipe1-S: full W4A4 quant_lr4 -> quant_lr6 after epoch2

Script: `tmp_scripts/run_recipe1_s_quantlr4_to6_epoch3_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_s_quantlr4_to6_epoch3_5ep_20260703`.
Motivation: R showed a tiny final gain using late quant_lr8 but hurt epoch3; test a milder late quant_lr6 schedule.

Observed markers:

- `Applied quant LR multiplier override: epoch=3, multiplier=6.0, groups=1`
- `Applied quant LR multiplier override: epoch=4, multiplier=6.0, groups=1`

Observed metrics:

| epoch | quant LR multiplier | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---:|---|
| 0 | 4 | 72.6880 | 91.3820 | same as x4 |
| 1 | 4 | 74.3160 | 92.1480 | same as x4 |
| 2 | 4 | 75.4120 | 92.6360 | same as x4 |
| 3 | 6 | 76.3500 | 92.9760 | near x4 epoch3 76.3460 |
| 4 | 6 | 76.6480 | 93.1820 | below R 76.690 and x4 best 76.684 |

Wall time: 1914 seconds. Decision: not better. Current best remains Recipe1-R at 76.690 Top-1.

## Recipe1-T: pre-QAT teacher-logit reconstruction warm start + Recipe1-R

Script: `tmp_scripts/run_recipe1_t_prerecon100_quantlr4_to8_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_t_prerecon100_quantlr4_to8_5ep_20260703`.
Infra change: added `--pre-qat-recon-updates` and `--pre-qat-recon-temperature`. Before epoch training, it runs teacher-logit reconstruction for quant/shift parameters only, preserving the raw W4A4 target setup.

Observed markers:

- `Starting pre-QAT teacher-logit reconstruction: updates=100, policy=quant, temperature=1.0`
- `PreQATRecon: update=1/100 loss=7.935340`
- `PreQATRecon: update=50/100 loss=6.485645`
- `PreQATRecon: update=100/100 loss=4.088405`

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.8220 | 91.3340 | above x4/R epoch0 72.6880 but lower Top-5 |
| 1 | 74.2220 | 92.0060 | below x4/R epoch1 74.3160 |

Decision: stopped after epoch1. Pre-QAT reconstruction lowers training loss and slightly improves epoch0 Top-1, but hurts subsequent validation; not a productive path in this simple quant-only logit form.

## Recipe1-U: full W4A4 quant_lr4 -> quant_lr8 only at final epoch

Script: `tmp_scripts/run_recipe1_u_quantlr4_to8_epoch4_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_u_quantlr4_to8_epoch4_5ep_20260703`.
Motivation: R improved final slightly but hurt epoch3 when quant_lr8 started at epoch3. U keeps the stable x4 trajectory through epoch3 and applies quant_lr8 only during epoch4.

Observed marker:

- `Applied quant LR multiplier override: epoch=4, multiplier=8.0, groups=1`

Observed metrics:

| epoch | quant LR multiplier | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---:|---|
| 0 | 4 | 72.6880 | 91.3820 | same as x4 |
| 1 | 4 | 74.3160 | 92.1480 | same as x4 |
| 2 | 4 | 75.4120 | 92.6360 | same as x4 |
| 3 | 4 | 76.3460 | 92.9960 | same as x4 |
| 4 | 8 | 76.7000 | 93.2260 | new best, +0.016 over x4 and +0.010 over R |

Wall time: 1914 seconds. Decision: current best valid full-W4A4 5-epoch raw result, but still far below the 80% gate. Do not mark goal complete.

## Recipe1-V: full W4A4 quant_lr4 -> quant_lr10 only at final epoch

Script: `tmp_scripts/run_recipe1_v_quantlr4_to10_epoch4_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_v_quantlr4_to10_epoch4_5ep_20260703`.
Motivation: U improved final by applying quant_lr8 only at epoch4; test whether an even stronger final-epoch quant/shift refinement with quant_lr10 helps.

Observed marker:

- `Applied quant LR multiplier override: epoch=4, multiplier=10.0, groups=1`

Observed metrics:

| epoch | quant LR multiplier | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---:|---|
| 0 | 4 | 72.6880 | 91.3820 | same as x4/U |
| 1 | 4 | 74.3160 | 92.1480 | same as x4/U |
| 2 | 4 | 75.4120 | 92.6360 | same as x4/U |
| 3 | 4 | 76.3460 | 92.9960 | same as x4/U |
| 4 | 10 | 76.6880 | 93.1760 | below U 76.7000 |

Wall time: 1914 seconds. Decision: not better. Current best remains Recipe1-U at 76.7000 Top-1.

## Recipe1-W: pre-QAT feature reconstruction warm start + Recipe1-U

Script: `tmp_scripts/run_recipe1_w_prefeatrecon100_quantlr4_to8e4_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_w_prefeatrecon100_quantlr4_to8e4_5ep_20260703`.
Motivation: APHQ/I&S-ViT style local feature/block output reconstruction before QAT. Run 100 updates of feature reconstruction on `features.5.5,features.7.1`, updating quant/shift parameters only, then use Recipe1-U.

Observed markers:

- `Starting pre-QAT feature reconstruction: updates=100, policy=quant, layers=('features.5.5', 'features.7.1')`
- `PreQATFeatRecon: update=1/100 loss=0.742389`
- `PreQATFeatRecon: update=50/100 loss=0.589160`
- `PreQATFeatRecon: update=100/100 loss=0.545850`

Outcome: runtime failure before valid validation. After pre-reconstruction, DDP failed with `Expected to have finished reduction... not compatible with static_graph=True`. This is an infrastructure issue caused by using `--static-graph` with a pre-training phase that masks gradients for most parameters. Need rerun without static graph.

## Recipe1-W2: pre-QAT feature reconstruction without static graph

Script: `tmp_scripts/run_recipe1_w2_prefeatrecon100_nostatic_quantlr4_to8e4_5ep_20260703.sh`.
Outcome: runtime failure during pre-QAT feature reconstruction. Without static graph, DDP still fails because the feature reconstruction loss only uses intermediate outputs, so many later parameters are unused in the reduction graph. Fix: execute pre-QAT reconstruction before wrapping the model with DDP, then run normal DDP QAT.

## Recipe1-W3: pre-QAT feature reconstruction before DDP + Recipe1-U

Script: `tmp_scripts/run_recipe1_w3_prefeatrecon100_preddp_quantlr4_to8e4_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_w3_prefeatrecon100_preddp_quantlr4_to8e4_5ep_20260703`.
Literature/code motivation: APHQ-ViT and I&S-ViT use cached FP block/layer outputs and local reconstruction before/for PTQ. W3 implements a lightweight variant in this codebase: before DDP wrapping and before epoch training, run 100 updates of feature-output reconstruction on late Swin blocks (`features.5.5,features.7.1`), updating quant/shift parameters only. Then train with Recipe1-U (full W4A4 quant_lr4, final epoch quant_lr8).

Observed pre-reconstruction markers:

- `Starting pre-QAT feature reconstruction: updates=100, policy=quant, layers=('features.5.5', 'features.7.1')`
- `PreQATFeatRecon: update=1/100 loss=0.742389`
- `PreQATFeatRecon: update=50/100 loss=0.589160`
- `PreQATFeatRecon: update=100/100 loss=0.545850`

Observed metrics:

| epoch | setting | raw Top-1 | raw Top-5 | note |
|---:|---|---:|---:|---|
| 0 | pre-feature recon + quant_lr4 | 72.7220 | 91.1720 | slightly above U/x4 Top-1 but lower Top-5 |
| 1 | quant_lr4 | 74.6220 | 92.2020 | clearly above U/x4 74.3160 |
| 2 | quant_lr4 | 75.7860 | 92.7760 | above U/x4 75.4120 |
| 3 | quant_lr4 | 76.4360 | 93.1040 | above U/x4 76.3460 |
| 4 | quant_lr8 | 76.7080 | 93.2300 | new best, +0.008 over U and +0.024 over original x4 |

Wall time: 1973 seconds. Decision: current best valid full-W4A4 5-epoch raw result, but still far below the 80% gate. Do not mark goal complete. This is the first positive result from paper-code-inspired PTQ-to-QAT reconstruction; next variants should tune reconstruction layers/updates/loss.

## Recipe1-W4: pre-QAT late feature reconstruction 200 updates + Recipe1-U

Script: `tmp_scripts/run_recipe1_w4_prefeatrecon200_preddp_quantlr4_to8e4_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_w4_prefeatrecon200_preddp_quantlr4_to8e4_5ep_20260703`.
Motivation: W3 with 100 pre-feature reconstruction updates was positive; test whether more reconstruction improves initialization.

Observed pre-reconstruction markers:

- `PreQATFeatRecon: update=1/200 loss=0.742389`
- `PreQATFeatRecon: update=100/200 loss=0.545850`
- `PreQATFeatRecon: update=200/200 loss=0.471344`

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.8000 | 91.3600 | above W3 epoch0 72.7220 |
| 1 | 74.4500 | 92.2360 | below W3 epoch1 74.6220 |

Decision: stopped after epoch1. More feature reconstruction lowers reconstruction loss but weakens QAT validation trajectory, likely overfitting quant/shift scales to the calibration feature objective. Next test fewer updates (50).

## Recipe1-W5: pre-QAT late feature reconstruction 50 updates + Recipe1-U

Script: `tmp_scripts/run_recipe1_w5_prefeatrecon50_preddp_quantlr4_to8e4_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_w5_prefeatrecon50_preddp_quantlr4_to8e4_5ep_20260703`.
Motivation: W3 with 100 reconstruction updates was positive and W4 with 200 updates weakened epoch1; test fewer updates.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.3640 | 91.4080 | below W3 and x4 |
| 1 | 74.2800 | 92.1260 | below W3 74.6220 and x4 74.3160 |

Decision: stopped after epoch1. 50 reconstruction updates is too weak/unstable. W3's 100 updates remains the best reconstruction setting so far.

## Recipe1-W7: sequential block-wise pre-QAT feature reconstruction + Recipe1-U

Script: `tmp_scripts/run_recipe1_w7_seqfeatrecon50x2_quantlr4_to8e4_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_w7_seqfeatrecon50x2_quantlr4_to8e4_5ep_20260703`.
Motivation: closer APHQ/I&S style sequential block reconstruction. Instead of W3's joint feature reconstruction over all selected layers, W7 reconstructs each layer sequentially and masks gradients to that layer's quant/shift parameters only.

Observed markers:

- `PreQATSeqFeatRecon: layer=features.5.5 update=50/50 loss=0.929804 kept=8337 masked=12309712`
- `PreQATSeqFeatRecon: layer=features.7.1 update=50/50 loss=0.299948 kept=16387 masked=27704859`

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.4980 | 91.3560 | below W3 and x4 |
| 1 | 74.2940 | 92.3080 | below W3 74.6220 and x4 74.3160 |

Decision: stopped after epoch1. Sequential per-layer quant/shift-only feature reconstruction is worse than W3's joint reconstruction; likely over-localizes quantizer adaptation and misses cross-block interactions. W3 remains the best paper-code-inspired reconstruction recipe at 76.708 Top-1.

## Recipe1-W8 / QSC-v1 module-all feature reconstruction

Script: `tmp_scripts/run_recipe1_w8_qscv1_moduleall_recon100_quantlr4_to8e4_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_w8_qscv1_moduleall_recon100_quantlr4_to8e4_5ep_20260703`.
Motivation: a more aggressive QSC-v1 variant where pre-QAT feature reconstruction updates all parameters inside the selected late blocks, not only quant/shift parameters.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.5820 | 91.3100 | below W3 72.7220 |
| 1 | 74.4620 | 92.1240 | below W3 74.6220 |

Decision: stopped after epoch1. Updating full local block parameters in the pre-QAT feature reconstruction is worse than updating only quant/shift. QSC best remains W3: joint late-block feature reconstruction, quant/shift-only, 100 updates.

## QSC best 10-epoch from scratch validation

Script: `tmp_scripts/run_recipe1_w3_best_10ep_prefeatrecon100_q4_final8_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_w3_best_10ep_prefeatrecon100_q4_final8_20260703`.
Purpose: after five QSC variants, run the current best QSC recipe for 10 epochs from scratch as requested. This is not used to satisfy the 5-epoch gate; it tests whether the QSC trajectory continues improving beyond the 5-epoch window.

Recipe:

- pre-QAT joint feature reconstruction, 100 updates, `features.5.5,features.7.1`
- reconstruction updates only quant/shift parameters
- full W4A4 QAT after reconstruction
- quant_lr_multiplier=4 for epochs 0-8
- final epoch quant_lr_multiplier=8
- KD hard+soft, no augmentation, bf16 AMP, global batch 2048

Observed metrics:

| epoch | raw Top-1 | raw Top-5 |
|---:|---:|---:|
| 0 | 72.2740 | 91.0340 |
| 1 | 73.3760 | 91.5620 |
| 2 | 74.4780 | 92.1440 |
| 3 | 74.7820 | 92.2480 |
| 4 | 75.5400 | 92.4900 |
| 5 | 76.0600 | 92.8720 |
| 6 | 76.4400 | 93.1700 |
| 7 | 76.7980 | 93.1700 |
| 8 | 77.1600 | 93.4940 |
| 9 | 77.3420 | 93.5400 |

Wall time: 3733 seconds. Decision: QSC continues improving to 77.342 at 10 epochs, but it does not satisfy the active 5-epoch >=80 goal. The 10-epoch result supports QSC as a stable training paradigm, but the 5-epoch acceleration gap remains large.

## Recipe1-W9 / QSC-v2: confidence-gated joint feature reconstruction

Script: `tmp_scripts/run_recipe1_w9_qscv2_conf1_recon100_quantlr4_to8e4_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_w9_qscv2_conf1_recon100_quantlr4_to8e4_5ep_20260703`.
Motivation: QSC-v2 attempts an original refinement to QSC-v0/W3. Instead of treating all calibration samples equally during pre-QAT feature reconstruction, it weights each sample by FP teacher confidence, hoping to align quant/shift states to the most stable teacher basin.

Observed pre-reconstruction markers:

- `Starting pre-QAT feature reconstruction: updates=100, policy=quant, confidence_power=1.0`
- `PreQATFeatRecon: update=1/100 loss=0.740891`
- `PreQATFeatRecon: update=50/100 loss=0.625952`
- `PreQATFeatRecon: update=100/100 loss=0.604389`

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.6580 | 91.3580 | below W3 epoch0 72.7220 |
| 1 | 74.2100 | 92.0140 | below W3 epoch1 74.6220 and x4 74.3160 |

Decision: stopped after epoch1. Confidence-gated feature reconstruction is worse; it likely over-weights easy samples and reduces coverage of quantization-sensitive regions. QSC-v0/W3 remains best.

## QSC current synthesis

Current best QSC recipe is W3:

1. Start from FP pretrained Swin-T.
2. Build target W4A4 fake-quant model immediately.
3. Before normal QAT, run 100-step joint late-block feature reconstruction on `features.5.5,features.7.1`, updating only quant/shift parameters.
4. Run normal full-parameter W4A4 QAT with KD hard+soft, no augmentation, global batch 2048, quant_lr4.
5. In the final epoch, increase quant/shift LR multiplier to 8 for target-state refinement.

Best 5-epoch result: 76.708 Top-1. Best 10-epoch result: 77.342 Top-1. The active 5-epoch >=80 goal is still unmet.

## Recipe1-W10 / QSC-v3: disagreement-guided joint feature reconstruction

Script: `tmp_scripts/run_recipe1_w10_qscv3_disagree_recon100_quantlr4_to8e4_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_w10_qscv3_disagree_recon100_quantlr4_to8e4_5ep_20260703`.
Motivation: QSC-v3 tries a quantization-sensitive sample weighting rule. During pre-QAT joint feature reconstruction, samples are weighted by current W4A4 student vs FP teacher logit KL, so reconstruction focuses on samples where quantization changes the model most.

Observed pre-reconstruction markers:

- `Starting pre-QAT feature reconstruction: updates=100, policy=quant, weight_mode=disagreement`
- `PreQATFeatRecon: update=1/100 loss=0.741889`
- `PreQATFeatRecon: update=50/100 loss=0.627988`
- `PreQATFeatRecon: update=100/100 loss=0.608001`

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.4360 | 91.1640 | below W3 72.7220 |
| 1 | 74.5480 | 92.0740 | below W3 74.6220 |

Decision: stopped after epoch1. Disagreement weighting is worse than uniform joint reconstruction. Likely it over-focuses on currently unstable samples and creates a noisier alignment target. QSC-v0/W3 remains best.

## Recipe1-X / QSS-v0: Quantizer Slow State

Script: `tmp_scripts/run_recipe1_x_qssv0_slowstate_q4_final8_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_x_qssv0_slowstate_q4_final8_5ep_20260703`.
Motivation: a new training mechanism to decouple weight learning and quantization-state learning. Maintain an EMA shadow state for quant/shift parameters, and periodically pull student quant/shift toward the slow state to reduce scale/shift oscillation.

Config:

- quant_slow_state_decay=0.99
- sync_interval=50 optimizer updates
- pull=0.1
- base recipe: full W4A4 quant_lr4, final epoch quant_lr8

Observed markers:

- `Initialized quant slow state: params=308, decay=0.99, sync_interval=50, pull=0.1`
- periodic `Applied quant slow state pull` every 50 updates

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.5620 | 91.1360 | below U/W3/x4 epoch0 |

Decision: stopped after epoch0. QSS-v0 hurts early validation; the slow-state pull likely over-damps necessary early quantizer adaptation.

## Recipe1-Y / QSS-v1: delayed quantizer slow state + final quant LR burst

Script: `tmp_scripts/run_recipe1_y_qssv1_delayed_e3_pull005_q4_final8_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_y_qssv1_delayed_e3_pull005_q4_final8_5ep_20260703`.
Motivation: QSS-v0 failed because slow-state pull started from step0 and over-damped early quantizer adaptation. QSS-v1 uses a more coherent training paradigm: early free quantizer adaptation, then late quantizer-state stabilization.

Recipe:

- full W4A4 from scratch
- KD hard+soft, no augmentation, global batch 2048
- quant_lr_multiplier=4
- QSS disabled for epochs 0-2
- QSS enabled from epoch3: EMA shadow over quant/shift params, decay=0.99, sync_interval=50 updates, pull=0.05
- final epoch quant_lr_multiplier=8

Observed markers:

- no QSS initialization during epochs 0-2
- `Initialized quant slow state: params=308, decay=0.99, sync_interval=50, pull=0.05` at epoch3
- periodic `Applied quant slow state pull` every 50 updates after epoch3
- `Applied quant LR multiplier override: epoch=4, multiplier=8.0, groups=1`

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.6880 | 91.3820 | same as x4/U |
| 1 | 74.3160 | 92.1480 | same as x4/U |
| 2 | 75.4120 | 92.6360 | same as x4/U |
| 3 | 76.2880 | 93.0200 | slightly below U epoch3 76.3460 |
| 4 | 76.7720 | 93.1800 | new best; above U 76.7000 and W3 76.7080 |

Wall time: 1913 seconds. Decision: current best valid full-W4A4 5-epoch raw result. QSS-v1 is the strongest original training-paradigm signal so far, but still below the 80% goal.

## Recipe1-Z / QSS-v2: final-epoch-only quantizer slow state

Script: `tmp_scripts/run_recipe1_z_qssv2_finalonly_e4_pull005_q4_final8_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_z_qssv2_finalonly_e4_pull005_q4_final8_5ep_20260703`.
Motivation: QSS-v1 improved final but slightly hurt epoch3. QSS-v2 delays slow-state stabilization until the final epoch only, so epochs 0-3 match the stable U/x4 trajectory, and epoch4 combines quant_lr8 with QSS.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.6880 | 91.3820 | same as U/x4 |
| 1 | 74.3160 | 92.1480 | same as U/x4 |
| 2 | 75.4120 | 92.6360 | same as U/x4 |
| 3 | 76.3460 | 92.9960 | same as U/x4 |
| 4 | 76.6760 | 93.1540 | below U 76.700 and QSS-v1/Y 76.772 |

Decision: QSS only in the final epoch is worse. The useful QSS window appears to need one full epoch of slow-state stabilization before the final quant_lr burst. Current best remains Recipe1-Y / QSS-v1 at 76.772 Top-1.

## Recipe1-AA / QSS-v3: observe from epoch0, pull from epoch3

Script: `tmp_scripts/run_recipe1_aa_qssv3_observe0_pull3_q4_final8_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_aa_qssv3_observe0_pull3_q4_final8_5ep_20260703`.
Motivation: QSS-v1 initializes the slow state at epoch3, so it does not track the free-adaptation trajectory from epochs 0-2. QSS-v3 separates observing from pulling: maintain shadow quant/shift EMA from epoch0, but only pull from epoch3 onward.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.6880 | 91.3820 | same as U/x4 |
| 1 | 74.3160 | 92.1480 | same as U/x4 |
| 2 | 75.4120 | 92.6360 | same as U/x4 |
| 3 | 76.3320 | 92.9620 | below U 76.346 and above Y 76.288 |
| 4 | 76.5420 | 93.2400 | below Y/QSS-v1 76.772 |

Decision: QSS-v3 is worse. Observing from epoch0 creates a shadow state that lags too far behind the useful late quantizer state; pulling toward it in epochs 3-4 over-regularizes. The simplest and best QSS remains QSS-v1: initialize/observe/pull from epoch3.

## Recipe1-AC: intended final-val QSS-v1 speed check

Script: `tmp_scripts/run_recipe1_ac_qssv1_true_valfinal_5ep_20260703.sh`.
Outcome: stopped early. The run still performed epoch0/1 validation, so `val_interval` did not alter the unified OFQ validation path as intended at that point. Since the relaxed time budget accepts <=2000s and Y/QSS-v1 already fits, this duplicate speed-only run was stopped.

## Recipe1-AD / QAS-v0: Quantization Align-and-Stabilize

Script: `tmp_scripts/run_recipe1_ad_qas_v0_align_then_qss_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_ad_qas_v0_align_then_qss_5ep_20260703`.
Motivation: combine the two simple original mechanisms: first align quant/shift state with pre-QAT feature reconstruction (QSC), then use delayed QSS for late quantizer stabilization. One-sentence recipe: align quantization state, freely train, then stabilize quantization state.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.4280 | 91.1900 | below W3 and U |
| 1 | 74.4320 | 92.2020 | below W3 74.6220 |

Decision: stopped after epoch1. Directly stacking QSC and QSS is worse than either best standalone mechanism. The two curricula appear to interfere: QSC's pre-aligned quant/shift state does not benefit from later QSS pull. Keep QSC and QSS as separate simple paradigms, not a combined recipe.

## Recipe1-AF / QSS-v4: delayed activation/shift-only slow state

Script: `tmp_scripts/run_recipe1_af_qssv4_activation_delayed_e3_pull005_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_af_qssv4_activation_delayed_e3_pull005_5ep_20260703`.
Motivation: QSS-v1 stabilizes all quant/shift parameters. QSS-v4 asks whether only activation quantizers and shift parameters need slow-state stabilization, leaving weight quantizers freer.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.6880 | 91.3820 | same as U/x4 |
| 1 | 74.3160 | 92.1480 | same as U/x4 |
| 2 | 75.4120 | 92.6360 | same as U/x4 |
| 3 | 76.3820 | 92.9880 | slightly above U epoch3 76.3460 |
| 4 | 76.6380 | 93.2000 | below Y/QSS-v1 76.7720 |

Decision: activation-only slow state is worse. QSS should stabilize the full quant/shift state, not only activation/shift. Current best remains Y/QSS-v1 at 76.772 Top-1.

## Recipe1-AG / QSS-start2: delayed full quant/shift slow state from epoch2

Script: `tmp_scripts/run_recipe1_ag_qss_start2_pull005_q4_final8_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_ag_qss_start2_pull005_q4_final8_5ep_20260703`.
Motivation: QSS-v1 starts full quant/shift slow-state stabilization at epoch3. AG tests whether starting one epoch earlier gives more stabilization time while keeping the same simple QSS paradigm.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.6880 | 91.3820 | same as U/x4 |
| 1 | 74.3160 | 92.1480 | same as U/x4 |
| 2 | 75.5860 | 92.6640 | above U/x4 75.4120 |
| 3 | 76.4380 | 93.0420 | above U/x4 76.3460 and Y epoch3 76.2880 |
| 4 | 76.7000 | 93.1920 | below Y/QSS-v1 76.7720 |

Wall time: 1923 seconds. Decision: starting QSS at epoch2 improves mid-epoch accuracy but final is worse than QSS-v1. Current best remains Y/QSS-v1 at 76.772 Top-1.

## Recipe1-AH / QSS-v1-light: delayed full quant/shift slow state with weaker pull

Script: `tmp_scripts/run_recipe1_ah_qssv1_pull0025_q4_final8_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_ah_qssv1_pull0025_q4_final8_5ep_20260703`.
Motivation: QSS-v1 with pull=0.05 is current best. AH tests whether weaker pull=0.025 reduces over-regularization while preserving late stabilization.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.6880 | 91.3820 | same as U/x4 |
| 1 | 74.3160 | 92.1480 | same as U/x4 |
| 2 | 75.4120 | 92.6360 | same as U/x4 |
| 3 | 76.2160 | 93.0220 | below Y/QSS-v1 76.2880 |
| 4 | 76.5940 | 93.1240 | below Y/QSS-v1 76.7720 |

Wall time: 1916 seconds. Decision: weaker pull is worse. QSS-v1 pull=0.05 remains best.

## Recipe1-AI / QSS-v1-strong: delayed full quant/shift slow state with stronger pull

Script: `tmp_scripts/run_recipe1_ai_qssv1_pull010_q4_final8_5ep_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/recipe1_ai_qssv1_pull010_q4_final8_5ep_20260703`.
Motivation: QSS-v1 with pull=0.05 is current best. AI tests whether stronger pull=0.10 improves late quantizer-state stabilization.

Observed metrics:

| epoch | raw Top-1 | raw Top-5 | note |
|---:|---:|---:|---|
| 0 | 72.6880 | 91.3820 | same as U/x4 |
| 1 | 74.3160 | 92.1480 | same as U/x4 |
| 2 | 75.4120 | 92.6360 | same as U/x4 |
| 3 | 76.4160 | 93.0000 | higher than Y epoch3 but lower than AG epoch3 |
| 4 | 76.6340 | 93.1860 | below Y/QSS-v1 76.7720 |

Wall time: 1917 seconds. Decision: stronger pull is worse. QSS-v1 pull=0.05 remains best. Pull strength window: 0.025 too weak, 0.10 too strong, 0.05 best so far.

## QSS-v1 best 100-epoch from scratch long run

Script: `tmp_scripts/run_qssv1_best_100ep_fromscratch_20260703.sh`.
Output: `/tmp/qat_recipe1_runs/qssv1_best_100ep_fromscratch_20260703`.
Log: `/tmp/train_qssv1_best_100ep_fromscratch_20260703.log`.
Purpose: new long-run goal to evaluate the long-training ceiling of the current best original QSS-v1 recipe.

Recipe:

- full W4A4 QAT from scratch
- KD hard+soft, no augmentation, global batch 2048
- bf16 AMP, weight_decay=0
- quant_lr_multiplier=4
- QSS-v1 enabled from epoch3: full quant/shift slow state, decay=0.99, sync_interval=50, pull=0.05
- final epoch quant_lr_multiplier=8 at epoch99
- checkpoint every 10 epochs
- full validation every 10 epochs
- output stays under `/tmp`

Status: launched. Initial training entered epoch0 normally with grouped LR and 8 GPU utilization. Awaiting checkpoint/validation at epoch10.

Update 2026-07-03 15:07 UTC:

- Worker: `975345`, reachable via SSH `fdbd:dccd:cdc2:12c8:0:138::` port `9680`.
- Training process is active on 8x H100; `nvidia-smi` shows all 8 GPUs at 100% utilization.
- Active command is the script above; output remains `/tmp/qat_recipe1_runs/qssv1_best_100ep_fromscratch_20260703` and log remains `/tmp/train_qssv1_best_100ep_fromscratch_20260703.log`.
- Code path checked: `epoch_checkpoint_interval=10` and `val_interval=10` are both applied through `(epoch + 1) % interval == 0`, so checkpoint/full validation should occur at epochs 10,20,...,100.
- Current observed progress: epoch0 and epoch1 completed; latest monitor snapshot is epoch2 around step 100/625. No checkpoint yet, as expected before epoch10.
- Local monitor started: `tmp_scripts/monitor_qssv1_100ep_20260703.sh`, writing `docs/qssv1_best_100ep_fromscratch_20260703_status.tsv`. This monitor only reads worker log/output and does not alter training.

Pending result table:

| epoch | raw Top-1 | raw Top-5 | checkpoint | note |
|---:|---:|---:|---|---|
| 10 | 74.4900 | 92.1500 | `/tmp/qat_recipe1_runs/qssv1_best_100ep_fromscratch_20260703/checkpoint-10.pth.tar` | raw ckpt full ImageNet eval-only; train-time val initially hit NCCL OOM before metric aggregation |
| 20 | 74.4280 | 92.1880 | `/tmp/qat_recipe1_runs/qssv1_best_100ep_fromscratch_20260703/checkpoint-20.pth.tar` | in-training full ImageNet val succeeded after validation hotfix |
| 30 | 75.5800 | 92.7360 | `/tmp/qat_recipe1_runs/qssv1_best_100ep_fromscratch_20260703/checkpoint-30.pth.tar` | in-training full ImageNet val |
| 40 | 76.2100 | 93.1240 | `/tmp/qat_recipe1_runs/qssv1_best_100ep_fromscratch_20260703/checkpoint-40.pth.tar` | in-training full ImageNet val |
| 50 | 76.8020 | 93.4600 | `/tmp/qat_recipe1_runs/qssv1_best_100ep_fromscratch_20260703/checkpoint-50.pth.tar` | in-training full ImageNet val |
| 60 | 77.4540 | 93.7500 | `/tmp/qat_recipe1_runs/qssv1_best_100ep_fromscratch_20260703/checkpoint-60.pth.tar` | in-training full ImageNet val |
| 70 | 77.9000 | 93.9880 | `/tmp/qat_recipe1_runs/qssv1_best_100ep_fromscratch_20260703/checkpoint-70.pth.tar` | in-training full ImageNet val |
| 80 | 78.1160 | 94.1600 | `/tmp/qat_recipe1_runs/qssv1_best_100ep_fromscratch_20260703/checkpoint-80.pth.tar` | in-training full ImageNet val |
| 90 | 78.2480 | 94.3200 | `/tmp/qat_recipe1_runs/qssv1_best_100ep_fromscratch_20260703/checkpoint-90.pth.tar` | in-training full ImageNet val |
| 100 | pending | pending | pending | |


Update 2026-07-03 16:03 UTC:

- `checkpoint-10.pth.tar` was saved successfully at `/tmp/qat_recipe1_runs/qssv1_best_100ep_fromscratch_20260703/checkpoint-10.pth.tar`.
- The in-training epoch10 validation initially failed during distributed metric aggregation with `NCCL Error 1: unhandled cuda error` after local validation; the checkpoint itself is valid and contains `state_dict`, `optimizer`, `lr_scheduler`, and `rng_state`.
- Minimal validation hotfix applied in `qat_launch.py`: free CUDA cache before metric reduction and replace the nonessential `dist.gather` of per-rank sample counts with scalar `all_reduce` summaries. `python -m py_compile qat_launch.py` passed.
- Independent full ImageNet distributed eval-only on raw `checkpoint-10` succeeded with batch size 64: Top-1 `74.4900`, Top-5 `92.1500`, `50000` samples, wall `29.413s`.
- Next action: strict resume from `checkpoint-10` to continue epochs 10-100 with the same QSS-v1 recipe and fixed validation path.

Update 2026-07-03 16:05 UTC:

- Strict resume from `checkpoint-10.pth.tar` launched with script `tmp_scripts/run_qssv1_best_100ep_resume_from_ckpt10_20260703.sh` and log `/tmp/train_qssv1_best_100ep_fromscratch_20260703_resume_from_ckpt10_20260703.log`.
- Resume evidence from log:
  - model loaded with `missing=0, unexpected=0`;
  - optimizer restored, `state entries=433`;
  - lr scheduler restored with last LR `[0.00019535184353590011, 0.0007814073741436005]`;
  - RNG state restored `True`;
  - resumed at `epoch=10` with global effective batch `2048`.
- Resume training is active on 8x H100 and has entered `Train: 10`.



Update 2026-07-03 17:06 UTC:

- `checkpoint-20.pth.tar` saved successfully.
- In-training full ImageNet distributed validation succeeded after the validation hotfix: Top-1 `74.4280`, Top-5 `92.1880`, samples `50000`.
- Resume job is still active and has continued into `epoch=20`; no validation/runtime error observed after checkpoint-20.


Update 2026-07-03 18:02 UTC:

- `checkpoint-30.pth.tar` saved successfully.
- In-training full ImageNet distributed validation: Top-1 `75.5800`, Top-5 `92.7360`, samples `50000`.
- Resume job remains active and has continued into `epoch=30`; no validation/runtime error observed.


Update 2026-07-03 19:03 UTC:

- `checkpoint-40.pth.tar` saved successfully.
- In-training full ImageNet distributed validation: Top-1 `76.2100`, Top-5 `93.1240`, samples `50000`.
- Resume job remains active and has continued into `epoch=40`; no validation/runtime error observed.


Update 2026-07-03 19:59 UTC:

- `checkpoint-50.pth.tar` saved successfully.
- In-training full ImageNet distributed validation: Top-1 `76.8020`, Top-5 `93.4600`, samples `50000`.
- Resume job remains active and has continued into `epoch=50`; no validation/runtime error observed.


Update 2026-07-03 20:55 UTC:

- `checkpoint-60.pth.tar` saved successfully.
- In-training full ImageNet distributed validation: Top-1 `77.4540`, Top-5 `93.7500`, samples `50000`.
- Resume job remains active and has continued into `epoch=60`; no validation/runtime error observed.


Update 2026-07-03 21:55 UTC:

- `checkpoint-70.pth.tar` saved successfully.
- In-training full ImageNet distributed validation: Top-1 `77.9000`, Top-5 `93.9880`, samples `50000`.
- Resume job remains active and has continued into `epoch=70`; no validation/runtime error observed.


Update 2026-07-03 22:51 UTC:

- `checkpoint-80.pth.tar` saved successfully.
- In-training full ImageNet distributed validation: Top-1 `78.1160`, Top-5 `94.1600`, samples `50000`.
- Resume job remains active and has continued into `epoch=80`; no validation/runtime error observed.


Update 2026-07-03 23:52 UTC:

- `checkpoint-90.pth.tar` saved successfully.
- In-training full ImageNet distributed validation: Top-1 `78.2480`, Top-5 `94.3200`, samples `50000`.
- Resume job remains active and has continued into `epoch=90`; no validation/runtime error observed.
