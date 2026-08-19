# ARIS-style QATs Current Analysis

Date: 2026-08-04

## Scope

This note uses the cloned ARIS workflow repository at:

```text
/mlx_devbox/users/quyanyi/playground/ARIS
```

Relevant ARIS workflow lens:

- `analyze-results`: locate experiment logs, build comparison tables, compute deltas, state observations and next experiments.
- `experiment-audit`: separate claimed conclusions from what logs and artifacts prove.

Target project:

```text
/mlx_devbox/users/quyanyi/playground/QATs
```

This analysis focuses on the current OFQ/Swin-T long-run branch, not the older July short-run recipe sweeps.

## Current Experiment Class

All rows below are Swin-T OFQ/QAT ImageNet runs using the public-family setup:

- `method=ofq`
- `wbits=4`, `abits=4`
- `wq_mode=statsq`, `aq_mode=lsq`
- `qk_reparam=true`
- ImageNet parquet data at `/tmp/imagenet1k_full_parquet`
- full validation rows use 50,000 samples

Bit-width label: these runs should be described as `W4A4-family`. This report did not re-audit first/last layer quantization, so it should not call them strict W4A4.

## Raw Comparison Table

Metrics below were extracted from `Test: [distributed-summary]` rows in the training logs.

| Run | Epoch rows | Epoch range | Best epoch | Best Top-1 | Best Top-5 | Last Top-1 | Last20 avg | Last10 avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ofq_100ep_fromscratch_original_ofq_public_control_20260714` | 100 | 0-99 | 81 | 80.7920 | 95.4100 | 80.6780 | 80.6916 | 80.7086 |
| `ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713` | 100 | 0-99 | 99 | 80.7720 | 95.4320 | 80.7720 | 80.6965 | 80.7316 |
| `ofq_100ep_fromscratch_teacher_sparse_attnkl_fixed_20260803` | 100 | 0-99 | 81 | 80.7920 | 95.4100 | 80.6780 | 80.6916 | 80.7086 |
| `ofq_200ep_fromscratch_fixedcycle_group_sparse_prevstep_refkl_20260731` | 200 | 0-199 | 194 | 80.8680 | 95.4420 | 80.7080 | 80.7375 | 80.7570 |
| `ofq_resume200_to300_fixedcycle_sparse_prevstep_refkl_20260802` | 100 | 200-299 | 209 | 80.8420 | 95.4820 | 80.6900 | 80.7227 | 80.7192 |

## Evidence Paths

Primary logs:

```text
/mlx_devbox/users/quyanyi/playground/train_ofq_100ep_fromscratch_original_ofq_public_control_20260714.log
/mlx_devbox/users/quyanyi/playground/train_ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713.log
/mlx_devbox/users/quyanyi/playground/train_ofq_100ep_fromscratch_teacher_sparse_attnkl_fixed_20260803.log
/mlx_devbox/users/quyanyi/playground/train_ofq_200ep_fromscratch_fixedcycle_group_sparse_prevstep_refkl_20260731.log
/mlx_devbox/users/quyanyi/playground/train_ofq_resume200_to300_fixedcycle_sparse_prevstep_refkl_20260802.log
```

Primary scripts:

```text
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_100ep_fromscratch_teacher_sparse_attnkl_fixed_20260803.sh
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_200ep_fromscratch_fixedcycle_group_sparse_prevstep_refkl_20260731.sh
/mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_ofq_resume200_to300_fixedcycle_sparse_prevstep_refkl_20260802.sh
```

Existing comparison docs:

```text
/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_100ep_kl_vs_nokl_trajectory_comparison_20260715.md
/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_200ep_fromscratch_fixedcycle_sparse_prevstep_refkl_experiment_20260731.md
/mlx_devbox/users/quyanyi/playground/QATs/docs/ofq_100ep_kl_nokl_teacher_fixed_fullval_curve_20260804.csv
```

## Key Findings

1. The 100epoch sparse prev-step KL branch does not clearly beat the no-KL control.

   Observation: 100ep no-KL best Top-1 is `80.7920`; 100ep sparse prev-step KL best Top-1 is `80.7720`. KL has slightly higher last10 average (`80.7316` vs `80.7086`), but the best checkpoint is lower by `0.0200`.

   Interpretation: the 100epoch result mainly shows that the public-family OFQ long-run recipe is strong. It does not prove that this sparse prev-step KL controller is beneficial in the 100epoch from-pretrained setting.

2. The teacher sparse attention-KL fixed run currently matches the no-KL control at full-val precision.

   Observation: `teacher_sparse_attnkl_fixed_20260803` and the no-KL control have identical Top-1 values for all 100 full-validation epochs. The extracted diff count for Top-1 is `0`.

   Important nuance: the teacher KL path did activate. Logs show nonzero `TeacherAttnKL` from epoch 5 through epoch 89 under weights `1e-6` or `2e-6`. However, the raw `TeacherAttnKL` is clipped around `20.0`, so the added loss scale is only about `2e-5` to `4e-5`. That is too small to move this run at reported validation precision.

   Interpretation: do not claim teacher-attention KL benefit from this run. The valid conclusion is that this schedule is effectively a near-control at the current weight scale.

3. The strongest single checkpoint in the current long-run set is the 200epoch fixed-cycle sparse prev-step KL run.

   Observation: `ofq_200ep_fromscratch_fixedcycle_group_sparse_prevstep_refkl_20260731` reaches best Top-1 `80.8680` at epoch 194.

   Interpretation: fixed-cycle sparse prev-step KL is the best observed single-checkpoint branch in this comparison, but it still does not prove KL benefit alone because a strict same-setting 200epoch no-KL from-scratch control is missing.

4. Extending the 200epoch checkpoint to 300 epochs did not create a new best.

   Observation: `resume200_to300` reaches best Top-1 `80.8420` at epoch 209, below the source 200epoch run best `80.8680`. Final epoch 299 is `80.6900`.

   Interpretation: this continuation behaves like low-yield late training. It should not be extended further in the same form unless a new intervention is added.

5. No active QAT training process was found during this analysis.

   Observation: process scan for `qat_launch`, `third_party/OFQ/train.py`, `torchrun`, and current `train_ofq` patterns returned no active training process.

## Integrity Notes

- All compared rows are from full ImageNet distributed summaries with `Samples: 50000`.
- End-of-run NCCL/TCPStore warnings appear after completed final validation and `wall_seconds` logging. Treat them as teardown warnings, not evidence that the training result is invalid.
- The `teacher_sparse_attnkl_fixed` log differs from the no-KL control log as a whole, but the extracted validation Top-1 trajectory is identical.
- The current workspace has many untracked docs/scripts. Avoid using git cleanliness as an evidence proxy here; use explicit logs, scripts, and checkpoint paths.

## Recommended Next Experiments

1. Run the missing 200epoch no-KL from-scratch control with the same 200epoch scheduler, batch size, augmentation, seed, data path, checkpoint cadence, and OFQ settings.

   Purpose: isolate whether `80.8680` is from fixed-cycle sparse prev-step KL or simply from extending the public-family recipe to 200 epochs.

2. If testing teacher attention KL again, increase it through a short-run calibration gate before spending a full 100epoch run.

   Suggested gate: keep the same heads and schedule family, but choose weights that move the loss by an observable amount, for example a target contribution around `1e-3` to `1e-2` instead of `2e-5` to `4e-5`. Run 5-10 epochs or a small continuation gate first, and check both `TeacherAttnKL` nonzero rate and validation delta.

3. Stop extending `resume200_to300` with the same sparse pulse schedule.

   The branch already failed to beat the epoch-194 source best. A further 300->400 continuation should require a changed hypothesis, not just more epochs.

4. Keep the current best checkpoint selection simple.

   Current best single checkpoint in this set:

   ```text
   /tmp/qat_public_repro/ofq_200ep_fromscratch_fixedcycle_group_sparse_prevstep_refkl_20260731/checkpoint-195.pth.tar
   ```

   Note: training logs are zero-based (`epoch=194`) while checkpoint naming in this workflow is one-based (`checkpoint-195`).

## Bottom Line

The current best evidence says the public-family OFQ recipe is strong and stable around `80.7-80.9` Top-1. The fixed-cycle sparse prev-step KL 200epoch run has the best observed single checkpoint at `80.8680`, but the decisive control is still missing. The teacher-attention KL fixed run should be treated as a near-control because its effective loss scale was too small to alter the validation trajectory.
