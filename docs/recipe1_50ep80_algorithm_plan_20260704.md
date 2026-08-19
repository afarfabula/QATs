# Recipe1 50-epoch 80% algorithm plan

Goal: make Swin-T W4A4 OFQ/QAT reach raw full ImageNet Top-1 >=80.0 within 50 epochs. Do not treat the QSS-v1 result as the main path; use it as negative evidence.

## Current evidence

### QSS-v1 long run

Run: `/tmp/qat_recipe1_runs/qssv1_best_100ep_fromscratch_20260703`.

Final full ImageNet validation at `checkpoint-100`:

| checkpoint | Top-1 | Top-5 | loss |
|---|---:|---:|---:|
| checkpoint-100 | 78.4020 | 94.3060 | 0.9126 |

Conclusion: QSS-v1 is not a strong long-training recipe. It slightly helped the 5-epoch search but underperformed the strict no-QSS stage1 100-epoch baseline (`78.4760`). Its slow-state pull appears to over-regularize quant/shift adaptation over long horizons.

### Strict stage1 100-epoch baseline

Run: `/mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage1_fromscratch_bs256_kd_noaug_strictckpt_100ep_20260702`.

Final full ImageNet validation:

| checkpoint | Top-1 | Top-5 |
|---|---:|---:|
| checkpoint-100 | 78.4760 | 94.3420 |

This remains the best same-codepath no-augmentation 100-epoch baseline.

### Older OFQ 300-epoch mainline

Run root: `/mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_mainline_300ep_20260613/swin_t_w4a4_imagenet1k_8gpu_300ep_mainline`.

The current directory retains only `checkpoint-300.pth.tar` and `last.pth.tar`; intermediate `checkpoint-49`, `checkpoint-50`, and `checkpoint-100` are no longer present. The 300-epoch run was not a clean QKR cold-start run. It resumed from an earlier QKR 50-epoch source run:

- source log: `/mlx_devbox/users/quyanyi/playground/QATs/logs/swin_t_w4a4_imagenet1k_8gpu_50ep_directverify_v2.log`
- source command included `--epochs 50 --batch-size 64 --qk_reparam --qk_reparam_type 0 --skip_validate`
- source checkpoint path used later: `/mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_unified_50ep_directverify_v2/swin_t_w4a4_imagenet1k_8gpu_50ep_directverify_v2/checkpoint-49.pth.tar`
- current status: that source output directory is no longer available locally, so the checkpoint artifact cannot currently be reused or inspected.

The closest logged validations around the source/continuation:

| log | approximate point | Top-1 | Top-5 |
|---|---:|---:|---:|
| `swin_t_w4a4_imagenet1k_8gpu_50ep_directverify_v2.log` | resumed/evaluated after `checkpoint-19`, before later source continuation | 78.7740 | 94.4400 |
| `swin_t_w4a4_ofq_mainline_300ep_20260613.log` / `rerun.log` | resumed from source `checkpoint-49`, validation before further continuation | 80.6460 | 95.3520 |
| `swin_t_w4a4_ofq_mainline_300ep_20260613_rerun2.log` | resumed from `checkpoint-51`, validation before further continuation | 78.8940 | 94.6460 |

Important caveat: this is evidence that a QKR 50-epoch lineage reached >=80, but the live checkpoint artifact is missing. It should guide reproduction, not be counted as a currently available successful checkpoint.

Additional source-run audit:

- `logs/swin_t_w4a4_imagenet1k_8gpu_50ep_directverify_v2.log` contains the original QKR 50-epoch command and checkpoint saves through at least `checkpoint-29`, then reaches `epoch: 39` and ends with an NCCL timeout. It does not contain a local `checkpoint-49` save line.
- `logs/swin_t_w4a4_ofq_mainline_300ep_20260613.log` and `..._rerun.log` both resume from `ofq_unified_50ep_directverify_v2/.../checkpoint-49.pth.tar` and validate Top-1 `80.6460`, so `checkpoint-49` did exist at the time.
- Current filesystem search under `/mlx_devbox/users/quyanyi` and `/tmp` did not find `ofq_unified_50ep_directverify_v2/.../checkpoint-49.pth.tar`. The only current `checkpoint-49.pth.tar` found is from a different no-augmentation stage1 directory and is not the QKR source artifact.
- Therefore the goal has historical evidence but no reusable checkpoint artifact.

Prepared reproduction plan:

1. Run `tmp_scripts/run_repro_qkr_50ep_history_keepckpt_20260704.sh` only after deciding to spend a full 50-epoch run. It intentionally mirrors the historical source run more closely than Candidate A:
   - QKR enabled with `--qk-reparam --qk-reparam-type 0`;
   - `--batch-size 64`, `--grad-accum-steps 1`, 8 GPUs;
   - `--skip_validate` during training;
   - `--checkpoint-hist 20`, `--epoch-checkpoint-interval 10`, so checkpoints needed for post-run verification should be retained.
2. Verify any produced checkpoint with `tmp_scripts/eval_repro_qkr_checkpoint_20260704.sh /path/to/checkpoint-N.pth.tar`.
3. Do not claim success from the training run alone. Success requires the eval script to report raw full ImageNet Top-1 >=80.0 at or before the 50-epoch checkpoint.

## Why QSS-v1 is weak

1. It does not enable QK reparameterization. QSS only pulls quant/shift parameters toward a slow EMA; it does not change the attention logit quantization graph.
2. It uses the large-batch no-augmentation Recipe1 setup: global batch 2048, 625 optimizer steps per epoch, no mixup/cutmix/randaugment/smoothing/reprob.
3. The older OFQ mainline used global batch 512, about 2502 optimizer steps per epoch, plus OFQ-style augmentation. Its 50 epochs contain about four times as many optimizer updates as Recipe1's 50 epochs.
4. Long-horizon quantizer adaptation should not be globally pulled from epoch3 onward. The QSS-v1 slow state likely damps useful later adaptation.

## QKR interpretation

OFQ's QK reparameterization is a structural attention fix, not just another regularizer.

For Swin attention, the QKR module splits `q`, `k`, and `v`, forms per-head `Wq^T @ Wk`, quantizes the composed QK weight, then computes attention logits through `x @ Wqk @ x^T`. This directly targets the query-key intertwined oscillation described by OFQ: independently quantized Q and K can jump between quantization bins and amplify attention-logit error. QSS-v1 does not address this failure mode.

The official OFQ W4A4 Swin/DeiT scripts enable `--qk_reparam --qk_reparam_type 0`, and the OFQ README reports Swin-T W4A4 `81.88` and DeiT-S W4A4 `81.10`. For this goal, QKR should be considered a required baseline component.

`qat_launch.py` currently supports QKR end to end:

- CLI flags: `--qk-reparam`, `--qk-reparam-type`
- runtime defaults: `qk_reparam`, `qk_reparam_type`
- subprocess command mapping: `--qk_reparam`, `--qk_reparam_type`
- in-process OFQ replacement path: passes QKR into `replace_module_by_qmodule_swin(...)`

No launcher patch is required before the first QKR candidate.

## Success criteria

Primary gate:

- raw full ImageNet validation Top-1 >=80.0
- reached by checkpoint at or before 50 epochs under the chosen epoch/update budget
- raw checkpoint metric, not EMA-only, soup-only, partial validation, or training loss proxy

Secondary reporting:

- Top-5, loss, sample count
- exact checkpoint path
- exact run command/script
- effective global batch
- optimizer updates per epoch and total optimizer updates to the gate
- whether QKR, augmentation, QSS/CGA/refmodel were enabled

## Epoch budget policy

The current project has two epoch conventions:

| convention | global batch | optimizer steps per epoch | 50-epoch update budget |
|---|---:|---:|---:|
| Recipe1 large batch | 2048 | ~625 | ~31k |
| older OFQ mainline | 512 | ~2502 | ~125k |

For an honest 50-epoch target, use both labels in reports:

1. `data_epoch`: one full pass over ImageNet.
2. `update_budget`: total optimizer updates.

The first serious candidate should follow the older OFQ effective update budget if the target is "match OFQ-like 50 epochs." If wall time is the limiting factor, keep global batch 2048 but do not compare its 50 epochs directly to old OFQ 50 epochs.

## Candidate order

### Candidate A: OFQ-QKR reproduction baseline

Purpose: recover the missing OFQ structural component before adding new mechanisms.

Config:

- Swin-T W4A4
- `qk_reparam=True`, `qk_reparam_type=0`
- StatsQ weights, LSQ activations, W4A4, per-channel, activation clip learnable
- KD hard+soft
- pretrained initialized
- no QSS
- OFQ-style augmentation restored: randaugment, mixup 0.8, cutmix 1.0, smoothing 0.1, reprob 0.25
- use old OFQ update budget where feasible: global batch 512 or equivalent update-count schedule

Gate:

- validate at epochs 10, 20, 30, 40, 50
- stop early if epoch20 is below 78.5 and trajectory is not above strict stage1 by at least 1.0 Top-1
- continue if epoch20 >=79.0 or epoch30 >=79.5

Expected value: highest chance to cross 80 by OFQ-style 50 epochs. This is the baseline the current QSS-v1 omitted.

### Candidate B: QKR plus large-batch speed recipe

Purpose: test whether QKR alone rescues the current faster 2048 batch pipeline.

Config:

- same as Candidate A, but global batch 2048
- no QSS initially
- optionally keep no-aug for first 1-3 epochs then switch augmentation on

Gate:

- if epoch5 is not above the best no-QKR 5-epoch result (`76.7720`), QKR integration or schedule is suspect
- if epoch20 remains below 78.5, large-batch update budget is insufficient for 50-epoch 80

Expected value: lower than Candidate A for the 50-epoch gate, but useful for wall-time discipline.

### Candidate C: QKR plus CGA-style stabilization

Purpose: use OFQ's native long-horizon stabilizer after QKR baseline is healthy.

Config:

- start from Candidate A checkpoint after a stable phase
- `qk_reparam_type=1`
- `boundaryRange=0.005`
- freeze high-confidence weights for the CGA stage

Gate:

- only run if Candidate A approaches but does not cross 80 by epoch50
- require positive validation delta within 5-10 CGA epochs

Expected value: better theoretical fit than QSS-v1 because CGA is designed for quantized weight oscillation, not just quantizer EMA smoothing.

### Candidate D: QKR plus selective QSS/refmodel

Purpose: salvage the useful part of QSS/refmodel without global over-regularization.

Config:

- QKR enabled
- no global QSS from epoch3
- if used, QSS only late and only on measured oscillating quant/shift tensors or attention-related modules
- optional weighted QK/attention relation loss with warmup; do not use the existing unweighted built-in QK KD because it previously dominated loss

Gate:

- must improve over QKR baseline, not over the failed no-QKR QSS-v1
- stop if validation drops for two consecutive validation points

Expected value: exploratory; not the first candidate.

## Recommended next concrete step

Do not launch a full 50-epoch run before a QKR smoke and 1-epoch check.

1. Verify QKR model construction with a short run: `max_train_updates=1`, no validation, confirm no runtime failure and checkpoint can save.
2. Run a 1-epoch QKR + OFQ augmentation sanity check on all 8 GPUs.
3. If sane, run Candidate A with validation every 10 epochs and checkpoint every 10 epochs.

Prepared scripts:

| script | purpose | status |
|---|---|---|
| `tmp_scripts/run_recipe1_qkr_smoke_1update_20260704.sh` | 8-GPU QKR construction/training smoke, 1 optimizer update, no validation, step checkpoint enabled | passed |
| `tmp_scripts/run_recipe1_qkr_ofqaug_1ep_sanity_20260704.sh` | 1-epoch QKR + OFQ augmentation sanity run with full validation | completed; poor initial validation |
| `tmp_scripts/run_recipe1_candidate_a_qkr_ofqaug_50ep_20260704.sh` | Candidate A full run: QKR + OFQ augmentation, global batch 512, 50 epochs, validation/checkpoint every 10 epochs | prepared; shell syntax checked; not launched |
| `tmp_scripts/run_repro_qkr_50ep_history_keepckpt_20260704.sh` | closer reproduction of historical `50ep_directverify_v2`: QKR, batch 64, `skip_validate`, keep checkpoints | prepared; shell syntax checked; not launched |
| `tmp_scripts/eval_repro_qkr_checkpoint_20260704.sh` | independent eval-only verifier for a reproduced checkpoint, defaulting to reproduced `checkpoint-49.pth.tar` | prepared; shell syntax checked; not launched |

Static parameter check:

- QKR enabled in all three scripts via `--qk-reparam --qk-reparam-type 0`.
- OFQ-style augmentation enabled in all three scripts: smoothing 0.1, mixup 0.8, cutmix 1.0, randaugment `rand-m9-mstd0.5-inc1`, color jitter 0.4, reprob 0.25.
- Candidate A uses `--batch-size 64` on 8 GPUs, giving global batch 512 and approximately the older OFQ update budget.
- Candidate A uses `--epochs 50 --scheduler-epochs 50 --val-interval 10 --epoch-checkpoint-interval 10`.

QKR smoke result:

- Command log: `/tmp/train_recipe1_qkr_smoke_1update_20260704.log`.
- Output: `/tmp/qat_recipe1_runs/recipe1_qkr_smoke_1update_20260704`.
- Bottom-line status: passed. The run completed one optimizer update and exited with code 0.
- Evidence:
  - bottom command contained `--qk_reparam --qk_reparam_type 0`;
  - effective batch alignment was `per_gpu_effective_batch=64`, `world_size=8`, `global_effective_batch=512`;
  - `TrainSummary: epoch=0 updates=1`;
  - `Stopped early after 1 optimizer updates in epoch 0`;
  - saved `checkpoint-1.pth.tar`, `last.pth.tar`, `step_checkpoints/step_0000.pth.tar`, and `step_checkpoints/step_0001.pth.tar`.
- Note: PyTorch/NCCL printed TCPStore heartbeat warnings during process shutdown after the controlled early exit. GPUs returned to idle and no QAT process remained, so this is treated as teardown noise rather than QKR smoke failure.

QKR + OFQ augmentation 1-epoch sanity result:

- Command log: `/tmp/train_recipe1_qkr_ofqaug_1ep_sanity_20260704.log`.
- Output: `/tmp/qat_recipe1_runs/recipe1_qkr_ofqaug_1ep_sanity_20260704`.
- Bottom-line status: completed but not healthy enough to justify launching Candidate A blindly.
- Evidence:
  - command contained `--qk_reparam --qk_reparam_type 0`;
  - effective batch alignment was `per_gpu_effective_batch=64`, `world_size=8`, `global_effective_batch=512`;
  - `TrainSummary: epoch=0 updates=2496 avg_step_time=0.216035s samples_per_step=512 samples_per_sec=2369.99`;
  - full ImageNet validation after one epoch: Top-1 `38.2700`, Top-5 `65.3000`, loss `2.8952`, samples `50000`;
  - saved `checkpoint-1.pth.tar` and `last.pth.tar`.
- Interpretation: this confirms QKR can train fast and complete validation, but the first-epoch accuracy is far below the no-QKR/no-augmentation stage1 epoch0/1 values. Before launching 50 epochs, diagnose whether this is expected QKR cold-start behavior under OFQ augmentation, a QKR initialization issue in the Swin module, or a mismatch between the unified path and the original OFQ script.

QKR initialization diagnosis:

- A code inspection found that `QAttention_swin_qkreparam` and `QAttention_swin_qkreparam_4_cga` wrapped `self.proj` from the newly constructed `ShiftedWindowAttention` instead of the original pretrained `m.proj`. They also did not copy `m.relative_position_bias_table` after reinitializing the table. This can destroy initial accuracy.
- A temporary diagnostic patch was tested in `third_party/OFQ/src/quantization/modules/swin_attention_and_mlp.py`:
  - use `m.proj` when constructing the QKR output projection `QLinear`;
  - copy `m.relative_position_bias_table` in both QKR and QKR+CGA Swin attention classes.
- Syntax check passed: `python3 -m py_compile third_party/OFQ/src/quantization/modules/swin_attention_and_mlp.py`.

Post-patch QKR initial eval:

- Script: `tmp_scripts/eval_recipe1_qkr_initial_20260704.sh`.
- Log: `/tmp/eval_recipe1_qkr_initial_eval_20260704.log`.
- Output: `/tmp/qat_recipe1_runs/recipe1_qkr_initial_eval_20260704`.
- Result: Top-1 `0.2460`, Top-5 `1.1440`, loss `7.7956`, samples `50000`.
- Interpretation: the simple initialization patch is not sufficient. The current unified QKR cold-start model is still not functionally initialized from the FP pretrained model. Candidate A must not be launched until QKR initialization/equivalence is fixed.
- Current hypotheses:
  1. The QKR attention logit algebra or tensor orientation does not preserve the FP attention operation when initialized from `m.qkv`.
  2. QKR requires a dedicated pretrained QKR checkpoint or a reconstruction/calibration phase, rather than direct FP checkpoint conversion.
  3. Some QKR quantizers or learnable bias terms need explicit calibration before full ImageNet eval.
  4. The original OFQ training logs that reached useful accuracy may not have evaluated the QKR cold-start model before substantial training/recovery.

Additional QKR cold-start diagnosis:

- Script: `tmp_scripts/debug_qkr_swin_attention_equivalence.py`.
- The first checked Swin attention layer showed that Q/K pretrained biases are nonzero (`q` bias norm about `8.61`, `k` bias norm about `3.03`).
- QKR was updated to preserve Q/K bias and add Q/K bias cross terms in the reparameterized attention logits. This improved local single-layer agreement with the normal quantized attention (`QKR_vs_QAttention` relative L2 improved from about `0.44` to about `0.33`), but did not fix full-model cold-start eval.
- Re-running QKR initial eval after the bias patch still gave Top-1 only `0.3040` / Top-5 `1.3580`, so Candidate A remains blocked as a cold-start run.
- The historical QKR 50-epoch success therefore appears to be a trained QKR lineage, not evidence that direct FP-to-QKR cold-start eval is healthy.
- The temporary QKR diagnostic changes were then reverted back to the historical QKR structure for the active reproduction: q/k use `bias=False`, QKR wraps `m=self.proj`, and relative-position bias is initialized as in the original module. This preserves the historical parameter count and makes the reproduction closer to `50ep_directverify_v2`.

Parameter-count audit:

- Historical OFQ logs print `Model swin_t created, param count:28593633`.
- Current `qat_launch.py` smoke prints `Model swin_t created, param count:28608256`.
- A static same-process check using `qat_launch.get_ofq_qat_model(...)` and native `replace_module_by_qmodule_swin(...)` reports the same pre-calibration QKR model size: `28593633`.
- The apparent `+14623` parameter difference is caused by print timing: current unified `qat_launch.py` prints after `setup_alpha(...)`, when LSQ activation scale parameters have been materialized; historical `train.py` printed before setup alpha. This is not evidence that QKR replacement structure differs.
- Diagnostic script: `tmp_scripts/debug_unified_vs_ofq_param_names.py`.

## Completion audit status

The active goal is not complete.

Current execution rules:

- Any candidate must pass epoch1 full ImageNet raw Top-1 `>72.0` before it may continue.
- A single run may advance at most 10 epochs beyond the last validated checkpoint.
- Every 10-epoch boundary must be saved and independently full-evaluated before continuing.
- Goal completion requires a raw checkpoint at or before epoch 50 with full ImageNet raw Top-1 `>=80.0`.
- Mechanisms such as QKR, QSS, CGA, and refmodel are tools only; if a mechanism breaks the epoch1 gate, it must be fixed or removed before any long run.

Concrete success criteria:

- produce or recover a Swin-T W4A4 QKR/OFQ checkpoint at or before epoch 50;
- independently run full ImageNet raw eval on that checkpoint;
- show raw Top-1 >=80.0, not EMA/soup/proxy/loss-only evidence;
- record the exact checkpoint, command/script, effective batch/update budget, QKR/augmentation/refmodel/QSS/CGA settings, Top-1/Top-5/loss, and sample count.

Current active run:

- script: `tmp_scripts/run_repro_qkr_50ep_history_keepckpt_20260704.sh`
- log: `/tmp/train_repro_qkr_50ep_history_keepckpt_20260704.log`
- output: `/tmp/qat_recipe1_runs/repro_qkr_50ep_history_keepckpt_20260704`
- config: QKR type 0, W4A4, StatsQ weights, LSQ activations, KD hard+soft, pretrained initialized, `skip_validate`, global batch 512, about 2496 optimizer updates per epoch, checkpoint every 10 epochs with history 20.
- latest inspected state at 2026-07-04 07:03 UTC: running, 8 H100 GPUs at 100% utilization, epoch 10 in progress.
- first checkpoint artifact exists: `/tmp/qat_recipe1_runs/repro_qkr_50ep_history_keepckpt_20260704/checkpoint-10.pth.tar`, size about 329 MiB, saved at 2026-07-04 06:57 UTC. `last.pth.tar` is a hardlink/copy of the same save.
- naming note: although the save condition triggers at zero-based epoch index 9, the saved file is named `checkpoint-10.pth.tar`.
- `checkpoint-10` was independently evaluated with `tmp_scripts/eval_repro_qkr_checkpoint_20260704.sh /tmp/qat_recipe1_runs/repro_qkr_50ep_history_keepckpt_20260704/checkpoint-10.pth.tar`. Full ImageNet raw result:
  - log: `/tmp/eval_repro_qkr_50ep_history_keepckpt_20260704_checkpoint-10_20260704.log`
  - Top-1 `0.1200`, Top-5 `0.5560`, loss `7.7495`, samples `50000`
  - wall time about 50s for validation, 196s including startup.
- This is not a successful result and is too low to be treated as a normal near-miss. However the closest historical validated point is not epoch 10: historical `50ep_directverify_v2` validated after resuming from `checkpoint-19.pth.tar` and reported Top-1 `78.7740`, Top-5 `94.4400`, loss about `0.9439`.
- The training-loss trajectory is broadly similar to the historical run through epoch 10:
  - historical `checkpoint-9` save was preceded by epoch 9 loss around `6.5541`;
  - current `checkpoint-10` save was preceded by epoch 9 loss around `6.6158`;
  - historical epoch 19 loss was around `6.2518`.
- Current interpretation: `checkpoint-10` full-val is strong negative evidence for early raw accuracy, but it is not yet the apples-to-apples historical checkpoint-19 comparison. The next decisive verification point is current `checkpoint-20.pth.tar`; evaluate it immediately after it is saved.

User-requested 1-epoch gate:

- New gate from 2026-07-04: before any more long-run claims, the chain must show full ImageNet raw Top-1 >72 at the 1-epoch sanity stage; otherwise the training/eval/QKR chain is considered broken.
- The active 50-epoch QKR reproduction was stopped after `checkpoint-10` because `checkpoint-10` full-val was near random and continuing to epoch 20/50 would waste compute before the gate is satisfied.
- Current-code non-QKR control passed the gate:
  - script: `tmp_scripts/run_gate_nonqkr_1ep_gt72_20260704.sh`
  - log: `/tmp/train_gate_nonqkr_1ep_gt72_20260704.log`
  - config: W4A4, StatsQ/LSQ, pretrained initialized, KD hard+soft, no QKR, no augmentation, global batch 2048, 1 epoch
  - result: Top-1 `73.3100`, Top-5 `91.4720`, loss `1.0998`, samples `50000`
  - conclusion: dataset, full-val evaluator, launcher, DDP, ordinary OFQ quantized training, and current GPU environment are not globally broken.
- Current-code QKR control failed the gate:
  - script: `tmp_scripts/run_gate_qkr_1ep_gt72_20260704.sh`
  - log: `/tmp/train_gate_qkr_1ep_gt72_20260704.log`
  - config: identical to non-QKR gate except `--qk-reparam --qk-reparam-type 0`
  - result: crashed before the first train step after model creation with repeated `terminate called without an active exception`; no validation metric produced
  - conclusion: the present QKR path is not ready for long runs. The failure is isolated to QKR construction/forward/backward/DDP interaction under this gate, not to the global data/eval stack.
- Historical note: `logs/swin_t_w4a4_real_imagenet1k_1ep.log` shows a QKR run reaching Top-1 `72.202` at `checkpoint-6` and `73.484` at `checkpoint-10`, but not at the first checkpoint. Therefore QKR has historical recovery evidence, but it does not satisfy the strict 1-epoch >72 gate in its current form.
- Next action: do not resume the 50-epoch QKR reproduction. Either fix QKR to pass the 1-epoch gate or pivot the 50-epoch/80 search to the non-QKR chain that already satisfies the gate, then add stabilization mechanisms only after the gate remains healthy.

Non-QKR epoch10 stage result:

- Because the non-QKR chain passed the epoch1 gate, it was advanced only to the allowed epoch10 boundary.
- Script: `tmp_scripts/run_gate_nonqkr_resume1_to10_20260704.sh`
- Log: `/tmp/train_gate_nonqkr_resume1_to10_20260704.log`
- Start checkpoint: `/tmp/qat_recipe1_runs/gate_nonqkr_1ep_gt72_20260704/checkpoint-1.pth.tar`
- Output checkpoint: `/tmp/qat_recipe1_runs/gate_nonqkr_resume1_to10_20260704/checkpoint-10.pth.tar`
- Resume details: model state restored with `missing=0`, `unexpected=0`; optimizer/scheduler were intentionally reset with `--no-resume-opt`; run started at epoch 1 and ended at epoch 10.
- Config: W4A4, StatsQ/LSQ, pretrained initialized, KD hard+soft, no QKR/QSS/CGA/refmodel, no augmentation, global batch 2048, about 624 optimizer updates per epoch.
- Full ImageNet raw validation by epoch:
  - epoch1 continuation eval: Top-1 `72.5380`, Top-5 `91.3180`, loss `1.1207`
  - epoch2: Top-1 `73.5640`, Top-5 `91.8780`, loss `1.0862`
  - epoch3: Top-1 `74.4160`, Top-5 `92.0820`, loss `1.0608`
  - epoch4: Top-1 `75.2420`, Top-5 `92.4720`, loss `1.0285`
  - epoch5: Top-1 `75.5360`, Top-5 `92.7120`, loss `1.0179`
  - epoch6: Top-1 `76.2120`, Top-5 `92.9760`, loss `0.9930`
  - epoch7: Top-1 `76.6160`, Top-5 `93.1560`, loss `0.9808`
  - epoch8: Top-1 `76.6200`, Top-5 `93.3200`, loss `0.9710`
  - epoch10 boundary: Top-1 `76.8180`, Top-5 `93.3940`, loss `0.9700`, samples `50000`
- Conclusion: the current non-QKR chain satisfies the gate and is stable through epoch10, but it plateaus around `76.8` by epoch10 and is still far from the 80% target. Continuing this exact no-augmentation/no-stabilizer recipe to epoch20 is allowed only after an explicit decision, but it is unlikely by itself to reach 80 by epoch50 without a training-paradigm change.

W3/QSC-style epoch10 and epoch20 stage result:

- Existing strongest current candidate before this gate update was W3/QSC-style: pre-QAT feature reconstruction plus quant LR multiplier.
- W3 original script: `tmp_scripts/run_recipe1_w3_best_10ep_prefeatrecon100_q4_final8_20260703.sh`
- W3 original log: `/tmp/train_recipe1_w3_best_10ep_prefeatrecon100_q4_final8_20260703.log`
- W3 original output checkpoint: `/tmp/qat_recipe1_runs/recipe1_w3_best_10ep_prefeatrecon100_q4_final8_20260703/checkpoint-10.pth.tar`
- Config: W4A4, StatsQ/LSQ, pretrained initialized, KD hard+soft, no QKR, no augmentation, global batch 2048, `quant_lr_multiplier=4`, `pre_qat_feature_recon_updates=100`, recon layers `features.5.5,features.7.1`, final epoch quant LR override `9:8`.
- W3 original full ImageNet raw validation:
  - epoch1 gate: Top-1 `72.2740`, Top-5 `91.0340`, loss `1.1372`, samples `50000`
  - epoch10 boundary: Top-1 `77.3420`, Top-5 `93.5400`, loss `0.9531`, samples `50000`
- Since W3 passed epoch1 and epoch10 boundaries, it was advanced only one allowed stage to epoch20.
- W3 epoch10->20 script: `tmp_scripts/run_w3_resume10_to20_20260704.sh`
- W3 epoch10->20 log: `/tmp/train_w3_resume10_to20_20260704.log`
- W3 epoch20 output checkpoint: `/tmp/qat_recipe1_runs/w3_resume10_to20_20260704/checkpoint-20.pth.tar`
- Resume details: model state restored from W3 checkpoint10 with `missing=0`, `unexpected=0`; optimizer/scheduler/scaler/RNG intentionally reset using `--no-resume-opt`; run started at epoch 10 and ended at epoch 20.
- W3 epoch10->20 full ImageNet raw validation:
  - epoch10 continuation eval: Top-1 `75.8540`, Top-5 `92.8140`, loss `1.0072`
  - epoch11: Top-1 `76.1880`, Top-5 `93.0480`, loss `0.9918`
  - epoch12: Top-1 `76.3560`, Top-5 `93.0940`, loss `0.9844`
  - epoch13: Top-1 `76.6540`, Top-5 `93.3280`, loss `0.9742`
  - epoch14: Top-1 `77.0460`, Top-5 `93.3540`, loss `0.9624`
  - epoch15: Top-1 `77.2840`, Top-5 `93.5040`, loss `0.9504`
  - epoch16: Top-1 `77.3640`, Top-5 `93.6100`, loss `0.9477`
  - epoch17: Top-1 `77.4320`, Top-5 `93.7020`, loss `0.9432`
  - epoch18: Top-1 `77.5480`, Top-5 `93.7260`, loss `0.9418`
  - epoch20 boundary: Top-1 `77.5920`, Top-5 `93.6580`, loss `0.9385`, samples `50000`
- Conclusion: W3 remains the strongest current non-QKR recipe, but even after an additional allowed 10-epoch stage it reaches only `77.5920`, far below the `80.0` goal. The `--no-resume-opt` continuation also caused an immediate drop from the original W3 checkpoint10 `77.3420` to `75.8540`, so future stage continuations should consider strict optimizer/scheduler resume or a deliberately designed stage schedule rather than a naive optimizer reset.

W3 strict-resume epoch10 to epoch20 control:

- Motivation: the W3 `--no-resume-opt` continuation dropped immediately from the original checkpoint10 Top-1 `77.3420` to `75.8540`. A strict-resume control tests whether preserving optimizer/scheduler/RNG avoids that regression.
- Script: `tmp_scripts/run_w3_strict_resume10_to20_20260704.sh`
- Log: `/tmp/train_w3_strict_resume10_to20_20260704.log`
- Output checkpoint: `/tmp/qat_recipe1_runs/w3_strict_resume10_to20_20260704/checkpoint-20.pth.tar`
- Resume details: model state restored with `missing=0`, `unexpected=0`; optimizer restored with `433` state entries; lr scheduler restored with last LR `[1.0000011324882508e-05, 8.000009059906007e-05]`; RNG restored.
- Config: same W3 candidate, no QKR/QSS/CGA/refmodel, no augmentation, global batch 2048, `quant_lr_multiplier=4`, final epoch quant LR override `19:8`.
- Full ImageNet raw validation:
  - epoch10 continuation eval: Top-1 `77.1060`, Top-5 `93.5040`, loss `0.9519`
  - epoch11: Top-1 `77.3420`, Top-5 `93.4920`, loss `0.9492`
  - epoch12: Top-1 `77.2820`, Top-5 `93.4880`, loss `0.9485`
  - epoch13: Top-1 `77.3060`, Top-5 `93.5880`, loss `0.9496`
  - epoch14: Top-1 `77.2980`, Top-5 `93.5340`, loss `0.9476`
  - epoch15: Top-1 `77.3860`, Top-5 `93.5880`, loss `0.9495`
  - epoch16: Top-1 `77.3920`, Top-5 `93.5640`, loss `0.9472`
  - epoch17: Top-1 `77.3320`, Top-5 `93.6240`, loss `0.9444`
  - epoch18: Top-1 `77.4720`, Top-5 `93.6360`, loss `0.9462`
  - epoch20 boundary: Top-1 `77.4120`, Top-5 `93.5640`, loss `0.9443`, samples `50000`
- Conclusion: strict resume avoids the severe immediate regression of `--no-resume-opt`, but it remains on a low-LR plateau around `77.3-77.5` and does not improve materially beyond the original W3 checkpoint10. W3/QSC is stable but insufficient; the next candidate should change the training paradigm, not merely extend W3.

Missing evidence:

- no current QKR-enabled checkpoint has independently full-evaluated at Top-1 >=80
- no raw full ImageNet Top-1 >=80.0 checkpoint exists from a <=50-epoch run under the agreed budget
- no current QKR run has passed the user-requested 1-epoch >72 chain gate
- current QKR cold-start is broken/unequivalent: post-patch initial full-val Top-1 is `0.2460`
- historical evidence exists for a QKR 50-epoch lineage reaching Top-1 `80.6460`, but its `checkpoint-49` artifact is currently missing locally.

The next phase should not launch Candidate A as a blind cold-start run. First make QKR satisfy the 1-epoch >72 gate, or pivot the optimization search to the current non-QKR Recipe1 chain that already satisfies the gate. Historical QKR evidence should guide diagnosis, but should not override the failed current gate.
