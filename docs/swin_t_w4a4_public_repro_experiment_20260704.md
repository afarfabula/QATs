# Swin-T W4A4 public reproduction experiment

Date: 2026-07-04

Goal: start public-baseline reproduction with auditability. Public numbers are not local success. Local success requires a checkpoint at epoch <=50 with full ImageNet raw Top-1 >=80.0.

## Rules

- First choice is VVTQ / Quantization Variation because official README and log report Swin-T W4A4-family `Acc@1 82.424 Acc@5 96.026`.
- VVTQ caveat: official code keeps patch embedding and final head at 8-bit; main transformer blocks are W4A4. Do not call it strict all-layer W4A4.
- If VVTQ official dependencies are unavailable, use OFQ as the runnable 1-epoch gate because it is closest to the current QATs/OFQ code and reports Swin-T W4A4 `81.88`.
- Any training must first pass epoch1 full ImageNet raw Top-1 >72.
- Advance at most 10 epochs per run.
- Record exact command, data path, output path, checkpoint path, GPU setting, Top-1/Top-5/loss/samples, and gate status.

## Environment snapshot

| item | value |
|---|---|
| QATs repo | `/mlx_devbox/users/quyanyi/playground/QATs` |
| VVTQ repo | `/mlx_devbox/users/quyanyi/playground/Quantization-Variation` |
| VVTQ commit | `b9349f8` |
| Python | `3.11.2` |
| PyTorch | `2.9.1` |
| torchvision | `0.24.1+cu129` |
| timm | `0.4.12` |
| CUDA | available, 8 GPUs |
| GPU state | 8x H100 idle at preflight |
| OFQ parquet ImageNet | `/tmp/imagenet1k_full_parquet` |
| VVTQ ImageFolder candidate | `/tmp/qats/imagenet1k/imagefolder` |

## VVTQ dependency check

VVTQ official training path requires:

- ImageFolder-style ImageNet with `train/` and `val/`.
- FKD soft labels under a path such as `FKD_soft_label_500_crops_marginal_smoothing_k_5`.
- For each image, `ImageFolder_FKD` loads a corresponding `.tar` file from the soft-label tree.
- Optional official Swin-T W4A4 checkpoint for eval-only reproduction.

Current local status:

| dependency | status | evidence |
|---|---|---|
| VVTQ repo | found | `/mlx_devbox/users/quyanyi/playground/Quantization-Variation` |
| VVTQ checkpoint files | not found locally | no `.pth`, `.pth.tar`, `.pt`, or `.bin` under the VVTQ repo |
| FKD soft labels | not found locally | search found only `utils_FKD.py` and unrelated temp/cache paths |
| ImageFolder train/val | not found in candidate path | `/tmp/qats/imagenet1k/imagefolder` currently has no visible `train/` or `val/` subdirectories |
| parquet ImageNet | found | `/tmp/imagenet1k_full_parquet/data/*.parquet` |

Conclusion: official VVTQ 1-epoch training is currently blocked by missing FKD soft labels and missing ImageFolder train/val data. Running VVTQ without those would not reproduce the public recipe.

## Prepared scripts

| script | purpose | status |
|---|---|---|
| `tmp_scripts/vvtq_preflight_20260704.sh` | no-train VVTQ dependency/model-build check | prepared |
| `tmp_scripts/run_public_repro_ofq_qkr_1ep_20260704.sh` | OFQ public-baseline 1epoch full-val gate | prepared |

## Run log

### VVTQ no-train preflight

Script: `tmp_scripts/vvtq_preflight_20260704.sh`

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/vvtq_preflight_20260704.sh
```

Log: `/tmp/vvtq_preflight_20260704.log`

Result:

| check | result |
|---|---|
| script exit | 0 |
| model build | passed, `SwinTransformer` |
| CUDA | available, 8 GPUs |
| `data_train_exists` | `False` |
| `data_val_exists` | `False` |
| `softlabel_exists` | `False` |
| patch embed bits | `[8.0]` |
| head bits | `[8.0]` |
| first block attention q bits | `[4.0]` |

Interpretation:

- VVTQ official model code can be imported and built in this environment.
- VVTQ official first/last layer caveat is confirmed locally: patch embedding and classifier head are 8-bit, main attention q projection is 4-bit.
- VVTQ official training/eval is blocked locally because `/tmp/qats/imagenet1k/imagefolder` has no visible `train/` and `val/` subdirectories and the FKD soft-label directory is missing.
- Do not run VVTQ 1epoch training without obtaining ImageFolder data and FKD soft labels, because that would not reproduce the public recipe.

### OFQ 1epoch gate

Reason for fallback: VVTQ official 1epoch is blocked by missing ImageFolder and FKD soft-label dependencies. OFQ is the closest runnable public-baseline family in this workspace because it supports the existing parquet ImageNet dataset.

Prepared script: `tmp_scripts/run_public_repro_ofq_qkr_1ep_20260704.sh`

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_public_repro_ofq_qkr_1ep_20260704.sh
```

Log: `/tmp/train_ofq_qkr_public_1ep_20260704.log`

Output: `/tmp/qat_public_repro/ofq_qkr_public_1ep_20260704`

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved, 344144224 bytes |
| `last.pth.tar` | saved, 344144224 bytes |

Key setting:

| item | value |
|---|---|
| QATs commit | `549141c` |
| data | `/tmp/imagenet1k_full_parquet` |
| devices | `0,1,2,3,4,5,6,7` |
| world size | 8 |
| effective global batch | 512 |
| updates | 2496 |
| avg step time | 0.215870s |
| samples/sec | 2371.79 |
| QKR | enabled, `qk_reparam_type=0` |
| weight quant | W4 StatsQ, per-channel |
| activation quant | A4 LSQ, per-channel, learnable clip |
| KD | enabled, hard+soft, Swin-T teacher checkpoint |
| augmentation | smoothing 0.1, mixup 0.8, cutmix 1.0, randaugment, color jitter 0.4, reprob 0.25 |
| wall time | 702s |

Full validation result:

| checkpoint | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| `checkpoint-1.pth.tar` | 2.8952 | 38.2700 | 65.3000 | 50000 | failed `epoch1 >72` |

Interpretation:

- The run completed and saved a checkpoint, so the distributed training/eval mechanics are functional.
- The result is far below the required epoch1 gate. This run must not be continued to 10 epochs.
- This is not a VVTQ reproduction. It is an OFQ-family fallback because VVTQ official dependencies are missing locally.
- The failure suggests a major setting/initialization mismatch in the attempted OFQ public-style 1epoch recipe. The likely suspects are using OFQ augmentation/QKR from epoch0, QKR initialization behavior, or mismatch with the historical high-epoch/skip-validate OFQ recipe.
- Next action should be diagnosis or dependency completion, not longer training.

### QKR-only delta gate

Reason: OFQ public-style 1epoch failed with Top-1 38.27, but that run changed several variables at once relative to the previous non-QKR passing gate: QKR, strong augmentation, effective batch 512, and scheduler horizon. This run isolates QKR by matching the known non-QKR passing gate except for `--qk-reparam --qk-reparam-type 0`.

Prepared script: `tmp_scripts/run_public_repro_qkr_delta_noaug_1ep_20260704.sh`

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_public_repro_qkr_delta_noaug_1ep_20260704.sh
```

Log: `/tmp/train_qkr_delta_noaug_1ep_20260704.log`

Output: `/tmp/qat_public_repro/qkr_delta_noaug_1ep_20260704`

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| checkpoint | not saved |

Key setting:

| item | value |
|---|---|
| QATs commit | `549141c` |
| data | `/tmp/imagenet1k_full_parquet` |
| devices | `0,1,2,3,4,5,6,7` |
| world size | 8 |
| effective global batch | 2048 |
| QKR | enabled, `qk_reparam_type=0` |
| weight quant | W4 StatsQ, per-channel |
| activation quant | A4 LSQ, per-channel, learnable clip |
| KD | enabled, hard+soft, Swin-T teacher checkpoint |
| augmentation | disabled: smoothing 0.0, mixup 0.0, cutmix 0.0, aa none, color jitter 0.0, reprob 0.0 |

Failure evidence:

```text
Model swin_t created, param count:28608256
Effective batch alignment: per_gpu_effective_batch=256, loader_batch=256, accum=1, world_size=8, global_effective_batch=2048
Trainable parameter policy: epoch=0, quant_only=False, policy=all, trainable=28608256, frozen=0
Trainable parameter update policy: epoch=0, update=0, mode=requires_grad, policy=all, trainable=28608256, frozen=0
terminate called without an active exception
```

Result:

| checkpoint | Top-1 | Top-5 | gate |
|---|---:|---:|---|
| none | n/a | n/a | failed before first train step |

Interpretation:

- QKR-only delta did not reach the first optimizer update and did not save a checkpoint.
- This reproduces the earlier QKR gate failure pattern and isolates the problem to the QKR replacement/initialization/runtime path under the otherwise passing no-augmentation gate.
- The public-style OFQ run that reached Top-1 38.27 should not be interpreted as a usable QKR baseline. A QKR path that cannot pass the no-augmentation 1epoch gate is not eligible for 10epoch continuation.
- After the crash, the run left GPU memory occupied; the process group `2800687` was terminated with `kill -TERM -2800687`, and GPUs returned to idle.

Next action:

- Do not continue QKR training.
- Debug QKR construction on a smaller/no-train path: inspect pretrained qkv -> q/k/v copying, attention equivalence before quantization, and whether the QKR module has a batch-size/static-graph/runtime assertion that appears as `terminate called without an active exception`.
- Alternatively, complete VVTQ dependencies and reproduce the VVTQ official checkpoint/eval path instead of spending more compute on the broken QKR route.

### QKR no-train initialization diagnosis

Reason: the QKR-only delta crashed before the first train step. Before spending more GPU time on a gate run, inspect whether QKR wraps pretrained Swin attention in an initialization-preserving way.

Commands:

```bash
python3 tmp_scripts/debug_qkr_swin_attention_equivalence.py --layer features.1.0.attn --device cuda:0 --seed 42
python3 tmp_scripts/debug_qkr_param_count.py
```

Initial diagnosis before fix:

| check | value |
|---|---:|
| original q bias norm | 8.61012 |
| original k bias norm | 3.03446 |
| original v bias norm | 0.839848 |
| QKR relative position bias max diff | 6.75307 |
| QKR proj weight max diff | 0.333704 |
| QKR vs QAttention mean abs diff | 0.163854 |
| QKR vs QAttention rel L2 | 1.12514 |

Static code findings:

- `QAttention_swin_qkreparam` and `QAttention_swin_qkreparam_4_cga` reinitialized `relative_position_bias_table` and did not copy from the FP module.
- They wrapped `self.proj`, which was newly created by `super().__init__`, instead of wrapping pretrained `m.proj`.
- They discarded q/k bias by constructing `q` and `k` with `bias=False`, while the source Swin `m.qkv` has nonzero q/k/v bias.

Patch applied:

- Copy `m.relative_position_bias_table` after re-registering Swin relative position index.
- Wrap `m.proj` rather than the randomly initialized `self.proj`.
- Store q/k bias as non-trainable buffers and add the corresponding q/k bias terms to QKR attention logits.
- Apply the same initialization fixes to the CGA QKR wrapper.

No-train diagnosis after fix:

| check | value |
|---|---:|
| QKR relative position bias max diff | 0 |
| QKR proj weight max diff | 0 |
| QKR vs QAttention mean abs diff | 0.0318878 |
| QKR vs QAttention rel L2 | 0.23086 |

Interpretation:

- The deterministic pretrained-initialization bugs are fixed.
- QKR is still not numerically identical to the regular QAttention path because it quantizes the composed QK path differently; however, the remaining gap is much smaller than before.
- A QKR 1epoch gate should only be retried after this no-train fix, and still must pass full ImageNet Top-1 >72 before any 10epoch continuation.

### QKR fixed 1-update smoke

Reason: after fixing deterministic initialization bugs, run the smallest training smoke before any 1epoch retry.

Prepared script: `tmp_scripts/run_public_repro_qkr_fixed_smoke_1update_20260704.sh`

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_public_repro_qkr_fixed_smoke_1update_20260704.sh
```

Log: `/tmp/train_qkr_fixed_smoke_1update_20260704.log`

Key setting:

| item | value |
|---|---|
| effective global batch | 2048 |
| QKR | enabled |
| augmentation | disabled |
| max train updates | 1 |
| validation | skipped |

Result:

| checkpoint | status |
|---|---|
| none | failed before first optimizer update |

Failure evidence:

```text
Model swin_t created, param count:28608256
Effective batch alignment: per_gpu_effective_batch=256, loader_batch=256, accum=1, world_size=8, global_effective_batch=2048
Trainable parameter update policy: epoch=0, update=0, mode=requires_grad, policy=all, trainable=28608256, frozen=0
terminate called without an active exception
```

Interpretation:

- Initialization bugs were real and reduced QKR no-train mismatch, but they did not fix the batch256/global2048 runtime abort.
- The abort likely depends on QKR's large composed-QK intermediate tensors or a static-graph/runtime constraint at the large batch setting.
- After the crash, the run left GPU memory occupied; process group `2809087` was terminated and GPUs returned to idle.
- Next diagnostic is a batch64/global512 1-update smoke, because the earlier public-style QKR run with batch64 completed one full epoch even though its accuracy was poor.

### QKR fixed batch64 1-update smoke

Reason: batch256/global2048 still aborts after the initialization fix. Test whether the fixed QKR path can at least execute one optimizer update at batch64/global512, matching the batch size of the earlier public-style run that completed but scored poorly.

Prepared script: `tmp_scripts/run_public_repro_qkr_fixed_smoke_b64_1update_20260704.sh`

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_public_repro_qkr_fixed_smoke_b64_1update_20260704.sh
```

Log: `/tmp/train_qkr_fixed_smoke_b64_1update_20260704.log`

Output: `/tmp/qat_public_repro/qkr_fixed_smoke_b64_1update_20260704`

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved, 344187416 bytes |
| `last.pth.tar` | saved, 344187416 bytes |
| `step_checkpoints/step_0000.pth.tar` | saved, 114925307 bytes |
| `step_checkpoints/step_0001.pth.tar` | saved, 344180259 bytes |

Key setting:

| item | value |
|---|---|
| effective global batch | 512 |
| QKR | enabled |
| augmentation | disabled |
| max train updates | 1 |
| validation | skipped |

Result:

```text
Effective batch alignment: per_gpu_effective_batch=64, loader_batch=64, accum=1, world_size=8, global_effective_batch=512
Train: 0 [   0/2502 (  0%)]  Loss: 15.790224 (15.7902)
TrainSummary: epoch=0 updates=1 avg_step_time=1.596262s samples_per_step=512 samples_per_sec=320.75
Stopped early after 1 optimizer updates in epoch 0.
```

Interpretation:

- After the initialization fix, QKR can execute one optimizer update at batch64/global512.
- QKR still aborts at batch256/global2048, so the QKR runtime path is batch/memory sensitive.
- This does not satisfy the epoch1 accuracy gate because validation was intentionally skipped.
- The next eligible experiment, if approved, is a batch64/global512 full 1epoch QKR gate with no augmentation to test whether the initialization fix raises Top-1 above 72. It must not be continued beyond 1 epoch unless the gate passes.

### QKR fixed batch64 no-augmentation 1epoch gate

Reason: the fixed QKR path can execute one optimizer update at batch64/global512. Run the required full ImageNet 1epoch gate before considering any longer run.

Prepared script: `tmp_scripts/run_public_repro_qkr_fixed_b64_noaug_1ep_gate_20260704.sh`

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_public_repro_qkr_fixed_b64_noaug_1ep_gate_20260704.sh
```

Log: `/tmp/train_qkr_fixed_b64_noaug_1ep_gate_20260704.log`

Output: `/tmp/qat_public_repro/qkr_fixed_b64_noaug_1ep_gate_20260704`

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved, 344187416 bytes |
| `last.pth.tar` | saved, 344187416 bytes |

Key setting:

| item | value |
|---|---|
| QATs commit | `549141c` |
| data | `/tmp/imagenet1k_full_parquet` |
| devices | `0,1,2,3,4,5,6,7` |
| world size | 8 |
| effective global batch | 512 |
| updates | 2496 |
| avg step time | 0.223401s |
| samples/sec | 2291.85 |
| QKR | enabled, fixed initialization, `qk_reparam_type=0` |
| augmentation | disabled: smoothing 0.0, mixup 0.0, cutmix 0.0, aa none, color jitter 0.0, reprob 0.0 |
| validation | full ImageNet raw validation |
| wall time | 719s |

Full validation result:

| checkpoint | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| `checkpoint-1.pth.tar` | 0.8985 | 77.8440 | 94.0720 | 50000 | passed `epoch1 >72` |

Interpretation:

- The QKR initialization fix recovered the QKR path from runtime abort / unusable accuracy to a valid 1epoch gate.
- The fixed QKR path is now better than the previous non-QKR 1epoch gate (`73.3100`) under a different batch/update budget, but it is still below the final success target of Top-1 >=80.
- This run is eligible for a controlled continuation, but only under the project rule of at most 10 epochs per run and with full validation at the next gate.
- Because batch256/global2048 still aborts, any continuation should keep batch64/global512 unless the QKR memory/runtime issue is separately fixed.

Next action:

- Continue from `checkpoint-1.pth.tar` to epoch10 with the same fixed-QKR batch64/global512 no-augmentation setting.
- Keep full validation at epoch10.
- Do not continue past epoch10 unless the result is reviewed.

### QKR fixed batch64 no-augmentation strict resume to epoch10

Reason: the fixed-QKR 1epoch gate passed with Top-1 77.844. Under the project rule, advance at most 10 epochs and stop for review.

Prepared script: `tmp_scripts/run_public_repro_qkr_fixed_b64_noaug_resume1_to10_20260704.sh`

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_public_repro_qkr_fixed_b64_noaug_resume1_to10_20260704.sh
```

Log: `/tmp/train_qkr_fixed_b64_noaug_resume1_to10_20260704.log`

Output: `/tmp/qat_public_repro/qkr_fixed_b64_noaug_resume1_to10_20260704`

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-2.pth.tar` ... `checkpoint-10.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-10 |

Resume state:

```text
Strict resume: loaded model from /tmp/qat_public_repro/qkr_fixed_b64_noaug_1ep_gate_20260704/checkpoint-1.pth.tar; missing=0, unexpected=0
Strict resume: restored optimizer state entries=445
Strict resume: restored lr scheduler state={'base_lr': 0.0002, 'min_lr': 1e-05, 'warmup_updates': 0, 'total_updates': 2502, 'last_lr': [1.0002695322036744e-05]}
Strict resume: restored RNG state=True
```

Important caveat: this strict resume restored the scheduler from the 1epoch gate run, whose `total_updates` was only one epoch. As a result, epochs 2-10 ran at `lr=1e-5`. This run is therefore a strict-resume low-LR continuation, not an ideal 10epoch cosine schedule from epoch1 to epoch10.

Validation curve:

| checkpoint | loss | Top-1 | Top-5 | samples |
|---|---:|---:|---:|---:|
| checkpoint-1 | 0.8985 | 77.8440 | 94.0720 | 50000 |
| checkpoint-2 | 0.8971 | 77.9520 | 94.0140 | 50000 |
| checkpoint-3 | 0.8918 | 78.1720 | 94.0760 | 50000 |
| checkpoint-4 | 0.8874 | 78.2880 | 94.1760 | 50000 |
| checkpoint-5 | 0.8872 | 78.3360 | 94.1320 | 50000 |
| checkpoint-6 | 0.8859 | 78.2360 | 94.1120 | 50000 |
| checkpoint-7 | 0.8850 | 78.3540 | 94.2200 | 50000 |
| checkpoint-8 | 0.8825 | 78.4980 | 94.2500 | 50000 |
| checkpoint-9 | 0.8822 | 78.3960 | 94.2200 | 50000 |
| checkpoint-10 | 0.8820 | 78.5640 | 94.2180 | 50000 |

Result:

- The run completed without crash and saved checkpoints through epoch10.
- It did not reach the public-reproduction success target of Top-1 >=80.
- The trajectory improved from 77.844 to 78.564, but plateaued under the restored min-LR schedule.
- This confirms the fixed QKR path is functional at batch64/global512, but strict resume from a 1epoch scheduler is not the right 10epoch optimization schedule.

Next action:

- Do not continue this strict-resume run beyond epoch10.
- If continuing QKR, run a controlled `--no-resume-opt` or resume-opt-force-lr variant from checkpoint-1 with a fresh 10epoch scheduler and full validation at epoch10.
- Keep batch64/global512 unless the QKR batch256/global2048 abort is separately fixed.

## Worker restart data restore

Reason: worker restart refreshed `/tmp`, so the previous parquet dataset and experiment outputs under `/tmp` are no longer available. The new goal requires restoring data before any training.

Command:

```bash
nohup env DATA_ROOT=/tmp/imagenet1k_full_parquet IMG_ROOT=/tmp/imagenet1k_full_parquet/imagefolder PARQUET_ROOT=/tmp/imagenet1k_full_parquet USE_PARQUET_EXPORT=0 bash check_dataset.sh > logs/check_dataset_public_repro_20260704.log 2>&1 &
```

Log: `logs/check_dataset_public_repro_20260704.log`

Result:

```text
[QATs] missing HF token. Please export HF_TOKEN (or HF_HUB_TOKEN / HUGGINGFACE_HUB_TOKEN) after accepting ImageNet terms on Hugging Face.
```

Status:

- `/tmp/imagenet1k_full_parquet` is not restored.
- No training was launched after worker restart.
- The next step is to provide/export a valid HF token or mount/copy an existing ImageNet parquet cache, then rerun `check_dataset.sh`.

### Data restore retry with HF token

Reason: user provided a Hugging Face token after the first restore attempt failed. The token was used only for this restore and is not recorded here.

Command:

```bash
DATA_ROOT=/tmp/imagenet1k_full_parquet IMG_ROOT=/tmp/imagenet1k_full_parquet/imagefolder PARQUET_ROOT=/tmp/imagenet1k_full_parquet USE_PARQUET_EXPORT=0 bash check_dataset.sh
```

Log: `logs/check_dataset_public_repro_20260704.log`

Result:

| item | value |
|---|---:|
| train shards | 294 |
| validation shards | 14 |
| missing train shards | 0 |
| missing validation shards | 0 |
| dataset size | 143G |

Status: data restore gate passed. `/tmp/imagenet1k_full_parquet/data/train-*.parquet` and `validation-*.parquet` are available.

### Fixed-QKR 1epoch gate after worker restart

Reason: rerun the known fixed-QKR 1epoch gate after restoring `/tmp`.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_public_repro_qkr_fixed_b64_noaug_1ep_gate_20260704.sh
```

Log: `/tmp/train_qkr_fixed_b64_noaug_1ep_gate_20260704.log`

Output: `/tmp/qat_public_repro/qkr_fixed_b64_noaug_1ep_gate_20260704`

Result:

| checkpoint | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| `checkpoint-1.pth.tar` | 0.8985 | 77.8440 | 94.0720 | 50000 | passed `epoch1 >77` |

Status: fixed-QKR 1epoch gate was reproduced after worker restart. It is eligible for at most a 10epoch continuation.

### Single-recipe 2epoch A: fixed-QKR no-aug scheduler2

Reason: first strict test under the new constraint: one continuous 2epoch recipe, no staging, no resume, full validation after epoch1, stop if epoch1 Top-1 <=77.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_a_fixed_qkr_noaug_20260704.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
output: /tmp/qat_public_repro/recipe2ep_a_fixed_qkr_noaug_20260704
log: /tmp/train_recipe2ep_a_fixed_qkr_noaug_20260704.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 2
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
weight_decay: 0.0
recipe: fixed-QKR, W4A4, statsq weights, LSQ activations, KD hard+soft, no augmentation
```

Data gate:

| item | value |
|---|---:|
| train shards | 294 |
| validation shards | 14 |

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9986 | 75.2760 | 92.9120 | 50000 | failed `epoch1 >77` |

Status:

- The run was stopped after epoch1 gate failure.
- No valid epoch2 result is accepted for this recipe.
- The likely cause is scheduler shape, not QKR init or data: the successful 1epoch gate ended epoch1 at LR `1.019e-05`, while this 2epoch cosine schedule ended epoch1 still at LR about `1.080e-04`; the model was under-converged for validation.
- Next recipe should keep the same fixed-QKR/no-aug functional path but restore the validated 1epoch LR decay shape while still running as one continuous 2epoch recipe.

### Single-recipe 2epoch B: fixed-QKR no-aug scheduler1

Reason: restore the validated 1epoch LR decay shape while still running a single continuous 2epoch recipe from pretrained. This tests whether the second epoch can improve the passed epoch1 checkpoint without using resume or staged training.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_b_fixed_qkr_noaug_scheduler1_20260704.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
output: /tmp/qat_public_repro/recipe2ep_b_fixed_qkr_noaug_scheduler1_20260704
log: /tmp/train_recipe2ep_b_fixed_qkr_noaug_scheduler1_20260704.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 1
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
weight_decay: 0.0
recipe: fixed-QKR, W4A4, statsq weights, LSQ activations, KD hard+soft, no augmentation
```

Data gate:

| item | value |
|---|---:|
| train shards | 294 |
| validation shards | 14 |

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.8985 | 77.8440 | 94.0720 | 50000 | passed `epoch1 >77` |
| 2 | 0.8978 | 77.9440 | 94.0040 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Status:

- B passed the epoch1 gate exactly matching the reproduced fixed-QKR 1epoch baseline.
- B did not reach the 2epoch success target. The second epoch improved Top-1 by only `+0.1000`.
- The failure mode is clear from LR: with `scheduler_epochs=1`, the second epoch runs at fixed `1e-5`, which is too conservative for a 2epoch target.
- Next recipe should keep the single continuous recipe and the fast first-epoch decay, but raise `min_lr` modestly so epoch2 has more effective learning without destroying the epoch1 gate.

### Single-recipe 2epoch C: fixed-QKR no-aug scheduler1 minlr3e-5

Reason: keep B's single continuous 2epoch structure but raise the global `min_lr` from `1e-5` to `3e-5`, so the second epoch has more effective learning rate while the first epoch still uses a fast 1epoch cosine decay.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_c_fixed_qkr_noaug_scheduler1_minlr3e5_20260704.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
output: /tmp/qat_public_repro/recipe2ep_c_fixed_qkr_noaug_scheduler1_minlr3e5_20260704
log: /tmp/train_recipe2ep_c_fixed_qkr_noaug_scheduler1_minlr3e5_20260704.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 1
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 3e-5
weight_decay: 0.0
recipe: fixed-QKR, W4A4, statsq weights, LSQ activations, KD hard+soft, no augmentation
```

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9113 | 77.4680 | 93.8940 | 50000 | passed `epoch1 >77` |
| 2 | 0.9125 | 77.6320 | 93.8500 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Status:

- C passed the epoch1 gate, but raising the global min LR hurt epoch1 by `-0.3760` vs B and ended below B at epoch2.
- This suggests global LR is not the right knob: base model updates at higher floor damage the already-good first epoch more than they help second-epoch quant adaptation.
- Next recipe should protect the base trajectory and give more capacity specifically to quant/shift parameters, for example via `quant_lr_multiplier`, while keeping the same single continuous 2epoch structure.

### Single-recipe 2epoch D: fixed-QKR no-aug scheduler1 quantlr2

Reason: protect the base-parameter LR trajectory from B while giving quant/shift parameters 2x LR. This tests whether targeted quant adaptation can improve both the epoch1 gate and epoch2 without raising global LR.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_d_fixed_qkr_noaug_scheduler1_quantlr2_20260704.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
output: /tmp/qat_public_repro/recipe2ep_d_fixed_qkr_noaug_scheduler1_quantlr2_20260704
log: /tmp/train_recipe2ep_d_fixed_qkr_noaug_scheduler1_quantlr2_20260704.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 1
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 2
weight_decay: 0.0
recipe: fixed-QKR, W4A4, statsq weights, LSQ activations, KD hard+soft, no augmentation
```

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.8897 | 77.9860 | 94.1160 | 50000 | passed `epoch1 >77` |
| 2 | 0.8860 | 78.0940 | 94.1540 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Status:

- D is the best 2epoch result so far: epoch2 Top-1 `78.0940`.
- Quant/shift parameter LR targeting improved both epoch1 and epoch2 versus B, unlike raising global `min_lr`.
- Still short of the 80 target by `1.9060` Top-1.
- Next recipe should continue the parameter-group direction with a stronger quant LR multiplier, while preserving the base schedule and the no-augmentation fixed-QKR setup.

### Single-recipe 2epoch E: fixed-QKR no-aug scheduler1 quantlr4

Reason: continue D's successful parameter-group direction by increasing quant/shift LR multiplier from 2 to 4, while keeping the same base schedule and one continuous 2epoch run.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_e_fixed_qkr_noaug_scheduler1_quantlr4_20260704.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
output: /tmp/qat_public_repro/recipe2ep_e_fixed_qkr_noaug_scheduler1_quantlr4_20260704
log: /tmp/train_recipe2ep_e_fixed_qkr_noaug_scheduler1_quantlr4_20260704.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 1
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
weight_decay: 0.0
recipe: fixed-QKR, W4A4, statsq weights, LSQ activations, KD hard+soft, no augmentation
```

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.8862 | 78.1660 | 94.0540 | 50000 | passed `epoch1 >77` |
| 2 | 0.8804 | 78.3680 | 94.1680 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Status:

- E is the best 2epoch result so far: epoch2 Top-1 `78.3680`.
- The parameter-group trend is monotonic so far in this fixed-QKR setting: B `77.9440`, D/x2 `78.0940`, E/x4 `78.3680`.
- Still short of the 80 target by `1.6320` Top-1.
- A stronger quant LR multiplier is the next direct test, but older non-fixed-QKR Recipe1 notes showed x8 can slightly hurt epoch0 versus x4, so the epoch1 gate must be enforced strictly.

### Single-recipe 2epoch F: fixed-QKR no-aug scheduler1 quantlr8

Reason: test whether stronger quant/shift adaptation continues the E trend by raising `quant_lr_multiplier` from 4 to 8, while keeping the same single continuous 2epoch fixed-QKR recipe.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_f_fixed_qkr_noaug_scheduler1_quantlr8_20260704.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
output: /tmp/qat_public_repro/recipe2ep_f_fixed_qkr_noaug_scheduler1_quantlr8_20260704
log: /tmp/train_recipe2ep_f_fixed_qkr_noaug_scheduler1_quantlr8_20260704.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 1
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 8
weight_decay: 0.0
recipe: fixed-QKR, W4A4, statsq weights, LSQ activations, KD hard+soft, no augmentation
```

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.8873 | 78.1120 | 94.0260 | 50000 | passed `epoch1 >77` |
| 2 | 0.8823 | 78.3280 | 94.1220 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Status:

- F passed the epoch1 gate, but underperformed E at both epoch1 and epoch2.
- Best current 2epoch result remains E: Top-1 `78.3680`.
- The useful quant LR range appears centered around x4 in this fixed-QKR setting; x8 is already too aggressive.
- Next recipe should do a fine search around x4, such as x5 or x6, or combine x4 with a non-LR single-recipe improvement. Do not continue increasing quant LR blindly.

### Single-recipe 2epoch G: fixed-QKR no-aug scheduler1 quantlr6

Reason: fine-search the quant/shift LR multiplier between E/x4 and F/x8 while keeping the same single continuous 2epoch recipe.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_g_fixed_qkr_noaug_scheduler1_quantlr6_20260704.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
output: /tmp/qat_public_repro/recipe2ep_g_fixed_qkr_noaug_scheduler1_quantlr6_20260704
log: /tmp/train_recipe2ep_g_fixed_qkr_noaug_scheduler1_quantlr6_20260704.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 1
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 6
weight_decay: 0.0
recipe: fixed-QKR, W4A4, statsq weights, LSQ activations, KD hard+soft, no augmentation
```

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.8873 | 78.1040 | 94.0100 | 50000 | passed `epoch1 >77` |
| 2 | 0.8792 | 78.1980 | 94.2000 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Status:

- G passed the epoch1 gate, but underperformed E/x4 at both epoch1 and epoch2.
- Best current 2epoch result remains E/x4: Top-1 `78.3680`.
- Quant LR multiplier sweep summary: x2 `78.0940`, x4 `78.3680`, x6 `78.1980`, x8 `78.3280`. The peak is x4, so further blind multiplier increases are not justified.
- Next recipe should keep quant LR multiplier x4 and add a single-recipe non-LR improvement, such as a fixed hard-label auxiliary or confidence-weighted KD, while preserving the no-staging constraint.

### Single-recipe 2epoch H plan: fixed-QKR no-aug scheduler1 quantlr4 EMA

Reason: keep the current best raw training recipe E/x4 and add student weight EMA as part of the same continuous 2epoch recipe. EMA is not a staged switch and does not change the loss schedule; it may provide a smoother candidate checkpoint. The unified runner validates the raw student during training and saves `.ema` checkpoints, but it does not automatically validate EMA, so any EMA claim must be backed by an extra full ImageNet eval.

Planned gate:

- Raw in-training epoch1 full-val must still pass `Top-1 >77`.
- If raw epoch1 fails, stop and do not use EMA as a loophole.
- If raw epoch1 passes, finish epoch2 and evaluate raw full-val from training logs.
- If raw epoch2 is below 80, run a separate full ImageNet validation on `checkpoint-2.ema.pth.tar` before judging whether EMA is useful.

### Single-recipe 2epoch H: fixed-QKR no-aug scheduler1 quantlr4 EMA

Reason: test whether student weight EMA can improve the current best E/x4 recipe without changing the raw training schedule. This remains one continuous 2epoch recipe; EMA is maintained throughout training and saved as sidecar checkpoints.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_h_fixed_qkr_noaug_scheduler1_quantlr4_ema_20260704.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
output: /tmp/qat_public_repro/recipe2ep_h_fixed_qkr_noaug_scheduler1_quantlr4_ema_20260704
log: /tmp/train_recipe2ep_h_fixed_qkr_noaug_scheduler1_quantlr4_ema_20260704.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 1
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
model_ema: true
model_ema_decay: 0.9998
weight_decay: 0.0
recipe: fixed-QKR, W4A4, statsq weights, LSQ activations, KD hard+soft, no augmentation
```

Raw in-training result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.8862 | 78.1660 | 94.0540 | 50000 | passed `epoch1 >77` |
| 2 | 0.8804 | 78.3680 | 94.1680 | 50000 | failed `epoch2 >=80` |

EMA eval:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python3 /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py \
  --method ofq --stage train \
  --config /mlx_devbox/users/quyanyi/playground/QATs/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml \
  --model swin_t --data /tmp/imagenet1k_full_parquet --dataset-format parquet \
  --output /tmp/qat_public_repro --experiment eval_recipe2ep_h_fixed_qkr_noaug_scheduler1_quantlr4_ema_ckpt2_extra_initial_20260704 \
  --devices 0,1,2,3,4,5,6,7 --nproc-per-node 8 --master-port 30447 --model-type swin \
  --teacher swin_t --teacher-type swin --teacher-pretrained \
  --epochs 2 --batch-size 64 --workers 8 \
  --warmup-lr 1e-6 --weight-decay 0.0 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 1 \
  --quantized --qk-reparam --qk-reparam-type 0 \
  --extra-arg=--eval-only \
  --extra-arg=--initial-checkpoint --extra-arg=/tmp/qat_public_repro/recipe2ep_h_fixed_qkr_noaug_scheduler1_quantlr4_ema_20260704/checkpoint-2.ema.pth.tar \
  --extra-arg=--cooldown-epochs --extra-arg=0 \
  --extra-arg=--log-interval --extra-arg=50
```

EMA eval result:

| checkpoint | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| `checkpoint-2.ema.pth.tar` | 1.4295 | 66.9640 | 87.2160 | 50000 | failed |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-1.ema.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `checkpoint-2.ema.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |
| `last.ema.pth.tar` | saved, same epoch as checkpoint-2 EMA |

Status:

- H raw result exactly matches E/x4: Top-1 `78.3680`.
- EMA with decay `0.9998` is much worse in this 2epoch short-run setting, likely because the EMA is too stale relative to rapid quantizer adaptation.
- Best current 2epoch result remains E/x4 or H raw: Top-1 `78.3680`.
- Do not use this EMA decay as a success path. If EMA is revisited, it needs a much lower decay, but this is unlikely to close a 1.632 Top-1 gap by itself.

### Single-recipe 2epoch I: fixed-QKR no-aug batch32 scheduler1 quantlr4

Reason: keep the best E/x4 recipe but reduce per-GPU batch size from 64 to 32, increasing optimizer updates per epoch from 2502 to 5004. This is still a single continuous 2epoch recipe with one fixed configuration, not staged training.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_i_fixed_qkr_noaug_b32_scheduler1_quantlr4_20260704.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
output: /tmp/qat_public_repro/recipe2ep_i_fixed_qkr_noaug_b32_scheduler1_quantlr4_20260704
log: /tmp/train_recipe2ep_i_fixed_qkr_noaug_b32_scheduler1_quantlr4_20260704.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 1
batch_size: 32 per GPU, global batch 256
updates_per_epoch: 5004
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
weight_decay: 0.0
recipe: fixed-QKR, W4A4, statsq weights, LSQ activations, KD hard+soft, no augmentation
```

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.8986 | 77.7980 | 93.9280 | 50000 | passed `epoch1 >77` |
| 2 | 0.8896 | 77.8720 | 94.1240 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Status:

- I passed the epoch1 gate, but underperformed E/H raw at both epoch1 and epoch2.
- More optimizer updates via smaller global batch did not improve this 2epoch setting; likely the added stochasticity from global batch 256 hurts validation more than the extra updates help.
- Best current 2epoch result remains E/x4 or H raw: Top-1 `78.3680`.
- Do not continue batch-size reduction as the main path.

### Single-recipe 2epoch J: fixed-QKR no-aug scheduler1 quantlr4 setupalpha4

Reason: keep the best E/x4 recipe but increase initial LSQ/setup-alpha calibration from 1 batch to 4 batches. This is a single fixed recipe; the calibration setting is constant and occurs before the same continuous 2epoch training.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_j_fixed_qkr_noaug_scheduler1_quantlr4_setupalpha4_20260704.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
output: /tmp/qat_public_repro/recipe2ep_j_fixed_qkr_noaug_scheduler1_quantlr4_setupalpha4_20260704
log: /tmp/train_recipe2ep_j_fixed_qkr_noaug_scheduler1_quantlr4_setupalpha4_20260704.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 1
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
setup_alpha_batches: 4
weight_decay: 0.0
recipe: fixed-QKR, W4A4, statsq weights, LSQ activations, KD hard+soft, no augmentation
```

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.8888 | 78.0440 | 94.0340 | 50000 | passed `epoch1 >77` |
| 2 | 0.8815 | 78.1860 | 94.1240 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Status:

- J passed the epoch1 gate, but underperformed E/H raw at both epoch1 and epoch2.
- Increasing setup-alpha calibration batches to 4 is not useful in this fixed-QKR 2epoch setting.
- Best current 2epoch result remains E/x4 or H raw: Top-1 `78.3680`.
- Do not continue setup-alpha batch sweeps as the main path.

### Single-recipe 2epoch K: fixed-QKR no-aug scheduler1 quantlr4 featnorm005

Reason: keep the best E/x4 recipe and add a weak normalized FP-teacher feature-output objective on late Swin stages. This tests a VVTQ/Q-ViT/APHQ-style representation alignment signal without staging; the feature loss is active from the first update through both epochs.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_k_fixed_qkr_noaug_scheduler1_quantlr4_featnorm005_20260704.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
output: /tmp/qat_public_repro/recipe2ep_k_fixed_qkr_noaug_scheduler1_quantlr4_featnorm005_20260704
log: /tmp/train_recipe2ep_k_fixed_qkr_noaug_scheduler1_quantlr4_featnorm005_20260704.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 1
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
teacher_feature_output_weight: 0.005
teacher_feature_output_layers: features.5.5,features.7.1
teacher_feature_output_loss: norm_mse
weight_decay: 0.0
recipe: fixed-QKR, W4A4, statsq weights, LSQ activations, KD hard+soft, no augmentation
```

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.8853 | 78.0900 | 94.0960 | 50000 | passed `epoch1 >77` |
| 2 | 0.8810 | 78.3580 | 94.2080 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Status:

- K passed the epoch1 gate and nearly matched E/H raw at epoch2, but did not improve it.
- Normalized teacher feature-output alignment is stable but too weak to close the remaining gap.
- Best current 2epoch result remains E/x4 or H raw: Top-1 `78.3680`.
- If this branch is revisited, a slightly higher feature weight may be tested, but prior records suggest feature alignment alone is unlikely to deliver the missing `1.632` Top-1.

### Single-recipe 2epoch L plan: fixed-QKR no-aug scheduler1 quantlr4 minlr2e-5

Reason: E/x4 is still best, while raising global `min_lr` all the way to `3e-5` hurt in C. L tests a milder LR floor, `2e-5`, to give the second epoch more effective learning than E without as much first-epoch damage as C. This is one fixed 2epoch scheduler, not a staged LR switch.

### Single-recipe 2epoch L: fixed-QKR no-aug scheduler1 quantlr4 minlr2e-5

Reason: execute the L plan: keep E/x4 and raise global `min_lr` from `1e-5` to `2e-5`.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_l_fixed_qkr_noaug_scheduler1_quantlr4_minlr2e5_20260704.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
output: /tmp/qat_public_repro/recipe2ep_l_fixed_qkr_noaug_scheduler1_quantlr4_minlr2e5_20260704
log: /tmp/train_recipe2ep_l_fixed_qkr_noaug_scheduler1_quantlr4_minlr2e5_20260704.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 1
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 2e-5
quant_lr_multiplier: 4
weight_decay: 0.0
recipe: fixed-QKR, W4A4, statsq weights, LSQ activations, KD hard+soft, no augmentation
```

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.8877 | 78.0600 | 94.0840 | 50000 | passed `epoch1 >77` |
| 2 | 0.8868 | 78.2640 | 94.1120 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Status:

- L passed the epoch1 gate, but underperformed E/H raw at both epoch1 and epoch2.
- A mild global min-LR increase is still harmful relative to E.
- Best current 2epoch result remains E/x4 or H raw: Top-1 `78.3680`.
- Do not continue min-LR sweeps as the main path.

### Single-recipe 2epoch M: fixed-QKR no-aug scheduler1 quantlr4 binreg1e-4

Reason: implement and test a VVTQ-inspired bin regularizer. The regularizer penalizes quantized-weight reconstruction error plus within-bin FP weight variance for quantized weight modules. It is fixed for the entire run, so it is compatible with the single-recipe constraint.

Implementation:

- Added `--bin-reg-weight` and `--bin-reg-variance-weight` to `qat_launch.py`.
- Default is disabled (`bin_reg_weight=0.0`), so existing recipes are unchanged.
- Smoke-tested with one optimizer update before launching the full run.

Smoke command:

```bash
PYTHONUNBUFFERED=1 python3 /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py \
  --method ofq --stage train \
  --config /mlx_devbox/users/quyanyi/playground/QATs/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml \
  --model swin_t --data /tmp/imagenet1k_full_parquet --dataset-format parquet \
  --output /tmp/qat_public_repro --experiment smoke_binreg_1update_v2_20260704 \
  --devices 0,1,2,3,4,5,6,7 --nproc-per-node 8 --master-port 30452 --model-type swin \
  --teacher swin_t --teacher-type swin --teacher-checkpoint /home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth --teacher-pretrained \
  --epochs 1 --scheduler-epochs 1 --batch-size 64 --workers 8 \
  --lr 2e-4 --min-lr 1e-5 --weight-decay 0.0 \
  --quant-lr-multiplier 4 \
  --bin-reg-weight 1e-4 --bin-reg-variance-weight 1.0 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 1 \
  --quantized --qk-reparam --qk-reparam-type 0 \
  --amp --amp-dtype bf16 \
  --extra-arg=--static-graph \
  --extra-arg=--smoothing --extra-arg=0.0 \
  --extra-arg=--mixup --extra-arg=0.0 \
  --extra-arg=--cutmix --extra-arg=0.0 \
  --extra-arg=--aa --extra-arg=none \
  --extra-arg=--color-jitter --extra-arg=0.0 \
  --extra-arg=--reprob --extra-arg=0.0 \
  --extra-arg=--max_train_updates --extra-arg=1 \
  --extra-arg=--skip_validate \
  --extra-arg=--log-interval --extra-arg=1 \
  --extra-arg=--seed --extra-arg=42
```

Smoke result:

```text
Enabled bin regularizer: weight=0.0001, variance_weight=1.0, pairs=77
Train: 0 [0/2502] ... BinReg: 8.281e-02 ...
Stopped early after 1 optimizer updates in epoch 0.
```

Full run command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_m_fixed_qkr_noaug_scheduler1_quantlr4_binreg1e4_20260704.sh
```

Status:

- Full run was intentionally stopped at epoch0 28% for efficiency.
- Observed step time was about `0.756s`, versus about `0.223s` for E/x4. This makes the full 2epoch run roughly 3.4x slower.
- The unweighted `BinReg` value was about `8.4e-02`; with `bin_reg_weight=1e-4`, the loss contribution is only about `8e-06`, too small to plausibly close the `1.632` Top-1 gap.
- Increasing the weight enough to matter would likely be much more invasive and still carry the same high compute overhead.
- No epoch1 or epoch2 validation result is accepted for M because the run was stopped before epoch1. It does not count as a failed gate; it is an efficiency/instrumentation rejection.
- Do not continue this bin-regularizer implementation as the main path unless it is reimplemented with a cheaper cached/vectorized approximation.

### Single-recipe 2epoch N: fixed-QKR no-aug scheduler1 quantlr4 prev-step dynamic attention KL

Reason: keep the best E/x4 recipe and add the strongest previously observed stabilization family: prev-step refmodel attention KL with dynamic custom-top5 head selection inside the oscillating top10 pool. This is active for the whole 2epoch run, so it is a single fixed recipe rather than a staged schedule.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_n_fixed_qkr_noaug_scheduler1_quantlr4_prevkl_20260704.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
output: /tmp/qat_public_repro/recipe2ep_n_fixed_qkr_noaug_scheduler1_quantlr4_prevkl_20260704
log: /tmp/train_recipe2ep_n_fixed_qkr_noaug_scheduler1_quantlr4_prevkl_20260704.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 1
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
train_scheme: ema_ref_attn_kl
ref_update: prev_step
ref_attn_kl_weight: 1e-4
ref_attn_loss: kl_ref
ref_head_mode: dynamic_custom_top5:oscillating_top10
weight_decay: 0.0
recipe: fixed-QKR, W4A4, statsq weights, LSQ activations, KD hard+soft, no augmentation
```

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.8900 | 78.0160 | 94.0380 | 50000 | passed `epoch1 >77` |
| 2 | 0.8813 | 78.3500 | 94.2460 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Status:

- N passed the epoch1 gate, but did not beat E/H raw.
- Prev-step attention KL is stable and improves Top-5, but it does not close the Top-1 gap in this 2epoch from-scratch fixed-QKR setting.
- Best current 2epoch result remains E/x4 or H raw: Top-1 `78.3680`.
- Do not continue this exact KL recipe as the main path; if attention stabilization is revisited, it needs a materially different formulation or interaction with stronger supervision.

### Implementation update: epoch1 automatic Top-1 gate

Reason: the current goal requires every 2epoch recipe to stop after epoch1 if full ImageNet raw validation Top-1 is not above 77.0. Manual log watching is error-prone, so the launcher now has an explicit gate.

Code change:

- Added `--epoch1-acc-gate` to `qat_launch.py`.
- The default is `0.0`, so existing recipes are unchanged.
- When enabled, after epoch0 validation the launcher prints `Epoch1AccGate: ...`.
- If `Acc@1 <= threshold`, `stopped_early=True` and the run exits before epoch2.

Verification:

```bash
python3 -m py_compile /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py
```

Dataset check before the next run:

| path | train shards | validation shards |
|---|---:|---:|
| `/tmp/imagenet1k_full_parquet/data` | 294 | 14 |

### First/last layer bit policy check

Reason: VVTQ's public Swin-T W4A4-family result keeps patch embedding and final head at 8-bit, so I checked whether our local OFQ Swin path was accidentally stricter.

Finding:

- `/mlx_devbox/users/quyanyi/playground/QATs/third_party/OFQ/src/quantization/modules/utils.py` already special-cases Swin `features.0.0` and `head`.
- Both QKR and non-QKR Swin replacement paths instantiate those two modules with `weight_bits=8` and `input_bits=8`.
- Therefore the current fixed-QKR W4A4 setting already has VVTQ-style first/last 8-bit. The gap to 80+ is not explained by first/last layers being forced to 4-bit.

### Single-recipe 2epoch O: fixed-QKR no-aug scheduler1 quantlr4 pure soft-KD

Reason: VVTQ relies heavily on soft-label supervision. O keeps the current best E/x4 recipe and changes only the KD criterion from hard+soft (`kd_hard_and_soft=1`) to pure teacher soft KD (`kd_hard_and_soft=0`). This is one continuous 2epoch recipe with no staging.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_o_fixed_qkr_noaug_scheduler1_quantlr4_softkd_20260704.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe2ep_o_fixed_qkr_noaug_scheduler1_quantlr4_softkd_20260704
log: /tmp/train_recipe2ep_o_fixed_qkr_noaug_scheduler1_quantlr4_softkd_20260704.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 1
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
epoch1_acc_gate: 77.0
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations, no augmentation
```

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.8862 | 78.1660 | 94.0540 | 50000 | passed `epoch1 >77`; `Epoch1AccGate: passed` |
| 2 | 0.8804 | 78.3680 | 94.1680 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223011s | 2295.85 |
| 2 | 2496 | 0.222813s | 2297.89 |

Status:

- O passed the epoch1 gate but exactly matched E/H raw: epoch1 `78.1660`, epoch2 `78.3680`.
- Post-run audit found that `args.yaml` saved `kd_hard_and_soft: 1`, not `0`.
- Root cause: `build_ofq_runtime_config` had a fallback that changed `kd_hard_and_soft==0` back to `1` whenever `--use-kd` was set.
- Therefore O is not a valid pure-soft-KD experiment; it is an accidental rerun of E/H raw with the new epoch1 gate enabled.
- Best current 2epoch result remains E/H/O raw: Top-1 `78.3680`.

### Bugfix: preserve explicit `kd_hard_and_soft=0`

Reason: recipe O was intended to test pure teacher soft KD, but the runtime config silently rewrote explicit `0` to `1`.

Fix:

- Changed the fallback so it only rewrites `kd_hard_and_soft` when the user did not explicitly pass `--kd-hard-and-soft`.
- This preserves explicit `--kd-hard-and-soft 0`.

Verification:

```bash
python3 -m py_compile /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py
```

Smoke command:

```bash
PYTHONUNBUFFERED=1 python3 /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py \
  --method ofq --stage train \
  --config /mlx_devbox/users/quyanyi/playground/QATs/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml \
  --model swin_t --data /tmp/imagenet1k_full_parquet --dataset-format parquet \
  --output /tmp/qat_public_repro --experiment smoke_true_softkd_1update_20260704 \
  --devices 0,1,2,3,4,5,6,7 --nproc-per-node 8 --master-port 30461 --model-type swin \
  --teacher swin_t --teacher-type swin --teacher-checkpoint /home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth --teacher-pretrained \
  --epochs 1 --scheduler-epochs 1 --batch-size 64 --workers 8 \
  --lr 2e-4 --min-lr 1e-5 --weight-decay 0.0 \
  --quant-lr-multiplier 4 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 0 \
  --quantized --qk-reparam --qk-reparam-type 0 \
  --amp --amp-dtype bf16 \
  --extra-arg=--static-graph \
  --extra-arg=--smoothing --extra-arg=0.0 \
  --extra-arg=--mixup --extra-arg=0.0 \
  --extra-arg=--cutmix --extra-arg=0.0 \
  --extra-arg=--aa --extra-arg=none \
  --extra-arg=--color-jitter --extra-arg=0.0 \
  --extra-arg=--reprob --extra-arg=0.0 \
  --extra-arg=--max_train_updates --extra-arg=1 \
  --extra-arg=--skip_validate \
  --extra-arg=--log-interval --extra-arg=1 \
  --extra-arg=--seed --extra-arg=42
```

Smoke result:

| item | value |
|---|---|
| `args.yaml kd_hard_and_soft` | `0` |
| first update loss | `7.907262` |
| E/O accidental first update loss | `15.790224` |

Status:

- The bugfix is effective.
- True pure soft-KD changes the optimization objective and is now eligible for a real 2epoch recipe.

### Single-recipe 2epoch P: fixed-QKR no-aug scheduler1 quantlr4 true pure soft-KD

Reason: after fixing the explicit `kd_hard_and_soft=0` override bug, rerun the intended pure teacher soft-KD recipe. This keeps the E/x4 training structure but removes hard-label CE from the KD loss. It is a single continuous 2epoch recipe with no staging.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_p_fixed_qkr_noaug_scheduler1_quantlr4_true_softkd_20260704.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe2ep_p_fixed_qkr_noaug_scheduler1_quantlr4_true_softkd_20260704
log: /tmp/train_recipe2ep_p_fixed_qkr_noaug_scheduler1_quantlr4_true_softkd_20260704.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 1
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
epoch1_acc_gate: 77.0
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations, no augmentation
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml kd_hard_and_soft` | `0` |
| first update loss | `7.907262` |
| E/O accidental first update loss | `15.790224` |

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9138 | 78.4840 | 94.4520 | 50000 | passed `epoch1 >77`; `Epoch1AccGate: passed` |
| 2 | 0.9049 | 78.8400 | 94.5540 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.222783s | 2298.20 |
| 2 | 2496 | 0.222839s | 2297.62 |

Status:

- P is the best valid 2epoch single recipe so far: Top-1 `78.8400`.
- It improves over E/O accidental hard+soft by `+0.4720` Top-1 at epoch2 and `+0.3180` at epoch1.
- The improvement confirms the useful direction is the soft teacher objective, not hard-label CE.
- It still misses the 80.0 target by `1.1600` Top-1.
- Next recipe should keep true soft-KD and change one additional continuous knob, most likely temperature, confidence weighting, or quant LR around the new optimum.

### Single-recipe 2epoch Q: fixed-QKR no-aug scheduler1 quantlr4 true soft-KD T=2

Reason: P showed that true pure teacher soft-KD is the first useful direction under the 2epoch single-recipe constraint. Q keeps P and changes one fixed continuous knob: teacher soft-KD temperature from `1.0` to `2.0`.

Smoke verification:

| item | value |
|---|---|
| smoke output | `/tmp/qat_public_repro/smoke_true_softkd_temp2_1update_20260705` |
| smoke log | `/tmp/smoke_true_softkd_temp2_1update_20260705.log` |
| first update loss | `28.740299` |
| P/T=1 first update loss | `7.907262` |

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_q_fixed_qkr_noaug_scheduler1_quantlr4_true_softkd_temp2_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe2ep_q_fixed_qkr_noaug_scheduler1_quantlr4_true_softkd_temp2_20260705
log: /tmp/train_recipe2ep_q_fixed_qkr_noaug_scheduler1_quantlr4_true_softkd_temp2_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 1
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.0
epoch1_acc_gate: 77.0
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations, no augmentation
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.0` |

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9042 | 78.9240 | 94.7160 | 50000 | passed `epoch1 >77`; `Epoch1AccGate: passed` |
| 2 | 0.8929 | 79.0980 | 94.8160 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.222926s | 2296.73 |
| 2 | 2496 | 0.223281s | 2293.07 |

Status:

- Q is the best valid 2epoch single recipe so far: Top-1 `79.0980`.
- It improves over P/T=1 by `+0.2580` Top-1 at epoch2 and `+0.4400` at epoch1.
- It improves over E/O accidental hard+soft by `+0.7300` Top-1 at epoch2.
- It still misses the 80.0 target by `0.9020` Top-1.
- Temperature is a real useful knob. Next recipe should keep true soft-KD and explore a nearby temperature or interaction with quant LR.

### Temperature smoke: true soft-KD T=3 and T=2.5

Reason: Q/T=2 improved over P/T=1, so I checked stronger temperatures before spending full 2epoch runs.

Smoke results:

| temperature | output | log | first update loss | status |
|---:|---|---|---:|---|
| 3.0 | `/tmp/qat_public_repro/smoke_true_softkd_temp3_1update_20260705` | `/tmp/smoke_true_softkd_temp3_1update_20260705.log` | 63.288391 | too aggressive for direct full run |
| 2.5 | `/tmp/qat_public_repro/smoke_true_softkd_temp25_1update_20260705` | `/tmp/smoke_true_softkd_temp25_1update_20260705.log` | 44.292557 | acceptable to test with epoch1 gate |

### Single-recipe 2epoch R: fixed-QKR no-aug scheduler1 quantlr4 true soft-KD T=2.5

Reason: T=2.0 improved the result; T=2.5 is the next nearby stronger temperature that still looked reasonable in the 1-update smoke. R keeps the same single continuous recipe structure as Q, changing only `teacher_soft_temperature` from `2.0` to `2.5`.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_r_fixed_qkr_noaug_scheduler1_quantlr4_true_softkd_temp25_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe2ep_r_fixed_qkr_noaug_scheduler1_quantlr4_true_softkd_temp25_20260705
log: /tmp/train_recipe2ep_r_fixed_qkr_noaug_scheduler1_quantlr4_true_softkd_temp25_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 1
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.5
epoch1_acc_gate: 77.0
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations, no augmentation
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.5` |

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.8943 | 79.3180 | 94.8780 | 50000 | passed `epoch1 >77`; `Epoch1AccGate: passed` |
| 2 | 0.8878 | 79.4100 | 94.9080 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.222964s | 2296.33 |
| 2 | 2496 | 0.222875s | 2297.25 |

Status:

- R is the best valid 2epoch single recipe so far: Top-1 `79.4100`.
- It improves over Q/T=2 by `+0.3120` Top-1 at epoch2 and `+0.3940` at epoch1.
- It improves over P/T=1 by `+0.5700` Top-1 at epoch2.
- It improves over E/O accidental hard+soft by `+1.0420` Top-1 at epoch2.
- It still misses the 80.0 target by `0.5900` Top-1.
- Stronger temperature helps up to 2.5; T=3 first-update loss is much larger, so the next efficient path is either a narrower temperature sweep around 2.5 or interaction with quant LR.

### Single-recipe 2epoch S: fixed-QKR no-aug scheduler1 quantlr6 true soft-KD T=2.5

Reason: R/T=2.5 became the best temperature setting. In earlier hard+soft experiments the quant LR multiplier optimum was near x4, but with true soft-KD the optimum could shift. S keeps R fixed and changes only `quant_lr_multiplier` from `4` to `6`.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_s_fixed_qkr_noaug_scheduler1_quantlr6_true_softkd_temp25_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe2ep_s_fixed_qkr_noaug_scheduler1_quantlr6_true_softkd_temp25_20260705
log: /tmp/train_recipe2ep_s_fixed_qkr_noaug_scheduler1_quantlr6_true_softkd_temp25_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 1
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 6
kd_hard_and_soft: 0
teacher_soft_temperature: 2.5
epoch1_acc_gate: 77.0
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations, no augmentation
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.5` |
| `args.yaml quant_lr_multiplier` | `6.0` |

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.8921 | 79.1100 | 94.8900 | 50000 | passed `epoch1 >77`; `Epoch1AccGate: passed` |
| 2 | 0.8834 | 79.4260 | 94.9660 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223202s | 2293.89 |
| 2 | 2496 | 0.223177s | 2294.14 |

Status:

- S is the best valid 2epoch single recipe so far: Top-1 `79.4260`.
- It improves over R/T=2.5/x4 by only `+0.0160` Top-1 at epoch2, while epoch1 drops from `79.3180` to `79.1100`.
- It improves over E/O accidental hard+soft by `+1.0580` Top-1 at epoch2.
- It still misses the 80.0 target by `0.5740` Top-1.
- Quant LR x6 is not a strong direction despite the tiny epoch2 gain. Prefer returning to x4 or trying x5, and continue the fine temperature/confidence-weight search.

### Current best design note

Wrote a standalone Chinese design summary for the current best recipe:

```text
/mlx_devbox/users/quyanyi/playground/swin_t_w4a4_best_recipe_design_20260705.md
```

### Single-recipe 2epoch T: fixed-QKR no-aug scheduler1 quantlr4 true soft-KD T=2.75

Reason: S showed quant LR x6 is not a strong direction despite a tiny epoch2 gain, while temperature remained the dominant useful knob. T returns to quant LR x4 and tests a finer temperature between R/T=2.5 and the too-aggressive T=3 smoke.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_t_fixed_qkr_noaug_scheduler1_quantlr4_true_softkd_temp275_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe2ep_t_fixed_qkr_noaug_scheduler1_quantlr4_true_softkd_temp275_20260705
log: /tmp/train_recipe2ep_t_fixed_qkr_noaug_scheduler1_quantlr4_true_softkd_temp275_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 1
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
epoch1_acc_gate: 77.0
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations, no augmentation
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.75` |
| `args.yaml quant_lr_multiplier` | `4.0` |

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.8926 | 79.1980 | 94.9120 | 50000 | passed `epoch1 >77`; `Epoch1AccGate: passed` |
| 2 | 0.8841 | 79.5180 | 95.0420 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223176s | 2294.16 |
| 2 | 2496 | 0.222963s | 2296.35 |

Status:

- T is the best valid 2epoch single recipe so far: Top-1 `79.5180`.
- It improves over S/T=2.5/x6 by `+0.0920` Top-1 at epoch2.
- It improves over R/T=2.5/x4 by `+0.1080` Top-1 at epoch2.
- It improves over E/O accidental hard+soft by `+1.1500` Top-1 at epoch2.
- It still misses the 80.0 target by `0.4820` Top-1.
- Temperature 2.75 is better than 2.5 under quant LR x4, despite a lower epoch1 than R. Next efficient test should stay near this temperature and adjust one mild knob, such as T=2.875 or quant LR x5.

### Single-recipe 2epoch U: fixed-QKR no-aug scheduler1 quantlr4 true soft-KD T=2.875

Reason: T=2.75 improved over T=2.5, while T=3 smoke looked too aggressive. U tests an intermediate temperature `2.875` with the same quant LR x4 recipe.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_u_fixed_qkr_noaug_scheduler1_quantlr4_true_softkd_temp2875_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe2ep_u_fixed_qkr_noaug_scheduler1_quantlr4_true_softkd_temp2875_20260705
log: /tmp/train_recipe2ep_u_fixed_qkr_noaug_scheduler1_quantlr4_true_softkd_temp2875_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 1
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.875
epoch1_acc_gate: 77.0
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations, no augmentation
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.875` |
| `args.yaml quant_lr_multiplier` | `4.0` |

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.8946 | 79.2140 | 94.8880 | 50000 | passed `epoch1 >77`; `Epoch1AccGate: passed` |
| 2 | 0.8852 | 79.4020 | 94.9640 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.222934s | 2296.65 |
| 2 | 2496 | 0.222870s | 2297.30 |

Status:

- U underperforms T: epoch2 Top-1 `79.4020` vs T `79.5180`.
- Temperature `2.875` appears too strong relative to `2.75`; do not continue this temperature direction without another compensating change.
- Best current 2epoch result remains T: Top-1 `79.5180`.

### Single-recipe 2epoch V: fixed-QKR no-aug scheduler1 quantlr5 true soft-KD T=2.75

Reason: T=2.75/x4 became the best recipe, and S/T=2.5/x6 showed that a higher quant LR may slightly improve epoch2 while hurting epoch1. V keeps T=2.75 and tests a milder quant LR multiplier x5.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_v_fixed_qkr_noaug_scheduler1_quantlr5_true_softkd_temp275_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe2ep_v_fixed_qkr_noaug_scheduler1_quantlr5_true_softkd_temp275_20260705
log: /tmp/train_recipe2ep_v_fixed_qkr_noaug_scheduler1_quantlr5_true_softkd_temp275_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 1
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 5
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
epoch1_acc_gate: 77.0
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations, no augmentation
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.75` |
| `args.yaml quant_lr_multiplier` | `5.0` |

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.8896 | 79.2580 | 94.9260 | 50000 | passed `epoch1 >77`; `Epoch1AccGate: passed` |
| 2 | 0.8835 | 79.4020 | 94.9980 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223190s | 2294.01 |
| 2 | 2496 | 0.223109s | 2294.84 |

Status:

- V underperforms T: epoch2 Top-1 `79.4020` vs T `79.5180`.
- Quant LR x5 does not help at T=2.75. The useful region remains T=2.75 with quant LR x4.
- Best current 2epoch result remains T: Top-1 `79.5180`.

### Confidence-weighted soft-KD smoke at T=2.75

Reason: after temperature tuning, test whether weighting samples by FP teacher confidence changes the effective objective enough to justify a full 2epoch run.

Smoke command:

```bash
PYTHONUNBUFFERED=1 python3 /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py \
  --method ofq --stage train \
  --config /mlx_devbox/users/quyanyi/playground/QATs/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml \
  --model swin_t --data /tmp/imagenet1k_full_parquet --dataset-format parquet \
  --output /tmp/qat_public_repro --experiment smoke_true_softkd_temp275_conf1_1update_20260705 \
  --devices 0,1,2,3,4,5,6,7 --nproc-per-node 8 --master-port 30472 --model-type swin \
  --teacher swin_t --teacher-type swin --teacher-checkpoint /home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth --teacher-pretrained \
  --epochs 1 --scheduler-epochs 1 --batch-size 64 --workers 8 \
  --lr 2e-4 --min-lr 1e-5 --weight-decay 0.0 \
  --quant-lr-multiplier 4 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 0 \
  --teacher-soft-temperature 2.75 --teacher-confidence-kd-power 1.0 \
  --quantized --qk-reparam --qk-reparam-type 0 \
  --amp --amp-dtype bf16 \
  --extra-arg=--static-graph \
  --extra-arg=--smoothing --extra-arg=0.0 \
  --extra-arg=--mixup --extra-arg=0.0 \
  --extra-arg=--cutmix --extra-arg=0.0 \
  --extra-arg=--aa --extra-arg=none \
  --extra-arg=--color-jitter --extra-arg=0.0 \
  --extra-arg=--reprob --extra-arg=0.0 \
  --extra-arg=--max_train_updates --extra-arg=1 \
  --extra-arg=--skip_validate \
  --extra-arg=--log-interval --extra-arg=1 \
  --extra-arg=--seed --extra-arg=42
```

Smoke result:

| item | value |
|---|---|
| output | `/tmp/qat_public_repro/smoke_true_softkd_temp275_conf1_1update_20260705` |
| log | `/tmp/smoke_true_softkd_temp275_conf1_1update_20260705.log` |
| first update loss | `53.363506` |
| T/T=2.75 first update loss | `53.359268` |

Status:

- Confidence weighting with power `1.0` barely changes the first-update objective relative to plain T=2.75.
- Do not prioritize a full 2epoch run for this knob until higher-impact options are exhausted.

### Single-recipe 2epoch W: fixed-QKR no-aug scheduler2 quantlr4 true soft-KD T=2.75

Reason: T=2.75/x4 became the best scheduler1 recipe, but epoch2 in scheduler1 spends most of its time at `min_lr=1e-5`. W keeps the same target and optimizer grouping as T, but changes the fixed cosine scheduler horizon from `1` to `2` epochs so epoch2 has more learning-rate budget. This is still one continuous fixed recipe, not a staged LR switch.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_w_fixed_qkr_noaug_scheduler2_quantlr4_true_softkd_temp275_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe2ep_w_fixed_qkr_noaug_scheduler2_quantlr4_true_softkd_temp275_20260705
log: /tmp/train_recipe2ep_w_fixed_qkr_noaug_scheduler2_quantlr4_true_softkd_temp275_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 2
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
epoch1_acc_gate: 77.0
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations, no augmentation
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.75` |
| `args.yaml quant_lr_multiplier` | `4.0` |
| `args.yaml scheduler_epochs` | `2` |

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9242 | 78.5060 | 94.5360 | 50000 | passed `epoch1 >77`; `Epoch1AccGate: passed` |
| 2 | 0.8734 | 79.7320 | 95.0580 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223172s | 2294.19 |
| 2 | 2496 | 0.222975s | 2296.22 |

Status:

- W is the best valid 2epoch single recipe so far: Top-1 `79.7320`.
- It improves over T/scheduler1 by `+0.2140` Top-1 at epoch2, despite a much lower epoch1.
- It improves over E/O accidental hard+soft by `+1.3640` Top-1 at epoch2.
- It still misses the 80.0 target by `0.2680` Top-1.
- Scheduler horizon is now a useful knob. The next efficient test should keep scheduler2 and tune one adjacent knob, e.g. quant LR x5 or a slightly lower temperature to recover epoch1 while preserving epoch2 learning budget.

### Single-recipe 2epoch X: fixed-QKR no-aug scheduler2 quantlr5 true soft-KD T=2.75

Reason: W/scheduler2 became the best recipe. X keeps scheduler2 and T=2.75, changing only `quant_lr_multiplier` from x4 to x5 to test whether extra quant/shift adaptation helps under the longer LR horizon.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_x_fixed_qkr_noaug_scheduler2_quantlr5_true_softkd_temp275_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe2ep_x_fixed_qkr_noaug_scheduler2_quantlr5_true_softkd_temp275_20260705
log: /tmp/train_recipe2ep_x_fixed_qkr_noaug_scheduler2_quantlr5_true_softkd_temp275_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 2
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 5
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
epoch1_acc_gate: 77.0
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations, no augmentation
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.75` |
| `args.yaml quant_lr_multiplier` | `5.0` |
| `args.yaml scheduler_epochs` | `2` |

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9220 | 78.6900 | 94.5820 | 50000 | passed `epoch1 >77`; `Epoch1AccGate: passed` |
| 2 | 0.8716 | 79.6480 | 95.0820 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223230s | 2293.59 |
| 2 | 2496 | 0.223061s | 2295.33 |

Status:

- X underperforms W: epoch2 Top-1 `79.6480` vs W `79.7320`.
- Quant LR x5 does not help under scheduler2 either.
- Best current 2epoch result remains W: Top-1 `79.7320`.

### Single-recipe 2epoch Y: fixed-QKR no-aug scheduler2 quantlr4 true soft-KD T=2.5

Reason: W/scheduler2 improved epoch2 but hurt epoch1. Y keeps scheduler2 and quant LR x4 but lowers temperature from `2.75` to `2.5` to test whether a less aggressive soft target recovers epoch1 while retaining scheduler2's second-epoch LR budget.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_y_fixed_qkr_noaug_scheduler2_quantlr4_true_softkd_temp25_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe2ep_y_fixed_qkr_noaug_scheduler2_quantlr4_true_softkd_temp25_20260705
log: /tmp/train_recipe2ep_y_fixed_qkr_noaug_scheduler2_quantlr4_true_softkd_temp25_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 2
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.5
epoch1_acc_gate: 77.0
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations, no augmentation
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.5` |
| `args.yaml quant_lr_multiplier` | `4.0` |
| `args.yaml scheduler_epochs` | `2` |

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9302 | 78.4060 | 94.5580 | 50000 | passed `epoch1 >77`; `Epoch1AccGate: passed` |
| 2 | 0.8745 | 79.5220 | 95.0780 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223108s | 2294.85 |
| 2 | 2496 | 0.223143s | 2294.50 |

Status:

- Y underperforms W: epoch2 Top-1 `79.5220` vs W `79.7320`.
- Lowering temperature to `2.5` does not recover enough epoch1 and loses epoch2 accuracy.
- Best current 2epoch result remains W: Top-1 `79.7320`.

### Single-recipe 2epoch Z: fixed-QKR no-aug scheduler2 quantlr4 true soft-KD T=2.75 minlr2e-5

Reason: W/scheduler2 improved epoch2 by preserving more LR budget. Z keeps W fixed and raises only the cosine LR floor from `1e-5` to `2e-5`, testing whether the late part of epoch2 benefits from a slightly higher floor.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_z_fixed_qkr_noaug_scheduler2_quantlr4_true_softkd_temp275_minlr2e5_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe2ep_z_fixed_qkr_noaug_scheduler2_quantlr4_true_softkd_temp275_minlr2e5_20260705
log: /tmp/train_recipe2ep_z_fixed_qkr_noaug_scheduler2_quantlr4_true_softkd_temp275_minlr2e5_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 2
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 2e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
epoch1_acc_gate: 77.0
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations, no augmentation
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.75` |
| `args.yaml quant_lr_multiplier` | `4.0` |
| `args.yaml scheduler_epochs` | `2` |
| `args.yaml min_lr` | `2e-5` |

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9273 | 78.4580 | 94.5140 | 50000 | passed `epoch1 >77`; `Epoch1AccGate: passed` |
| 2 | 0.8701 | 79.6520 | 95.0700 | 50000 | failed `epoch2 >=80` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223346s | 2292.41 |
| 2 | 2496 | 0.223164s | 2294.28 |

Status:

- Z underperforms W: epoch2 Top-1 `79.6520` vs W `79.7320`.
- Raising `min_lr` to `2e-5` does not help; the W LR floor is already adequate.
- Best current 2epoch result remains W: Top-1 `79.7320`.

### Single-recipe 2epoch AA: fixed-QKR no-aug scheduler2 lr2.2e-4 quantlr4 true soft-KD T=2.75

Reason: W/scheduler2 is current best. AA keeps W fixed and changes only the base LR from `2e-4` to `2.2e-4`, testing whether a slightly stronger LR can improve epoch2.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_aa_fixed_qkr_noaug_scheduler2_lr22e5_quantlr4_true_softkd_temp275_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe2ep_aa_fixed_qkr_noaug_scheduler2_lr22e5_quantlr4_true_softkd_temp275_20260705
log: /tmp/train_recipe2ep_aa_fixed_qkr_noaug_scheduler2_lr22e5_quantlr4_true_softkd_temp275_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 2
batch_size: 64 per GPU, global batch 512
lr: 2.2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
epoch1_acc_gate: 77.0
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations, no augmentation
```

Observed result before interruption:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9248 | 78.3160 | 94.5860 | 50000 | passed `epoch1 >77`; `Epoch1AccGate: passed` |

Status:

- AA did not complete epoch2: the log stops during epoch2 training and the output directory only has `checkpoint-1.pth.tar` / `last.pth.tar`.
- No epoch2 result is accepted for AA.
- Epoch1 underperformed W (`78.3160` vs W `78.5060`), so rerunning AA is lower priority than other W-neighborhood tests.
- Best current complete 2epoch result remains W: Top-1 `79.7320`.

### Single-recipe 2epoch AB: fixed-QKR no-aug scheduler3 quantlr4 true soft-KD T=2.75

Reason: W showed scheduler horizon is useful. AB keeps W fixed and changes only `scheduler_epochs` from `2` to `3`, testing whether a longer cosine horizon gives epoch2 enough LR budget to reach the 79.9 gate.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_ab_fixed_qkr_noaug_scheduler3_quantlr4_true_softkd_temp275_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe2ep_ab_fixed_qkr_noaug_scheduler3_quantlr4_true_softkd_temp275_20260705
log: /tmp/train_recipe2ep_ab_fixed_qkr_noaug_scheduler3_quantlr4_true_softkd_temp275_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 3
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
epoch1_acc_gate: 77.0
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations, no augmentation
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.75` |
| `args.yaml quant_lr_multiplier` | `4.0` |
| `args.yaml scheduler_epochs` | `3` |

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9510 | 77.9460 | 94.3740 | 50000 | passed `epoch1 >77`; `Epoch1AccGate: passed` |
| 2 | 0.8957 | 79.4320 | 94.9780 | 50000 | failed `epoch2 >=79.9` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223232s | 2293.57 |
| 2 | 2496 | 0.223059s | 2295.36 |

Status:

- AB underperforms W: epoch2 Top-1 `79.4320` vs W `79.7320`.
- Extending the scheduler horizon from 2 to 3 preserves too much LR too early: epoch1 drops sharply, and epoch2 does not recover enough.
- Do not continue to scheduler4 as the main path.
- Best current complete 2epoch result remains W: Top-1 `79.7320`.

### Single-recipe 2epoch AC: fixed-QKR no-aug scheduler2 quantlr4 true soft-KD T=2.625

Reason: AB showed scheduler3 is too slow. AC returns to W's scheduler2 and changes only `teacher_soft_temperature` from `2.75` to `2.625`, a narrow temperature test between Y/T=2.5 and W/T=2.75.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_ac_fixed_qkr_noaug_scheduler2_quantlr4_true_softkd_temp2625_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe2ep_ac_fixed_qkr_noaug_scheduler2_quantlr4_true_softkd_temp2625_20260705
log: /tmp/train_recipe2ep_ac_fixed_qkr_noaug_scheduler2_quantlr4_true_softkd_temp2625_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 2
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.625
epoch1_acc_gate: 77.0
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations, no augmentation
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.625` |
| `args.yaml quant_lr_multiplier` | `4.0` |
| `args.yaml scheduler_epochs` | `2` |

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9313 | 78.4300 | 94.5180 | 50000 | passed `epoch1 >77`; `Epoch1AccGate: passed` |
| 2 | 0.8717 | 79.7160 | 95.0620 | 50000 | failed `epoch2 >=79.9` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223054s | 2295.40 |
| 2 | 2496 | 0.223038s | 2295.58 |

Status:

- AC underperforms W slightly: epoch2 Top-1 `79.7160` vs W `79.7320`.
- Lowering temperature from `2.75` to `2.625` does not help, though it is much closer to W than Y/T=2.5.
- Best current complete 2epoch result remains W: Top-1 `79.7320`.

### Single-recipe 2epoch AD: fixed-QKR no-aug scheduler2 quantlr4 true soft-KD T=2.8125

Reason: AC showed the lower-temperature side does not beat W. AD keeps W fixed and changes only `teacher_soft_temperature` from `2.75` to `2.8125`, a narrow test between W/T=2.75 and U/T=2.875.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_2ep_recipe_ad_fixed_qkr_noaug_scheduler2_quantlr4_true_softkd_temp28125_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe2ep_ad_fixed_qkr_noaug_scheduler2_quantlr4_true_softkd_temp28125_20260705
log: /tmp/train_recipe2ep_ad_fixed_qkr_noaug_scheduler2_quantlr4_true_softkd_temp28125_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 2
scheduler_epochs: 2
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.8125
epoch1_acc_gate: 77.0
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations, no augmentation
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.8125` |
| `args.yaml quant_lr_multiplier` | `4.0` |
| `args.yaml scheduler_epochs` | `2` |

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9194 | 78.6720 | 94.6500 | 50000 | passed `epoch1 >77`; `Epoch1AccGate: passed` |
| 2 | 0.8713 | 79.6160 | 95.0980 | 50000 | failed `epoch2 >=79.9` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-2 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223054s | 2295.41 |
| 2 | 2496 | 0.223204s | 2293.86 |

Status:

- AD underperforms W: epoch2 Top-1 `79.6160` vs W `79.7320`.
- Narrow temperature search around W did not beat W: AC/T=2.625 `79.7160`, W/T=2.75 `79.7320`, AD/T=2.8125 `79.6160`.
- Best current complete 2epoch result remains W: Top-1 `79.7320`.

## New 4epoch / 5epoch public-style W4A4-family goal

Updated goal: stop optimizing for marginal 2epoch gains. Use public-style W4A4-family fixed-QKR recipes, where first/last layers stay 8-bit and main transformer blocks are W4A4, to reach:

- Stage 1: 4epoch full ImageNet raw validation Top-1 >= `80.5`.
- Stage 2: after Stage 1 passes, extend the same best paradigm to 5epoch full ImageNet raw validation Top-1 >= `81.0`.

The main search should prioritize training-paradigm changes such as stronger teacher supervision, QAT warm starts, quantization curriculum, or refmodel stabilization. Do not claim strict all-layer W4A4.

### Single-recipe 3epoch A: QSC warm-start fixed-QKR soft-KD T=2.75

Reason: move from 2epoch hyperparameter search to a paradigm candidate. This recipe keeps the strongest public-family W path (fixed-QKR, true pure soft-KD T=2.75, no augmentation) and adds a QSC/APHQ-style pre-QAT late feature reconstruction warm start before DDP wrapping. The warm start reconstructs FP teacher outputs at `features.5.5,features.7.1`, updating only quant/shift parameters for 100 updates.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_3ep_public_family_qsc_warmstart_a_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe3ep_a_qsc_warmstart_fixed_qkr_softkd_t275_20260705
log: /tmp/train_recipe3ep_a_qsc_warmstart_fixed_qkr_softkd_t275_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 3
scheduler_epochs: 3
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
pre_qat_feature_recon_updates: 100
pre_qat_feature_recon_layers: features.5.5,features.7.1
pre_qat_feature_recon_policy: quant
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations, no augmentation
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml epochs` | `3` |
| `args.yaml scheduler_epochs` | `3` |
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.75` |
| `args.yaml quant_lr_multiplier` | `4.0` |
| `args.yaml pre_qat_feature_recon_updates` | `100` |
| `args.yaml pre_qat_feature_recon_layers` | `features.5.5,features.7.1` |
| `args.yaml pre_qat_feature_recon_policy` | `quant` |

Pre-QAT reconstruction markers:

```text
PreQATFeatRecon: update=1/100 loss=0.739306 kept=67764 masked=27767356
PreQATFeatRecon: update=50/100 loss=0.616958 kept=67764 masked=27767356
PreQATFeatRecon: update=100/100 loss=0.596983 kept=67764 masked=27767356
```

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9513 | 77.9840 | 94.2920 | 50000 | below W epoch1 trajectory |
| 2 | 0.9014 | 79.3800 | 94.9980 | 50000 | below W epoch2 `79.7320` |
| 3 | 0.8640 | 79.9460 | 95.1780 | 50000 | failed `epoch3 >=80.5` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `checkpoint-3.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-3 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.222987s | 2296.09 |
| 2 | 2496 | 0.222843s | 2297.58 |
| 3 | 2496 | 0.222829s | 2297.73 |

Status:

- A is a valid 3epoch public-style W4A4-family result, but it fails the `80.5` gate by `0.5540` Top-1.
- It should not be extended to 5epoch under the current rule.
- The QSC late feature warm start is stable and reaches `79.9460`, but it does not provide enough acceleration over W to hit the new 3epoch target.

### Single-recipe 3epoch B: QSS-start2 fixed-QKR soft-KD T=2.75

Reason: A/QSC was stable but insufficient. B tests a different training paradigm: delayed quantizer slow-state stabilization. It keeps W's fixed-QKR + true soft-KD recipe and enables full quant/shift slow-state EMA only from epoch2, so epochs 0-1 remain free adaptation and epoch2 stabilizes quant/shift states.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_3ep_public_family_qss_start2_b_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe3ep_b_qss_start2_fixed_qkr_softkd_t275_20260705
log: /tmp/train_recipe3ep_b_qss_start2_fixed_qkr_softkd_t275_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 3
scheduler_epochs: 3
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
quant_slow_state_decay: 0.99
quant_slow_state_sync_interval: 50
quant_slow_state_pull: 0.05
quant_slow_state_policy: all
quant_slow_state_observe_start_epoch: 2
quant_slow_state_start_epoch: 2
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations, no augmentation
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml epochs` | `3` |
| `args.yaml scheduler_epochs` | `3` |
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.75` |
| `args.yaml quant_lr_multiplier` | `4.0` |
| `args.yaml quant_slow_state_decay` | `0.99` |
| `args.yaml quant_slow_state_observe_start_epoch` | `2` |
| `args.yaml quant_slow_state_start_epoch` | `2` |

QSS markers:

```text
Initialized quant slow state: params=308, policy=all, decay=0.99, sync_interval=50, pull=0.05
Applied quant slow state pull: update=5000, tensors=308, pull=0.05
...
Applied quant slow state pull: update=7450, tensors=308, pull=0.05
```

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9510 | 77.9460 | 94.3740 | 50000 | same as scheduler3 baseline before QSS |
| 2 | 0.8957 | 79.4320 | 94.9780 | 50000 | same as scheduler3 baseline before QSS |
| 3 | 0.8615 | 79.9520 | 95.1800 | 50000 | failed `epoch3 >=80.5` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `checkpoint-3.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-3 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.222763s | 2298.41 |
| 2 | 2496 | 0.222852s | 2297.49 |
| 3 | 2496 | 0.224895s | 2276.62 |

Status:

- B is a valid 3epoch public-style W4A4-family result, but it fails the `80.5` gate by `0.5480` Top-1.
- It should not be extended to 5epoch under the current rule.
- QSS-start2 is correctly active in epoch3, but it only matches A/QSC-level outcome (`79.9520` vs A `79.9460`) and does not provide enough acceleration.
- Current best 3epoch result is B: Top-1 `79.9520`, Top-5 `95.1800`.

### Single-recipe 3epoch C plan: quant-first100 fixed-QKR soft-KD T=2.75

Reason: A/QSC and B/QSS are stable but only reach about `79.95`. C tests a different training-paradigm family: quant/backbone responsibility separation. For the first 100 optimizer updates, gradients are masked so only quant/shift parameters update; after that, training switches to all parameters. This gives quantizers a short adaptation window before full backbone updates.

Planned command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_3ep_public_family_quantfirst100_c_20260705.sh
```

Key planned config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
output: /tmp/qat_public_repro/recipe3ep_c_quantfirst100_fixed_qkr_softkd_t275_20260705
log: /tmp/train_recipe3ep_c_quantfirst100_fixed_qkr_softkd_t275_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 3
scheduler_epochs: 3
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
trainable_policy: quant
trainable_policy_update_overrides: 100:all
trainable_policy_update_mode: grad_mask
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations, no augmentation
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml epochs` | `3` |
| `args.yaml scheduler_epochs` | `3` |
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.75` |
| `args.yaml quant_lr_multiplier` | `4.0` |
| `args.yaml trainable_policy` | `quant` |
| `args.yaml trainable_policy_update_mode` | `grad_mask` |
| trainable policy marker | `epoch=0, update=0, mode=grad_mask, policy=quant` |
| trainable policy marker | `epoch=0, update=100, mode=grad_mask, policy=all` |

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9437 | 77.9760 | 94.3640 | 50000 | below B/QSS `77.9460` only by noise-scale |
| 2 | 0.8885 | 79.3920 | 95.0180 | 50000 | below B/QSS `79.4320` |
| 3 | 0.8619 | 79.8880 | 95.1820 | 50000 | failed `epoch3 >=80.5` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `checkpoint-3.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-3 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223123s | 2294.70 |
| 2 | 2496 | 0.222865s | 2297.36 |
| 3 | 2496 | 0.222982s | 2296.15 |

Status:

- C is a valid 3epoch public-style W4A4-family result, but it fails the `80.5` gate by `0.6120` Top-1.
- It should not be extended to 5epoch under the current rule.
- Quant-first100 hurts the final 3epoch result relative to A/QSC and B/QSS; do not continue this exact quant-first schedule as the main path.
- Current best 3epoch result remains B: Top-1 `79.9520`, Top-5 `95.1800`.

### Single-recipe 3epoch D: late feature-output normalized alignment fixed-QKR soft-KD T=2.75

Reason: A/B/C show that light warm-start or quant-state stabilization is not enough. D adds stronger online teacher supervision throughout training by aligning late Swin feature outputs (`features.5.5,features.7.1`) with the FP teacher using normalized MSE, while keeping the same fixed-QKR pure soft-KD recipe.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_3ep_public_family_featnorm_d_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe3ep_d_featnorm005_fixed_qkr_softkd_t275_20260705
log: /tmp/train_recipe3ep_d_featnorm005_fixed_qkr_softkd_t275_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 3
scheduler_epochs: 3
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
teacher_feature_output_weight: 0.005
teacher_feature_output_layers: features.5.5,features.7.1
teacher_feature_output_loss: norm_mse
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations, no augmentation
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml epochs` | `3` |
| `args.yaml scheduler_epochs` | `3` |
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.75` |
| `args.yaml quant_lr_multiplier` | `4.0` |
| `args.yaml teacher_feature_output_weight` | `0.005` |
| `args.yaml teacher_feature_output_layers` | `features.5.5,features.7.1` |
| `args.yaml teacher_feature_output_loss` | `norm_mse` |

Feature-output markers:

```text
Teacher feature-output hooks: layers=('features.5.5', 'features.7.1'), student_matches=('features.5.5', 'features.7.1'), teacher_matches=('features.5.5', 'features.7.1')
Teacher feature-output debug: student_count=2, teacher_count=2, 0:s=(64, 14, 14, 384) t=(64, 14, 14, 384) mse=2.918e+01; 1:s=(64, 7, 7, 768) t=(64, 7, 7, 768) mse=9.878e+00
TeacherFeatOut: first update 7.393e-01; final epoch late training around 2.3e-01
```

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9506 | 78.0080 | 94.4080 | 50000 | slight early improvement vs B `77.9460` |
| 2 | 0.8793 | 79.4500 | 94.9980 | 50000 | slight improvement vs B `79.4320` |
| 3 | 0.8633 | 80.0080 | 95.1060 | 50000 | failed `epoch3 >=80.5` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `checkpoint-3.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-3 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223802s | 2287.74 |
| 2 | 2496 | 0.223439s | 2291.45 |
| 3 | 2496 | 0.223491s | 2290.92 |

Status:

- D is a valid 3epoch public-style W4A4-family result, but it fails the `80.5` gate by `0.4920` Top-1.
- It should not be extended to 5epoch under the current rule.
- D is the current best 3epoch result: Top-1 `80.0080`, Top-5 `95.1060`.
- Online late feature alignment is the first tested paradigm to cross 80 at 3epoch, but the effect size remains too small. Next work should use stronger teacher supervision, not more 2epoch-style hyperparameter search.

### Single-recipe 3epoch E: augmented teacher distill + late feature-output alignment

Reason: D crossed 80 but still missed the 80.5 gate. E strengthens teacher supervision by applying online augmented views to both student and teacher, while keeping late feature-output normalized alignment. This is a lightweight FKD/VVTQ-style move toward richer teacher targets, without relying on missing offline FKD soft-label files. Mixup/cutmix remain disabled to avoid mixing teacher soft-target semantics.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_3ep_public_family_featnorm_aug_e_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe3ep_e_featnorm_aug_fixed_qkr_softkd_t275_20260705
log: /tmp/train_recipe3ep_e_featnorm_aug_fixed_qkr_softkd_t275_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 3
scheduler_epochs: 3
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
teacher_feature_output_weight: 0.005
teacher_feature_output_layers: features.5.5,features.7.1
teacher_feature_output_loss: norm_mse
smoothing: 0.1
mixup: 0.0
cutmix: 0.0
aa: rand-m9-mstd0.5-inc1
color_jitter: 0.4
reprob: 0.25
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml epochs` | `3` |
| `args.yaml scheduler_epochs` | `3` |
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.75` |
| `args.yaml teacher_feature_output_weight` | `0.005` |
| `args.yaml aa` | `rand-m9-mstd0.5-inc1` |
| `args.yaml color_jitter` | `0.4` |
| `args.yaml reprob` | `0.25` |
| `args.yaml mixup/cutmix` | `0.0 / 0.0` |

Feature-output markers:

```text
Teacher feature-output hooks: layers=('features.5.5', 'features.7.1'), student_matches=('features.5.5', 'features.7.1'), teacher_matches=('features.5.5', 'features.7.1')
Teacher feature-output debug epoch0: mse=2.828e+01 / 9.951e+00
Teacher feature-output debug epoch2: mse=1.170e+01 / 2.069e+00
```

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9261 | 78.5760 | 94.4900 | 50000 | best early result so far |
| 2 | 0.8735 | 79.6400 | 95.0200 | 50000 | best epoch2 result so far |
| 3 | 0.8583 | 80.0880 | 95.1220 | 50000 | failed `epoch3 >=80.5` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `checkpoint-3.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-3 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223673s | 2289.05 |
| 2 | 2496 | 0.223437s | 2291.47 |
| 3 | 2496 | 0.223513s | 2290.69 |

Status:

- E is a valid 3epoch public-style W4A4-family result, but it fails the `80.5` gate by `0.4120` Top-1.
- It should not be extended to 5epoch under the current rule.
- E is the current best 3epoch result: Top-1 `80.0880`, Top-5 `95.1220`.
- Online augmented teacher distillation plus late feature alignment is the best direction so far. The next candidate should build on E rather than returning to QSC/QSS or quant-first.

### Single-recipe 3epoch F: E plus small hard-label CE anchor

Reason: E improved early and final accuracy, but still missed the gate. F keeps E and adds a small hard-label CE auxiliary (`clean_start_target_loss_weight=0.02`) to test whether augmented teacher-distillation needs a weak class-label anchor. This is not a return to hard+soft KD; the primary loss remains true pure soft-KD.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_3ep_public_family_featnorm_aug_auxce_f_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe3ep_f_featnorm_aug_auxce002_fixed_qkr_softkd_t275_20260705
log: /tmp/train_recipe3ep_f_featnorm_aug_auxce002_fixed_qkr_softkd_t275_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 3
scheduler_epochs: 3
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
teacher_feature_output_weight: 0.005
teacher_feature_output_layers: features.5.5,features.7.1
teacher_feature_output_loss: norm_mse
clean_start_target_loss_weight: 0.02
smoothing: 0.1
mixup: 0.0
cutmix: 0.0
aa: rand-m9-mstd0.5-inc1
color_jitter: 0.4
reprob: 0.25
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml epochs` | `3` |
| `args.yaml scheduler_epochs` | `3` |
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.75` |
| `args.yaml teacher_feature_output_weight` | `0.005` |
| `args.yaml clean_start_target_loss_weight` | `0.02` |
| `args.yaml aa` | `rand-m9-mstd0.5-inc1` |
| `args.yaml color_jitter` | `0.4` |
| `args.yaml reprob` | `0.25` |

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9468 | 78.0600 | 94.3900 | 50000 | below E `78.5760` |
| 2 | 0.8792 | 79.6140 | 94.8760 | 50000 | below E `79.6400` |
| 3 | 0.8559 | 80.0160 | 95.0660 | 50000 | failed `epoch3 >=80.5` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `checkpoint-3.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-3 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223449s | 2291.36 |
| 2 | 2496 | 0.223341s | 2292.46 |
| 3 | 2496 | 0.223424s | 2291.60 |

Status:

- F is a valid 3epoch public-style W4A4-family result, but it fails the `80.5` gate by `0.4840` Top-1.
- It should not be extended to 5epoch under the current rule.
- Small hard-label CE anchor hurts relative to E; do not continue this auxiliary-CE direction as the main path.
- Current best 3epoch result remains E: Top-1 `80.0880`, Top-5 `95.1220`.

### Single-recipe 3epoch G: E plus attention-output structural distillation

Reason: E remains the best tested paradigm, and F shows hard-label anchoring is not useful. G keeps E's augmented teacher distillation plus late feature alignment, and adds teacher attention-output MSE on later Swin attention layers. This is stronger structural teacher supervision and uses an existing implementation path, avoiding a new offline FKD dependency.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_3ep_public_family_feat_attnout_aug_g_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe3ep_g_feat_attnout_aug_fixed_qkr_softkd_t275_20260705
log: /tmp/train_recipe3ep_g_feat_attnout_aug_fixed_qkr_softkd_t275_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 3
scheduler_epochs: 3
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
teacher_feature_output_weight: 0.005
teacher_feature_output_layers: features.5.5,features.7.1
teacher_feature_output_loss: norm_mse
teacher_attn_output_weight: 0.001
teacher_attn_output_layers: 6,7,8,9,10,11
smoothing: 0.1
mixup: 0.0
cutmix: 0.0
aa: rand-m9-mstd0.5-inc1
color_jitter: 0.4
reprob: 0.25
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml epochs` | `3` |
| `args.yaml scheduler_epochs` | `3` |
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.75` |
| `args.yaml teacher_feature_output_weight` | `0.005` |
| `args.yaml teacher_attn_output_weight` | `0.001` |
| `args.yaml teacher_attn_output_layers` | `6,7,8,9,10,11` |
| `args.yaml aa` | `rand-m9-mstd0.5-inc1` |
| `args.yaml color_jitter` | `0.4` |
| `args.yaml reprob` | `0.25` |

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9264 | 78.4200 | 94.5120 | 50000 | below E `78.5760` |
| 2 | 0.8810 | 79.5620 | 94.9040 | 50000 | below E `79.6400` |
| 3 | 0.8600 | 80.0240 | 95.1160 | 50000 | failed `epoch3 >=80.5` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `checkpoint-3.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-3 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223880s | 2286.94 |
| 2 | 2496 | 0.223512s | 2290.71 |
| 3 | 2496 | 0.223943s | 2286.29 |

Status:

- G is a valid 3epoch public-style W4A4-family result, but it fails the `80.5` gate by `0.4760` Top-1.
- It should not be extended to 5epoch under the current rule.
- Attention-output structural distillation hurts relative to E; do not continue this direction as the main path.
- Current best 3epoch result remains E: Top-1 `80.0880`, Top-5 `95.1220`.

### Single-recipe 3epoch H: E plus late QSS-start2

Reason: E remains the best tested paradigm, while B showed late QSS-start2 is stable and close to the non-QSS 3epoch baseline. H combines E's augmented teacher distillation plus late feature alignment with late quant slow-state stabilization. This tests whether the best teacher-supervision paradigm benefits from a smoother quant-state trajectory without adding another local KL loss.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_3ep_public_family_featnorm_aug_qss_h_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe3ep_h_featnorm_aug_qss_fixed_qkr_softkd_t275_20260705
log: /tmp/train_recipe3ep_h_featnorm_aug_qss_fixed_qkr_softkd_t275_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 3
scheduler_epochs: 3
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
teacher_feature_output_weight: 0.005
teacher_feature_output_layers: features.5.5,features.7.1
teacher_feature_output_loss: norm_mse
quant_slow_state_decay: 0.99
quant_slow_state_sync_interval: 50
quant_slow_state_pull: 0.05
quant_slow_state_policy: all
quant_slow_state_observe_start_epoch: 2
quant_slow_state_start_epoch: 2
smoothing: 0.1
mixup: 0.0
cutmix: 0.0
aa: rand-m9-mstd0.5-inc1
color_jitter: 0.4
reprob: 0.25
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations
```

Runtime verification:

| item | value |
|---|---|
| `args.yaml epochs` | `3` |
| `args.yaml scheduler_epochs` | `3` |
| `args.yaml kd_hard_and_soft` | `0` |
| `args.yaml teacher_soft_temperature` | `2.75` |
| `args.yaml teacher_feature_output_weight` | `0.005` |
| `args.yaml quant_slow_state_decay` | `0.99` |
| `args.yaml quant_slow_state_sync_interval` | `50` |
| `args.yaml quant_slow_state_pull` | `0.05` |
| `args.yaml quant_slow_state_policy` | `all` |
| `args.yaml quant_slow_state_observe_start_epoch` | `2` |
| `args.yaml quant_slow_state_start_epoch` | `2` |

QSS markers:

```text
Initialized quant slow state: params=308, policy=all, decay=0.99, sync_interval=50, pull=0.05
Applied quant slow state pull: update=5000, tensors=308, pull=0.05
...
Applied quant slow state pull: update=7450, tensors=308, pull=0.05
```

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9261 | 78.5760 | 94.4900 | 50000 | same as E |
| 2 | 0.8735 | 79.6400 | 95.0200 | 50000 | same as E |
| 3 | 0.8619 | 80.0840 | 95.1500 | 50000 | below E `80.0880` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `checkpoint-3.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-3 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223658s | 2289.21 |
| 2 | 2496 | 0.223418s | 2291.67 |
| 3 | 2496 | 0.225687s | 2268.63 |

Status:

- H is a valid 3epoch public-style W4A4-family result, but late QSS does not improve over E.
- Under the updated user gate, the next Stage 1 target is now `4epoch >=80.5`, so H is retained as evidence but not the main 4epoch candidate.
- Current best 3epoch result remains E: Top-1 `80.0880`, Top-5 `95.1220`.

### Single-recipe 4epoch I: E extended to 4epoch

Reason: The active Stage 1 gate is now 4epoch Top-1 >= `80.5`. E is still the strongest 3epoch paradigm, while H/QSS failed to improve over E. I extends E directly to 4epoch with the same augmented teacher distillation plus late feature-output alignment, avoiding unrelated tuning drift.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_4ep_public_family_featnorm_aug_i_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe4ep_i_featnorm_aug_fixed_qkr_softkd_t275_20260705
log: /tmp/train_recipe4ep_i_featnorm_aug_fixed_qkr_softkd_t275_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 4
scheduler_epochs: 4
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
teacher_feature_output_weight: 0.005
teacher_feature_output_layers: features.5.5,features.7.1
teacher_feature_output_loss: norm_mse
smoothing: 0.1
mixup: 0.0
cutmix: 0.0
aa: rand-m9-mstd0.5-inc1
color_jitter: 0.4
reprob: 0.25
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations
```

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9424 | 78.0400 | 94.3020 | 50000 | below E epoch1 `78.5760` |
| 2 | 0.9058 | 79.1340 | 94.8760 | 50000 | below E epoch2 `79.6400` |
| 3 | 0.8697 | 79.8920 | 95.1460 | 50000 | below E epoch3 `80.0880` |
| 4 | 0.8561 | 80.0860 | 95.1840 | 50000 | failed `epoch4 >=80.5` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `checkpoint-3.pth.tar` | saved |
| `checkpoint-4.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-4 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223586s | 2289.95 |
| 2 | 2496 | 0.223501s | 2290.82 |
| 3 | 2496 | 0.223644s | 2289.35 |
| 4 | 2496 | 0.223660s | 2289.19 |

Status:

- I is a valid 4epoch public-style W4A4-family result, but fails the `80.5` gate by `0.4140` Top-1.
- Extending both training and cosine scheduler to 4 epochs slows early progress and does not beat E/J.

### Single-recipe 4epoch J: E extended to 4epoch with scheduler3

Reason: I showed that stretching the cosine schedule to 4 epochs hurts early progress. J keeps E's `scheduler_epochs=3` and extends only the training loop to 4 epochs. This tests whether the best known 3epoch trajectory can gain from one extra low-LR epoch without changing the first three epochs.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_4ep_public_family_featnorm_aug_sched3_j_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe4ep_j_featnorm_aug_sched3_fixed_qkr_softkd_t275_20260705
log: /tmp/train_recipe4ep_j_featnorm_aug_sched3_fixed_qkr_softkd_t275_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 4
scheduler_epochs: 3
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
teacher_feature_output_weight: 0.005
teacher_feature_output_layers: features.5.5,features.7.1
teacher_feature_output_loss: norm_mse
smoothing: 0.1
mixup: 0.0
cutmix: 0.0
aa: rand-m9-mstd0.5-inc1
color_jitter: 0.4
reprob: 0.25
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations
```

Implementation note:

- During J setup, `--scheduler-epochs` was found to be parsed by `qat_launch.py` but not forwarded to the OFQ subprocess.
- Fixed `qat_launch.py` to pass `--scheduler-epochs`, and added `--scheduler-epochs` support to `third_party/OFQ/train.py` so timm's LR scheduler can use a different horizon while the train loop still runs `--epochs`.
- Verified by downstream command containing `--epochs 4 --scheduler-epochs 3` and by epoch1/2/3 metrics exactly matching E.

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9261 | 78.5760 | 94.4900 | 50000 | same as E |
| 2 | 0.8735 | 79.6400 | 95.0200 | 50000 | same as E |
| 3 | 0.8583 | 80.0880 | 95.1220 | 50000 | same as E |
| 4 | 0.8585 | 80.0360 | 95.1920 | 50000 | failed `epoch4 >=80.5` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `checkpoint-3.pth.tar` | saved |
| `checkpoint-4.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-4 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223451s | 2291.33 |
| 2 | 2496 | 0.223315s | 2292.73 |
| 3 | 2496 | 0.223368s | 2292.18 |
| 4 | 2496 | 0.223560s | 2290.22 |

Status:

- J is a valid 4epoch public-style W4A4-family result, but fails the `80.5` gate by `0.4640` Top-1.
- Because J reproduces E exactly through epoch3 and then drops slightly at epoch4, simply adding a low-LR epoch is not a useful direction.
- Do not extend I or J to 5epoch under the current rule.

### Single-recipe 4epoch K: E plus pre-QAT feature reconstruction warm start

Reason: A/QSC warm-start alone did not beat E, but E remains the best online teacher-supervision paradigm. K combines the two ideas as a curriculum: first align quant/shift parameters to FP teacher late features before formal QAT, then run E's augmented soft-KD plus late feature-output supervision. This is a training-paradigm change rather than another epoch/scheduler extension.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_4ep_public_family_featnorm_aug_prerecon_k_20260705.sh
```

Key config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
train_shards: 294
validation_shards: 14
output: /tmp/qat_public_repro/recipe4ep_k_featnorm_aug_prerecon_fixed_qkr_softkd_t275_20260705
log: /tmp/train_recipe4ep_k_featnorm_aug_prerecon_fixed_qkr_softkd_t275_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 4
scheduler_epochs: 3
pre_qat_feature_recon_updates: 100
pre_qat_feature_recon_layers: features.5.5,features.7.1
pre_qat_feature_recon_policy: quant
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
teacher_feature_output_weight: 0.005
teacher_feature_output_layers: features.5.5,features.7.1
teacher_feature_output_loss: norm_mse
smoothing: 0.1
mixup: 0.0
cutmix: 0.0
aa: rand-m9-mstd0.5-inc1
color_jitter: 0.4
reprob: 0.25
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations
```

Pre-QAT reconstruction markers:

```text
PreQATFeatRecon: update=1/100 loss=0.744460 kept=67764 masked=27767356
PreQATFeatRecon: update=50/100 loss=0.620180 kept=67764 masked=27767356
PreQATFeatRecon: update=100/100 loss=0.606486 kept=67764 masked=27767356
```

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9325 | 78.4080 | 94.5280 | 50000 | below E/J epoch1 `78.5760` |
| 2 | 0.8773 | 79.7220 | 95.0020 | 50000 | above E/J epoch2 `79.6400` |
| 3 | 0.8598 | 80.0480 | 95.1540 | 50000 | below E/J epoch3 `80.0880` |
| 4 | 0.8571 | 80.0320 | 95.1340 | 50000 | failed `epoch4 >=80.5` |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `checkpoint-2.pth.tar` | saved |
| `checkpoint-3.pth.tar` | saved |
| `checkpoint-4.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-4 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223236s | 2293.54 |
| 2 | 2496 | 0.223194s | 2293.96 |
| 3 | 2496 | 0.223210s | 2293.80 |
| 4 | 2496 | 0.223421s | 2291.64 |

Status:

- K is a valid 4epoch public-style W4A4-family result, but fails the `80.5` gate by `0.4680` Top-1.
- Pre-QAT feature reconstruction improves the reconstruction objective and briefly improves epoch2 versus E/J, but it does not improve epoch3/4 accuracy.
- Do not extend K to 5epoch under the current rule.

### Single-recipe 4epoch L plan: E plus teacher-confidence weighted soft-KD

Reason: Directly extending E and adding quant-state curriculum did not pass the 4epoch gate. L returns to the strongest E/J trajectory and changes the teacher supervision itself: soft-KD samples are weighted by FP teacher confidence (`teacher_confidence_kd_power=1.0`). This is a public-reference-compatible teacher supervision change and tests whether high-confidence teacher signals should dominate the short QAT window.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_4ep_public_family_featnorm_aug_confkd_l_20260705.sh
```

Key planned config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
output: /tmp/qat_public_repro/recipe4ep_l_featnorm_aug_confkd_fixed_qkr_softkd_t275_20260705
log: /tmp/train_recipe4ep_l_featnorm_aug_confkd_fixed_qkr_softkd_t275_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 4
scheduler_epochs: 3
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
teacher_confidence_kd_power: 1.0
teacher_feature_output_weight: 0.005
teacher_feature_output_layers: features.5.5,features.7.1
teacher_feature_output_loss: norm_mse
smoothing: 0.1
mixup: 0.0
cutmix: 0.0
aa: rand-m9-mstd0.5-inc1
color_jitter: 0.4
reprob: 0.25
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations
```

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 1.0237 | 76.8100 | 93.7060 | 50000 | failed early sanity; stopped |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-1 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223387s | 2291.99 |

Status:

- L is a valid early-stop public-style W4A4-family result, but confidence-weighted soft-KD is much worse than E/J at epoch1 (`76.8100` vs `78.5760`).
- Stopped after epoch1 to avoid wasting compute; do not continue this main-KD confidence weighting direction.

### Single-recipe 4epoch M plan: E plus mid/late feature pyramid distillation

Reason: Confidence-weighted main KD is too aggressive, and pre-QAT feature reconstruction does not transfer into final Top-1. M keeps E/J's stable soft-KD objective and strengthens only the auxiliary structural supervision by adding a mid-stage feature layer to the existing late feature-output distillation. This tests whether a wider feature pyramid can guide quantized intermediate representations without distorting the main soft target.

Command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_4ep_public_family_featpyramid_aug_m_20260705.sh
```

Key planned config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
output: /tmp/qat_public_repro/recipe4ep_m_featpyramid_aug_fixed_qkr_softkd_t275_20260705
log: /tmp/train_recipe4ep_m_featpyramid_aug_fixed_qkr_softkd_t275_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 4
scheduler_epochs: 3
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
teacher_feature_output_weight: 0.005
teacher_feature_output_layers: features.3.1,features.5.5,features.7.1
teacher_feature_output_loss: norm_mse
smoothing: 0.1
mixup: 0.0
cutmix: 0.0
aa: rand-m9-mstd0.5-inc1
color_jitter: 0.4
reprob: 0.25
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4, statsq weights, LSQ activations
```

Hook verification:

```text
Teacher feature-output hooks: layers=('features.3.1', 'features.5.5', 'features.7.1'), student_matches=('features.3.1', 'features.5.5', 'features.7.1'), teacher_matches=('features.3.1', 'features.5.5', 'features.7.1')
Teacher feature-output debug: student_count=3, teacher_count=3
```

Result:

| epoch | loss | Top-1 | Top-5 | samples | gate |
|---|---:|---:|---:|---:|---|
| 1 | 0.9463 | 78.2840 | 94.3900 | 50000 | below E/J `78.5760`; stopped |

Artifacts:

| artifact | status |
|---|---|
| `args.yaml` | saved |
| `checkpoint-1.pth.tar` | saved |
| `last.pth.tar` | saved, same epoch as checkpoint-1 |

Timing:

| epoch | updates | avg step time | samples/sec |
|---|---:|---:|---:|
| 1 | 2496 | 0.223748s | 2288.29 |

Status:

- M is a valid early-stop public-style W4A4-family result, but adding a mid-stage feature layer hurts epoch1 relative to E/J.
- Stopped after epoch1; do not continue wider feature-pyramid auxiliary loss as the main path.

### Single-recipe 4epoch N plan: activation-only progressive quant curriculum

Reason: I/J/K/L/M show that extending epochs, pre-QAT feature reconstruction, confidence-weighted main KD, and wider feature auxiliary losses do not pass the 4epoch gate. Prior progressive-bit work showed full W8/6 to W4 switching is unstable, especially for weights. N keeps target W4 weights from the beginning and only relaxes activation quantization during epoch0 (`W4A8 -> W4A4`), then runs the stable E/J teacher supervision. This is a quantization curriculum with final W4A4-family evaluation.

Planned command:

```bash
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/run_4ep_public_family_actcurr_aug_n_20260705.sh
```

Key planned config:

```text
commit: 549141c
data: /tmp/imagenet1k_full_parquet
output: /tmp/qat_public_repro/recipe4ep_n_actcurr_aug_fixed_qkr_softkd_t275_20260705
log: /tmp/train_recipe4ep_n_actcurr_aug_fixed_qkr_softkd_t275_20260705.log
devices: 0,1,2,3,4,5,6,7
epochs: 4
scheduler_epochs: 3
progressive_bit_schedule: 0:4:8,1:4:4
batch_size: 64 per GPU, global batch 512
lr: 2e-4
min_lr: 1e-5
quant_lr_multiplier: 4
kd_hard_and_soft: 0
teacher_soft_temperature: 2.75
teacher_feature_output_weight: 0.005
teacher_feature_output_layers: features.5.5,features.7.1
teacher_feature_output_loss: norm_mse
smoothing: 0.1
mixup: 0.0
cutmix: 0.0
aa: rand-m9-mstd0.5-inc1
color_jitter: 0.4
reprob: 0.25
recipe: fixed-QKR, first/last 8-bit, main blocks W4A4 final target, epoch0 W4A8 activation curriculum
```
