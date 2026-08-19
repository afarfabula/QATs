# Long-Train Acc Logs, Misc

This directory archives training logs found outside `experiment_logs/fullval_ge10`
that meet the broader preservation rule:

- training spans more than 20 epochs; and
- the log contains at least one `Acc@1` validation record.

These are kept separate from `fullval_ge10` because not all of them contain at
least 10 `Test: [distributed-summary]` full-validation records.

## Contents

| file | train epochs | validation points | best Acc@1 | last Acc@1 | note |
|---|---:|---:|---:|---:|---|
| `playground_OFQ_out__swin_t_w4a4_qkr_30ep_bs512_eff.log` | 0-39 | 40 | 80.466 | 80.360 | Old-format full validation lines; use final running average per epoch. |
| `qats_logs__swin_t_w4a4_imagenet1k_8gpu_50ep_directverify_v2.log` | 0-39 | 1 | 78.774 | 78.774 | Long training log, but launched with `--skip_validate`; only one validation block is present. |
| `qats_logs__swin_t_w4a4_ofq_mainline_300ep_20260613_rerun2.log` | 52-309 | 1 | 78.894 | 78.894 | OFQ 300ep mainline resume log with `--skip_validate`; the Acc@1 block is the resume-source validation, not an epoch-by-epoch trajectory. |

See `manifest_long_train_acc_ge20_misc.csv` for source paths and SHA256 hashes.
