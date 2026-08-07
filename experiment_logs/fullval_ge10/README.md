# Full-Val >=10 Training Logs Archive

Archived logs from `/mlx_devbox/users/quyanyi/playground` with at least 10 `Test: [distributed-summary]` entries.
Each `Test: [distributed-summary]` line is treated as one complete validation point.

- log_count: 23
- manifest: `manifest_fullval_ge10.csv`

Top by best Acc@1:

| file | fullval_count | best_epoch | best_acc1 | last_epoch | last_acc1 |
|---|---:|---:|---:|---:|---:|
| `playground__launch_ofq_200ep_fixedcycle_refkl_20260731.nohup.log` | 200 | 194 | 80.868 | 199 | 80.708 |
| `playground__train_ofq_200ep_fromscratch_fixedcycle_group_sparse_prevstep_refkl_20260731.log` | 200 | 194 | 80.868 | 199 | 80.708 |
| `playground__launch_ofq_resume200_to300_fixedcycle_refkl_20260802.nohup.log` | 100 | 209 | 80.842 | 299 | 80.69 |
| `playground__train_ofq_resume200_to300_fixedcycle_sparse_prevstep_refkl_20260802.log` | 100 | 209 | 80.842 | 299 | 80.69 |
| `playground__train_ofq_resume10_to210_late_sparse_prevstep_refkl_20260712.log` | 200 | 98 | 80.828 | 209 | 80.614 |
| `playground__launch_ofq_100ep_fromscratch_teacher_sparse_attnkl_clipgrad_20260804.nohup.log` | 100 | 99 | 80.818 | 99 | 80.818 |
| `playground__train_ofq_100ep_fromscratch_teacher_sparse_attnkl_clipgrad_20260804.log` | 100 | 99 | 80.818 | 99 | 80.818 |
| `playground__launch_ofq_100ep_fromscratch_teacher_sparse_attnkl_fixed_20260803.nohup.log` | 100 | 81 | 80.792 | 99 | 80.678 |
| `playground__train_ofq_100ep_fromscratch_original_ofq_public_control_20260714.log` | 100 | 81 | 80.792 | 99 | 80.678 |
| `playground__train_ofq_100ep_fromscratch_teacher_sparse_attnkl_fixed_20260803.log` | 100 | 81 | 80.792 | 99 | 80.678 |
| `playground__train_ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713.log` | 100 | 99 | 80.772 | 99 | 80.772 |
| `playground__train_ofq_resume10_to110_dynamic_sparse_prevstep_refkl_20260710.log` | 100 | 99 | 80.76 | 109 | 80.66 |
| `playground__train_ofq_resume10_to110_original_ofq_public_20260711.log` | 100 | 101 | 80.752 | 109 | 80.686 |
| `playground__train_ofq_resume10_to60_original_ofq_public_20260710.log` | 50 | 51 | 80.724 | 59 | 80.57 |
| `playground__train_ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709.log` | 50 | 53 | 80.682 | 59 | 80.562 |
| `playground__train_ofq_public_resume10_to30_20260709.log` | 20 | 26 | 80.598 | 29 | 80.42 |
| `playground__launch_ofq_100ep_fromscratch_teacher_sparse_attnkl_latepolish_20260805.nohup.log` | 66 | 59 | 80.526 | 65 | 80.48 |
| `playground__train_ofq_100ep_fromscratch_teacher_sparse_attnkl_latepolish_20260805.log` | 66 | 59 | 80.526 | 65 | 80.48 |
| `playground__train_ofq_resume10_to30_late_prevstep_refkl_20260709.log` | 16 | 25 | 80.484 | 25 | 80.484 |
| `playground__train_recipe100ep_e_featnorm_aug_fixed_qkr_softkd_t275_20260705.log` | 62 | 59 | 80.414 | 61 | 80.406 |
| `playground__train_recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706.log` | 10 | 9 | 80.364 | 9 | 80.364 |
| `playground__launch_ofq_100ep_fromscratch_teacher_sparse_attnkl_20260803.nohup.log` | 37 | 36 | 79.964 | 36 | 79.964 |
| `playground__train_ofq_100ep_fromscratch_teacher_sparse_attnkl_20260803.log` | 37 | 36 | 79.964 | 36 | 79.964 |
