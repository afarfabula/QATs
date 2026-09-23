# 在本机 8×RTX3090 上跑 OFQ Swin-T QAT：agent 运行手册

> 面向对象：要在这台机器上跑 Swin-T W4A4 OFQ QAT 实验的其他 agent。
> 最后验证：2026-09-20（GPU 6，真实 ImageNet folder 数据，32 步冒烟，成功）

## 0. 最短路径

```bash
# 1) 先看哪张卡空。注意：agent 沙箱里看不到 GPU 设备，下面两条都要提权执行
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv

# 2) 起一次 32 步冒烟（把 --devices 换成空卡号，--master-port 用没被占的端口）
cd /home_ext/quyanyi/tiger/resume_repos/QATs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
setsid nohup /tmp/qat_env/bin/python qat_launch.py \
  --method ofq --stage train --config third_party/OFQ/configs/swin_t_imagenet.attn_q.yml \
  --model swin_t --data /datadisk2/linyichen/OFQ/ImageNet-1K --dataset-format folder \
  --output /tmp/qat_runs --experiment doc_smoke_gpu6 \
  --devices 6 --nproc-per-node 1 --master-port 30621 \
  --model-type swin --teacher swin_t --teacher-type swin \
  --epochs 1 --scheduler-epochs 1 --batch-size 32 --workers 8 \
  --lr 2e-4 --min-lr 1e-5 --weight-decay 0.0 \
  --epoch-checkpoint-interval 10000 --checkpoint-hist 0 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq \
  --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --use-kd --kd-hard-and-soft 1 --quantized --qk-reparam --qk-reparam-type 0 \
  --amp --amp-dtype bf16 \
  --extra-arg=--skip_validate --extra-arg=--max_train_updates --extra-arg=32 \
  --extra-arg=--log-interval --extra-arg=8 --extra-arg=--seed --extra-arg=42 \
  > /tmp/qat_doc_smoke_gpu6.log 2>&1 & echo "PID=$!"
```

## 1. 关键事实速查

| 项 | 值 |
|---|---|
| 代码仓库 | `/home_ext/quyanyi/tiger/resume_repos/QATs` |
| 统一入口 | `qat_launch.py`（内部转发给 `third_party/OFQ/train.py`，不要直接调 train.py） |
| Python 环境 | **`/datadisk2/quyanyi/envs/qat_env/bin/python`**（持久；Python 3.10.18、torch 2.5.1+cu121、timm 0.5.4）<br>旧路径 `/tmp/qat_env` 仍在，但它是易失的 overlay，别再用 |
| 数据（folder） | `/datadisk2/linyichen/OFQ/ImageNet-1K`（**别人的目录，只读，禁止写入**） |
| 数据（parquet） | 需要自己准备，例如 `/tmp/imagenet1k_full_parquet`（**本机当前没有**，见第 3 节） |
| 历史假数据 | `/tmp/qat_fake_data`（8192 张，易失） |
| 输出目录 | 自己指定 `--output`，建议 `/tmp/qat_runs/<实验名>` 或仓库外自有目录 |
| 单卡速度 | batch 32 稳定态 **0.33 s/step，约 97 img/s**（3090 空闲时实测） |
| 单卡一个 epoch | 约 **3.7 小时**（1,281,167 张 ÷ 97 img/s ≈ 13,200 s） |
| 单卡一个 epoch 步数 | 40,036 步（batch 32） |
| GPU 权限 | agent 沙箱内没有 `/dev/nvidia*`：`nvidia-smi` 与 `torch.cuda.is_available()` 都不可用，**任何 GPU 操作都必须提权执行** |

## 2. 机器与资源

- 8× NVIDIA GeForce RTX 3090（24GB），CPU 80 核，内存约 503GB，**多人共用**。
- 8 张卡在硬件层确认存在：`lspci | grep -i nvidia`、`/proc/driver/nvidia/gpus`（`1a:00.0`、`1b:00.0`、`3d:00.0`、`3e:00.0`、`88:00.0`、`89:00.0`、`b1:00.0`、`b2:00.0`）。
- 跑之前必须查空卡，别抢别人正在跑的卡；显存与算力是节点级稀缺资源。
- 判断「空」的标准：`utilization.gpu = 0%` 且 `memory.used` 很小（约 20MiB 以下）。只有 `0%` 但占着几 GB 的卡通常是别人的进程在等数据，别用。

## 3. 数据

### 3.1 folder 模式（本机现在唯一可用，已跑通）

```text
/datadisk2/linyichen/OFQ/ImageNet-1K
├── train/   1000 类，1,281,167 张
├── val/     1000 类，50,000 张
├── test/    100,000 张
└── ILSVRC2012_devkit_t12/
```

- 权限：全局可读，`linyichen` 所有，**只读使用，绝对不要往里写任何东西**。
- 用法：`--data /datadisk2/linyichen/OFQ/ImageNet-1K --dataset-format folder`。

### 3.2 parquet 模式（历史 100ep 长跑用的格式，本机当前缺失）

- 期望结构：`<root>/data/train-*.parquet` 与 `<root>/data/validation-*.parquet`（代码也接受直接放在 `<root>/` 下），每行含 `image.bytes` + `label`。
- 参数：`--data <root> --dataset-format parquet`（`parquet` 是默认值，不写 `--dataset-format` 就是它）。
- 本机 `/tmp/imagenet1k_full_parquet` 已不存在（`/tmp` 被清理）。要复现历史长跑，需要重新转换或另找分片。

### 3.3 预训练权重

- torchvision 的 `swin_t-704ceda3.pth`（109MB）在本机**没有缓存**，`~/.cache/torch/hub/checkpoints/` 目录都不存在。
- KD 的 FP teacher 需要它；没有权重时 teacher 是随机初始化，**只能测速、不能出精度结论**。
- 沙箱内无网络，下载权重需要提权。

## 4. 代码与环境的坑

1. **`qat_launch.py` 有未提交的本地改动**（`git status` 显示 `M qat_launch.py`）：为支持 folder 数据做了三处 dataset 名映射（`torch/imagenet` → `torch/image_folder`，约 510 / 1259 / 1591 行）和 `"val_split": "val"`（1261 行）。**不要 `git checkout` 覆盖它**，否则 folder 模式会挂。
2. **venv 的持久副本**（2026-09-20 迁移）：
   - `/datadisk2/quyanyi/envs/qat_env` 是**可用**的持久环境，脚本默认用它；
   - 结构：`include-system-site-packages = true`，`home = /datadisk2/quyanyi/envs/llava/bin`，
     也就是 torch / torchvision 这些大件（6.9G）来自 `/datadisk2/quyanyi/envs/llava`（持久），
     qat_env 自己只装了 465MB 的 overlay（timm 0.5.4、huggingface_hub、matplotlib、pyarrow 等 63 个包）；
   - 旧的 `/tmp/qat_env` 是同一份 overlay 的易失副本，`/tmp` 被清就会消失（需要重建时按上面的包列表装）。
     迁移方式：`cp -a /tmp/qat_env <新路径>`，然后改掉 `bin/activate*` 里的 `VIRTUAL_ENV`
     和 `bin/*` 里 7 个脚本的 shebang（本次已改完，`grep -rl '/tmp/qat_env'` 为 0）。
3. 验证集路径在本机**还没冒烟过**（历史冒烟都带 `--skip_validate`）。第一次跑全量验证要盯一下 `val` 读取。
4. 启动器把 `--extra-arg=X` 原样透传给底层 `train.py`；OFQ 专属开关（`--skip_validate`、`--max_train_updates`、`--log-interval`、`--seed`、`--static-graph`、`--smoothing` 等）都走这个通道。
5. 日志第 2 行 `[QATs] command=...` 是实际透传到底层的完整命令，排障先看它。

## 5. 常用参数

| 参数 | 含义 | 本机建议 |
|---|---|---|
| `--method ofq --stage train` | 走 OFQ 训练流程 | 固定 |
| `--config third_party/OFQ/configs/swin_t_imagenet.attn_q.yml` | Swin-T 配置 | 固定 |
| `--model swin_t --model-type swin` | 学生模型 | 固定 |
| `--data <路径>` | 数据根 | folder 用 linyichen 那份 |
| `--dataset-format folder` | 数据格式 | 本机目前只能 folder |
| `--output <目录>` | 输出根目录 | 放自己名下目录 |
| `--experiment <名字>` | 实验名（子目录） | 带日期和卡号，便于追溯 |
| `--devices 6` 或 `--devices 0,1,2,3` | 可见 GPU 列表 | 只选空卡；多卡 DDP 会据此推 world_size |
| `--nproc-per-node N` | 每节点进程数 | 单卡 1；多卡等于卡数 |
| `--master-port <端口>` | DDP 通信端口 | 多人共用机器，**必须各用各的**（可按卡号编，如 30620+卡号） |
| `--batch-size 32` | 单进程 batch | 3090 24GB 上 bs32 + KD + 注意力矩阵已验证可跑 |
| `--workers 8` | dataloader 进程数 | 8–16；folder 模式靠它做 JPEG 解码，Data 时间应接近 0 |
| `--epochs` / `--scheduler-epochs` | 训练轮数 / lr 调度总轮数 | 冒烟用 1 |
| `--wbits 4 --abits 4` | W4A4 | 固定 |
| `--wq-mode statsq --aq-mode lsq` | 量化器 | 固定 |
| `--wq-per-channel --aq-per-channel --aq-clip-learnable` | per-channel + 可学 clip | 固定 |
| `--use-kd --kd-hard-and-soft 1` | 知识蒸馏 | 固定 |
| `--quantized --qk-reparam --qk-reparam-type 0` | 量化 + QK reparam | 固定 |
| `--amp --amp-dtype bf16` | 混合精度 | 固定 |
| `--checkpoint-hist 0 --epoch-checkpoint-interval 10000` | 不存历史 epoch checkpoint | 冒烟必备，否则会写大盘 |
| `--extra-arg=--skip_validate` | 跳过验证 | 只测速 / 冒烟时用 |
| `--extra-arg=--max_train_updates N` | 跑 N 步就停 | 冒烟用 24 或 32 |

## 6. 后台跑与观察

```bash
# 起（setsid 脱离会话，nohup 防挂断）
cd /home_ext/quyanyi/tiger/resume_repos/QATs
... setsid nohup /tmp/qat_env/bin/python qat_launch.py ... > /tmp/<log>.log 2>&1 &

# 看进度
grep "TrainSummary" /tmp/<log>.log | tail -3            # 每个 epoch 的平均速度
grep -o "Time: [0-9.]*s," /tmp/<log>.log | tail -3      # 实时 s/step
tail -f /tmp/<log>.log

# 确认进程与卡
ps -ef | grep "[q]at_launch.py"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv

# 收工：按 PID 精确 kill，别用 pkill 误杀别人的训练
kill <PID>
```

日志里要盯的行：

- `TrainSummary: epoch=N updates=... avg_step_time=...s samples_per_sec=...`：每 epoch 平均速度。
- `Test: [distributed-summary] ... Samples: 50000`：全量验证点（历史长跑每 epoch 一个）。
- `Stopped early after N optimizer updates in epoch 0.`：`--max_train_updates` 生效，冒烟成功结束。
- 注意：即使 `--checkpoint-hist 0`，每次跑完仍会写 `last.pth.tar` 和 `checkpoint-<epoch>.pth.tar`（32 步冒烟一次约 329MB）。冒烟后记得删掉自己的输出目录。

## 7. 速度基准（一个 epoch 要多久）

| 配置 | 机器 | s/step | 吞吐 | 一个 epoch |
|---|---|---:|---:|---|
| bs32、W4A4 + KD + qk-reparam + bf16、folder 真实数据 | 1×3090 | 稳定态 0.33（首步 3.7–5.4s 是预热） | ~97 img/s | **约 3.7 小时**（40,036 步） |
| 同上，但假数据（8192 张） | 1×3090 | 0.335 | ~95 img/s | —（每 epoch 只 256 步） |
| bs64、global batch 512、parquet 真实数据 | 8×H100（**另一台机器**） | 0.232 | ~2,200 img/s | 约 9.7 分钟 |

说明：

- 3090 的数字是 2026-09-20 在空闲卡上实测；bs32 基本是算力瓶颈（Data 时间约 0.004s），想快只能加卡或改精度方案。
- 多卡 DDP 近似线性：8 卡理想情况把 3.7 小时压到约 30 分钟，但 folder 模式的 JPEG 解码、DDP 通信会打折，以实测为准。
- 长跑按自己盘位设 `--epoch-checkpoint-interval 1 --checkpoint-hist 2` 这类小保留策略。

## 8. 已知坑与事故记录

1. **2026-08-31 GPU 5 掉驱动（单卡硬件/驱动级故障）**：W4A4 QAT 假数据测速，epoch 0 的 256 步正常结束后，epoch 1 第一步 `.cuda()` 报 `CUDA driver error: unknown error`；随后 `nvidia-smi` 报 `Unable to determine the device handle for GPU5`，并扩散到节点级——其他 7 张卡可见但任何新进程都建不了 CUDA 上下文（存量长任务不受影响）。9/1 节点重启后恢复，9/3 复核 8 卡健康。
   - 判断：Xid 级硬件/驱动事件，非训练代码问题。诱因很可能是那张卡当天先被别人跑到 79% 满载（317W），紧接着的满载压测成了「最后一根稻草」。
   - 对策：长跑前/中挂 `nvidia-smi dmon` 留证据；别用刚被跑满的卡；OOM 后不要立刻换配置重试；若再复现，报管理员跑 `dmesg | grep -i xid` 或 `nvidia-bug-report.sh`。
2. **同参数在 GPU 2 复现（9/3）**：3 epoch × 256 步完整跑完，未复现；0.335 s/step。日志 `/tmp/qat_repro_gpu2_3ep.log`。
3. **`/tmp` 会被清理**：venv、parquet 数据、输出都可能消失。长期产物放 `/datadisk2/quyanyi/` 或自己名下目录。
4. **master-port 冲突**：多人同时起 DDP 必须用不同端口，否则握手失败。
5. **OOM**：3090 只有 24GB，bs32 + KD 已接近上限；OOM 是干净退出、不伤卡，但别马上加大 batch 重试。
6. **预训练权重缺失**：见 3.3；没有 teacher 权重就只有测速意义，不能出精度结论。

## 9. 2 卡 2 epoch 复现实测（2026-09-20，进行中）

目标：在这台机器上用 GPU 6+7 跑 2 个 epoch，实测 wall time，同时验证完整训练链路
（folder 数据 + 预训练权重 + KD teacher + W4A4 QAT + DDP）在 3090 上是否跑得通。

### 两次尝试

| | 第 1 次（已停） | 第 2 次（当前，4 卡） |
|---|---|---|
| 脚本 | `tmp_scripts/run_repro_2ep_2gpu_3090_20260920.sh` | `tmp_scripts/run_repro_2ep_4gpu_3090_20260920.sh` |
| 卡 | GPU 6,7 | GPU 2,5,6,7 |
| accum | 传 8 → 实际 **4** | 传 16 → 实际 **4** |
| 每个优化器 step | 32×2×4 = 256 | 32×4×4 = **512** |
| 每 epoch 微步 | 20,018 | 10,009 |
| 每 epoch 优化器 step | 5,004 | **2,502** |
| `max_train_updates=5004` 含义 | 只够 **1 个 epoch** | 正好 **2 个 epoch** |
| 结果 | 跑到 12,100/20,018 微步手动停掉，`wall_seconds=10733`（2.98h） | 进行中 |

第 1 次停掉的原因见上文"启动器会把 `--grad-accum-steps` 再除以 world_size"。

### 4 卡版本实测（启动后 6 分钟）

```text
Effective batch alignment: per_gpu_effective_batch=32, loader_batch=32, accum=4,
world_size=4, global_effective_batch=128
```

| 指标 | 实测 |
|---|---|
| 微步时间 | 0.800 s（10,009 微步/epoch） |
| 聚合吞吐 | **160 img/s**（每卡 40 img/s） |
| 4 张卡 | 全部 100% 利用率、显存 14.2GB/卡、212–320W |
| 单 epoch | 10,009 × 0.8 s ≈ **2.2 h**（+ 约 5 min 全量验证） |
| **2 epoch 预期 wall** | **≈ 4.5 h** |

注：这里的 160 img/s 是**含数据卡顿的长期平均**。按区间分布看，不吃卡顿时的瞬时吞吐是
128 张 / 0.32 s = **400 img/s 聚合（每卡 100 img/s）**，与单卡基准一致；
所以多卡效率本身是好的，被拉低的是被 11.8% 的、3–13 s 的数据等待区间拖出来的平均值。

### 第 1 次（2 卡）配置留档

```text
日志   /datadisk2/quyanyi/qat_runs/repro_2ep_2gpu_3090_20260920.log
输出   /datadisk2/quyanyi/qat_runs/repro_2ep_2gpu_3090_20260920/
权重   /home_ext/quyanyi/qat_weights/swin_t-704ceda3.pth（113,445,839 B，torchvision 官方）
```

配置：bs32/卡 × 2 卡 × grad-accum 8 = 每个优化器 step 512 张（与历史 8×H100 run 的优化语义一致），
lr 2e-4、100 epoch 的 cosine 调度、KD soft temperature 2.75、W4A4 + QK-reparam + bf16、
`--max_train_updates 5004`（= 2 epoch × 2502 个优化器 step），每个 epoch 结束做 50k 全量验证。

### 与历史 8×H100 run 的已知差异

| 项 | 历史 run | 本机复现 |
|---|---|---|
| 数据 | parquet 分片 | folder（linyichen 那份，只读） |
| 卡 | 8 × H100 80GB | 2 × 3090 24GB |
| 单卡 batch | 64 | 32 + grad-accum 8 |
| `--static-graph` | 开 | **关**（torch 2.5.1 的 DDP reducer 会 assert） |
| torch | 2.9.1 | 2.5.1 |

所以这次只能对齐"每个优化器 step 的全局 batch"，不能对齐单卡 batch 和静态图；精度数字只能做链路校验，
不能直接和历史的 77.7080 / 78.2980 逐位对比。

### 实测：把三个配置放在同一天同一节点上对比

为了分清"配置本身慢"还是"节点被挤"，在 GPU 2/5（空卡）上跑了两次 40 步对照：

| 配置 | 微步时间 | 聚合吞吐 | 每卡 |
|---|---:|---:|---:|
| 1 卡 bs32，accum=1 | 0.331 s | 96.6 img/s | 96.6 img/s |
| 2 卡 bs32，accum=1（KD=1） | 0.349 s | 183 img/s | 91.5 img/s |
| 2 卡 bs32，accum=1（KD=0） | 0.343 s | 186 img/s | 93 img/s |
| **2 卡 bs32，accum=4（本次 run）** | **0.78–0.85 s** | **63–75 img/s** | ~35 img/s |

结论：

1. **不是节点被挤，也不是多人共用**：单卡实测 0.331 s/step 与第 7 节基准（0.33 s）完全一致；
   双卡 DDP 近线性加速（183/96.6 = 1.9×）。GPU 6/7 上只有本任务的两个进程，没有别人共享。
   （`gdm` 是系统 GNOME 登录管理器，uid 128，GPU 显存 0、CPU 0%，与本任务无关。）
2. **不是梯度累积**（更正见下一节）：accum=4 的中位步时间和 accum=1、单卡完全一致。
3. GPU 侧没有硬件瓶颈：利用率 94–100%、显存 14.1GB/卡、59℃、未降频。

### ⚠️ 更正：长期平均是被输入流水线的周期性卡顿拉高的，不是累积

早先只看了长期平均值（0.8–0.87 s/微步）就归因到梯度累积的通信放大，**这个结论是错的**。
把每个 log 区间的耗时按分布拆开看（2 卡那次，n=246 个区间）：

| 统计量 | 步时 | 其中 Data 等待 |
|---|---:|---:|
| min | 0.300 s | 0.002 s |
| **median** | **0.321 s** | 0.004 s |
| p90 | 3.343 s | 1.450 s |
| max | 9.925 s | 9.491 s |
| >3 s 的区间占比 | **11.8%** | — |

4 卡那次同样是 median 0.320 s / max 13.487 s，Data 等待 max 13.219 s。也就是说：

- **数据一到手，微步就是 0.32 s**——和单卡基准 0.331 s、accum=1 的 0.343 s 一致，
  所以 accum=4 和 4 卡 DDP 都没有可测的额外开销；
- 但平均每 8~10 个区间就有一次 3~13 s 的**数据等待**，长期平均因此被抬到 0.8–0.87 s/微步；
- 磁盘不是带宽瓶颈：实测 `/datadisk2`(sdd) 读只有 5–7 MB/s，buff/cache 有 413 GB。
  更可能是共用节点的 CPU/调度争用（load ≈ 60、9 个用户）打在下游 32 个 dataloader worker 上，
  叠加共享盘上的偶发读延迟。

影响：不吃卡顿的话单 epoch 只需 10,009 × 0.32 s = **53 分钟**；现在实测 ~2.2 h。
想提速的方向是加 worker 数、减少增强的 CPU 开销、或把数据放到更快的盘上——
而不是像这篇文档早先写的那样去动梯度累积。

### ⚠️ 启动器会把 `--grad-accum-steps` 再除以 world_size

本次按 `--grad-accum-steps 8` 提交，但日志里实际生效的是：

```text
Effective batch alignment: per_gpu_effective_batch=32, loader_batch=32, accum=4,
world_size=2, global_effective_batch=64
```

即 `ceil(8 / world_size=2) = 4`。后果有两个：

1. 真实每个优化器 step 的样本数是 32×2×4 = **256**，不是我原本想要的 512；
2. 20,018 个微步 ÷ 4 = **5,004 个优化器 step = 1 个 epoch**，
   而脚本里 `--max_train_updates 5004` 是按"2 epoch"算的——所以这次实际只会跑 **1 个 epoch**。

要多卡 + 指定全局 batch，必须把 world_size 的除法算进去（例如想要 accum=8、2 卡，就要传
`--grad-accum-steps 16`）。

### 完成后要补的数字

脚本结束时会自动把 `wall_seconds=`、`end_epoch=` 追加到同一份日志，`TrainSummary:` 行给出每个 epoch 的
`avg_step_time` / `samples_per_sec`。实测完成后回填到本节。

### 本次踩到的两个坑（已在脚本里规避）

1. **teacher 权重格式**：OFQ 的 `swin_t` 是 torchvision 移植实现（`features.*` 键名），必须用
   `download.pytorch.org/models/swin_t-704ceda3.pth`；用 timm 命名（`patch_embed.*` / `layers.*`）的权重
   会在 `load_checkpoint(..., strict=True)` 直接报 Missing/Unexpected keys。
   `--pretrained` 会自动下载并 `check_hash=True` 校验，缓存在 `<TORCH_HOME>/hub/checkpoints/`。
2. **`--static-graph` 不能用**：torch 2.5.1 的 DDP reducer 会抛
   `RuntimeError: expect_autograd_hooks_ INTERNAL ASSERT FAILED ... reducer.cpp:1603`，
   启动 60 秒内必崩。去掉即可正常训练（代价是少量通信优化）。

## 10. 交付前 checklist

- [ ] `nvidia-smi` 确认目标卡真的空着，且没和别人共用。
- [ ] `--output` 和日志都在自己名下，没往 `/datadisk2/linyichen` 等他人目录写东西。
- [ ] `--master-port` 唯一。
- [ ] 先跑 32 步冒烟，看到 `TrainSummary` 和正常下降的 loss，再起长跑。
- [ ] 长跑前确认 checkpoint 策略与磁盘余量。
- [ ] 起长跑时挂 `nvidia-smi dmon -o T -s pucvmet -d 5 > <log>` 留故障证据。
- [ ] 记录：实验名、卡号、batch、s/step、epoch 时长、日志路径。
