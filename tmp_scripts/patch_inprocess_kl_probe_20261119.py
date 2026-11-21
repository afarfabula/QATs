import io, shutil, sys, time

P = "/home/quyanyi/tiger/resume_repos/QATs/qat_launch.py"
src = io.open(P, encoding="utf-8").read()

if "_dsh_teacher_kl_probe" in src:
    print("already patched")
    sys.exit(0)

FUNC = '''

def _dsh_teacher_kl_probe(model, teacher, runtime_args, epoch, head_mode, n_batches=6):
    """[DSH] One-off diagnostic: measure the teacher-attention KL in-process, before saving."""
    import os as _os
    inner = maybe_unwrap_ddp(model)
    was_training = inner.training
    cfg = resolve_data_config(vars(runtime_args), model=None)
    ds = create_dataset_compat(
        runtime_args.dataset,
        root=runtime_args.data_dir,
        split=runtime_args.val_split,
        is_training=False,
        batch_size=32,
        subset_ratio=1.0,
        rank=0,
        world_size=1,
    )
    ld = create_loader_compat(
        ds,
        input_size=cfg["input_size"],
        batch_size=32,
        is_training=False,
        use_prefetcher=False,
        interpolation=cfg["interpolation"],
        mean=cfg["mean"],
        std=cfg["std"],
        num_workers=4,
        distributed=False,
        crop_pct=cfg["crop_pct"],
        pin_memory=False,
    )

    def _run(use_amp):
        inner.eval()
        vals, stats = [], []
        try:
            with torch.no_grad():
                for i, batch in enumerate(ld):
                    if i >= n_batches:
                        break
                    imgs = batch[0] if not isinstance(batch, torch.Tensor) else batch
                    imgs = imgs.cuda(non_blocking=True)
                    if use_amp:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            s_out = inner(imgs)
                            t_out = teacher(imgs)
                            loss = attention_kl_consistency_loss(
                                s_out[1], t_out[1], head_mode=head_mode,
                                loss_type="kl_ref", clip_value=0.0, temperature=2.0)
                    else:
                        s_out = inner(imgs)
                        t_out = teacher(imgs)
                        loss = attention_kl_consistency_loss(
                            s_out[1], t_out[1], head_mode=head_mode,
                            loss_type="kl_ref", clip_value=0.0, temperature=2.0)
                    vals.append(float(loss))
                    lst = extract_attn_prob_list(s_out[1])
                    for li in (8, 9, 10, 11):
                        if li < len(lst) and torch.is_tensor(lst[li]):
                            x = lst[li].float()
                            ssum = x.sum(-1)
                            p = x / ssum.clamp_min(1e-8).unsqueeze(-1)
                            h = -(p * p.clamp_min(1e-8).log()).sum(-1).mean()
                            stats.append((float(ssum.mean()), float((x == 0).float().mean()), float(h)))
            kl = sum(vals) / max(len(vals), 1)
            mass = sum(t[0] for t in stats) / max(len(stats), 1)
            zero = sum(t[1] for t in stats) / max(len(stats), 1)
            ent = sum(t[2] for t in stats) / max(len(stats), 1)
            print(f"[DSH-PROBE] epoch={epoch} amp={int(use_amp)} KL={kl:.3f} "
                  f"collected_attn mass={mass:.4f} zero={zero:.4f} H={ent:.3f} n={len(vals)}", flush=True)
        finally:
            inner.train(was_training)

    _run(False)
    _run(True)
    s_list = [p.detach().float().flatten() for n, p in inner.named_parameters()
              if n.endswith("quan_a_softmax_fn.s")]
    if s_list:
        all_s = torch.cat(s_list)
        print(f"[DSH-PROBE] softmax_scale s: mean={all_s.mean():.6f} min={all_s.min():.6f} "
              f"max={all_s.max():.6f} n={all_s.numel()}", flush=True)

'''
anchor_def = "def train_one_epoch_ofq("
idx = src.index(anchor_def)
src = src[:idx] + FUNC.lstrip("\n") + "\n" + src[idx:]

CALL = '''            if os.environ.get("DSH_TEACHER_KL_PROBE") == "1" and runtime_args.local_rank == 0:
                try:
                    _dsh_teacher_kl_probe(
                        model, teacher, runtime_args, epoch,
                        getattr(runtime_args, "ref_head_mode", "all"),
                    )
                except Exception as _probe_err:
                    print(f"[DSH-PROBE] failed: {_probe_err!r}", flush=True)
'''
call_anchor = "                confidence_ref_model,\n            )\n"
cidx = src.index(call_anchor) + len(call_anchor)
src = src[:cidx] + CALL + src[cidx:]

shutil.copyfile(P, P + ".bak_dsh")
io.open(P, "w", encoding="utf-8", newline="\n").write(src)
print("patched", P)
