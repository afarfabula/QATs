import io, shutil, sys

P = "/home/quyanyi/tiger/resume_repos/QATs/qat_launch.py"
src = io.open(P, encoding="utf-8").read()

if "_dsh_topk_recall_loss" in src:
    print("already patched")
    sys.exit(0)

FUNCS = '''

def _dsh_quantize_attn_info(attn_info, model):
    """[DSH] 用学生同一套 post-softmax 量化器把教师注意力过一遍（按层号对齐）。"""
    if attn_info is None:
        return None
    mods = {}
    for name, mod in maybe_unwrap_ddp(model).named_modules():
        if not name.endswith("quan_a_softmax_fn"):
            continue
        if not hasattr(mod, "forward_detached_scale"):
            continue
        idx = next((int(p) for p in name.split(".") if p.isdigit()), None)
        if idx is not None:
            mods[idx] = mod
    out = []
    for layer, item in enumerate(attn_info):
        if item is None or not torch.is_tensor(item):
            out.append(item)
            continue
        mod = mods.get(layer)
        if mod is None:
            out.append(item)
            continue
        q = mod(item.float())
        out.append(q.to(item.dtype))
    return out


def _dsh_topk_recall_loss(student_attn_info, teacher_attn_info, head_mode, k=5):
    """[DSH] 教师 top-k 支撑被学生 top-k 覆盖的比例（1 - recall）。"""
    s_list = extract_attn_prob_list(student_attn_info)
    t_list = extract_attn_prob_list(teacher_attn_info)
    sel = parse_ref_head_mode(head_mode)
    if sel is None:
        items = [(i, None) for i in range(min(len(s_list), len(t_list)))]
    else:
        items = list(sel)
    total = None
    count = 0
    for layer, head in items:
        if layer >= len(s_list) or layer >= len(t_list):
            continue
        s_attn, t_attn = s_list[layer], t_list[layer]
        if s_attn is None or t_attn is None or s_attn.ndim < 4:
            continue
        if head is not None:
            if head >= s_attn.shape[1] or head >= t_attn.shape[1]:
                continue
            s_attn = s_attn[:, head:head + 1]
            t_attn = t_attn[:, head:head + 1]
        kk = min(int(k), s_attn.shape[-1], t_attn.shape[-1])
        t_idx = t_attn.float().topk(kk, dim=-1).indices
        s_idx = s_attn.float().topk(kk, dim=-1).indices
        hit = (s_idx.unsqueeze(-1) == t_idx.unsqueeze(-2)).any(dim=-1).float()
        layer_loss = 1.0 - hit.sum(dim=-1).mean() / float(kk)
        total = layer_loss if total is None else total + layer_loss
        count += 1
    if total is None:
        return torch.zeros((), device="cuda")
    return total / max(count, 1)

'''
anchor = "def train_one_epoch_ofq("
idx = src.index(anchor)
src = src[:idx] + FUNCS.lstrip("\n") + "\n" + src[idx:]

OLD_CALL = '''                    teacher_attn_kl_loss = attention_kl_consistency_loss(
                        student_attn_info,
                        teacher_attn_info_for_kl,
                        head_mode=runtime_args.ref_head_mode,
                        loss_type=runtime_args.ref_attn_loss,
                        clip_value=runtime_args.ref_attn_kl_clip,
                        temperature=runtime_args.teacher_attn_kl_temperature,
                    )'''
NEW_CALL = '''                    if os.environ.get("DSH_TEACHER_KL_TARGET", "") == "quantized":
                        teacher_attn_info_for_kl = _dsh_quantize_attn_info(teacher_attn_info_for_kl, model)
                    _dsh_topk_w = float(os.environ.get("DSH_TOPK_RECALL_WEIGHT", "0") or 0.0)
                    if _dsh_topk_w > 0.0:
                        runtime_args.teacher_attn_kl_weight = _dsh_topk_w
                        teacher_attn_kl_loss = _dsh_topk_recall_loss(
                            student_attn_info,
                            teacher_attn_info_for_kl,
                            runtime_args.ref_head_mode,
                            k=int(os.environ.get("DSH_TOPK_RECALL_K", "5") or 5),
                        )
                    else:
                        teacher_attn_kl_loss = attention_kl_consistency_loss(
                            student_attn_info,
                            teacher_attn_info_for_kl,
                            head_mode=runtime_args.ref_head_mode,
                            loss_type=runtime_args.ref_attn_loss,
                            clip_value=runtime_args.ref_attn_kl_clip,
                            temperature=runtime_args.teacher_attn_kl_temperature,
                        )'''

if OLD_CALL not in src:
    print("ERROR: KL call anchor not found")
    sys.exit(1)
src = src.replace(OLD_CALL, NEW_CALL, 1)

CALL_EPOCH = "            train_metrics, total_update_count, stopped_early = train_one_epoch_ofq("
FREEZE = '''            if os.environ.get("DSH_FREEZE_ATTN_S", "") == "1":
                _n_frozen = freeze_attention_softmax_scales(model)
                if runtime_args.local_rank == 0:
                    print(f"[DSH] froze attention softmax scales: {_n_frozen} tensors", flush=True)

'''
if CALL_EPOCH not in src:
    print("ERROR: epoch call anchor not found")
    sys.exit(1)
src = src.replace(CALL_EPOCH, FREEZE + CALL_EPOCH, 1)

shutil.copyfile(P, P + ".bak_dsh2")
io.open(P, "w", encoding="utf-8", newline="\n").write(src)
print("patched ok")
