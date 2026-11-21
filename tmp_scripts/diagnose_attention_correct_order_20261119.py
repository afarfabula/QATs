import argparse, sys, torch, contextlib
import torch.nn.functional as F
from pathlib import Path
QATS = Path('/home/quyanyi/tiger/resume_repos/QATs')
for p in (QATS, QATS/'third_party'/'OFQ'/'tools', QATS/'third_party'/'OFQ'):
    sys.path.insert(0, str(p))
import qat_launch as ql
from offline_attention_oscillation import build_args, build_probe_loader
from timm.models import create_model, load_checkpoint

HEAD = 'custom_subset:8:0,8:1,8:2,9:0,9:1,9:2,10:0,10:1,10:2,11:0,11:1,11:2'
ap = argparse.ArgumentParser()
ap.add_argument('--args-yaml'); ap.add_argument('--checkpoint'); ap.add_argument('--tag')
ap.add_argument('--data', default='/datadisk2/linyichen/OFQ/ImageNet-1K'); ap.add_argument('--batches', type=int, default=8)
a = ap.parse_args()
ra = build_args(a.args_yaml); ra.batch_size = 32; ra.workers = 4; ra.prefetcher = False; ra.collect_attention = True
m = create_model(ra.model, num_classes=ra.num_classes, drop_rate=ra.drop, pretrained=False, qqkkvv=False)
m = ql.get_ofq_qat_model(m, ra); ql.enable_attention_collection(m); m = m.cuda()
ql.set_attention_mode(m, collect_attention=True, qqkkvv=False)
ql.set_attention_collection_target(m, 'post_quant')

# --- 关键：先 setup_alpha 建出 s，再 load checkpoint ---
ds_tr = ql.create_dataset_compat(ra.dataset, root=ra.data_dir, split=ra.train_split, is_training=True,
                                 batch_size=32, subset_ratio=1.0)
ld_tr = ql.create_loader_compat(ds_tr, input_size=224, batch_size=32, is_training=True, use_prefetcher=False,
    no_aug=False, re_prob=ra.reprob, re_mode=ra.remode, re_count=ra.recount, re_split=ra.resplit,
    scale=ra.scale, ratio=ra.ratio, hflip=ra.hflip, vflip=ra.vflip, color_jitter=ra.color_jitter,
    auto_augment=ra.aa, num_aug_splits=ra.aug_splits, num_aug_repeats=ra.num_aug_repeats,
    interpolation='bicubic', mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225),
    num_workers=4, distributed=False, collate_fn=None, pin_memory=False, use_multi_epochs_loader=False)
amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
ra.setup_alpha_batches = 1
ql.setup_alpha(m, ld_tr, ra, amp)
n_s = len([1 for n,_ in m.named_parameters() if n.endswith('quan_a_softmax_fn.s')])
load_checkpoint(m, a.checkpoint, strict=False)
n_s2 = len([1 for n,_ in m.named_parameters() if n.endswith('quan_a_softmax_fn.s')])
print(f'[{a.tag}] s params after setup_alpha={n_s} after load={n_s2}')

teacher = ql.create_ofq_teacher_model(ra)
ql.set_attention_mode(teacher, collect_attention=True, qqkkvv=False)
ql.set_attention_collection_target(teacher, 'post_quant'); teacher = teacher.cuda().eval()
loader = build_probe_loader(ra, a.data)
m.eval()
kls, mass, zero, top5 = [], [], [], []
caps, hs = ql.install_attention_quantization_hooks(m)
with torch.no_grad():
    for i, batch in enumerate(loader):
        if i >= a.batches: break
        imgs = batch[0] if not isinstance(batch, torch.Tensor) else batch
        imgs = imgs.cuda(non_blocking=True)
        caps.clear()
        s_out = m(imgs); t_out = teacher(imgs)
        kls.append(float(ql.attention_kl_consistency_loss(s_out[1], t_out[1], head_mode=HEAD,
                                                          loss_type='kl_ref', clip_value=0.0, temperature=2.0)))
        t_attn = t_out[1]
        for name, pre, post in caps:
            layer = next(int(p) for p in name.split('.') if p.isdigit())
            if layer not in (8,9,10,11) or t_attn[layer] is None: continue
            x = post.float(); s = x.sum(-1)
            p = x / s.clamp_min(1e-8).unsqueeze(-1)
            mass.append(float(s.mean())); zero.append(float((x==0).float().mean()))
            t = t_attn[layer].float()
            k = min(5, x.shape[-1])
            if k > 1:
                ia = x.topk(k, -1).indices; ib = t.topk(k, -1).indices
                hit = (ia.unsqueeze(-1) == ib.unsqueeze(-2)).any(-1).float()
                top5.append(float(hit.sum(-1).mean() / k))
for h in hs: h.remove()
n = lambda v: sum(v)/max(len(v),1)
print(f'[{a.tag}] KL(trainer formula)={n(kls):.2f}  mass={n(mass):.3f} zero={n(zero):.3f} top5_overlap={n(top5):.3f}')
