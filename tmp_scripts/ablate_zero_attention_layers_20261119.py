import argparse, sys, torch
from pathlib import Path
QATS = Path('/home/quyanyi/tiger/resume_repos/QATs')
for p in (QATS, QATS/'third_party'/'OFQ'/'tools', QATS/'third_party'/'OFQ'):
    sys.path.insert(0, str(p))
import qat_launch as ql
from offline_attention_oscillation import build_args, build_probe_loader
from timm.models import create_model, load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument('--args-yaml'); ap.add_argument('--checkpoint'); ap.add_argument('--tag')
ap.add_argument('--layers', default='8,9,10,11')
ap.add_argument('--data', default='/datadisk2/linyichen/OFQ/ImageNet-1K')
ap.add_argument('--batches', type=int, default=0)
a = ap.parse_args()
ra = build_args(a.args_yaml); ra.batch_size = 64; ra.workers = 8; ra.prefetcher = False
m = create_model(ra.model, num_classes=ra.num_classes, drop_rate=ra.drop, pretrained=False, qqkkvv=False)
m = ql.get_ofq_qat_model(m, ra); ql.enable_attention_collection(m); m = m.cuda()
ql.set_attention_mode(m, collect_attention=True, qqkkvv=False)
ql.set_attention_collection_target(m, 'post_quant')
ds_tr = ql.create_dataset_compat(ra.dataset, root=ra.data_dir, split=ra.train_split, is_training=True, batch_size=64, subset_ratio=1.0)
ld_tr = ql.create_loader_compat(ds_tr, input_size=224, batch_size=64, is_training=True, use_prefetcher=False,
    no_aug=False, re_prob=ra.reprob, re_mode=ra.remode, re_count=ra.recount, re_split=ra.resplit,
    scale=ra.scale, ratio=ra.ratio, hflip=ra.hflip, vflip=ra.vflip, color_jitter=ra.color_jitter,
    auto_augment=ra.aa, num_aug_splits=ra.aug_splits, num_aug_repeats=ra.num_aug_repeats,
    interpolation='bicubic', mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225),
    num_workers=8, distributed=False, collate_fn=None, pin_memory=False, use_multi_epochs_loader=False)
ra.setup_alpha_batches = 1
ql.setup_alpha(m, ld_tr, ra, lambda: torch.autocast('cuda', dtype=torch.bfloat16))
load_checkpoint(m, a.checkpoint, strict=False)
m.eval()
targets = {int(x) for x in a.layers.split(',')}
loader = build_probe_loader(ra, a.data)

def eval_acc(tag):
    correct = total = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            imgs, tgt = batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True)
            out = m(imgs)
            logits = out[0] if isinstance(out, tuple) else out
            if logits.shape[-1] == 2000:
                logits = (logits[:, :1000] + logits[:, 1000:]) / 2
            correct += (logits.argmax(1) == tgt).sum().item(); total += tgt.numel()
            if a.batches and i + 1 >= a.batches: break
    print(f'[{a.tag}] {tag}: top1={100.0*correct/total:.3f}  n={total}', flush=True)

eval_acc('原样')
handles = []
for name, mod in m.named_modules():
    if not name.endswith('quan_a_softmax_fn'): continue
    layer = next(int(p) for p in name.split('.') if p.isdigit())
    if layer in targets:
        handles.append(mod.register_forward_hook(lambda _m, _i, out: torch.zeros_like(out)))
eval_acc(f'把 block {sorted(targets)} 的注意力输出置零')
for h in handles: h.remove()
