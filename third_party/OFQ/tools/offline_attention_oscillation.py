import argparse
import os
import sys
from collections import OrderedDict

import torch
import torch.serialization

from timm.models import create_model, load_checkpoint

torch.serialization.add_safe_globals([argparse.Namespace])

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train import (  # noqa: E402
    get_qat_model,
    create_dataset_compat,
    create_loader_compat,
    resolve_data_config,
    enable_attention_collection,
)


SWIN_STAGE_SPECS = [
    (56, 56, 7, 2),
    (28, 28, 7, 2),
    (14, 14, 7, 6),
    (7, 7, 7, 2),
]


def parse_cli():
    parser = argparse.ArgumentParser(description='Offline attention oscillation probe for Swin')
    parser.add_argument('--checkpoint-a', type=str, default='')
    parser.add_argument('--checkpoint-b', type=str, default='')
    parser.add_argument('--probe-dir', required=True, type=str)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--topk', type=int, default=4)
    parser.add_argument('--topk-list', type=str, default='')
    parser.add_argument('--max-batches', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--train-args', type=str, default='')
    parser.add_argument('--checkpoint-dir', type=str, default='')
    parser.add_argument('--adjacent-output-root', type=str, default='')
    return parser.parse_args()


def build_args(train_args_path):
    if not train_args_path:
        raise ValueError('train_args is required')
    import yaml
    with open(train_args_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['collect_attention'] = True
    cfg['resume'] = ''
    cfg['initial_checkpoint'] = ''
    cfg['no_resume_opt'] = True
    cfg['prefetcher'] = not cfg.get('no_prefetcher', False)
    args = argparse.Namespace(**cfg)
    return args


def build_model(args, checkpoint_path, device):
    model = create_model(
        args.model,
        drop_path=args.drop_path,
        num_classes=args.num_classes,
        pretrained=False,
        qqkkvv=False,
    )
    if args.quantized:
        model = get_qat_model(model, args)
    enable_attention_collection(model)
    load_checkpoint(model, checkpoint_path, strict=False)
    model.to(device)
    model.eval()
    return model


def build_probe_loader(args, probe_dir):
    dataset = create_dataset_compat(
        args.dataset,
        root=probe_dir,
        split=args.val_split,
        is_training=False,
        batch_size=args.batch_size,
    )
    data_config = resolve_data_config(vars(args), model=None)
    loader = create_loader_compat(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=False,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )
    return loader


def get_block_metas():
    metas = []
    for stage_id, (h, w, ws, depth) in enumerate(SWIN_STAGE_SPECS):
        num_windows = (h // ws) * (w // ws)
        for block_id in range(depth):
            metas.append({
                'stage': stage_id,
                'block_in_stage': block_id,
                'h': h,
                'w': w,
                'window_size': ws,
                'num_windows': num_windows,
                'shift_size': 0 if block_id % 2 == 0 else ws // 2,
            })
    return metas


def make_global_indices(meta, batch_size, device):
    h = meta['h']
    w = meta['w']
    ws = meta['window_size']
    num_windows = meta['num_windows']
    grid = torch.arange(h * w, device=device).reshape(h, w)
    if meta['shift_size'] > 0:
        grid = torch.roll(grid, shifts=(-meta['shift_size'], -meta['shift_size']), dims=(0, 1))
    windows = grid.view(h // ws, ws, w // ws, ws).permute(0, 2, 1, 3).reshape(num_windows, ws * ws)
    windows = windows.unsqueeze(0).expand(batch_size, -1, -1)
    return windows.reshape(batch_size * num_windows, ws * ws)


def make_valid_attention_mask(meta, batch_size, device):
    h = meta['h']
    w = meta['w']
    ws = meta['window_size']
    num_windows = meta['num_windows']

    window_ids = torch.arange(num_windows, device=device).reshape(h // ws, w // ws)
    window_ids = window_ids.repeat_interleave(ws, dim=0).repeat_interleave(ws, dim=1)
    if meta['shift_size'] > 0:
        window_ids = torch.roll(window_ids, shifts=(-meta['shift_size'], -meta['shift_size']), dims=(0, 1))
    window_ids = window_ids.view(h // ws, ws, w // ws, ws).permute(0, 2, 1, 3).reshape(num_windows, ws * ws)
    valid = window_ids.unsqueeze(1) == window_ids.unsqueeze(2)
    valid = valid.unsqueeze(0).expand(batch_size, -1, -1, -1)
    return valid.reshape(batch_size * num_windows, ws * ws, ws * ws)


def get_block_candidate_stats():
    stats = []
    for meta in get_block_metas():
        valid = make_valid_attention_mask(meta, batch_size=1, device=torch.device('cpu'))[0]
        counts = valid.sum(dim=-1)
        stats.append({
            'stage': meta['stage'],
            'block_in_stage': meta['block_in_stage'],
            'shift_size': meta['shift_size'],
            'max_candidates': int(counts.max().item()),
            'min_candidates': int(counts.min().item()),
            'unique_candidates': [int(v) for v in torch.unique(counts).tolist()],
        })
    return stats


@torch.no_grad()
def collect_topk(model, loader, topk, device, max_batches=0):
    block_metas = get_block_metas()
    outputs = [list() for _ in block_metas]
    for batch_idx, (images, _) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        if not isinstance(images, torch.Tensor):
            images = images[0]
        images = images.to(device)
        _, attn_list = model(images)
        batch_size = images.shape[0]
        for i, attn in enumerate(attn_list):
            if attn is None:
                continue
            meta = block_metas[i]
            idx_map = make_global_indices(meta, batch_size, device)
            valid_mask = make_valid_attention_mask(meta, batch_size, device)
            min_candidates = int(valid_mask.sum(dim=-1).min().item())
            if topk > min_candidates:
                raise ValueError(
                    f'topk={topk} exceeds minimum valid candidates={min_candidates} '
                    f"for stage_{meta['stage']}_block_{meta['block_in_stage']}"
                )
            masked_attn = attn.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))
            k = min(topk, attn.shape[-1])
            topk_idx = masked_attn.topk(k, dim=-1).indices
            global_idx = idx_map.unsqueeze(1).unsqueeze(1).expand(-1, attn.shape[1], attn.shape[2], -1)
            mapped = torch.gather(global_idx, dim=-1, index=topk_idx)
            outputs[i].append(mapped.cpu())
    return [torch.cat(x, dim=0) if x else None for x in outputs]


def oscillation(topk_a, topk_b):
    scores = OrderedDict()
    for i, (a, b) in enumerate(zip(topk_a, topk_b)):
        if a is None or b is None:
            continue
        if a.shape != b.shape:
            raise ValueError(f'shape mismatch on block {i}: {a.shape} vs {b.shape}')
        inter = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(dim=-1).sum(dim=-1).float()
        k = a.shape[-1]
        score_per_head = 1.0 - inter.mean(dim=(0, 2)) / k
        scores[i] = score_per_head
    return scores


def parse_topk_values(cli):
    if cli.topk_list:
        values = [int(x.strip()) for x in cli.topk_list.split(',') if x.strip()]
        if not values:
            raise ValueError('topk_list is empty after parsing')
        return sorted(dict.fromkeys(values))
    return [int(cli.topk)]


def slice_topk_outputs(outputs, k):
    sliced = []
    for item in outputs:
        if item is None:
            sliced.append(None)
        else:
            sliced.append(item[..., :k])
    return sliced


def format_scores(scores):
    lines = []
    for block_id, values in scores.items():
        meta = get_block_metas()[block_id]
        values_str = ', '.join(f'{v.item():.6f}' for v in values)
        lines.append(
            f"stage_{meta['stage']}_block_{meta['block_in_stage']}(global_block_{block_id}): [{values_str}]"
        )
    return '\n'.join(lines)


def write_output(content, output_path):
    if not output_path:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content + '\n')


def write_adjacent_outputs(prev_name, curr_name, topk_prev, topk_curr, topk_values, output_root):
    pair_name = f'{prev_name}_to_{curr_name}'
    rendered = []
    for k in topk_values:
        scores = oscillation(slice_topk_outputs(topk_prev, k), slice_topk_outputs(topk_curr, k))
        content = format_scores(scores)
        rendered.append((k, content))
        if output_root:
            output_path = os.path.join(output_root, f'topk_{k}', f'{pair_name}.txt')
            write_output(content, output_path)
    return rendered


def run_adjacent_mode(cli, args, loader, device, topk_values):
    checkpoint_dir = cli.checkpoint_dir
    if not checkpoint_dir:
        raise ValueError('checkpoint_dir is required for adjacent mode')
    checkpoint_paths = sorted(
        [os.path.join(checkpoint_dir, name) for name in os.listdir(checkpoint_dir) if name.startswith('step_') and name.endswith('.pth.tar')]
    )
    if len(checkpoint_paths) < 2:
        raise ValueError(f'need at least 2 step checkpoints under {checkpoint_dir}')

    max_topk = max(topk_values)
    prev_path = checkpoint_paths[0]
    prev_name = os.path.basename(prev_path).replace('.pth.tar', '')
    prev_model = build_model(args, prev_path, device)
    prev_topk = collect_topk(prev_model, loader, max_topk, device, cli.max_batches)
    del prev_model

    for curr_path in checkpoint_paths[1:]:
        curr_name = os.path.basename(curr_path).replace('.pth.tar', '')
        curr_model = build_model(args, curr_path, device)
        curr_topk = collect_topk(curr_model, loader, max_topk, device, cli.max_batches)
        del curr_model

        rendered = write_adjacent_outputs(prev_name, curr_name, prev_topk, curr_topk, topk_values, cli.adjacent_output_root)
        multi_content = '\n\n'.join([f'[{prev_name}_to_{curr_name}][topk={k}]\n{content}' for k, content in rendered])
        print(multi_content, flush=True)

        prev_name = curr_name
        prev_topk = curr_topk


def main():
    cli = parse_cli()
    topk_values = parse_topk_values(cli)
    max_topk = max(topk_values)
    args = build_args(cli.train_args)
    args.data_dir = cli.probe_dir
    args.batch_size = cli.batch_size
    args.workers = cli.workers
    device = torch.device(cli.device)
    loader = build_probe_loader(args, cli.probe_dir)

    if cli.checkpoint_dir:
        run_adjacent_mode(cli, args, loader, device, topk_values)
        return

    if not cli.checkpoint_a or not cli.checkpoint_b:
        raise ValueError('checkpoint_a and checkpoint_b are required unless checkpoint_dir is provided')

    model_a = build_model(args, cli.checkpoint_a, device)
    model_b = build_model(args, cli.checkpoint_b, device)
    topk_a = collect_topk(model_a, loader, max_topk, device, cli.max_batches)
    topk_b = collect_topk(model_b, loader, max_topk, device, cli.max_batches)

    if len(topk_values) == 1:
        scores = oscillation(slice_topk_outputs(topk_a, topk_values[0]), slice_topk_outputs(topk_b, topk_values[0]))
        content = format_scores(scores)
        print(content)
        write_output(content, cli.output)
        return

    rendered = []
    for k in topk_values:
        scores = oscillation(slice_topk_outputs(topk_a, k), slice_topk_outputs(topk_b, k))
        content = format_scores(scores)
        rendered.append((k, content))

    multi_content = '\n\n'.join([f'[topk={k}]\n{content}' for k, content in rendered])
    print(multi_content)
    if cli.output:
        for k, content in rendered:
            if '{topk}' in cli.output:
                output_path = cli.output.format(topk=k)
            else:
                output_root, output_ext = os.path.splitext(cli.output)
                output_path = f'{output_root}_topk{k}{output_ext}'
            write_output(content, output_path)


if __name__ == '__main__':
    main()
