#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", default="features.1.0.attn")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-scale", type=float, default=1.0)
    args = parser.parse_args()

    qats = Path(__file__).resolve().parents[1]
    ofq = qats / "third_party" / "OFQ"
    sys.path.insert(0, str(ofq))

    from src.swin import ShiftedWindowAttention, swin_t  # noqa: WPS433
    from src.quantization.modules.swin_attention_and_mlp import (  # noqa: WPS433
        QAttention_swin,
        QAttention_swin_qkreparam,
    )

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = swin_t(pretrained=True, num_classes=1000).to(device).eval()
    module = dict(model.named_modules())[args.layer]
    if not isinstance(module, ShiftedWindowAttention):
        raise TypeError(f"{args.layer} is {type(module)}, not ShiftedWindowAttention")
    module = module.to(device).eval()

    # Use the expected NHWC shape for a Swin-T block at the selected layer.
    dim = int(module.dim)
    if args.layer.startswith("features.1."):
        h = w = 56
    elif args.layer.startswith("features.3."):
        h = w = 28
    elif args.layer.startswith("features.5."):
        h = w = 14
    elif args.layer.startswith("features.7."):
        h = w = 7
    else:
        h = w = 56
    x = torch.randn(2, h, w, dim, device=device) * args.input_scale

    q_attn = QAttention_swin(
        module,
        weight_bits=4,
        input_bits=4,
        aq_learnable=True,
        wq_learnable=True,
        weight_channelwise=True,
        input_channelwise=True,
        weight_quant_method="statsq",
        input_quant_method="lsq",
        pretrained_initialized=True,
    ).to(device).eval()
    qkr_attn = QAttention_swin_qkreparam(
        module,
        weight_bits=4,
        input_bits=4,
        aq_learnable=True,
        wq_learnable=True,
        weight_channelwise=True,
        input_channelwise=True,
        weight_quant_method="statsq",
        input_quant_method="lsq",
        pretrained_initialized=True,
    ).to(device).eval()

    with torch.no_grad():
        y_fp, _ = module(x)
        y_q, _ = q_attn(x)
        y_qkr, _ = qkr_attn(x)

    def summarize(name: str, y: torch.Tensor, ref: torch.Tensor) -> None:
        diff = (y - ref).float()
        denom = ref.float().norm().clamp_min(1e-12)
        print(
            f"{name}: mean={y.float().mean().item():.6g} std={y.float().std().item():.6g} "
            f"max_abs_diff={diff.abs().max().item():.6g} "
            f"mean_abs_diff={diff.abs().mean().item():.6g} "
            f"rel_l2={diff.norm().div(denom).item():.6g}"
        )

    print(f"layer={args.layer} dim={dim} input_shape={tuple(x.shape)}")
    print(
        "qkv_bias_norms: "
        f"q={module.qkv.bias[:dim].norm().item():.6g} "
        f"k={module.qkv.bias[dim:2*dim].norm().item():.6g} "
        f"v={module.qkv.bias[2*dim:3*dim].norm().item():.6g}"
    )
    print(
        "relative_position_bias_diff_after_wrap: "
        f"q={float((q_attn.relative_position_bias_table - module.relative_position_bias_table).abs().max()):.6g} "
        f"qkr={float((qkr_attn.relative_position_bias_table - module.relative_position_bias_table).abs().max()):.6g}"
    )
    print(
        "proj_weight_diff_after_wrap: "
        f"q={float((q_attn.proj.weight - module.proj.weight).abs().max()):.6g} "
        f"qkr={float((qkr_attn.proj.weight - module.proj.weight).abs().max()):.6g}"
    )
    summarize("QAttention_vs_FP", y_q, y_fp)
    summarize("QKR_vs_FP", y_qkr, y_fp)
    summarize("QKR_vs_QAttention", y_qkr, y_q)


if __name__ == "__main__":
    main()
