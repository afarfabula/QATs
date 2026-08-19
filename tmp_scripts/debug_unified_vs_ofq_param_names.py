#!/usr/bin/env python3
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def main() -> None:
    qats = Path(__file__).resolve().parents[1]
    ofq = qats / "third_party" / "OFQ"
    sys.path.insert(0, str(qats))
    sys.path.insert(0, str(ofq))

    import qat_launch  # noqa: WPS433
    from src.swin import swin_t  # noqa: WPS433
    from src.quantization.modules.utils import replace_module_by_qmodule_swin  # noqa: WPS433

    model_a = swin_t(pretrained=True, num_classes=1000).eval()
    model_b = swin_t(pretrained=True, num_classes=1000).eval()

    runtime_args = SimpleNamespace(
        qmodules=[
            "features.0.0",
            "features.1.0.attn",
            "features.1.0.mlp",
            "features.1.1.attn",
            "features.1.1.mlp",
            "features.2.reduction",
            "features.3.0.attn",
            "features.3.0.mlp",
            "features.3.1.attn",
            "features.3.1.mlp",
            "features.4.reduction",
            "features.5.0.attn",
            "features.5.0.mlp",
            "features.5.1.attn",
            "features.5.1.mlp",
            "features.5.2.attn",
            "features.5.2.mlp",
            "features.5.3.attn",
            "features.5.3.mlp",
            "features.5.4.attn",
            "features.5.4.mlp",
            "features.5.5.attn",
            "features.5.5.mlp",
            "features.6.reduction",
            "features.7.0.attn",
            "features.7.0.mlp",
            "features.7.1.attn",
            "features.7.1.mlp",
            "head",
        ],
        wq_mode="statsq",
        wq_enable=True,
        wq_bitw=4,
        aq_enable=True,
        wq_asym=False,
        wq_per_channel=True,
        wq_clip_learnable=False,
        aq_mode="lsq",
        aq_bitw=4,
        aq_per_channel=True,
        aq_clip_learnable=True,
        apply_q_attn_dropout=0,
        act_layer="gelu",
        model_type="swin",
        pretrained_initialized=True,
        qk_reparam=True,
        qk_reparam_type=0,
    )
    unified = qat_launch.get_ofq_qat_model(model_a, runtime_args)
    qconfigs = qat_launch.build_ofq_qconfigs(runtime_args)
    native = replace_module_by_qmodule_swin(
        model_b,
        qconfigs,
        pretrained_initialized=True,
        qk_reparam=True,
        qk_reparam_type=0,
    )

    print(f"unified_total={sum(p.numel() for p in unified.parameters())}")
    print(f"native_total={sum(p.numel() for p in native.parameters())}")
    up = dict(unified.named_parameters())
    np = dict(native.named_parameters())
    for name in sorted(set(up) | set(np)):
        u = up.get(name)
        n = np.get(name)
        us = tuple(u.shape) if u is not None else None
        ns = tuple(n.shape) if n is not None else None
        if us != ns:
            unum = u.numel() if u is not None else 0
            nnum = n.numel() if n is not None else 0
            print(f"DIFF {name}: unified={us} native={ns} delta={unum - nnum}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
