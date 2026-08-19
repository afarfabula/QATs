#!/usr/bin/env python3
import sys
import copy
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import yaml


def main() -> None:
    qats = Path(__file__).resolve().parents[1]
    ofq = qats / "third_party" / "OFQ"
    sys.path.insert(0, str(ofq))

    from src.swin import ShiftedWindowAttention, swin_t  # noqa: WPS433
    from src.quantization.modules.swin_attention_and_mlp import (  # noqa: WPS433
        QAttention_swin,
        QAttention_swin_qkreparam,
    )
    from src.quantization.modules.utils import replace_module_by_qmodule_swin  # noqa: WPS433

    model = swin_t(pretrained=True, num_classes=1000).eval()
    fp_total = sum(p.numel() for p in model.parameters())
    print(f"fp_total={fp_total}")
    cfg = yaml.safe_load((ofq / "configs" / "swin_t_imagenet.attn_q.yml").read_text())
    qconfigs = {}
    for module_name in cfg["qmodules"]:
        wcfg = {
            "mode": "statsq",
            "bit": 4,
            "all_positive": False,
            "symmetric": True,
            "per_channel": True,
            "normalize_first": False,
            "learnable": False,
        }
        acfg = {
            "enable": True,
            "mode": "lsq",
            "bit": 4,
            "per_channel": True,
            "normalize_first": False,
            "learnable": True,
        }
        qconfigs[module_name] = {"weight": wcfg, "act": acfg, "q_attn_dropout": 0, "act_layer": nn.GELU}

    qkr_model = copy.deepcopy(model)
    replace_module_by_qmodule_swin(qkr_model, qconfigs, pretrained_initialized=True, qk_reparam=True, qk_reparam_type=0)
    print(f"qkr_replaced_total={sum(p.numel() for p in qkr_model.parameters())}")
    q_model = copy.deepcopy(model)
    replace_module_by_qmodule_swin(q_model, qconfigs, pretrained_initialized=True, qk_reparam=False, qk_reparam_type=0)
    print(f"q_replaced_total={sum(p.numel() for p in q_model.parameters())}")

    for name, module in model.named_modules():
        if isinstance(module, ShiftedWindowAttention):
            q = QAttention_swin(module, weight_bits=4, input_bits=4, pretrained_initialized=True)
            qkr = QAttention_swin_qkreparam(module, weight_bits=4, input_bits=4, pretrained_initialized=True)
            print(f"layer={name}")
            print(f"  fp={sum(p.numel() for p in module.parameters())}")
            print(f"  q ={sum(p.numel() for p in q.parameters())}")
            print(f"  qkr={sum(p.numel() for p in qkr.parameters())}")
            groups = defaultdict(int)
            for pname, p in qkr.named_parameters():
                groups[pname.split(".")[0]] += p.numel()
            for key in sorted(groups):
                print(f"    {key}: {groups[key]}")
            break


if __name__ == "__main__":
    with torch.no_grad():
        main()
