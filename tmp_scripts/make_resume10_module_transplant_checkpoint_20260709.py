#!/usr/bin/env python3
"""Create a single-checkpoint module/state transplant candidate for resume10 clean LSQ runs."""

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch


def load_checkpoint(path: str) -> dict:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    return torch.load(path, map_location="cpu", weights_only=False)


def state_dict_of(checkpoint: dict) -> Dict[str, torch.Tensor]:
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a state_dict")
    return state


def strip_module(name: str) -> str:
    return name[len("module.") :] if name.startswith("module.") else name


def module_matches(key: str, modules: Tuple[str, ...]) -> bool:
    name = strip_module(key)
    for module in modules:
        if name == module or name.startswith(f"{module}."):
            return True
    return False


def parse_modules(text: str) -> Tuple[str, ...]:
    modules = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if not modules:
        raise ValueError("--modules must not be empty")
    return modules


def parse_optional_csv(text: str | None) -> Tuple[str, ...]:
    if not text:
        return ()
    return tuple(part.strip() for part in str(text).split(",") if part.strip())


def suffix_matches(key: str, modules: Tuple[str, ...], include_suffixes: Tuple[str, ...]) -> bool:
    if not include_suffixes:
        return True
    name = strip_module(key)
    for module in modules:
        if name == module:
            relative = ""
        elif name.startswith(f"{module}."):
            relative = name[len(module) + 1 :]
        else:
            continue
        if relative in include_suffixes:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--donor", required=True)
    parser.add_argument("--modules", required=True)
    parser.add_argument(
        "--include-suffixes",
        default="",
        help="Optional comma-separated tensor suffixes relative to each module, e.g. weight,lsqw_fn.s.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    modules = parse_modules(args.modules)
    include_suffixes = parse_optional_csv(args.include_suffixes)
    base = load_checkpoint(args.base)
    donor = load_checkpoint(args.donor)
    base_state = state_dict_of(base)
    donor_state = state_dict_of(donor)

    output = copy.deepcopy(base)
    output_state = state_dict_of(output)
    copied = []
    missing = []
    for key, value in list(output_state.items()):
        if not torch.is_tensor(value) or not module_matches(key, modules):
            continue
        if not suffix_matches(key, modules, include_suffixes):
            continue
        donor_value = donor_state.get(key)
        if donor_value is None and key.startswith("module."):
            donor_value = donor_state.get(strip_module(key))
        elif donor_value is None:
            donor_value = donor_state.get(f"module.{key}")
        if donor_value is None or not torch.is_tensor(donor_value) or tuple(donor_value.shape) != tuple(value.shape):
            missing.append(key)
            continue
        output_state[key] = donor_value.detach().clone()
        copied.append(key)

    if not copied:
        raise ValueError(f"no tensors copied for modules={modules}")
    output["state_dict"] = output_state
    meta = output.setdefault("module_transplant_20260709", {})
    meta.update(
        {
            "base": args.base,
            "donor": args.donor,
            "modules": modules,
            "include_suffixes": include_suffixes,
            "copied_tensors": len(copied),
            "missing_tensors": len(missing),
        }
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    torch.save(output, tmp_path)
    tmp_path.replace(out_path)
    summary = {
        "output": str(out_path),
        "base": args.base,
        "donor": args.donor,
        "modules": modules,
        "include_suffixes": include_suffixes,
        "copied_tensors": len(copied),
        "missing_tensors": len(missing),
        "first_copied": copied[:20],
        "first_missing": missing[:20],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
