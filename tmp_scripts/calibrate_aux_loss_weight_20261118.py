#!/usr/bin/env python3
"""Compute an auxiliary-loss weight so its initial share is a target fraction."""

from __future__ import annotations

import argparse
import re


def parse_value(line: str, field: str) -> float:
    match = re.search(rf"{re.escape(field)}:\s*([0-9.eE+-]+)\s*\(([0-9.eE+-]+)\)", line)
    if not match:
        raise ValueError(f"field {field!r} not found in training line")
    return float(match.group(2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--aux-field", required=True)
    parser.add_argument("--target-share", type=float, default=0.05)
    args = parser.parse_args()

    lines = [line for line in open(args.log, encoding="utf-8") if line.startswith("Train:")]
    if not lines:
        raise SystemExit("no Train lines found")
    last = lines[-1]
    base = parse_value(last, "BaseLoss")
    aux = parse_value(last, args.aux_field)
    if base <= 0 or aux <= 0:
        raise SystemExit(f"non-positive values: base={base} aux={aux}")
    if not 0.0 < args.target_share < 1.0:
        raise SystemExit("target_share must be in (0, 1)")
    weight = (args.target_share / (1.0 - args.target_share)) * (base / aux)
    print(f"{weight:.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
