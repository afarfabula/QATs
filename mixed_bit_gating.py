"""Differentiable multi-bit mixed-precision gating for OFQ QAT.

Provides tensor-level multi-branch quantizer wrappers (Ternary / 2-bit / 3-bit)
with independent quantizer parameters (e.g. per-branch LSQ step sizes `s`),
continuous gated drift during search, and exact-budget hardening for static
mixed-precision fine-tuning.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

CANDIDATE_BITS: Tuple[int, ...] = (0, 2, 3)
EFFECTIVE_BITS: Tuple[float, ...] = (math.log2(3.0), 2.0, 3.0)
BIT_LABELS: Tuple[str, ...] = ("T", "2", "3")


def parse_candidate_bits(
    spec: Optional[object],
) -> Tuple[Tuple[int, ...], Tuple[float, ...], Tuple[str, ...]]:
    """Parse mixed-bit candidate specification into (bits, effective_bits, labels)."""
    if spec is None:
        return CANDIDATE_BITS, EFFECTIVE_BITS, BIT_LABELS
    if isinstance(spec, str):
        raw_items = [item.strip() for item in spec.split(",") if item.strip()]
        if not raw_items:
            return CANDIDATE_BITS, EFFECTIVE_BITS, BIT_LABELS
    else:
        raw_items = [str(item).strip() for item in spec if str(item).strip()]
        if not raw_items:
            return CANDIDATE_BITS, EFFECTIVE_BITS, BIT_LABELS

    cand_bits: List[int] = []
    eff_bits: List[float] = []
    labels: List[str] = []
    for tok in raw_items:
        if tok.upper() == "T" or tok == "0":
            cand_bits.append(0)
            eff_bits.append(math.log2(3.0))
            labels.append("T")
        else:
            b = int(tok)
            if b < 2:
                raise ValueError(f"Mixed-bit candidate integer must be >= 2 or T/0, got {tok!r}")
            cand_bits.append(b)
            eff_bits.append(float(b))
            labels.append(str(b))

    if len(cand_bits) < 2:
        raise ValueError(f"Mixed-bit candidates must contain at least 2 entries, got {spec!r}")
    if any(eff_bits[i] >= eff_bits[i + 1] for i in range(len(eff_bits) - 1)):
        raise ValueError(f"Mixed-bit candidates must be strictly ascending in bit-width, got {spec!r}")
    return tuple(cand_bits), tuple(eff_bits), tuple(labels)

WEIGHT_QUANT_ATTRS: Tuple[str, ...] = ("statsq_fn", "qk_quant", "v_quant", "lsqw_fn")
ACT_QUANT_ATTRS: Tuple[str, ...] = (
    "input_quant_fn",
    "quan_a_qkx_fn",
    "quan_a_v_fn",
    "quan_a_softmax_fn",
    "quan_a_q_fn",
    "quan_a_k_fn",
)


def maybe_unwrap_ddp(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _depth_block_name(module_name: str) -> Optional[str]:
    for suffix in (".attn", ".mlp"):
        if module_name.endswith(suffix):
            return module_name[: -len(suffix)]
    return None


def _natural_sort_key(name: str) -> Tuple[Tuple[int, object], ...]:
    import re

    tokens = []
    for part in re.split(r"(\d+)", str(name)):
        if not part:
            continue
        tokens.append((0, int(part)) if part.isdigit() else (1, part))
    return tuple(tokens)


def _clone_quantizer_for_bit(orig_q: nn.Module, bit: int) -> nn.Module:
    cls_name = type(orig_q).__name__
    if cls_name == "StatsQuantizer":
        from third_party.OFQ.src.quantization.quantizer.statsq import StatsQuantizer

        device = orig_q.clip_val.device if hasattr(orig_q, "clip_val") else torch.device("cpu")
        cloned = StatsQuantizer(num_bits=int(bit), clip_learnable=False).to(device)
        if hasattr(orig_q, "aoq_scale_ratio"):
            cloned.aoq_scale_ratio = float(orig_q.aoq_scale_ratio)
        return cloned
    if cls_name in {"LsqQuantizer", "LsqQuantizer4v", "LsqQuantizerWeight"}:
        cloned = type(orig_q)(
            bit=int(bit),
            all_positive=bool(getattr(orig_q, "all_positive", False)),
            per_channel=bool(getattr(orig_q, "per_channel", True)),
            learnable=bool(getattr(orig_q, "learnable", True)),
        )
        return cloned
    raise TypeError(f"Unsupported quantizer class for MultiBitGatedQuantizer: {cls_name}")


class MultiBitGatedQuantizer(nn.Module):
    """Wraps T/2/3-bit quantizer branches with tensor-level continuous gating."""

    def __init__(
        self,
        branches: Sequence[nn.Module],
        gate_owner: nn.Module,
        module_name: str,
        is_weight: bool,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(list(branches))
        # Store gate_owner outside nn.Module._modules to avoid module cycle while preserving deepcopy memoization.
        object.__setattr__(self, "_gate_owner", gate_owner)
        self.module_name = str(module_name)
        self.is_weight = bool(is_weight)
        self.gate_active: bool = False
        self.tau: float = 1.5
        self.use_hard: bool = False
        self.frozen_idx: Optional[int] = None
        self.eval_use_argmax: bool = False

    def _gate_probs(self, dtype: torch.dtype) -> torch.Tensor:
        alpha = self._gate_owner.mixed_bit_alpha
        if not self.gate_active:
            alpha = alpha.detach()
        if not self.training:
            if self.eval_use_argmax or self.use_hard:
                idx = int(alpha.detach().argmax().item())
                return F.one_hot(
                    torch.tensor(idx, device=alpha.device), num_classes=len(self.branches)
                ).to(dtype=dtype)
            return F.softmax(alpha.detach() / max(float(self.tau), 1e-4), dim=0).to(dtype=dtype)

        y_soft = F.softmax(alpha / max(float(self.tau), 1e-4), dim=0)
        if self.use_hard:
            idx = alpha.detach().argmax()
            y_hard = F.one_hot(idx, num_classes=len(self.branches)).to(dtype=y_soft.dtype)
            probs = (y_hard - y_soft).detach() + y_soft
        else:
            probs = y_soft
        return probs.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen_idx is not None:
            return self.branches[int(self.frozen_idx)](x)

        outs = [branch(x) for branch in self.branches]
        probs = self._gate_probs(outs[0].dtype)
        mixed = probs[0] * outs[0]
        for idx in range(1, len(outs)):
            mixed = mixed + probs[idx] * outs[idx]
        return mixed


def _compute_submodule_param_weight(owner: nn.Module) -> float:
    total = 0
    for mod in owner.modules():
        if isinstance(mod, nn.Linear):
            weight = getattr(mod, "weight", None)
            if torch.is_tensor(weight):
                total += int(weight.numel())
    return float(max(1, total))


def install_mixed_bit_gating(model: nn.Module, runtime_args: SimpleNamespace) -> int:
    if not bool(getattr(runtime_args, "mixed_bit_gate_enable", False)):
        return 0
    root = maybe_unwrap_ddp(model)
    cand_bits, eff_bits, bit_labels = parse_candidate_bits(getattr(runtime_args, "mixed_bit_candidates", None))
    init_logits = tuple(float(v) for v in getattr(runtime_args, "mixed_bit_init_logits", (0.0, 1.5, 0.0)))
    if len(init_logits) != len(cand_bits):
        raise ValueError(f"mixed_bit_init_logits must have {len(cand_bits)} entries for {cand_bits}, got {init_logits}")

    qmodules = sorted(
        (str(name) for name in getattr(runtime_args, "qmodules", ()) if _depth_block_name(str(name)) is not None),
        key=_natural_sort_key,
    )
    installed = 0
    gate_entries: List[Tuple[str, float]] = []
    for module_name in qmodules:
        try:
            owner = root.get_submodule(module_name)
        except AttributeError:
            continue
        if not hasattr(owner, "mixed_bit_alpha"):
            owner.mixed_bit_alpha = nn.Parameter(torch.tensor(init_logits, dtype=torch.float32))
        param_weight = _compute_submodule_param_weight(owner)
        owner._mixed_bit_param_weight = param_weight
        owner._mixed_bit_module_name = module_name
        owner._mixed_bit_candidate_bits = cand_bits
        owner._mixed_bit_effective_bits = eff_bits
        owner._mixed_bit_labels = bit_labels
        gate_entries.append((module_name, param_weight))

        for submodule in owner.modules():
            for attr in WEIGHT_QUANT_ATTRS:
                orig_q = getattr(submodule, attr, None)
                if orig_q is None or isinstance(orig_q, MultiBitGatedQuantizer):
                    continue
                branches = [_clone_quantizer_for_bit(orig_q, b) for b in cand_bits]
                setattr(submodule, attr, MultiBitGatedQuantizer(branches, owner, module_name, is_weight=True))
                installed += 1
            for attr in ACT_QUANT_ATTRS:
                orig_q = getattr(submodule, attr, None)
                if orig_q is None or isinstance(orig_q, MultiBitGatedQuantizer):
                    continue
                branches = [_clone_quantizer_for_bit(orig_q, b) for b in cand_bits]
                setattr(submodule, attr, MultiBitGatedQuantizer(branches, owner, module_name, is_weight=False))
                installed += 1

    runtime_args._mixed_bit_candidate_bits = cand_bits
    runtime_args._mixed_bit_effective_bits = eff_bits
    runtime_args._mixed_bit_labels = bit_labels
    runtime_args._mixed_bit_gate_entries = gate_entries
    runtime_args._mixed_bit_frozen = False
    return installed


def iter_gate_owners(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    root = maybe_unwrap_ddp(model)
    owners: List[Tuple[str, nn.Module]] = []
    for name, module in root.named_modules():
        if hasattr(module, "mixed_bit_alpha"):
            owners.append((str(name), module))
    owners.sort(key=lambda item: _natural_sort_key(item[0]))
    return owners


def is_mixed_bit_gate_parameter(name: str) -> bool:
    return name.endswith("mixed_bit_alpha") or ".mixed_bit_alpha" in name


def _anneal_value(start: float, end: float, progress: float) -> float:
    progress = min(1.0, max(0.0, float(progress)))
    return float(start) + (float(end) - float(start)) * progress


def mixed_bit_schedule_for_epoch(
    epoch: int, runtime_args: SimpleNamespace
) -> Dict[str, object]:
    search_start = int(getattr(runtime_args, "mixed_bit_search_start_epoch", 2))
    hard_start = int(getattr(runtime_args, "mixed_bit_hard_start_epoch", 10))
    freeze_epoch = int(getattr(runtime_args, "mixed_bit_freeze_epoch", 14))
    ramp_epochs = max(1, int(getattr(runtime_args, "mixed_bit_weight_ramp_epochs", 4)))
    max_bit_weight = float(getattr(runtime_args, "mixed_bit_budget_weight", 20.0))
    tau_start = float(getattr(runtime_args, "mixed_bit_tau_start", 1.5))
    tau_mid = float(getattr(runtime_args, "mixed_bit_tau_mid", 0.4))
    tau_end = float(getattr(runtime_args, "mixed_bit_tau_end", 0.25))

    if epoch < search_start:
        return {
            "phase": "warmup",
            "gate_active": False,
            "use_hard": False,
            "tau": tau_start,
            "bit_weight": 0.0,
            "should_freeze": False,
        }
    if epoch < hard_start:
        soft_prog = float(epoch - search_start) / float(max(1, hard_start - search_start - 1))
        ramp_prog = float(epoch - search_start + 1) / float(ramp_epochs)
        return {
            "phase": "soft",
            "gate_active": True,
            "use_hard": False,
            "tau": _anneal_value(tau_start, tau_mid, soft_prog),
            "bit_weight": _anneal_value(0.0, max_bit_weight, ramp_prog),
            "should_freeze": False,
        }
    if epoch < freeze_epoch:
        hard_prog = float(epoch - hard_start) / float(max(1, freeze_epoch - hard_start - 1))
        return {
            "phase": "st_hard",
            "gate_active": True,
            "use_hard": True,
            "use_hard_budget": bool(getattr(runtime_args, "mixed_bit_hard_budget", True)),
            "tau": _anneal_value(tau_mid, tau_end, hard_prog),
            "bit_weight": max_bit_weight,
            "should_freeze": False,
        }
    return {
        "phase": "frozen",
        "gate_active": False,
        "use_hard": True,
        "tau": tau_end,
        "bit_weight": 0.0,
        "should_freeze": True,
    }


def apply_mixed_bit_schedule_to_model(
    model: nn.Module,
    gate_active: bool,
    tau: float,
    use_hard: bool,
    eval_use_argmax: bool = False,
) -> None:
    root = maybe_unwrap_ddp(model)
    for mod in root.modules():
        if isinstance(mod, MultiBitGatedQuantizer):
            mod.gate_active = bool(gate_active)
            mod.tau = float(tau)
            mod.use_hard = bool(use_hard)
            mod.eval_use_argmax = bool(eval_use_argmax)


def set_gate_optimizer_lr_scale(optimizer: torch.optim.Optimizer, gate_lr_scale: float) -> None:
    for group in optimizer.param_groups:
        if group.get("is_mixed_bit_gate", False):
            group["lr_scale"] = float(gate_lr_scale)
            if float(gate_lr_scale) == 0.0:
                group["lr"] = 0.0


def compute_mixed_bit_budget_loss(
    model: nn.Module,
    runtime_args: SimpleNamespace,
    tau: float,
    bit_weight: float,
    use_hard: bool = False,
) -> Tuple[torch.Tensor, float, float]:
    owners = iter_gate_owners(model)
    if not owners:
        zero = torch.zeros((), device="cuda" if torch.cuda.is_available() else "cpu")
        return zero, 0.0, 0.0

    device = owners[0][1].mixed_bit_alpha.device
    eff_bits_tuple = tuple(getattr(owners[0][1], "_mixed_bit_effective_bits", EFFECTIVE_BITS))
    eff_bits = torch.tensor(eff_bits_tuple, device=device, dtype=torch.float32)
    soft_num = torch.zeros((), device=device, dtype=torch.float32)
    budget_num = torch.zeros((), device=device, dtype=torch.float32)
    hard_num = 0.0
    denom = 0.0
    inv_tau = 1.0 / max(float(tau), 1e-4)

    for _name, owner in owners:
        w = float(getattr(owner, "_mixed_bit_param_weight", 1.0))
        alpha = owner.mixed_bit_alpha
        y_soft = F.softmax(alpha * inv_tau, dim=0)
        soft_num = soft_num + w * (y_soft * eff_bits).sum()
        hard_idx = int(alpha.detach().argmax().item())
        hard_num += w * float(eff_bits_tuple[hard_idx])
        if use_hard:
            y_hard = F.one_hot(alpha.detach().argmax(), num_classes=len(eff_bits_tuple)).to(dtype=y_soft.dtype)
            budget_probs = (y_hard - y_soft).detach() + y_soft
        else:
            budget_probs = y_soft
        budget_num = budget_num + w * (budget_probs * eff_bits).sum()
        denom += w

    soft_mean_bits = soft_num / max(denom, 1.0)
    budget_mean_bits = budget_num / max(denom, 1.0)
    hard_mean_bits = hard_num / max(denom, 1.0)

    target_bits = float(getattr(runtime_args, "mixed_bit_target_bits", 2.0))
    tol = float(getattr(runtime_args, "mixed_bit_budget_tol", 0.015))
    if float(bit_weight) <= 0.0:
        return torch.zeros_like(budget_mean_bits), float(soft_mean_bits.detach().item()), hard_mean_bits

    excess = F.relu(torch.abs(budget_mean_bits - target_bits) - tol)
    loss = float(bit_weight) * (excess ** 2)
    return loss, float(soft_mean_bits.detach().item()), hard_mean_bits


def _exact_budget_round_choices(
    alphas: Sequence[torch.Tensor],
    weights: Sequence[float],
    target_bits: float,
    tol_low: float = 0.02,
    tol_high: float = 0.015,
    effective_bits: Sequence[float] = EFFECTIVE_BITS,
) -> List[int]:
    choices = [int(a.detach().argmax().item()) for a in alphas]
    total_w = max(1e-8, sum(weights))

    def mean_bits(ch: Sequence[int]) -> float:
        return sum(w * effective_bits[c] for w, c in zip(weights, ch)) / total_w

    cur = mean_bits(choices)
    max_iters = len(choices) * 2
    iters = 0
    while (cur > target_bits + tol_high or cur < target_bits - tol_low) and iters < max_iters:
        iters += 1
        need_lower = cur > target_bits + tol_high
        best_move = None
        for idx, (alpha, cur_c) in enumerate(zip(alphas, choices)):
            vals = alpha.detach().cpu().tolist()
            if need_lower and cur_c > 0:
                cand_c = cur_c - 1
            elif (not need_lower) and cur_c < len(effective_bits) - 1:
                cand_c = cur_c + 1
            else:
                continue
            logit_penalty = float(vals[cur_c] - vals[cand_c])
            new_mean = cur + (weights[idx] * (effective_bits[cand_c] - effective_bits[cur_c])) / total_w
            overshoots = (new_mean < target_bits - tol_low) if need_lower else (new_mean > target_bits + tol_high)
            dist_to_target = abs(new_mean - target_bits)
            score = (1 if overshoots else 0, logit_penalty, dist_to_target)
            if best_move is None or score < best_move[0]:
                best_move = (score, idx, cand_c)
        if best_move is None:
            break
        _, move_idx, move_c = best_move
        choices[move_idx] = move_c
        cur = mean_bits(choices)
    return choices


def freeze_mixed_bit_gate(
    model: nn.Module,
    runtime_args: SimpleNamespace,
    model_ema: Optional[nn.Module] = None,
) -> Tuple[float, Dict[str, int]]:
    owners = iter_gate_owners(model)
    if not owners:
        return 0.0, {}

    alphas = [owner.mixed_bit_alpha for _, owner in owners]
    weights = [float(getattr(owner, "_mixed_bit_param_weight", 1.0)) for _, owner in owners]
    eff_bits = tuple(getattr(owners[0][1], "_mixed_bit_effective_bits", EFFECTIVE_BITS))
    target_bits = float(getattr(runtime_args, "mixed_bit_target_bits", 2.0))

    if bool(getattr(runtime_args, "local_rank", 0) == 0):
        choices_list = _exact_budget_round_choices(alphas, weights, target_bits, effective_bits=eff_bits)
    else:
        choices_list = [0] * len(owners)

    if bool(getattr(runtime_args, "distributed", False)) and dist.is_initialized():
        choice_tensor = torch.tensor(choices_list, dtype=torch.long, device=alphas[0].device)
        dist.broadcast(choice_tensor, src=0)
        choices_list = choice_tensor.cpu().tolist()

    choice_map = {name: int(c) for (name, _), c in zip(owners, choices_list)}
    for name, owner in owners:
        owner.mixed_bit_alpha.requires_grad_(False)
        chosen = choice_map[name]
        for mod in owner.modules():
            if isinstance(mod, MultiBitGatedQuantizer):
                mod.frozen_idx = chosen
                mod.gate_active = False
                mod.use_hard = True
                for b_idx, branch in enumerate(mod.branches):
                    if b_idx != chosen:
                        for p in branch.parameters():
                            p.requires_grad_(False)

    if model_ema is not None:
        for name, owner in iter_gate_owners(model_ema):
            if name in choice_map:
                chosen = choice_map[name]
                for mod in owner.modules():
                    if isinstance(mod, MultiBitGatedQuantizer):
                        mod.frozen_idx = chosen
                        mod.gate_active = False
                        mod.use_hard = True

    total_w = max(1e-8, sum(weights))
    hard_mean = sum(w * eff_bits[c] for w, c in zip(weights, choices_list)) / total_w
    runtime_args._mixed_bit_frozen = True
    runtime_args._mixed_bit_frozen_choices = choice_map
    return hard_mean, choice_map


def summarize_mixed_bit_gate(model: nn.Module, tau: float) -> Dict[str, object]:
    owners = iter_gate_owners(model)
    if not owners:
        return {}
    eff_bits = tuple(getattr(owners[0][1], "_mixed_bit_effective_bits", EFFECTIVE_BITS))
    labels = tuple(getattr(owners[0][1], "_mixed_bit_labels", BIT_LABELS))
    inv_tau = 1.0 / max(float(tau), 1e-4)
    soft_sum = 0.0
    hard_sum = 0.0
    denom = 0.0
    counts = {lbl: 0 for lbl in labels}
    tokens = []
    for name, owner in owners:
        w = float(getattr(owner, "_mixed_bit_param_weight", 1.0))
        alpha = owner.mixed_bit_alpha.detach()
        probs = F.softmax(alpha * inv_tau, dim=0)
        frozen_idx = None
        for mod in owner.modules():
            if isinstance(mod, MultiBitGatedQuantizer):
                frozen_idx = mod.frozen_idx
                break
        hard_idx = int(frozen_idx) if frozen_idx is not None else int(alpha.argmax().item())
        soft_sum += w * sum(float(probs[i].item()) * eff_bits[i] for i in range(len(eff_bits)))
        hard_sum += w * eff_bits[hard_idx]
        denom += w
        label = labels[hard_idx]
        counts[label] += 1
        short = name.replace("blocks.", "b").replace(".attn", "A").replace(".mlp", "M")
        tokens.append(f"{short}:{label}")
    return {
        "soft_mean_bits": soft_sum / max(denom, 1.0),
        "hard_mean_bits": hard_sum / max(denom, 1.0),
        "counts": counts,
        "profile": ",".join(tokens),
    }

def enforce_frozen_mixed_bit_grad(model: nn.Module) -> None:
    for name, owner in iter_gate_owners(model):
        owner.mixed_bit_alpha.requires_grad_(False)
        for mod in owner.modules():
            if isinstance(mod, MultiBitGatedQuantizer) and mod.frozen_idx is not None:
                chosen = int(mod.frozen_idx)
                for b_idx, branch in enumerate(mod.branches):
                    if b_idx != chosen:
                        for p in branch.parameters():
                            p.requires_grad_(False)
