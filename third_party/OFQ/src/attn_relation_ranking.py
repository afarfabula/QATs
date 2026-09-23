"""Attention-relation ranking：把教师注意力的大小关系迁移到学生注意力上。

与模型无关（DeiT / Swin 通用）：只要 attention module 在 `collect_attention` 打开时返回
softmax 后的注意力矩阵 `[*, heads, tokens, tokens]`，本模块就能直接用。
教师只提供"谁该排在谁前面"，不提供注意力数值。
"""
import torch
import torch.nn.functional as F


def extract_attention_list(attn_info):
    """把模型返回的注意力信息规整成 list[Tensor|None]。

    兼容三种形式：None、单层就是 Tensor、单层是 (attn, q, k, v) 这类元组。
    """
    if attn_info is None:
        return []
    if torch.is_tensor(attn_info):
        return [attn_info]
    extracted = []
    for layer_info in attn_info:
        if torch.is_tensor(layer_info):
            extracted.append(layer_info)
        elif isinstance(layer_info, (tuple, list)) and layer_info and torch.is_tensor(layer_info[0]):
            extracted.append(layer_info[0])
        else:
            extracted.append(None)
    return extracted


def attention_relation_ranking_loss(student_attn_info, teacher_attn_info, heads=None, topk: int = 1, eps: float = 1e-8):
    """教师注意力大小关系的成对排序损失。

    教师在每个 (layer, head, query) 行内取 top-k 个 key，与所有教师分数严格更低的 key 组成有向对,
    用 softplus(-(log s_c - log s_d)) 要求学生在同样两个 key 上保持同样顺序。
    log 概率差等于 softmax 前的分数差, 因此等价于约束 QK 分数的大小关系。
    teacher 值 ≤ 0 的位置（mask / dropout / 该 key 未被注意）与并列（ties）都不计入。

    heads: None 表示使用所有 (layer, head)；也可以传 ((layer_idx, head_idx), ...) 只约束部分 head。
    返回 (loss, 有效对数)；没有任何有效对时返回 loss=0。
    """
    student_list = extract_attention_list(student_attn_info)
    teacher_list = extract_attention_list(teacher_attn_info)
    if not student_list or not teacher_list:
        return torch.zeros((), device="cuda"), 0

    if heads is None:
        heads = tuple(
            (layer_idx, head_idx)
            for layer_idx, attn in enumerate(student_list)
            if torch.is_tensor(attn)
            for head_idx in range(attn.shape[1])
        )

    total = None
    used_heads = 0
    pair_count_total = 0
    for layer_idx, head_idx in heads:
        if layer_idx >= len(student_list) or layer_idx >= len(teacher_list):
            continue
        student_attn, teacher_attn = student_list[layer_idx], teacher_list[layer_idx]
        if not torch.is_tensor(student_attn) or not torch.is_tensor(teacher_attn) or student_attn.ndim < 4:
            continue
        if head_idx is None or head_idx >= student_attn.shape[1] or head_idx >= teacher_attn.shape[1]:
            continue
        k = min(int(topk), teacher_attn.shape[-1] - 1)
        if k < 1:
            continue

        student_head = student_attn[:, head_idx]                       # [B, Q, K]
        teacher_head = teacher_attn[:, head_idx].detach().float()      # [B, Q, K]
        top_values, top_indices = teacher_head.topk(k, dim=-1)         # [B, Q, k]
        teacher_keys = teacher_head.unsqueeze(-2)
        valid = (top_values.unsqueeze(-1) > teacher_keys) & (teacher_keys > 0)
        pair_count = valid.sum(dim=(-1, -2))                           # [B, Q]
        rows = pair_count > 0
        if not bool(rows.any()):
            continue

        delta = (
            torch.log(student_head.gather(-1, top_indices).clamp_min(eps)).unsqueeze(-1)
            - torch.log(student_head.clamp_min(eps)).unsqueeze(-2)
        )                                                              # [B, Q, k, K]
        per_row = (F.softplus(-delta) * valid).sum(dim=(-1, -2)) / pair_count.clamp_min(1)
        head_loss = per_row[rows].mean()
        total = head_loss if total is None else total + head_loss
        used_heads += 1
        pair_count_total += int(pair_count.sum())

    if total is None:
        return torch.zeros((), device="cuda"), 0
    return total / used_heads, pair_count_total


__all__ = ["attention_relation_ranking_loss", "extract_attention_list"]
