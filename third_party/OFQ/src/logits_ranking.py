"""Classification-logit ranking：把教师 Top-k 类别的顺序迁移到学生 logits 上。

与模型无关：输入就是 (student_logits, teacher_logits)，形状 [B, C]；DeiT 蒸馏模型训练时
返回 (cls, dist) 元组，这里按仓库既有 KD 辅助函数的约定取第一个 (cls)。
教师只提供类别索引与"谁排在谁前面"，不提供分数数值。
"""
import torch
import torch.nn.functional as F


def primary_logits(logits):
    """DeiT 蒸馏模型训练时返回 (cls, dist)，取 cls（与仓库里 KD 辅助函数处理一致）。"""
    if isinstance(logits, (tuple, list)):
        return logits[0]
    return logits


def logits_ranking_loss(student_logits, teacher_logits, topk: int = 5):
    """教师 Top-k 类别的成对排序损失。

    对每个样本取教师分数最高的 topk 个类别，与所有教师分数严格更低的类别组成有向对,
    用 softplus(-(s_c - s_d)) 要求学生在同样两个类别上保持同样顺序。
    并列（教师分数相等）的对不计入；先按样本对有效对取平均，再对样本取平均。
    返回 (loss, 有效对数)；没有任何有效对时返回 0。
    """
    student_logits = primary_logits(student_logits)
    teacher_logits = primary_logits(teacher_logits).detach()
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("Expected paired [B, C] logits")

    k = min(int(topk), teacher_logits.shape[1] - 1)
    if k < 1:
        return torch.zeros((), device=student_logits.device), 0

    top_values, top_indices = teacher_logits.topk(k, dim=-1)             # [B, k]
    student_top = student_logits.gather(1, top_indices)                  # [B, k]，按教师给的类别索引取
    valid = top_values.unsqueeze(-1) > teacher_logits.unsqueeze(-2)      # [B, k, C]
    delta = student_top.unsqueeze(-1) - student_logits.unsqueeze(-2)     # [B, k, C]
    counts = valid.sum(dim=(1, 2))                                       # [B]
    rows = counts > 0
    if not bool(rows.any()):
        return torch.zeros((), device=student_logits.device), 0

    per_sample = (F.softplus(-delta) * valid).sum(dim=(1, 2)) / counts.clamp_min(1)
    return per_sample[rows].mean(), int(counts.sum())


__all__ = ["logits_ranking_loss", "primary_logits"]
