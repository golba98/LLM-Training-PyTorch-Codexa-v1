"""Optional target-token weighting with unchanged assistant-token normalization."""

from collections.abc import Mapping
import math

import torch
import torch.nn.functional as F


def validate_target_weights(weights: Mapping[int, float] | None) -> None:
    """Reject invalid weighting configuration before training begins."""
    if weights is None:
        return
    if not isinstance(weights, Mapping):
        raise ValueError('Target weights must be a mapping of token IDs to positive finite weights.')
    if any(type(token) is not int or token < 0 or not isinstance(weight, (int, float))
           or not math.isfinite(weight) or weight <= 0 for token, weight in weights.items()):
        raise ValueError('Target weights must use nonnegative integer IDs and positive finite weights.')


def weighted_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: Mapping[int, float],
) -> torch.Tensor:
    """Return sum(weight × CE) / actual supervised tokens, excluding padding."""
    validate_target_weights(weights)
    if any(token >= logits.shape[-1] for token in weights):
        raise ValueError('Weighted target ID exceeds vocabulary.')
    targets = labels.reshape(-1)
    supervised = targets.ne(-100)
    if not supervised.any():
        raise ValueError('Weighted batch contains no target tokens.')
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets,
                           ignore_index=-100, reduction='none')
    multipliers = torch.ones_like(loss)
    for token, weight in weights.items():
        multipliers = torch.where(targets.eq(token), weight, multipliers)
    return (loss * multipliers).sum() / supervised.sum()
