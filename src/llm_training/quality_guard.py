"""Configurable behavioral regression thresholds for conversational runs."""

import math
from typing import Mapping


def behavioral_regression(
    current: Mapping[str, float],
    reference: Mapping[str, float],
    policy: Mapping[str, float] | None = None,
) -> bool:
    """Compare to the retained reference; the caller applies sustained-check patience."""
    validate_regression_policy(policy)
    limits = policy or {}
    return (
        current['macro_task_accuracy'] < reference['macro_task_accuracy'] - limits.get('task_drop', .1)
        or current['severe_degeneration_rate'] > reference['severe_degeneration_rate'] + limits.get('degeneration_increase', .2)
        or current['END_rate'] < reference['END_rate'] - limits.get('END_drop', .2)
        or current['empty_rate'] > reference.get('empty_rate', 0) + limits.get('empty_increase', .2)
        or current['role_switch_rate'] > reference.get('role_switch_rate', 0) + limits.get('role_switch_increase', .2)
        or current.get('special_token_dominated_rate', 0) > limits.get('special_token_maximum', .25)
        or (
            reference.get('mean_generated_tokens', 0) > 0
            and current.get('mean_generated_tokens', float('inf'))
                < reference['mean_generated_tokens'] * limits.get('length_ratio_minimum', .3)
            and current['macro_task_accuracy'] <= reference['macro_task_accuracy']
        )
        or any(current.get('category_pass_rates', {}).get(category, 0) <
               reference.get('category_pass_rates', {}).get(category, 0) - limits[category+'_drop']
               for category in ('instruction','factual') if category+'_drop' in limits)
        or ('correct_END_drop' in limits and current.get('correct_END_rate', 0) <
            reference.get('correct_END_rate', 0) - limits['correct_END_drop'])
    )


def validate_regression_policy(policy: Mapping[str, float] | None) -> None:
    """Reject malformed thresholds before a training job allocates its model."""
    if policy is not None and not isinstance(policy, Mapping):
        raise ValueError('Behavioral regression policy must be a mapping.')
    limits = policy or {}
    allowed = {'task_drop', 'degeneration_increase', 'END_drop', 'empty_increase',
               'role_switch_increase', 'special_token_maximum', 'length_ratio_minimum',
               'instruction_drop', 'factual_drop', 'correct_END_drop'}
    if set(limits) - allowed:
        raise ValueError('Unknown behavioral regression threshold.')
    if any(not isinstance(value, (int, float)) or not math.isfinite(value)
           or not 0 <= value <= 1 for value in limits.values()):
        raise ValueError('Behavioral regression thresholds must be finite values in [0, 1].')
