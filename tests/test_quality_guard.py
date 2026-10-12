"""Behavioral guard checks based on observed conversational failure patterns."""

import pytest

from llm_training.quality_guard import behavioral_regression


def test_custom_policy_detects_observed_deterioration_legacy_policy_tolerates():
    baseline = dict(macro_task_accuracy=.142857, END_rate=.6875,
                    severe_degeneration_rate=.359375, empty_rate=0., role_switch_rate=0.)
    observed = dict(macro_task_accuracy=.125, END_rate=.5625,
                    severe_degeneration_rate=.515625, empty_rate=0., role_switch_rate=0.)
    assert not behavioral_regression(observed, baseline)
    assert behavioral_regression(observed, baseline,
                                 dict(task_drop=.05, END_drop=.10, degeneration_increase=.10))


def test_shorter_correct_answers_are_not_length_collapse():
    baseline = dict(macro_task_accuracy=.1, END_rate=.5, severe_degeneration_rate=.2,
                    empty_rate=0., role_switch_rate=0., mean_generated_tokens=100.)
    better = dict(baseline, macro_task_accuracy=.4, mean_generated_tokens=10.)
    assert not behavioral_regression(better, baseline)
    assert behavioral_regression(dict(better, macro_task_accuracy=.1), baseline)


@pytest.mark.parametrize('policy', [{'END_drop': -.1}, {'task_drop': float('nan')}, {'typo': .1}])
def test_invalid_thresholds_cannot_silently_disable_guard(policy):
    with pytest.raises(ValueError):
        behavioral_regression({}, {}, policy)


def test_bad_policy_fails_before_model_or_run_directory_creation(tmp_path):
    from llm_training.chat_run import run_chat
    spec = {'evaluation': {'regression_patience': 0}}
    output = tmp_path / 'never-created'
    with pytest.raises(ValueError, match='patience'):
        run_chat(spec, output, None)
    assert not output.exists()
