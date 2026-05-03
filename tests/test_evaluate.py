import math

import pytest

from pmcts.evaluate import PsychometricEvaluator


def test_calculate_irt_2pl_returns_half_when_ability_equals_difficulty() -> None:
    probability = PsychometricEvaluator.calculate_irt_2pl(theta=1.0, a=1.0, b=1.0)

    assert probability == pytest.approx(0.5)


def test_calculate_hakes_gain_uses_average_pre_and_post_scores() -> None:
    gain = PsychometricEvaluator.calculate_hakes_gain(
        pre_test_scores=[50, 60],
        post_test_scores=[75, 80],
        max_score=100,
    )

    assert gain == pytest.approx(0.5)


def test_calculate_hedges_g_for_length_ten_samples() -> None:
    treatment = [80, 82, 84, 86, 88, 90, 92, 94, 96, 98]
    control = [70, 72, 74, 76, 78, 80, 82, 84, 86, 88]

    effect_size = PsychometricEvaluator.calculate_hedges_g(treatment, control)

    assert math.isfinite(effect_size)
    assert effect_size > 0


def test_calculate_hedges_g_rejects_zero_pooled_variance() -> None:
    with pytest.raises(ValueError, match="pooled standard deviation"):
        PsychometricEvaluator.calculate_hedges_g([80] * 10, [80] * 10)


def test_calculate_hakes_gain_returns_zero_when_pre_average_is_max_score() -> None:
    gain = PsychometricEvaluator.calculate_hakes_gain(
        pre_test_scores=[100, 100],
        post_test_scores=[100, 100],
        max_score=100,
    )

    assert gain == 0.0
