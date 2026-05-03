"""Psychometric evaluation utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence


class PsychometricEvaluator:
    """Pure mathematical evaluators for learning benchmark analysis."""

    @staticmethod
    def calculate_irt_2pl(theta: float, a: float, b: float) -> float:
        """Return the 2PL IRT probability of a correct response."""

        z = a * (theta - b)
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))

        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

    @staticmethod
    def calculate_hakes_gain(
        pre_test_scores: Sequence[float],
        post_test_scores: Sequence[float],
        max_score: float = 100.0,
    ) -> float:
        """Return Hake's normalized learning gain."""

        _validate_non_empty(pre_test_scores, "pre_test_scores")
        _validate_non_empty(post_test_scores, "post_test_scores")
        if max_score <= 0:
            raise ValueError("max_score must be positive")

        pre_avg = _mean(pre_test_scores)
        post_avg = _mean(post_test_scores)
        denominator = max_score - pre_avg

        if math.isclose(denominator, 0.0):
            return 0.0
        if denominator < 0:
            raise ValueError("pre-test average cannot exceed max_score")

        return (post_avg - pre_avg) / denominator

    @staticmethod
    def calculate_hedges_g(
        treatment_scores: Sequence[float],
        control_scores: Sequence[float],
    ) -> float:
        """Return Hedges' g with small-sample correction."""

        _validate_minimum_size(treatment_scores, "treatment_scores", minimum=2)
        _validate_minimum_size(control_scores, "control_scores", minimum=2)

        n_treatment = len(treatment_scores)
        n_control = len(control_scores)
        treatment_variance = _sample_variance(treatment_scores)
        control_variance = _sample_variance(control_scores)

        pooled_variance = (
            ((n_treatment - 1) * treatment_variance)
            + ((n_control - 1) * control_variance)
        ) / (n_treatment + n_control - 2)

        if pooled_variance < 0:
            raise ValueError("pooled variance cannot be negative")

        pooled_sd = math.sqrt(pooled_variance)
        if math.isclose(pooled_sd, 0.0):
            raise ValueError("pooled standard deviation must be non-zero")

        cohen_d = (_mean(treatment_scores) - _mean(control_scores)) / pooled_sd
        correction = 1.0 - (3.0 / (4.0 * (n_treatment + n_control) - 9.0))
        return correction * cohen_d


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _sample_variance(values: Sequence[float]) -> float:
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _validate_non_empty(values: Sequence[float], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    _validate_finite(values, name)


def _validate_minimum_size(values: Sequence[float], name: str, *, minimum: int) -> None:
    if len(values) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} scores")
    _validate_finite(values, name)


def _validate_finite(values: Sequence[float], name: str) -> None:
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values")
