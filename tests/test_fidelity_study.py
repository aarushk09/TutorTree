from fidelity_study import (
    FidelityResult,
    alignment_accuracy,
    binomial_tail_p_value,
    confusion_matrix,
    is_aligned,
)
from pmcts.models import Action, InterventionType


def test_is_aligned_matches_icap_expectations() -> None:
    direct = Action(
        intervention_type=InterventionType.DIRECT_ANSWER,
        content="The answer is 5.",
    )
    socratic = Action(
        intervention_type=InterventionType.SOCRATIC_PROMPT,
        content="What should we try first?",
    )

    assert is_aligned(direct, "Passive")
    assert is_aligned(direct, "Active")
    assert not is_aligned(direct, "Constructive")
    assert is_aligned(socratic, "Constructive")
    assert is_aligned(socratic, "Interactive")
    assert not is_aligned(socratic, "Passive")


def test_alignment_accuracy_and_confusion_matrix() -> None:
    results = [
        FidelityResult("s1", "Math", "direct_answer", "low", "Passive", "novice", True, ""),
        FidelityResult("s1", "Math", "socratic_prompt", "high", "Constructive", "developing", True, ""),
        FidelityResult("s2", "Python", "direct_answer", "low", "Constructive", "developing", False, ""),
    ]

    matrix = confusion_matrix(results)

    assert alignment_accuracy(results) == 2 / 3
    assert matrix["direct_answer"]["Passive"] == 1
    assert matrix["direct_answer"]["Constructive"] == 1
    assert matrix["socratic_prompt"]["Constructive"] == 1


def test_binomial_tail_p_value_is_small_for_high_alignment() -> None:
    p_value = binomial_tail_p_value(successes=54, trials=60, chance=0.5)

    assert p_value < 0.001
