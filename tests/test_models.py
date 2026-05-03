import pytest
from pydantic import ValidationError

from pmcts.models import (
    Action,
    CognitiveLevel,
    InterventionType,
    Message,
    Node,
    Role,
    State,
)


def test_state_append_returns_new_state_with_incremented_turn() -> None:
    state = State(
        learning_objective="Solve one-step linear equations",
        cognitive_level=CognitiveLevel.NOVICE,
    )

    next_state = state.append(
        Role.STUDENT,
        "I do not know how to isolate x.",
        confidence="low",
    )

    assert state.turn_index == 0
    assert state.chat_history == ()
    assert next_state.turn_index == 1
    assert next_state.chat_history[0] == Message(
        role=Role.STUDENT,
        content="I do not know how to isolate x.",
        metadata={"confidence": "low"},
    )


def test_action_strips_content_and_validates_empty_optional_text() -> None:
    action = Action(
        intervention_type=InterventionType.SOCRATIC_PROMPT,
        content="  What operation would undo adding 3?  ",
        rationale="  Encourage constructive reasoning  ",
    )

    assert action.content == "What operation would undo adding 3?"
    assert action.rationale == "Encourage constructive reasoning"

    with pytest.raises(ValidationError):
        Action(
            intervention_type=InterventionType.HINT,
            content="Try inverse operations.",
            expected_student_activity="   ",
        )


def test_node_value_estimate_uses_backed_up_mean() -> None:
    state = State(learning_objective="Solve equations")
    node = Node(state=state)

    assert node.value_estimate == 0.0

    node.visits = 4
    node.value_sum = 3.0

    assert node.value_estimate == pytest.approx(0.75)


def test_node_rejects_invalid_bookkeeping_values() -> None:
    state = State(learning_objective="Solve equations")

    with pytest.raises(ValidationError):
        Node(state=state, visits=-1)

    with pytest.raises(ValidationError):
        Node(state=state, prior_probability=1.5)
