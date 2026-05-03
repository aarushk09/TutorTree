from data_pipeline import (
    PROFILE_TO_OPTIMAL,
    BenchmarkScenario,
    _batch_prompt,
    _format_openai_status_error,
    _profile_for_index,
    validate_jsonl,
)
from pmcts.models import CognitiveLevel, Role, State


def test_benchmark_scenario_maps_to_state() -> None:
    scenario = BenchmarkScenario(
        scenario_id="scenario_001",
        domain="Python Programming",
        learning_objective="Explain why a Python loop skips an expected item",
        student_profile="developing Python learner with moderate debugging confidence",
        cognitive_level=CognitiveLevel.DEVELOPING,
        mastery_estimate=0.35,
        misconception="The student thinks range(5) includes 5.",
        chat_history=[
            {"role": "student", "content": "My loop never prints 5.", "metadata": {}},
            {"role": "teacher", "content": "What values do you expect range(5) to include?", "metadata": {}},
            {"role": "student", "content": "I thought it was 0 through 5, so I am lost.", "metadata": {}},
        ],
    )

    state = scenario.to_state()

    assert isinstance(state, State)
    assert state.metadata["scenario_id"] == "scenario_001"
    assert state.metadata["domain"] == "Python Programming"
    assert state.metadata["profile_type"] == "Standard Beginner"
    assert state.metadata["optimal_intervention"] == "hint"
    assert state.chat_history[-1].role == Role.STUDENT
    assert state.turn_index == 3


def test_validate_jsonl_loads_state_rows(tmp_path) -> None:
    state = State(
        learning_objective="Identify a contradiction in a logic puzzle",
        cognitive_level=CognitiveLevel.NOVICE,
        chat_history=(
            {"role": "student", "content": "I think both statements can be true.", "metadata": {}},
        ),
        metadata={"scenario_id": "scenario_002", "domain": "Logical Reasoning"},
    )
    path = tmp_path / "benchmark.jsonl"
    path.write_text(state.model_dump_json() + "\n", encoding="utf-8")

    loaded = validate_jsonl(path)

    assert loaded == [state]


def test_openai_403_message_is_actionable() -> None:
    class DummyStatusError(Exception):
        status_code = 403

    message = _format_openai_status_error(DummyStatusError(), model="gpt-4o-mini")

    assert "permission denied" in message.lower()
    assert "--model" in message


def test_adaptive_profile_schedule_has_equal_blocks() -> None:
    assert _profile_for_index(0, count_per_profile=100) == "Standard Beginner"
    assert _profile_for_index(100, count_per_profile=100) == "Advanced Student"
    assert _profile_for_index(200, count_per_profile=100) == "Frustrated/Overloaded Learner"
    assert _profile_for_index(300, count_per_profile=100) == "Deep Misconception"
    assert _profile_for_index(400, count_per_profile=100) == "Passive Learner"


def test_adaptive_prompt_forces_profile_and_optimal_intervention() -> None:
    prompt = _batch_prompt(
        start_index=201,
        count=10,
        profile_type="Frustrated/Overloaded Learner",
    )

    assert 'profile_type="Frustrated/Overloaded Learner"' in prompt
    assert f'"{PROFILE_TO_OPTIMAL["Frustrated/Overloaded Learner"]}"' in prompt
    assert "fixed Socratic prompting must NOT be optimal for all scenarios" in prompt
