import csv

import pytest

import run_experiment
from run_experiment import (
    MULTIMODEL_RUNNER_TYPES,
    MultiModelRow,
    RUNNER_TYPES,
    fallback_sim_student_response,
    preflight_models,
    parse_reward_response,
    read_multimodel_csv,
    resolve_model_name,
    parse_sim_student_response,
    run_all,
    summarize,
    summarize_multimodel,
    write_csv,
    write_multimodel_csv,
)
from pmcts.models import Action, InterventionType, Message, Role, State


def test_run_all_produces_one_row_per_runner_per_state() -> None:
    states = [
        State(
            learning_objective="Solve 2x + 5 = 15",
            chat_history=(Message(role=Role.STUDENT, content="I do not know the first step."),),
            metadata={"scenario_id": "scenario_001", "domain": "High School Math"},
        ),
        State(
            learning_objective="Debug a Python TypeError",
            chat_history=(Message(role=Role.STUDENT, content="Why did adding text and a number fail?"),),
            metadata={"scenario_id": "scenario_002", "domain": "Python Programming"},
        ),
    ]

    rows = run_all(
        states,
        backend="heuristic",
        model="unused",
        pmcts_iterations=3,
        shallow_iterations=1,
        max_depth=2,
        seed=7,
    )

    assert len(rows) == len(states) * len(RUNNER_TYPES)
    assert {row.runner_type for row in rows} == set(RUNNER_TYPES)
    assert all(row.scenario_id.startswith("scenario_") for row in rows)
    assert all(row.total_tokens_used >= 0 for row in rows)


def test_write_csv_uses_required_columns(tmp_path) -> None:
    states = [
        State(
            learning_objective="Find a logical contradiction",
            chat_history=(Message(role=Role.STUDENT, content="I think both claims work."),),
            metadata={"scenario_id": "scenario_001", "domain": "Logical Reasoning"},
        )
    ]
    rows = run_all(
        states,
        backend="heuristic",
        model="unused",
        pmcts_iterations=2,
        shallow_iterations=1,
        max_depth=1,
        seed=7,
    )
    output = tmp_path / "results.csv"

    write_csv(rows, output)

    with output.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        loaded = list(reader)

    assert reader.fieldnames == [
        "scenario_id",
        "domain",
        "runner_type",
        "selected_action",
        "student_state",
        "pedagogical_reward",
        "total_tokens_used",
    ]
    assert len(loaded) == len(RUNNER_TYPES)


def test_summarize_returns_all_runners() -> None:
    states = [
        State(
            learning_objective="Use range in Python",
            chat_history=(Message(role=Role.STUDENT, content="Does range(5) include 5?"),),
            metadata={"scenario_id": "scenario_001", "domain": "Python Programming"},
        )
    ]
    rows = run_all(
        states,
        backend="heuristic",
        model="unused",
        pmcts_iterations=2,
        shallow_iterations=1,
        max_depth=1,
        seed=7,
    )

    summary = summarize(rows)

    assert [runner for runner, _, _ in summary] == RUNNER_TYPES


def test_write_multimodel_csv_uses_phase_6_columns(tmp_path) -> None:
    rows = [
        MultiModelRow(
            scenario_id="scenario_001",
            domain="High School Math",
            student_profile="Standard Beginner",
            model_name="gpt-4o-mini",
            runner_type="baseline_socratic_prompt",
            selected_action="socratic_prompt",
            engagement_score=0.8,
            frustration_score=0.2,
            efficiency_score=0.5,
            total_reward=0.85,
        )
    ]
    output = tmp_path / "multimodel.csv"

    write_multimodel_csv(rows, output)

    with output.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        loaded = list(reader)

    assert reader.fieldnames == [
        "scenario_id",
        "domain",
        "student_profile",
        "model_name",
        "runner_type",
        "selected_action",
        "engagement_score",
        "frustration_score",
        "efficiency_score",
        "total_reward",
    ]
    assert loaded[0]["model_name"] == "gpt-4o-mini"


def test_summarize_multimodel_groups_by_model_and_runner() -> None:
    rows = [
        MultiModelRow("s1", "Math", "Standard", "model-a", "baseline_socratic_prompt", "socratic_prompt", 1, 0, 1, 1.0),
        MultiModelRow("s1", "Math", "Standard", "model-a", "pmcts_full", "hint", 0.7, 0.1, 0.9, 1.0),
        MultiModelRow("s2", "Math", "Standard", "model-a", "baseline_socratic_prompt", "socratic_prompt", 0.2, 0.9, 0.2, -0.6),
        MultiModelRow("s2", "Math", "Standard", "model-a", "pmcts_full", "direct_answer", 0.4, 0.1, 0.8, 0.7),
    ]

    summary = summarize_multimodel(rows)

    assert summary == [
        ("model-a", "baseline_socratic_prompt", 0.2, 2),
        ("model-a", "pmcts_full", 0.85, 2),
    ]


def test_preflight_models_reports_missing_provider_keys(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        preflight_models(["claude-3-haiku-20240307"])


def test_resolve_model_name_maps_retired_claude_haiku() -> None:
    assert resolve_model_name("claude-3-haiku-20240307") == "claude-haiku-4-5-20251001"
    assert resolve_model_name("groq/llama3-8b-8192") == "groq/llama-3.1-8b-instant"
    assert resolve_model_name("gpt-4o-mini") == "gpt-4o-mini"


def test_parse_sim_student_response_normalizes_beginner_alias() -> None:
    parsed = parse_sim_student_response(
        """
        {
          "reply": "I guess I should subtract 3 first, then what?",
          "activity": "constructive",
          "frustration_score": 0.6,
          "cognitive_level": "beginner",
          "mastery_estimate": 0.4
        }
        """
    )

    assert parsed.cognitive_level.value == "novice"
    assert parsed.activity == "Constructive"


def test_parse_sim_student_response_normalizes_profile_label_misfiled_as_level() -> None:
    parsed = parse_sim_student_response(
        """
        {
          "reply": "Okay, got it.",
          "activity": "Passive",
          "frustration_score": 0.2,
          "cognitive_level": "Standard Beginner",
          "mastery_estimate": 0.4
        }
        """
    )

    assert parsed.cognitive_level.value == "novice"


def test_parse_sim_student_response_normalizes_activity_label_misfiled_as_level() -> None:
    parsed = parse_sim_student_response(
        """
        {
          "reply": "Oh, I see.",
          "activity": "Passive",
          "frustration_score": "20%",
          "cognitive_level": "Passive",
          "mastery_estimate": 30
        }
        """
    )

    assert parsed.cognitive_level.value == "novice"
    assert parsed.frustration_score == 0.2
    assert parsed.mastery_estimate == 1.0


def test_parse_sim_student_response_normalizes_capitalized_valid_level() -> None:
    parsed = parse_sim_student_response(
        """
        {
          "reply": "Oh, I see.",
          "activity": "Passive",
          "frustration_score": 0.2,
          "cognitive_level": "Novice",
          "mastery_estimate": 0.3
        }
        """
    )

    assert parsed.cognitive_level.value == "novice"


def test_parse_sim_student_response_normalizes_qualitative_mastery() -> None:
    parsed = parse_sim_student_response(
        """
        {
          "reply": "I can try.",
          "activity": "Active",
          "frustration_score": "High",
          "cognitive_level": "novice",
          "mastery_estimate": "Low"
        }
        """
    )

    assert parsed.frustration_score == 0.75
    assert parsed.mastery_estimate == 0.25


def test_parse_sim_student_response_normalizes_cognitive_word_as_mastery() -> None:
    parsed = parse_sim_student_response(
        """
        {
          "reply": "I can handle that.",
          "activity": "Constructive",
          "frustration_score": "low",
          "cognitive_level": "advanced",
          "mastery_estimate": "advanced"
        }
        """
    )

    assert parsed.cognitive_level.value == "advanced"
    assert parsed.mastery_estimate == 0.9


def test_parse_sim_student_response_normalizes_socratic_activity_label() -> None:
    parsed = parse_sim_student_response(
        """
        ```json
        {
          "reply": "I think I need to store the result as I go.",
          "activity": "Socratic",
          "frustration_score": 0.4,
          "cognitive_level": "Constructive",
          "mastery_estimate": 0.2
        }
        ```
        """
    )

    assert parsed.activity == "Constructive"
    assert parsed.cognitive_level.value == "developing"


def test_parse_reward_response_extracts_scores_from_malformed_json() -> None:
    state = State(
        learning_objective="Debug factorial loops",
        chat_history=(Message(role=Role.STUDENT, content="I do not get the loop."),),
        metadata={"optimal_intervention": "hint"},
    )
    action = Action(
        intervention_type=InterventionType.HINT,
        content="Look at where the accumulator changes.",
    )
    next_state = State(
        learning_objective=state.learning_objective,
        chat_history=state.chat_history,
        metadata={"activity": "Constructive", "frustration": 0.2},
    )

    parsed = parse_reward_response(
        """
        {
          "engagement_score": 0.8,
          "frustration_score": 0.1,
          "efficiency_score": 0.7,
          "rationale": "The student said "I can try that" which broke JSON"
        }
        """,
        state,
        action,
        next_state,
    )

    assert parsed["engagement_score"] == 0.8
    assert parsed["frustration_score"] == 0.1
    assert parsed["efficiency_score"] == 0.7


def test_fallback_sim_student_response_penalizes_socratic_when_frustrated() -> None:
    state = State(
        learning_objective="Solve equations",
        chat_history=(Message(role=Role.STUDENT, content="I am completely lost."),),
        metadata={"profile_type": "Frustrated/Overloaded Learner"},
    )
    action = Action(
        intervention_type=InterventionType.SOCRATIC_PROMPT,
        content="What should the first step be?",
    )

    parsed = fallback_sim_student_response(state, action)

    assert parsed.activity == "Active"
    assert parsed.frustration_score > 0.8


def test_safe_litellm_completion_retries_transient_failure(monkeypatch) -> None:
    calls = {"count": 0}

    class Usage:
        total_tokens = 3

    class Message:
        content = "{}"

    class Choice:
        message = Message()

    class Response:
        usage = Usage()
        choices = [Choice()]

    def flaky_completion(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("temporary")
        return Response()

    monkeypatch.setattr(run_experiment.litellm, "completion", flaky_completion)
    monkeypatch.setattr(run_experiment.time, "sleep", lambda seconds: None)

    response = run_experiment.safe_litellm_completion(
        model="test-model",
        temperature=0.0,
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response is not None
    assert calls["count"] == 2


def test_safe_litellm_completion_does_not_retry_decommissioned_model(monkeypatch) -> None:
    calls = {"count": 0}

    def bad_completion(**kwargs):
        calls["count"] += 1
        raise RuntimeError("model_decommissioned: llama3-8b-8192")

    monkeypatch.setattr(run_experiment.litellm, "completion", bad_completion)
    monkeypatch.setattr(run_experiment.time, "sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="model_decommissioned"):
        run_experiment.safe_litellm_completion(
            model="groq/llama3-8b-8192",
            temperature=0.0,
            messages=[{"role": "user", "content": "hi"}],
        )

    assert calls["count"] == 1


def test_read_multimodel_csv_roundtrips_written_rows(tmp_path) -> None:
    rows = [
        MultiModelRow(
            scenario_id="scenario_001",
            domain="High School Math",
            student_profile="Standard Beginner",
            model_name="gpt-4o-mini",
            runner_type="pmcts_full",
            selected_action="hint",
            engagement_score=0.7,
            frustration_score=0.1,
            efficiency_score=0.9,
            total_reward=1.0,
        )
    ]
    output = tmp_path / "rows.csv"

    write_multimodel_csv(rows, output)

    assert read_multimodel_csv(output) == rows
