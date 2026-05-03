import csv

from human_eval import export_human_eval, load_divergent_cases
from pmcts.models import Message, Role, State


def test_load_divergent_cases_filters_matching_actions(tmp_path) -> None:
    state = State(
        learning_objective="Solve a linear equation",
        chat_history=(Message(role=Role.STUDENT, content="I am stuck."),),
        metadata={"scenario_id": "scenario_001", "profile_type": "Standard Beginner"},
    )
    results = tmp_path / "results.csv"
    results.write_text(
        "\n".join(
            [
                "scenario_id,domain,student_profile,model_name,runner_type,selected_action,engagement_score,frustration_score,efficiency_score,total_reward",
                "scenario_001,Math,Standard Beginner,model-a,baseline_socratic_prompt,socratic_prompt,1,0,1,1",
                "scenario_001,Math,Standard Beginner,model-a,pmcts_full,hint,0.8,0.1,1,1",
                "scenario_002,Math,Standard Beginner,model-a,baseline_socratic_prompt,socratic_prompt,1,0,1,1",
                "scenario_002,Math,Standard Beginner,model-a,pmcts_full,socratic_prompt,1,0,1,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_divergent_cases(results, {"scenario_001": state})

    assert len(cases) == 1
    assert cases[0].baseline_action == "socratic_prompt"
    assert cases[0].pmcts_action == "hint"


def test_export_human_eval_blinds_sources_and_writes_key(tmp_path) -> None:
    state = State(
        learning_objective="Solve a linear equation",
        chat_history=(Message(role=Role.STUDENT, content="I am stuck."),),
        metadata={"scenario_id": "scenario_001", "profile_type": "Standard Beginner"},
    )
    from human_eval import DivergentCase

    case = DivergentCase(
        scenario_id="scenario_001",
        model_name="model-a",
        baseline_action="socratic_prompt",
        pmcts_action="hint",
        state=state,
    )
    task_path = tmp_path / "task.csv"
    key_path = tmp_path / "key.csv"

    export_human_eval([case], sample_size=1, task_path=task_path, key_path=key_path, seed=1)

    with task_path.open("r", encoding="utf-8", newline="") as handle:
        task_rows = list(csv.DictReader(handle))
    with key_path.open("r", encoding="utf-8", newline="") as handle:
        key_rows = list(csv.DictReader(handle))

    assert task_rows[0]["Expert_Preference"] == ""
    assert task_rows[0]["Reasoning"] == ""
    assert key_rows[0]["Intervention_A_Source"] in {"pmcts_full", "baseline_socratic_prompt"}
    assert key_rows[0]["Intervention_B_Source"] in {"pmcts_full", "baseline_socratic_prompt"}
