"""Create blinded human-evaluation tasks from divergent model decisions."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pmcts.models import State
from run_experiment import action_for_type


@dataclass(frozen=True)
class DivergentCase:
    scenario_id: str
    model_name: str
    baseline_action: str
    pmcts_action: str
    state: State


def load_states_by_id(path: Path) -> dict[str, State]:
    states: dict[str, State] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            state = State.model_validate_json(line)
            states[str(state.metadata["scenario_id"])] = state
    return states


def load_divergent_cases(results_path: Path, states_by_id: dict[str, State]) -> list[DivergentCase]:
    grouped: dict[tuple[str, str], dict[str, str]] = {}
    with results_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["scenario_id"], row["model_name"])
            grouped.setdefault(key, {})[row["runner_type"]] = row["selected_action"]

    cases_by_scenario: dict[str, DivergentCase] = {}
    for (scenario_id, model_name), runners in sorted(grouped.items()):
        baseline = runners.get("baseline_socratic_prompt")
        pmcts = runners.get("pmcts_full")
        if baseline is None or pmcts is None or baseline == pmcts:
            continue
        if scenario_id not in states_by_id:
            continue
        cases_by_scenario.setdefault(
            scenario_id,
            DivergentCase(
                scenario_id=scenario_id,
                model_name=model_name,
                baseline_action=baseline,
                pmcts_action=pmcts,
                state=states_by_id[scenario_id],
            ),
        )
    return list(cases_by_scenario.values())


def export_human_eval(
    cases: list[DivergentCase],
    *,
    sample_size: int,
    task_path: Path,
    key_path: Path,
    seed: int,
) -> list[DivergentCase]:
    if len(cases) < sample_size:
        raise ValueError(f"need {sample_size} divergent cases, found {len(cases)}")

    rng = random.Random(seed)
    sampled = rng.sample(cases, sample_size)

    task_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    with task_path.open("w", newline="", encoding="utf-8") as task_handle, key_path.open(
        "w", newline="", encoding="utf-8"
    ) as key_handle:
        task_writer = csv.DictWriter(
            task_handle,
            fieldnames=[
                "Scenario_ID",
                "Student_Profile",
                "Chat_Context",
                "Intervention_A",
                "Intervention_B",
                "Expert_Preference",
                "Reasoning",
            ],
        )
        key_writer = csv.DictWriter(
            key_handle,
            fieldnames=[
                "Scenario_ID",
                "Intervention_A_Source",
                "Intervention_B_Source",
            ],
        )
        task_writer.writeheader()
        key_writer.writeheader()
        for case in sampled:
            baseline_text = intervention_text(case.state, case.baseline_action)
            pmcts_text = intervention_text(case.state, case.pmcts_action)
            pmcts_is_a = rng.choice([True, False])
            intervention_a = pmcts_text if pmcts_is_a else baseline_text
            intervention_b = baseline_text if pmcts_is_a else pmcts_text
            source_a = "pmcts_full" if pmcts_is_a else "baseline_socratic_prompt"
            source_b = "baseline_socratic_prompt" if pmcts_is_a else "pmcts_full"

            task_writer.writerow(
                {
                    "Scenario_ID": case.scenario_id,
                    "Student_Profile": str(case.state.metadata.get("profile_type", case.state.student_profile)),
                    "Chat_Context": format_chat_context(case.state),
                    "Intervention_A": intervention_a,
                    "Intervention_B": intervention_b,
                    "Expert_Preference": "",
                    "Reasoning": "",
                }
            )
            key_writer.writerow(
                {
                    "Scenario_ID": case.scenario_id,
                    "Intervention_A_Source": source_a,
                    "Intervention_B_Source": source_b,
                }
            )
    return sampled


def intervention_text(state: State, selected_action: str) -> str:
    action = action_for_type(state, selected_action)
    return f"{action.intervention_type.value}: {action.content}"


def format_chat_context(state: State) -> str:
    return "\n".join(f"{message.role.value}: {message.content}" for message in state.chat_history)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "multimodel_adaptive_results.csv")
    parser.add_argument("--benchmark", type=Path, default=PROJECT_ROOT / "adaptive_benchmark_500.jsonl")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--task-output", type=Path, default=PROJECT_ROOT / "human_rating_task.csv")
    parser.add_argument("--key-output", type=Path, default=PROJECT_ROOT / "human_rating_key.csv")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    states_by_id = load_states_by_id(args.benchmark)
    cases = load_divergent_cases(args.results, states_by_id)
    sampled = export_human_eval(
        cases,
        sample_size=args.sample_size,
        task_path=args.task_output,
        key_path=args.key_output,
        seed=args.seed,
    )
    print(f"Found divergent scenarios: {len(cases)}")
    print(f"Exported blinded scenarios: {len(sampled)}")
    print(f"Task CSV: {args.task_output}")
    print(f"Hidden key CSV: {args.key_output}")


if __name__ == "__main__":
    main()
