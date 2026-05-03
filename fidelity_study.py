"""Evaluate LiveSimStudent alignment with ICAP expectations."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pmcts.models import Action, InterventionType, State
from pmcts.sim_student import LiveSimStudent


LOW_ENGAGEMENT = {"Passive", "Active"}
HIGH_ENGAGEMENT = {"Constructive", "Interactive"}
ACTION_ORDER = ["direct_answer", "socratic_prompt"]
ACTIVITY_ORDER = ["Passive", "Active", "Constructive", "Interactive", "Unknown"]


class StudentSimulator(Protocol):
    def predict_next_state(self, state: State, action: Action) -> State:
        """Return the simulated next state for a state-action pair."""


@dataclass(frozen=True)
class FidelityResult:
    scenario_id: str
    domain: str
    action_type: str
    expected_band: str
    observed_activity: str
    cognitive_level: str
    aligned: bool
    student_reply: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def load_states(path: Path, *, limit: int) -> list[State]:
    states: list[State] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            states.append(State.model_validate_json(line))
            if len(states) >= limit:
                break
    return states


def build_test_actions(state: State) -> list[Action]:
    return [
        Action(
            intervention_type=InterventionType.DIRECT_ANSWER,
            content=(
                "The answer is the correct final result. Here is the full "
                f"step-by-step solution for {state.learning_objective}."
            ),
            rationale="Directly supplies the solution path.",
            expected_student_activity="Passive",
        ),
        Action(
            intervention_type=InterventionType.SOCRATIC_PROMPT,
            content=(
                "That is a great question. What do you think the first step "
                "should be based on our previous rule?"
            ),
            rationale="Elicits student reasoning before revealing the answer.",
            expected_student_activity="Constructive",
        ),
    ]


def run_fidelity_study(
    states: list[State],
    simulator: StudentSimulator,
    *,
    output_path: Path,
) -> list[FidelityResult]:
    output_path.write_text("", encoding="utf-8")
    results: list[FidelityResult] = []

    for state_index, state in enumerate(states, start=1):
        for action in build_test_actions(state):
            next_state = simulator.predict_next_state(state, action)
            result = make_fidelity_result(state, action, next_state)
            results.append(result)
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(result.to_json() + "\n")
        print(f"Processed {state_index}/{len(states)} scenarios")

    return results


def make_fidelity_result(state: State, action: Action, next_state: State) -> FidelityResult:
    activity = str(next_state.metadata.get("activity", "Unknown"))
    expected_band = expected_engagement_band(action)
    aligned = is_aligned(action, activity)
    return FidelityResult(
        scenario_id=str(state.metadata.get("scenario_id", "unknown")),
        domain=str(state.metadata.get("domain", "unknown")),
        action_type=action.intervention_type.value,
        expected_band=expected_band,
        observed_activity=activity,
        cognitive_level=next_state.cognitive_level.value,
        aligned=aligned,
        student_reply=next_state.chat_history[-1].content if next_state.chat_history else "",
    )


def expected_engagement_band(action: Action) -> str:
    if action.intervention_type == InterventionType.DIRECT_ANSWER:
        return "low"
    if action.intervention_type == InterventionType.SOCRATIC_PROMPT:
        return "high"
    return "unknown"


def is_aligned(action: Action, observed_activity: str) -> bool:
    if action.intervention_type == InterventionType.DIRECT_ANSWER:
        return observed_activity in LOW_ENGAGEMENT
    if action.intervention_type == InterventionType.SOCRATIC_PROMPT:
        return observed_activity in HIGH_ENGAGEMENT
    return False


def confusion_matrix(results: list[FidelityResult]) -> dict[str, Counter[str]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for result in results:
        action_type = result.action_type
        activity = result.observed_activity
        if activity not in ACTIVITY_ORDER:
            activity = "Unknown"
        matrix[action_type][activity] += 1
    return matrix


def alignment_accuracy(results: list[FidelityResult]) -> float:
    if not results:
        raise ValueError("cannot calculate accuracy for empty results")
    return sum(result.aligned for result in results) / len(results)


def binomial_tail_p_value(successes: int, trials: int, *, chance: float = 0.5) -> float:
    if trials < 1:
        raise ValueError("trials must be positive")
    return sum(
        math.comb(trials, k) * (chance**k) * ((1.0 - chance) ** (trials - k))
        for k in range(successes, trials + 1)
    )


def print_report(results: list[FidelityResult]) -> None:
    matrix = confusion_matrix(results)
    accuracy = alignment_accuracy(results)
    successes = sum(result.aligned for result in results)
    p_value = binomial_tail_p_value(successes, len(results), chance=0.5)

    print("Simulator Fidelity Study")
    print("=" * 25)
    print("Confusion Matrix: action_type x observed ICAP activity")
    print(f"{'Action':18} " + " ".join(f"{activity:>13}" for activity in ACTIVITY_ORDER))
    print("-" * 91)
    for action_type in ACTION_ORDER:
        row = matrix[action_type]
        print(f"{action_type:18} " + " ".join(f"{row[activity]:13d}" for activity in ACTIVITY_ORDER))
    print("-" * 91)
    print(f"Aligned outcomes: {successes}/{len(results)}")
    print(f"Simulator Fidelity Score: {accuracy * 100:.2f}%")
    print(f"One-sided binomial p-value vs 50% chance: {p_value:.6f}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "benchmark_100.jsonl")
    parser.add_argument("--subset", type=int, default=30)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "simulator_fidelity_results.jsonl",
    )
    parser.add_argument("--min-accuracy", type=float, default=0.80)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set to run the live fidelity study")

    states = load_states(args.input, limit=args.subset)
    if len(states) < args.subset:
        raise ValueError(f"requested {args.subset} states but loaded only {len(states)}")

    simulator = LiveSimStudent(model=args.model, temperature=args.temperature)
    results = run_fidelity_study(states, simulator, output_path=args.output)
    print_report(results)

    accuracy = alignment_accuracy(results)
    if accuracy < args.min_accuracy:
        raise SystemExit(
            f"Simulator fidelity below threshold: {accuracy * 100:.2f}% "
            f"< {args.min_accuracy * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
