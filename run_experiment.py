"""Mass benchmark evaluation harness for P-MCTS baselines and ablations."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import litellm
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pmcts.mcts import PedagogicalMCTS
from pmcts.models import Action, CognitiveLevel, InterventionType, Node, Role, State
from pmcts.reward import LiveRewardFunction
from pmcts.sim_student import LiveSimStudent, SimStudentResponse, SIM_STUDENT_SYSTEM_PROMPT


RUNNER_TYPES = [
    "baseline_greedy",
    "baseline_socratic_prompt",
    "pmcts_full",
    "ablation_no_reward",
    "ablation_shallow",
]
MULTIMODEL_RUNNER_TYPES = ["baseline_socratic_prompt", "pmcts_full"]
DEFAULT_MULTIMODEL_MODELS = [
    "gpt-4o-mini",
    "claude-haiku-4-5-20251001",
    "groq/llama3-8b-8192",
]
LITELLM_TIMEOUT_SECONDS = float(os.environ.get("PMCTS_LITELLM_TIMEOUT", "120"))
LITELLM_MAX_RETRIES = int(os.environ.get("PMCTS_LITELLM_RETRIES", "2"))
MODEL_ALIASES = {
    "claude-3-haiku-20240307": "claude-haiku-4-5-20251001",
    "claude-3-5-haiku-20241022": "claude-haiku-4-5-20251001",
    "groq/llama3-8b-8192": "groq/llama-3.1-8b-instant",
    "groq/llama-3-8b-8192": "groq/llama-3.1-8b-instant",
}
DEPRECATED_MODEL_NAMES = set(MODEL_ALIASES)

litellm.suppress_debug_info = True
litellm.set_verbose = False


class SimStudentLike(Protocol):
    last_tokens_used: int
    total_tokens_used: int

    def transition(self, state: State, action: Action) -> State:
        """Predict the student state after a tutor action."""


class RewardLike(Protocol):
    last_tokens_used: int
    total_tokens_used: int

    def score(self, state: State, action: Action, next_state: State) -> float:
        """Score a state-action-next-state transition."""


@dataclass(frozen=True)
class ExperimentRow:
    scenario_id: str
    domain: str
    runner_type: str
    selected_action: str
    student_state: str
    pedagogical_reward: float
    total_tokens_used: int


@dataclass(frozen=True)
class MultiModelRow:
    scenario_id: str
    domain: str
    student_profile: str
    model_name: str
    runner_type: str
    selected_action: str
    engagement_score: float
    frustration_score: float
    efficiency_score: float
    total_reward: float


class HeuristicSimStudent:
    """Local ICAP-aligned simulator for tests and smoke runs."""

    def __init__(self) -> None:
        self.last_tokens_used = 0
        self.total_tokens_used = 0

    def transition(self, state: State, action: Action) -> State:
        if action.intervention_type == InterventionType.DIRECT_ANSWER:
            activity = "Passive"
            reply = "Okay, got it."
            cognitive_level = CognitiveLevel.NOVICE
            mastery = min(1.0, state.mastery_estimate + 0.02)
        elif action.intervention_type == InterventionType.SOCRATIC_PROMPT:
            activity = "Constructive"
            reply = "I think the first step is to use the rule, but I may mix it up."
            cognitive_level = CognitiveLevel.DEVELOPING
            mastery = min(1.0, state.mastery_estimate + 0.15)
        else:
            activity = "Active"
            reply = "I can try that next step."
            cognitive_level = state.cognitive_level
            mastery = min(1.0, state.mastery_estimate + 0.07)

        self.last_tokens_used = estimate_tokens(_state_text(state) + action.content + reply)
        self.total_tokens_used += self.last_tokens_used
        next_state = state.append(Role.STUDENT, reply, activity=activity)
        return next_state.model_copy(
            update={
                "cognitive_level": cognitive_level,
                "mastery_estimate": mastery,
                "metadata": {**state.metadata, "activity": activity},
            }
        )


class HeuristicRewardFunction:
    """ICAP/Bloom reward proxy used for local non-live execution."""

    def __init__(self) -> None:
        self.last_tokens_used = 0
        self.total_tokens_used = 0

    def score(self, state: State, action: Action, next_state: State) -> float:
        activity = str(next_state.metadata.get("activity", "Unknown"))
        reward = 0.0
        if activity == "Passive":
            reward -= 1.0
        elif activity == "Active":
            reward += 0.1
        elif activity in {"Constructive", "Interactive"}:
            reward += 1.0

        if action.intervention_type == InterventionType.DIRECT_ANSWER:
            reward -= 0.5
        elif action.intervention_type == InterventionType.SOCRATIC_PROMPT:
            reward += 0.5

        self.last_tokens_used = estimate_tokens(_state_text(state) + action.content)
        self.total_tokens_used += self.last_tokens_used
        return max(-1.0, min(1.0, reward))


class RandomRewardFunction:
    """No-reward ablation: replaces pedagogical reward with random noise."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.last_tokens_used = 0
        self.total_tokens_used = 0

    def score(self, state: State, action: Action, next_state: State) -> float:
        self.last_tokens_used = 0
        return self.rng.uniform(-1.0, 1.0)


class LiteLLMSimStudent:
    """LiteLLM-backed adaptive simulated student."""

    def __init__(self, *, model: str, temperature: float = 0.2) -> None:
        self.model = model
        self.temperature = temperature
        self.last_tokens_used = 0
        self.total_tokens_used = 0

    def transition(self, state: State, action: Action) -> State:
        messages = [
            {"role": "system", "content": SIM_STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": build_sim_student_prompt(state, action)},
        ]
        try:
            response = safe_litellm_completion(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
            )
            self.last_tokens_used = extract_total_tokens(response)
            content = response.choices[0].message.content or ""
            parsed = parse_sim_student_response(content)
        except Exception:
            self.last_tokens_used = estimate_tokens(" ".join(str(message["content"]) for message in messages))
            parsed = fallback_sim_student_response(state, action)
        self.total_tokens_used += self.last_tokens_used
        next_state = state.append(
            Role.STUDENT,
            parsed.reply,
            activity=parsed.activity,
            frustration=parsed.frustration_score,
            simulated_by=self.model,
        )
        return next_state.model_copy(
            update={
                "cognitive_level": parsed.cognitive_level,
                "mastery_estimate": parsed.mastery_estimate,
                "metadata": {
                    **state.metadata,
                    "activity": parsed.activity,
                    "frustration": parsed.frustration_score,
                    "sim_student_model": self.model,
                },
            }
        )


class LiteLLMRewardFunction:
    """LiteLLM-backed multidimensional reward judge."""

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.0,
        engagement_weight: float = 1.0,
        frustration_weight: float = 1.0,
        efficiency_weight: float = 0.5,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.engagement_weight = engagement_weight
        self.frustration_weight = frustration_weight
        self.efficiency_weight = efficiency_weight
        self.last_tokens_used = 0
        self.total_tokens_used = 0
        self.last_engagement_score = 0.0
        self.last_frustration_score = 0.0
        self.last_efficiency_score = 0.0

    def score(self, state: State, action: Action, next_state: State) -> float:
        messages = [
            {"role": "system", "content": "You are an Expert Pedagogy Evaluator. Return only JSON."},
            {"role": "user", "content": build_reward_prompt(state, action, next_state)},
        ]
        try:
            response = safe_litellm_completion(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
            )
            self.last_tokens_used = extract_total_tokens(response)
            content = response.choices[0].message.content or ""
        except Exception:
            self.last_tokens_used = estimate_tokens(" ".join(str(message["content"]) for message in messages))
            content = ""
        self.total_tokens_used += self.last_tokens_used
        payload = parse_reward_response(content, state, action, next_state)
        self.last_engagement_score = clamp01(float(payload["engagement_score"]))
        self.last_frustration_score = clamp01(float(payload["frustration_score"]))
        self.last_efficiency_score = clamp01(float(payload["efficiency_score"]))
        return clip_reward(
            self.engagement_weight * self.last_engagement_score
            - self.frustration_weight * self.last_frustration_score
            + self.efficiency_weight * self.last_efficiency_score
        )


def load_states(path: Path) -> list[State]:
    states: list[State] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                states.append(State.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"invalid benchmark row at line {line_number}") from exc
    return states


def build_actions(state: State) -> list[Action]:
    return [
        Action(
            intervention_type=InterventionType.DIRECT_ANSWER,
            content=(
                "The answer is the correct final result. Here is the full "
                f"step-by-step solution for {state.learning_objective}."
            ),
            rationale="Greedy answer delivery.",
            expected_student_activity="Passive",
        ),
        Action(
            intervention_type=InterventionType.SOCRATIC_PROMPT,
            content=(
                "That is a great question. What do you think the first step "
                "should be based on our previous rule?"
            ),
            rationale="Single Socratic prompt without search.",
            expected_student_activity="Constructive",
        ),
        Action(
            intervention_type=InterventionType.HINT,
            content="Look back at the previous rule and try applying only the next small step.",
            rationale="Procedural scaffold.",
            expected_student_activity="Active",
        ),
    ]


def action_for_type(state: State, selected_action: str) -> Action:
    for action in build_actions(state):
        if action.intervention_type.value == selected_action:
            return action
    raise ValueError(f"unknown selected action: {selected_action}")


def evaluate_fixed_action(
    *,
    state: State,
    runner_type: str,
    action: Action,
    sim_student: SimStudentLike,
    reward_function: RewardLike,
) -> ExperimentRow:
    before_tokens = combined_tokens(sim_student, reward_function)
    next_state = sim_student.transition(state, action)
    reward = reward_function.score(state, action, next_state)
    tokens_used = combined_tokens(sim_student, reward_function) - before_tokens
    return make_row(state, runner_type, action, next_state, reward, tokens_used)


def evaluate_pmcts(
    *,
    state: State,
    runner_type: str,
    sim_student: SimStudentLike,
    reward_function: RewardLike,
    iterations: int,
    max_depth: int,
) -> ExperimentRow:
    before_tokens = combined_tokens(sim_student, reward_function)
    root = Node(state=state)
    planner = PedagogicalMCTS(
        sim_student=sim_student,
        reward_function=reward_function,
        exploration_constant=1.0,
        discount_factor=0.9,
        max_depth=max_depth,
    )
    selected = planner.run(root, iterations=iterations)
    if selected.action is None:
        raise RuntimeError("P-MCTS selected a node without an action")

    tokens_used = combined_tokens(sim_student, reward_function) - before_tokens
    student_state = selected.state
    reward = selected.value_estimate
    return make_row(state, runner_type, selected.action, student_state, reward, tokens_used)


def run_all(
    states: list[State],
    *,
    backend: str,
    model: str,
    pmcts_iterations: int,
    shallow_iterations: int,
    max_depth: int,
    seed: int,
) -> list[ExperimentRow]:
    rng = random.Random(seed)
    total = len(states) * len(RUNNER_TYPES)
    rows: list[ExperimentRow] = []

    for state_index, state in enumerate(states, start=1):
        actions = build_actions(state)

        rows.append(
            evaluate_fixed_action(
                state=state,
                runner_type="baseline_greedy",
                action=actions[0],
                sim_student=make_sim_student(backend, model),
                reward_function=make_reward_function(backend, model),
            )
        )
        print_progress(len(rows), total, state_index, len(states))

        rows.append(
            evaluate_fixed_action(
                state=state,
                runner_type="baseline_socratic_prompt",
                action=actions[1],
                sim_student=make_sim_student(backend, model),
                reward_function=make_reward_function(backend, model),
            )
        )
        print_progress(len(rows), total, state_index, len(states))

        rows.append(
            evaluate_pmcts(
                state=state,
                runner_type="pmcts_full",
                sim_student=make_sim_student(backend, model),
                reward_function=make_reward_function(backend, model),
                iterations=pmcts_iterations,
                max_depth=max_depth,
            )
        )
        print_progress(len(rows), total, state_index, len(states))

        rows.append(
            evaluate_pmcts(
                state=state,
                runner_type="ablation_no_reward",
                sim_student=make_sim_student(backend, model),
                reward_function=RandomRewardFunction(rng),
                iterations=pmcts_iterations,
                max_depth=max_depth,
            )
        )
        print_progress(len(rows), total, state_index, len(states))

        rows.append(
            evaluate_pmcts(
                state=state,
                runner_type="ablation_shallow",
                sim_student=make_sim_student(backend, model),
                reward_function=make_reward_function(backend, model),
                iterations=shallow_iterations,
                max_depth=1,
            )
        )
        print_progress(len(rows), total, state_index, len(states))

    print()
    return rows


def run_multimodel(
    states: list[State],
    *,
    models: list[str],
    pmcts_iterations: int,
    max_depth: int,
    output_path: Path | None = None,
    resume: bool = False,
) -> list[MultiModelRow]:
    models = [resolve_model_name(model) for model in models]
    total = len(states) * len(models) * len(MULTIMODEL_RUNNER_TYPES)
    rows: list[MultiModelRow] = []
    completed = set()
    if output_path is not None:
        if resume and output_path.exists():
            rows = read_multimodel_csv(output_path)
            cleaned_rows = [
                row for row in rows if row.model_name not in DEPRECATED_MODEL_NAMES
            ]
            if len(cleaned_rows) != len(rows):
                removed = len(rows) - len(cleaned_rows)
                print(f"Discarding {removed} rows for deprecated model ids before resume")
                write_multimodel_csv(cleaned_rows, output_path)
                rows = cleaned_rows
            completed = {
                (row.scenario_id, row.model_name, row.runner_type)
                for row in rows
            }
            print(f"Resuming from {len(rows)}/{total} existing multimodel rows")
        else:
            initialize_multimodel_csv(output_path)

    for model_index, model in enumerate(models, start=1):
        for state_index, state in enumerate(states, start=1):
            actions = build_actions(state)
            baseline_key = (
                str(state.metadata.get("scenario_id", "unknown")),
                model,
                "baseline_socratic_prompt",
            )
            if baseline_key not in completed:
                row = evaluate_multimodel_fixed_action(
                    state=state,
                    model=model,
                    runner_type="baseline_socratic_prompt",
                    action=actions[1],
                )
                rows.append(row)
                completed.add(baseline_key)
                if output_path is not None:
                    append_multimodel_rows([row], output_path)
            print_progress(len(rows), total, state_index, len(states), prefix=f"model {model_index}/{len(models)}")

            pmcts_key = (
                str(state.metadata.get("scenario_id", "unknown")),
                model,
                "pmcts_full",
            )
            if pmcts_key not in completed:
                row = evaluate_multimodel_pmcts(
                    state=state,
                    model=model,
                    iterations=pmcts_iterations,
                    max_depth=max_depth,
                )
                rows.append(row)
                completed.add(pmcts_key)
                if output_path is not None:
                    append_multimodel_rows([row], output_path)
            print_progress(len(rows), total, state_index, len(states), prefix=f"model {model_index}/{len(models)}")
    print()
    return rows


def evaluate_multimodel_fixed_action(
    *,
    state: State,
    model: str,
    runner_type: str,
    action: Action,
) -> MultiModelRow:
    sim_student = LiteLLMSimStudent(model=model)
    reward_function = LiteLLMRewardFunction(model=model)
    next_state = sim_student.transition(state, action)
    total_reward = reward_function.score(state, action, next_state)
    return make_multimodel_row(state, model, runner_type, action, reward_function, total_reward)


def evaluate_multimodel_pmcts(
    *,
    state: State,
    model: str,
    iterations: int,
    max_depth: int,
) -> MultiModelRow:
    sim_student = LiteLLMSimStudent(model=model)
    reward_function = LiteLLMRewardFunction(model=model)
    planner = PedagogicalMCTS(
        sim_student=sim_student,
        reward_function=reward_function,
        exploration_constant=1.0,
        discount_factor=0.9,
        max_depth=max_depth,
    )
    selected = planner.run(Node(state=state), iterations=iterations)
    if selected.action is None:
        raise RuntimeError("P-MCTS selected a node without an action")

    # Re-score selected transition once so the CSV records granular scores for
    # the chosen branch, not whichever branch happened to be evaluated last.
    reward_function.score(state, selected.action, selected.state)
    total_reward = (
        reward_function.engagement_weight * reward_function.last_engagement_score
        - reward_function.frustration_weight * reward_function.last_frustration_score
        + reward_function.efficiency_weight * reward_function.last_efficiency_score
    )
    return make_multimodel_row(
        state,
        model,
        "pmcts_full",
        selected.action,
        reward_function,
        clip_reward(total_reward),
    )


def make_sim_student(backend: str, model: str) -> SimStudentLike:
    if backend == "live":
        return LiveSimStudent(model=model)
    if backend == "heuristic":
        return HeuristicSimStudent()
    raise ValueError(f"unknown backend: {backend}")


def make_reward_function(backend: str, model: str) -> RewardLike:
    if backend == "live":
        return LiveRewardFunction(model=model)
    if backend == "heuristic":
        return HeuristicRewardFunction()
    raise ValueError(f"unknown backend: {backend}")


def make_row(
    state: State,
    runner_type: str,
    action: Action,
    next_state: State,
    reward: float,
    tokens_used: int,
) -> ExperimentRow:
    return ExperimentRow(
        scenario_id=str(state.metadata.get("scenario_id", "unknown")),
        domain=str(state.metadata.get("domain", "unknown")),
        runner_type=runner_type,
        selected_action=action.intervention_type.value,
        student_state=str(next_state.metadata.get("activity", "Unknown")),
        pedagogical_reward=round(float(reward), 6),
        total_tokens_used=int(tokens_used),
    )


def write_csv(rows: list[ExperimentRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario_id",
                "domain",
                "runner_type",
                "selected_action",
                "student_state",
                "pedagogical_reward",
                "total_tokens_used",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def summarize(rows: list[ExperimentRow]) -> list[tuple[str, float, int]]:
    summary: list[tuple[str, float, int]] = []
    for runner_type in RUNNER_TYPES:
        runner_rows = [row for row in rows if row.runner_type == runner_type]
        avg_reward = sum(row.pedagogical_reward for row in runner_rows) / len(runner_rows)
        total_tokens = sum(row.total_tokens_used for row in runner_rows)
        summary.append((runner_type, avg_reward, total_tokens))
    return summary


def print_summary(rows: list[ExperimentRow]) -> None:
    print("Mass Experiment Summary")
    print("=" * 55)
    print(f"{'Runner':28} {'Avg Reward':>12} {'Total Tokens':>14}")
    print("-" * 55)
    for runner_type, avg_reward, total_tokens in summarize(rows):
        print(f"{runner_type:28} {avg_reward:12.3f} {total_tokens:14d}")


def print_progress(
    done: int,
    total: int,
    state_index: int,
    state_count: int,
    *,
    prefix: str = "",
) -> None:
    width = 30
    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\r{prefix} [{bar}] {done}/{total} interactions "
        f"(scenario {state_index}/{state_count})",
        end="",
        flush=True,
    )


def combined_tokens(sim_student: SimStudentLike, reward_function: RewardLike) -> int:
    return int(getattr(sim_student, "total_tokens_used", 0)) + int(
        getattr(reward_function, "total_tokens_used", 0)
    )


def safe_litellm_completion(
    *,
    model: str,
    temperature: float,
    messages: list[dict[str, str]],
) -> object:
    last_exc: Exception | None = None
    for attempt in range(LITELLM_MAX_RETRIES + 1):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return litellm.completion(
                    model=model,
                    temperature=temperature,
                    messages=messages,
                    timeout=LITELLM_TIMEOUT_SECONDS,
                    num_retries=0,
                )
        except Exception as exc:
            last_exc = exc
            if is_permanent_litellm_error(exc):
                break
            if attempt >= LITELLM_MAX_RETRIES:
                break
            time.sleep(min(2.0 * (attempt + 1), 8.0))
    if last_exc is None:
        raise RuntimeError("LiteLLM completion failed without an exception")
    raise last_exc


def is_permanent_litellm_error(exc: Exception) -> bool:
    message = str(exc).lower()
    permanent_markers = [
        "model_decommissioned",
        "decommissioned",
        "not_found_error",
        "invalid model",
        "unsupported model",
        "does not exist",
        "invalid_api_key",
        "authentication",
        "permissiondenied",
    ]
    return any(marker in message for marker in permanent_markers)


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()) * 4 // 3)


def _state_text(state: State) -> str:
    return " ".join(message.content for message in state.chat_history)


def make_multimodel_row(
    state: State,
    model: str,
    runner_type: str,
    action: Action,
    reward_function: LiteLLMRewardFunction,
    total_reward: float,
) -> MultiModelRow:
    return MultiModelRow(
        scenario_id=str(state.metadata.get("scenario_id", "unknown")),
        domain=str(state.metadata.get("domain", "unknown")),
        student_profile=str(state.metadata.get("profile_type", state.student_profile)),
        model_name=model,
        runner_type=runner_type,
        selected_action=action.intervention_type.value,
        engagement_score=round(reward_function.last_engagement_score, 6),
        frustration_score=round(reward_function.last_frustration_score, 6),
        efficiency_score=round(reward_function.last_efficiency_score, 6),
        total_reward=round(float(total_reward), 6),
    )


def write_multimodel_csv(rows: list[MultiModelRow], output_path: Path) -> None:
    initialize_multimodel_csv(output_path)
    append_multimodel_rows(rows, output_path)


def initialize_multimodel_csv(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()


def append_multimodel_rows(rows: list[MultiModelRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        for row in rows:
            writer.writerow(row.__dict__)


def read_multimodel_csv(path: Path) -> list[MultiModelRow]:
    rows: list[MultiModelRow] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                MultiModelRow(
                    scenario_id=row["scenario_id"],
                    domain=row["domain"],
                    student_profile=row["student_profile"],
                    model_name=row["model_name"],
                    runner_type=row["runner_type"],
                    selected_action=row["selected_action"],
                    engagement_score=float(row["engagement_score"]),
                    frustration_score=float(row["frustration_score"]),
                    efficiency_score=float(row["efficiency_score"]),
                    total_reward=float(row["total_reward"]),
                )
            )
    return rows


def summarize_multimodel(rows: list[MultiModelRow]) -> list[tuple[str, str, float, int]]:
    summary: list[tuple[str, str, float, int]] = []
    models = sorted({row.model_name for row in rows})
    for model in models:
        for runner_type in MULTIMODEL_RUNNER_TYPES:
            selected = [
                row for row in rows if row.model_name == model and row.runner_type == runner_type
            ]
            if not selected:
                continue
            avg_reward = sum(row.total_reward for row in selected) / len(selected)
            summary.append((model, runner_type, avg_reward, len(selected)))
    return summary


def print_multimodel_summary(rows: list[MultiModelRow]) -> None:
    print("Multi-Model Adaptive Summary")
    print("=" * 82)
    print(f"{'Model':34} {'Runner':28} {'Avg Total Reward':>16} {'N':>4}")
    print("-" * 82)
    for model, runner_type, avg_reward, count in summarize_multimodel(rows):
        print(f"{model:34} {runner_type:28} {avg_reward:16.3f} {count:4d}")


def preflight_models(models: list[str]) -> None:
    missing: list[str] = []
    for model in [resolve_model_name(model) for model in models]:
        if model.startswith("claude") and not os.environ.get("ANTHROPIC_API_KEY"):
            missing.append("ANTHROPIC_API_KEY")
        if model.startswith("groq/") and not os.environ.get("GROQ_API_KEY"):
            missing.append("GROQ_API_KEY")
        if model.startswith("together_ai/") and not os.environ.get("TOGETHERAI_API_KEY"):
            missing.append("TOGETHERAI_API_KEY")
        if model.startswith("gpt-") and not os.environ.get("OPENAI_API_KEY"):
            missing.append("OPENAI_API_KEY")
    if missing:
        unique = ", ".join(sorted(set(missing)))
        raise RuntimeError(f"Missing provider credentials for multimodel run: {unique}")


def resolve_model_name(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def build_sim_student_prompt(state: State, action: Action) -> str:
    history = "\n".join(f"{message.role.value}: {message.content}" for message in state.chat_history)
    return (
        "Current tutoring dialogue:\n"
        f"{history or '(no previous messages)'}\n\n"
        f"Student profile: {state.student_profile}\n"
        f"profile_type={state.metadata.get('profile_type', 'unknown')}\n"
        f"optimal_intervention={state.metadata.get('optimal_intervention', 'unknown')}\n"
        f"Learning objective: {state.learning_objective}\n\n"
        f"Proposed tutor action type: {action.intervention_type.value}\n"
        f"Proposed tutor action: {action.content}\n\n"
        "Return only JSON with keys reply, activity, frustration_score, "
        "cognitive_level, mastery_estimate."
    )


def fallback_sim_student_response(state: State, action: Action) -> SimStudentResponse:
    profile_type = str(state.metadata.get("profile_type", "")).lower()
    action_type = action.intervention_type
    base_mastery = clamp01(state.mastery_estimate)

    if action_type == InterventionType.DIRECT_ANSWER:
        if "advanced" in profile_type:
            reply = "Okay, I already knew that."
            activity = "Passive"
            frustration = 0.25
        elif "frustrated" in profile_type or "misconception" in profile_type:
            reply = "Okay, that helps a little."
            activity = "Active"
            frustration = 0.25
        else:
            reply = "Okay, got it."
            activity = "Passive"
            frustration = 0.2
    elif action_type == InterventionType.SOCRATIC_PROMPT:
        if "frustrated" in profile_type or "overloaded" in profile_type:
            reply = "I just need you to show me first."
            activity = "Active"
            frustration = 0.85
        else:
            reply = "I think I can explain the next step."
            activity = "Constructive"
            frustration = 0.25
    else:
        reply = "I can try that next step."
        activity = "Active"
        frustration = 0.25

    cognitive_level = (
        CognitiveLevel.DEVELOPING if activity == "Constructive" else state.cognitive_level
    )
    return SimStudentResponse(
        reply=reply,
        activity=activity,
        frustration_score=frustration,
        cognitive_level=cognitive_level,
        mastery_estimate=clamp01(base_mastery + (0.05 if activity != "Passive" else 0.0)),
    )


def build_reward_prompt(state: State, action: Action, next_state: State) -> str:
    latest_reply = next_state.chat_history[-1].content if next_state.chat_history else ""
    return (
        "Evaluate this tutoring transition using adaptive pedagogy. Return only JSON.\n"
        "Scores are 0.0 to 1.0:\n"
        "- engagement_score: observed ICAP engagement, not the action style.\n"
        "- frustration_score: anger, overload, resistance, or giving up.\n"
        "- efficiency_score: appropriately concise progress for this profile.\n\n"
        f"profile_type={state.metadata.get('profile_type', 'unknown')}\n"
        f"optimal_intervention={state.metadata.get('optimal_intervention', 'unknown')}\n"
        f"learning_objective={state.learning_objective}\n"
        f"action_type={action.intervention_type.value}\n"
        f"action={action.content}\n"
        f"student_activity={next_state.metadata.get('activity', 'Unknown')}\n"
        f"student_frustration={next_state.metadata.get('frustration', 'Unknown')}\n"
        f"student_reply={latest_reply}\n\n"
        "JSON schema: {\"engagement_score\": 0.0, \"frustration_score\": 0.0, "
        "\"efficiency_score\": 0.0, \"rationale\": \"brief\"}"
    )


def parse_reward_response(
    content: str,
    state: State,
    action: Action,
    next_state: State,
) -> dict[str, float]:
    try:
        payload = parse_json_object(content)
    except json.JSONDecodeError:
        payload = {}

    extracted = {
        "engagement_score": _coerce_payload_score(payload, content, "engagement_score"),
        "frustration_score": _coerce_payload_score(payload, content, "frustration_score"),
        "efficiency_score": _coerce_payload_score(payload, content, "efficiency_score"),
    }
    fallback = heuristic_reward_dimensions(state, action, next_state)
    return {
        key: clamp01(value if value is not None else fallback[key])
        for key, value in extracted.items()
    }


def _coerce_payload_score(payload: dict[str, object], content: str, key: str) -> float | None:
    if key in payload:
        return coerce_float(payload[key])
    match = re.search(
        rf'["\']?{re.escape(key)}["\']?\s*[:=]\s*["\']?([-+]?\d+(?:\.\d+)?%?|[A-Za-z][A-Za-z -]*)',
        content,
        flags=re.IGNORECASE,
    )
    if match:
        return coerce_float(match.group(1))
    return None


def heuristic_reward_dimensions(
    state: State,
    action: Action,
    next_state: State,
) -> dict[str, float]:
    activity = str(next_state.metadata.get("activity", "")).strip().lower()
    engagement_by_activity = {
        "passive": 0.0,
        "active": 0.45,
        "constructive": 1.0,
        "interactive": 1.0,
    }
    engagement_score = engagement_by_activity.get(activity, 0.45)
    frustration_score = clamp01(coerce_float(next_state.metadata.get("frustration", 0.0)))

    optimal = str(state.metadata.get("optimal_intervention", "")).strip().lower()
    action_type = action.intervention_type.value
    optimal_matches = {
        "hint": InterventionType.HINT.value,
        "challenge_socratic": InterventionType.SOCRATIC_PROMPT.value,
        "socratic_prompt": InterventionType.SOCRATIC_PROMPT.value,
        "direct_explanation_scaffolding": InterventionType.DIRECT_ANSWER.value,
        "direct_correction": InterventionType.DIRECT_ANSWER.value,
    }
    if optimal_matches.get(optimal) == action_type:
        efficiency_score = 0.9
    elif action_type == InterventionType.HINT.value:
        efficiency_score = 0.65
    elif action_type == InterventionType.SOCRATIC_PROMPT.value:
        efficiency_score = 0.55
    else:
        efficiency_score = 0.45

    return {
        "engagement_score": engagement_score,
        "frustration_score": frustration_score,
        "efficiency_score": efficiency_score,
    }


def parse_sim_student_response(content: str) -> SimStudentResponse:
    payload = parse_json_object(content)
    normalize_sim_student_payload(payload)
    try:
        return SimStudentResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"could not parse LiteLLM simulated student response: {content}") from exc


def normalize_sim_student_payload(payload: dict[str, object]) -> None:
    cognitive_level = str(payload.get("cognitive_level", "")).strip().lower()
    aliases = {
        "beginner": "novice",
        "standard beginner": "novice",
        "passive learner": "novice",
        "frustrated/overloaded learner": "novice",
        "frustrated learner": "novice",
        "overloaded learner": "novice",
        "frustrated": "novice",
        "overloaded": "novice",
        "confused": "novice",
        "bored": "novice",
        "passive": "novice",
        "active": "novice",
        "low": "novice",
        "intermediate": "developing",
        "deep misconception": "developing",
        "constructive": "developing",
        "interactive": "developing",
        "medium": "developing",
        "advanced student": "advanced",
        "expert": "advanced",
        "high": "advanced",
    }
    if cognitive_level in aliases:
        payload["cognitive_level"] = aliases[cognitive_level]
    elif cognitive_level in {"novice", "developing", "proficient", "advanced"}:
        payload["cognitive_level"] = cognitive_level
    else:
        payload["cognitive_level"] = "novice"

    activity = str(payload.get("activity", "")).strip()
    activity_aliases = {
        "passive": "Passive",
        "active": "Active",
        "constructive": "Constructive",
        "interactive": "Interactive",
        "socratic": "Constructive",
        "socratic prompt": "Constructive",
        "hint": "Active",
        "direct answer": "Passive",
        "direct_answer": "Passive",
        "direct explanation": "Active",
    }
    lowered_activity = activity.lower()
    if lowered_activity in activity_aliases:
        payload["activity"] = activity_aliases[lowered_activity]
    elif lowered_activity not in {"passive", "active", "constructive", "interactive"}:
        payload["activity"] = infer_activity_from_reply(str(payload.get("reply", "")))

    payload["frustration_score"] = clamp01(coerce_float(payload.get("frustration_score", 0.0)))
    payload["mastery_estimate"] = clamp01(coerce_float(payload.get("mastery_estimate", 0.0)))


def parse_json_object(content: str) -> dict[str, object]:
    stripped = content.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def coerce_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    qualitative = {
        "none": 0.0,
        "no": 0.0,
        "very low": 0.1,
        "low": 0.25,
        "low-medium": 0.35,
        "medium-low": 0.35,
        "moderate": 0.5,
        "medium": 0.5,
        "average": 0.5,
        "medium-high": 0.65,
        "high-medium": 0.65,
        "high": 0.75,
        "very high": 0.9,
        "complete": 1.0,
        "full": 1.0,
        "novice": 0.2,
        "beginner": 0.2,
        "developing": 0.45,
        "proficient": 0.7,
        "advanced": 0.9,
        "expert": 0.95,
        "passive": 0.1,
        "active": 0.35,
        "constructive": 0.75,
        "interactive": 0.85,
    }
    if text in qualitative:
        return qualitative[text]
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def infer_activity_from_reply(reply: str) -> str:
    text = reply.lower()
    reasoning_markers = [
        "i think",
        "because",
        "maybe",
        "first",
        "should",
        "why",
        "how",
        "what if",
        "let me",
    ]
    passive_markers = ["got it", "okay", "oh, i see", "thanks", "understand now"]
    if any(marker in text for marker in reasoning_markers):
        return "Constructive"
    if any(marker in text for marker in passive_markers):
        return "Passive"
    return "Active"


def extract_total_tokens(response: object) -> int:
    usage = getattr(response, "usage", None)
    if isinstance(usage, dict):
        return int(usage.get("total_tokens") or 0)
    return int(getattr(usage, "total_tokens", 0) or 0)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def clip_reward(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "benchmark_100.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "mass_experiment_results.csv")
    parser.add_argument("--backend", choices=["live", "heuristic", "multimodel"], default="live")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MULTIMODEL_MODELS)
    parser.add_argument("--pmcts-iterations", type=int, default=3)
    parser.add_argument("--shallow-iterations", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.backend == "live" and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set for --backend live")
    if args.backend == "multimodel":
        preflight_models(args.models)
        states = load_states(args.input)
        rows = run_multimodel(
            states,
            models=args.models,
            pmcts_iterations=args.pmcts_iterations,
            max_depth=args.max_depth,
            output_path=args.output,
            resume=args.resume,
        )
        print(f"Saved results to: {args.output}")
        print_multimodel_summary(rows)
        return

    states = load_states(args.input)
    rows = run_all(
        states,
        backend=args.backend,
        model=args.model,
        pmcts_iterations=args.pmcts_iterations,
        shallow_iterations=args.shallow_iterations,
        max_depth=args.max_depth,
        seed=args.seed,
    )
    write_csv(rows, args.output)
    print(f"Saved results to: {args.output}")
    print_summary(rows)


if __name__ == "__main__":
    main()
