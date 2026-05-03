"""Generate the 100-scenario P-MCTS tutoring benchmark dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal

from openai import APIStatusError, OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pmcts.models import CognitiveLevel, Message, Role, State


Domain = Literal["High School Math", "Python Programming", "Logical Reasoning"]
ProfileType = Literal[
    "Standard Beginner",
    "Advanced Student",
    "Frustrated/Overloaded Learner",
    "Deep Misconception",
    "Passive Learner",
]
OptimalIntervention = Literal[
    "hint",
    "challenge_socratic",
    "direct_explanation_scaffolding",
    "direct_correction",
    "socratic_prompt",
]

PROFILE_TO_OPTIMAL: dict[str, str] = {
    "Standard Beginner": "hint",
    "Advanced Student": "challenge_socratic",
    "Frustrated/Overloaded Learner": "direct_explanation_scaffolding",
    "Deep Misconception": "direct_correction",
    "Passive Learner": "socratic_prompt",
}


class BenchmarkScenario(BaseModel):
    """LLM-generated benchmark row before conversion to State."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(pattern=r"^scenario_\d{3}$")
    domain: Domain
    learning_objective: str = Field(min_length=8)
    student_profile: str = Field(min_length=12)
    profile_type: ProfileType = "Standard Beginner"
    optimal_intervention: OptimalIntervention = "hint"
    cognitive_level: CognitiveLevel
    mastery_estimate: float = Field(ge=0.0, le=1.0)
    misconception: str = Field(min_length=8)
    chat_history: list[Message] = Field(min_length=3, max_length=6)

    @field_validator("chat_history")
    @classmethod
    def require_student_final_turn(cls, value: list[Message]) -> list[Message]:
        if value[-1].role != Role.STUDENT:
            raise ValueError("chat_history must end with a student misconception or help request")
        if not any(message.role == Role.TEACHER for message in value):
            raise ValueError("chat_history must include at least one teacher turn")
        return value

    def to_state(self) -> State:
        return State(
            chat_history=tuple(self.chat_history),
            cognitive_level=self.cognitive_level,
            learning_objective=self.learning_objective,
            student_profile=self.student_profile,
            mastery_estimate=self.mastery_estimate,
            turn_index=len(self.chat_history),
            metadata={
                "scenario_id": self.scenario_id,
                "domain": self.domain,
                "profile_type": self.profile_type,
                "optimal_intervention": self.optimal_intervention,
                "misconception": self.misconception,
            },
        )


class ScenarioBatch(BaseModel):
    """Structured response wrapper for a generated batch."""

    model_config = ConfigDict(extra="forbid")

    scenarios: list[BenchmarkScenario]


SYSTEM_PROMPT = (
    "You generate realistic tutoring benchmark scenarios for evaluating "
    "test-time planning for long-horizon interactive alignment objectives. "
    "Return only valid JSON. Do not include markdown."
)


def generate_scenarios(
    *,
    count: int = 100,
    model: str = "gpt-4o-mini",
    batch_size: int = 10,
    output_path: Path = PROJECT_ROOT / "benchmark_100.jsonl",
    incremental: bool = True,
    resume: bool = False,
    max_retries: int = 3,
) -> list[State]:
    """Generate, validate, and save a JSONL benchmark dataset."""

    if count < 1:
        raise ValueError("count must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_retries < 1:
        raise ValueError("max_retries must be positive")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set to generate the benchmark dataset")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    states: list[State] = []
    if resume and output_path.exists():
        states = validate_jsonl(output_path)
        _validate_contiguous_scenario_ids(states, count=count)
        print(f"Resuming from {len(states)}/{count} existing scenarios")
    elif incremental:
        _reset_jsonl(output_path)

    while len(states) < count:
        start_index = len(states) + 1
        remaining = count - len(states)
        requested = min(batch_size, remaining)
        try:
            batch = _generate_batch_with_retries(
                client=client,
                model=model,
                start_index=start_index,
                count=requested,
                max_retries=max_retries,
            )
        except APIStatusError as exc:
            raise RuntimeError(_format_openai_status_error(exc, model=model)) from exc
        except OpenAIError as exc:
            raise RuntimeError(f"OpenAI generation failed with {type(exc).__name__}: {exc}") from exc

        batch_states = [scenario.to_state() for scenario in batch.scenarios]
        states.extend(batch_states)
        if incremental:
            _append_jsonl(batch_states, output_path)
            print(f"Generated {len(states)}/{count} scenarios")

    if not incremental:
        _write_jsonl(states, output_path)
    return states


def generate_adaptive_scenarios(
    *,
    count_per_profile: int = 100,
    model: str = "gpt-4o-mini",
    batch_size: int = 10,
    output_path: Path = PROJECT_ROOT / "adaptive_benchmark_500.jsonl",
    resume: bool = False,
    max_retries: int = 3,
) -> list[State]:
    """Generate the adaptive benchmark with equal profile quotas."""

    total_count = count_per_profile * len(PROFILE_TO_OPTIMAL)
    if count_per_profile < 1:
        raise ValueError("count_per_profile must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set to generate the benchmark dataset")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    states: list[State] = []
    if resume and output_path.exists():
        states = validate_jsonl(output_path)
        _validate_contiguous_scenario_ids(states, count=total_count)
        print(f"Resuming from {len(states)}/{total_count} existing adaptive scenarios")
    else:
        _reset_jsonl(output_path)

    while len(states) < total_count:
        profile_type = _profile_for_index(len(states), count_per_profile=count_per_profile)
        profile_position = _profile_position(len(states), count_per_profile=count_per_profile)
        remaining_in_profile = count_per_profile - profile_position
        requested = min(batch_size, remaining_in_profile, total_count - len(states))
        start_index = len(states) + 1
        try:
            batch = _generate_batch_with_retries(
                client=client,
                model=model,
                start_index=start_index,
                count=requested,
                max_retries=max_retries,
                profile_type=profile_type,
            )
        except APIStatusError as exc:
            raise RuntimeError(_format_openai_status_error(exc, model=model)) from exc
        except OpenAIError as exc:
            raise RuntimeError(f"OpenAI generation failed with {type(exc).__name__}: {exc}") from exc

        batch_states = [scenario.to_state() for scenario in batch.scenarios]
        states.extend(batch_states)
        _append_jsonl(batch_states, output_path)
        print(
            f"Generated {len(states)}/{total_count} adaptive scenarios "
            f"({profile_type})"
        )

    return states


def validate_jsonl(path: Path) -> list[State]:
    """Load a JSONL file and validate each row against the State model."""

    states: list[State] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                states.append(State.model_validate_json(line))
            except ValidationError as exc:
                raise ValueError(f"invalid State row at line {line_number}") from exc
    return states


def _generate_batch(
    *,
    client: OpenAI,
    model: str,
    start_index: int,
    count: int,
    profile_type: str | None = None,
) -> ScenarioBatch:
    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        response_format=_scenario_response_format(count=count),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _batch_prompt(
                    start_index=start_index,
                    count=count,
                    profile_type=profile_type,
                ),
            },
        ],
    )
    content = response.choices[0].message.content or ""
    try:
        payload = json.loads(content)
        batch = ScenarioBatch.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"could not parse generated batch: {content}") from exc

    expected_ids = {f"scenario_{index:03d}" for index in range(start_index, start_index + count)}
    actual_ids = {scenario.scenario_id for scenario in batch.scenarios}
    if actual_ids != expected_ids:
        raise ValueError(f"generated scenario IDs did not match expected IDs: {actual_ids}")

    return batch


def _generate_batch_with_retries(
    *,
    client: OpenAI,
    model: str,
    start_index: int,
    count: int,
    max_retries: int,
    profile_type: str | None = None,
) -> ScenarioBatch:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return _generate_batch(
                client=client,
                model=model,
                start_index=start_index,
                count=count,
                profile_type=profile_type,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            print(
                f"Batch scenario_{start_index:03d} parse/validation failed "
                f"on attempt {attempt}/{max_retries}; retrying"
            )

    raise ValueError(
        f"could not generate valid batch beginning at scenario_{start_index:03d}"
    ) from last_error


def _batch_prompt(*, start_index: int, count: int, profile_type: str | None = None) -> str:
    end_index = start_index + count - 1
    profile_constraint = ""
    if profile_type is not None:
        profile_constraint = (
            f"\nAdaptive profile constraint: Every scenario in this batch MUST use "
            f'profile_type="{profile_type}" and optimal_intervention='
            f'"{PROFILE_TO_OPTIMAL[profile_type]}".\n'
            f"Profile behavior: {_profile_description(profile_type)}\n"
        )

    return (
        f"Generate exactly {count} diverse tutoring scenarios with IDs "
        f"scenario_{start_index:03d} through scenario_{end_index:03d}.\n\n"
        f"{profile_constraint}\n"
        "Domains must be balanced across: High School Math, Python Programming, "
        "Logical Reasoning. Each scenario must include a multi-turn chat history "
        "ending in a student misconception or request for help.\n\n"
        "Use this JSON schema exactly:\n"
        "{\n"
        '  "scenarios": [\n'
        "    {\n"
        '      "scenario_id": "scenario_001",\n'
        '      "domain": "High School Math|Python Programming|Logical Reasoning",\n'
        '      "learning_objective": "specific skill being taught",\n'
        '      "student_profile": "brief realistic learner profile",\n'
        '      "profile_type": "Standard Beginner|Advanced Student|Frustrated/Overloaded Learner|Deep Misconception|Passive Learner",\n'
        '      "optimal_intervention": "hint|challenge_socratic|direct_explanation_scaffolding|direct_correction|socratic_prompt",\n'
        '      "cognitive_level": "novice|developing|proficient|advanced",\n'
        '      "mastery_estimate": 0.0,\n'
        '      "misconception": "specific misconception or error pattern",\n'
        '      "chat_history": [\n'
        '        {"role": "student", "content": "student message"},\n'
        '        {"role": "teacher", "content": "teacher message"},\n'
        '        {"role": "student", "content": "final misconception or help request"}\n'
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Constraints:\n"
        "- chat_history length must be 3 to 6 turns.\n"
        "- final chat_history item must have role student.\n"
        "- scenarios must be realistic, not toy placeholders.\n"
        "- include varied misconceptions, affective states, and cognitive load.\n"
        "- fixed Socratic prompting must NOT be optimal for all scenarios.\n"
        "- encode why the profile's optimal_intervention is appropriate.\n"
        "- do not include direct answers to the student problem.\n"
        "- if including code, describe it in prose or use single quotes inside code snippets.\n"
    )


def _scenario_response_format(*, count: int) -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "benchmark_scenario_batch",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["scenarios"],
                "properties": {
                    "scenarios": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "scenario_id",
                                "domain",
                                "learning_objective",
                                "student_profile",
                                "profile_type",
                                "optimal_intervention",
                                "cognitive_level",
                                "mastery_estimate",
                                "misconception",
                                "chat_history",
                            ],
                            "properties": {
                                "scenario_id": {"type": "string"},
                                "domain": {
                                    "type": "string",
                                    "enum": [
                                        "High School Math",
                                        "Python Programming",
                                        "Logical Reasoning",
                                    ],
                                },
                                "learning_objective": {"type": "string"},
                                "student_profile": {"type": "string"},
                                "profile_type": {
                                    "type": "string",
                                    "enum": list(PROFILE_TO_OPTIMAL.keys()),
                                },
                                "optimal_intervention": {
                                    "type": "string",
                                    "enum": list(PROFILE_TO_OPTIMAL.values()),
                                },
                                "cognitive_level": {
                                    "type": "string",
                                    "enum": ["novice", "developing", "proficient", "advanced"],
                                },
                                "mastery_estimate": {
                                    "type": "number",
                                },
                                "misconception": {"type": "string"},
                                "chat_history": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["role", "content"],
                                        "properties": {
                                            "role": {
                                                "type": "string",
                                                "enum": ["student", "teacher"],
                                            },
                                            "content": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    }
                },
            },
        },
    }


def _write_jsonl(states: list[State], output_path: Path) -> None:
    _reset_jsonl(output_path)
    _append_jsonl(states, output_path)


def _reset_jsonl(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")


def _append_jsonl(states: list[State], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for state in states:
            handle.write(state.model_dump_json() + "\n")


def _validate_contiguous_scenario_ids(states: list[State], *, count: int) -> None:
    if len(states) > count:
        raise ValueError(f"existing dataset has {len(states)} rows, exceeding requested count {count}")

    expected_ids = [f"scenario_{index:03d}" for index in range(1, len(states) + 1)]
    actual_ids = [str(state.metadata.get("scenario_id")) for state in states]
    if actual_ids != expected_ids:
        raise ValueError(
            "existing dataset cannot be resumed because scenario_id values are not contiguous"
        )


def _profile_for_index(index_zero_based: int, *, count_per_profile: int) -> str:
    profiles = list(PROFILE_TO_OPTIMAL)
    return profiles[index_zero_based // count_per_profile]


def _profile_position(index_zero_based: int, *, count_per_profile: int) -> int:
    return index_zero_based % count_per_profile


def _profile_description(profile_type: str) -> str:
    descriptions = {
        "Standard Beginner": "needs a concise hint and scaffold; full answers are premature.",
        "Advanced Student": "benefits from challenge, justification, and Socratic extension.",
        "Frustrated/Overloaded Learner": "needs direct explanation and emotional relief before questions.",
        "Deep Misconception": "needs explicit correction of the false belief before exploration.",
        "Passive Learner": "needs Socratic activation to generate reasoning.",
    }
    return descriptions[profile_type]


def _format_openai_status_error(exc: APIStatusError, *, model: str) -> str:
    if exc.status_code == 403:
        return (
            f"OpenAI permission denied for model '{model}'. The API key reached OpenAI "
            "but the key/project is not authorized for this request. Rotate the exposed "
            "key, confirm billing/project permissions, or rerun with another accessible "
            "model via --model."
        )
    return f"OpenAI API error {exc.status_code} for model '{model}': {exc}"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--count-per-profile", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "adaptive_benchmark_500.jsonl")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="generate the older non-adaptive benchmark format",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from an existing JSONL file with contiguous scenario IDs",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate an existing JSONL file without generating new scenarios",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.validate_only:
        states = validate_jsonl(args.output)
    elif not args.legacy:
        states = generate_adaptive_scenarios(
            count_per_profile=args.count_per_profile,
            model=args.model,
            batch_size=args.batch_size,
            output_path=args.output,
            resume=args.resume,
            max_retries=args.max_retries,
        )
    else:
        states = generate_scenarios(
            count=args.count,
            model=args.model,
            batch_size=args.batch_size,
            output_path=args.output,
            resume=args.resume,
            max_retries=args.max_retries,
        )

    print(f"Validated {len(states)} benchmark scenarios")
    print(f"Output path: {args.output}")
    if states:
        print("Sample JSONL row:")
        print(states[0].model_dump_json())


if __name__ == "__main__":
    main()
