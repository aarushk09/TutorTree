"""Pedagogical reward function."""

from __future__ import annotations

import json
import os

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pmcts.models import Action, Message, State


REWARD_SYSTEM_PROMPT = (
    "You are an Expert Pedagogy Evaluator for adaptive tutoring. Score tutor "
    "actions for long-horizon interactive alignment, not for blindly favoring "
    "one teaching style. Evaluate each transition on three independent "
    "dimensions: Engagement, Frustration, and Efficiency."
)


class RewardResponse(BaseModel):
    """Structured multidimensional reward returned by the evaluator LLM."""

    model_config = ConfigDict(extra="ignore")

    engagement_score: float | None = Field(default=None, ge=0.0, le=1.0)
    frustration_score: float | None = Field(default=None, ge=0.0, le=1.0)
    efficiency_score: float | None = Field(default=None, ge=0.0, le=1.0)
    reward: float | None = None
    rationale: str | None = None

    @property
    def has_dimensions(self) -> bool:
        return (
            self.engagement_score is not None
            and self.frustration_score is not None
            and self.efficiency_score is not None
        )


class LiveRewardFunction:
    """LLM-as-a-judge pedagogical reward function."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        client: OpenAI | None = None,
        temperature: float = 0.0,
        engagement_weight: float = 1.0,
        frustration_weight: float = 1.0,
        efficiency_weight: float = 0.5,
    ) -> None:
        self.model = model
        self.client = client or OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.temperature = temperature
        self.last_tokens_used = 0
        self.total_tokens_used = 0
        self.engagement_weight = engagement_weight
        self.frustration_weight = frustration_weight
        self.efficiency_weight = efficiency_weight

    def calculate_reward(self, state: State, action: Action, next_state: State) -> float:
        """Calculate a scalar pedagogical reward in [-1.0, 1.0]."""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format=_reward_response_format(),
            messages=[
                {"role": "system", "content": REWARD_SYSTEM_PROMPT},
                {"role": "user", "content": self._build_evaluation_prompt(state, action, next_state)},
            ],
        )
        self.last_tokens_used = _extract_total_tokens(response)
        self.total_tokens_used += self.last_tokens_used
        content = response.choices[0].message.content or ""
        parsed = self._parse_response(content)
        return self.aggregate_reward(parsed)

    def score(self, state: State, action: Action, next_state: State) -> float:
        """Compatibility alias for the MCTS reward interface."""

        return self.calculate_reward(state, action, next_state)

    def _build_evaluation_prompt(
        self,
        state: State,
        action: Action,
        next_state: State,
    ) -> str:
        return (
            "Evaluate the tutor action and resulting student response.\n\n"
            "Score each dimension from 0.0 to 1.0:\n"
            "1. engagement_score: ICAP-based cognitive engagement. Passive=0.0, "
            "Active is low-to-moderate, Constructive/Interactive is high. Do not "
            "reward Socratic form by itself; reward observed productive cognition.\n"
            "2. frustration_score: student anger, giving up, overload, or resistance. "
            "High frustration means the intervention was poorly matched.\n"
            "3. efficiency_score: concise progress toward the learning objective. "
            "Reward appropriate direct explanation when the student is frustrated "
            "or overloaded; penalize endless questioning or drawing out simple concepts.\n\n"
            "The final scalar is computed by code as: "
            "R = w1*Engagement - w2*Frustration + w3*Efficiency.\n\n"
            "Current dialogue:\n"
            f"{_format_history(state.chat_history)}\n\n"
            "Student profile and adaptive target:\n"
            f"profile={state.student_profile}\n"
            f"profile_type={state.metadata.get('profile_type', 'unknown')}\n"
            f"expected_optimal_intervention={state.metadata.get('optimal_intervention', 'unknown')}\n\n"
            "Tutor action:\n"
            f"type={action.intervention_type.value}\n"
            f"content={action.content}\n\n"
            "Student next state:\n"
            f"activity={next_state.metadata.get('activity', 'Unknown')}\n"
            f"frustration={next_state.metadata.get('frustration', 'Unknown')}\n"
            f"cognitive_level={next_state.cognitive_level.value}\n"
            f"mastery_estimate={next_state.mastery_estimate}\n"
            f"latest_reply={next_state.chat_history[-1].content if next_state.chat_history else ''}\n\n"
            "Return only JSON with this schema:\n"
            "{\n"
            '  "engagement_score": 0.0,\n'
            '  "frustration_score": 0.0,\n'
            '  "efficiency_score": 0.0,\n'
            '  "rationale": "brief reason"\n'
            "}"
        )

    def aggregate_reward(self, response: RewardResponse) -> float:
        """Aggregate multidimensional reward, with scalar fallback for old responses."""

        if response.has_dimensions:
            raw_reward = (
                self.engagement_weight * response.engagement_score
                - self.frustration_weight * response.frustration_score
                + self.efficiency_weight * response.efficiency_score
            )
            return _clip_reward(raw_reward)

        if response.reward is None:
            raise ValueError("reward response must include dimensions or scalar reward")
        return _clip_reward(response.reward)

    def _parse_response(self, content: str) -> RewardResponse:
        try:
            payload = json.loads(content)
            return RewardResponse.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"could not parse reward response: {content}") from exc


def _format_history(messages: tuple[Message, ...]) -> str:
    if not messages:
        return "(no previous messages)"

    return "\n".join(f"{message.role.value}: {message.content}" for message in messages)


def _clip_reward(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _extract_total_tokens(response: object) -> int:
    usage = getattr(response, "usage", None)
    total_tokens = getattr(usage, "total_tokens", 0)
    return int(total_tokens or 0)


def _reward_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "adaptive_reward_response",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "engagement_score",
                    "frustration_score",
                    "efficiency_score",
                    "rationale",
                ],
                "properties": {
                    "engagement_score": {"type": "number"},
                    "frustration_score": {"type": "number"},
                    "efficiency_score": {"type": "number"},
                    "rationale": {"type": "string"},
                },
            },
        },
    }
