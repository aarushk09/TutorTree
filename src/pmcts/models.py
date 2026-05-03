"""Foundational data models for the P-MCTS pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class Role(StrEnum):
    """Speaker roles in the tutoring dialogue."""

    SYSTEM = "system"
    TEACHER = "teacher"
    STUDENT = "student"
    JUDGE = "judge"


class CognitiveLevel(StrEnum):
    """Coarse student cognitive states used by the pedagogical environment."""

    NOVICE = "novice"
    DEVELOPING = "developing"
    PROFICIENT = "proficient"
    ADVANCED = "advanced"


class InterventionType(StrEnum):
    """Pedagogical action families available to the planner."""

    DIRECT_ANSWER = "direct_answer"
    SOCRATIC_PROMPT = "socratic_prompt"
    HINT = "hint"
    EXAMPLE = "example"
    ANALOGY = "analogy"
    ERROR_DIAGNOSIS = "error_diagnosis"
    METACOGNITIVE_PROMPT = "metacognitive_prompt"


class Message(BaseModel):
    """One message in the multi-turn tutoring history."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Role
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message content cannot be empty")
        return stripped


class State(BaseModel):
    """Environment state containing dialogue context and student estimates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chat_history: tuple[Message, ...] = Field(default_factory=tuple)
    cognitive_level: CognitiveLevel = CognitiveLevel.NOVICE
    learning_objective: str = Field(min_length=1)
    student_profile: str = Field(
        default="struggling 9th-grade algebra student with high cognitive load",
        min_length=1,
    )
    mastery_estimate: float = Field(default=0.0, ge=0.0, le=1.0)
    turn_index: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("learning_objective", "student_profile")
    @classmethod
    def strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text fields cannot be empty")
        return stripped

    def append(self, role: Role, content: str, **metadata: Any) -> State:
        """Return a new state with one additional dialogue message."""

        message = Message(role=role, content=content, metadata=metadata)
        return self.model_copy(
            update={
                "chat_history": (*self.chat_history, message),
                "turn_index": self.turn_index + 1,
            }
        )


class Action(BaseModel):
    """A pedagogical intervention candidate considered by P-MCTS."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intervention_type: InterventionType
    content: str = Field(min_length=1)
    rationale: str | None = None
    expected_student_activity: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content", "rationale", "expected_student_activity")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        stripped = value.strip()
        if not stripped:
            raise ValueError("text fields cannot be empty")
        return stripped


class Node(BaseModel):
    """Search-tree node for MCTS bookkeeping.

    The node is intentionally mutable because visit counts and value estimates
    are updated during tree search.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    state: State
    action: Action | None = None
    node_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_id: str | None = None
    children: list[Node] = Field(default_factory=list)
    visits: int = Field(default=0, ge=0)
    value_sum: float = 0.0
    prior_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    depth: int = Field(default=0, ge=0)
    terminal: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def value_estimate(self) -> float:
        """Mean backed-up value, defaulting to zero before any visits."""

        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


Node.model_rebuild()
