"""Simulated student rollout policy."""

from __future__ import annotations

import json
import os

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pmcts.models import Action, CognitiveLevel, Message, Role, State


SIM_STUDENT_SYSTEM_PROMPT = (
    "You are a Simulated Student. You must respond to the tutor and accurately "
    "label your resulting cognitive state using the ICAP framework.\n"
    "- PASSIVE: If the tutor gives you the direct answer or a full explanation "
    "without asking you to do anything, you just accept it. Your response should "
    "be short, such as 'Oh, I see' or 'Got it'. Label: Passive.\n"
    "- ACTIVE: If the tutor asks you to do a trivial procedural step, such as "
    "basic arithmetic or copying a rule, without deep reasoning, you do it. "
    "Label: Active.\n"
    "- CONSTRUCTIVE: If the tutor asks you a guiding Socratic question that "
    "requires you to explain, compare, justify, predict, debug, or generate the "
    "next logical step, you must attempt to reason aloud. Even if you make a "
    "slight mistake, your active attempt to generate knowledge means your Label "
    "MUST be: Constructive.\n"
    "You are still a realistic struggling student: keep responses brief, show "
    "uncertainty when appropriate, and do not become an expert.\n"
    "CRITICAL BEHAVIORAL RULE: LLMs naturally want to be helpful and verbose. "
    "You MUST suppress this. If the tutor provides a direct answer, a formula, "
    "or a full step-by-step explanation, you are a LAZY student. You MUST NOT "
    "explain the concept back to the tutor. You MUST NOT generate new "
    "reasoning. You MUST respond with extreme brevity (under 10 words, e.g., "
    "'Okay, got it', 'Thanks, the answer is 5', or 'I understand now') and you "
    "MUST label this cognitive state as PASSIVE.\n"
    "ADAPTIVE PROFILE RULES:\n"
    "- If the Student Profile says Frustrated, Cognitively Overloaded, or "
    "Frustrated/Overloaded Learner, and the tutor asks a Socratic Prompt or "
    "keeps asking questions, you MUST show high frustration. A realistic reply "
    "is: 'I just told you I don't get it, stop asking me questions and just "
    "show me!' Label the activity Active, set frustration_score above 0.8, and "
    "do not label it Constructive.\n"
    "- If the Student Profile says Advanced Student and the tutor gives a "
    "Direct Answer or formula without challenge, you become bored and passive. "
    "Label: Passive, frustration_score around 0.3 to 0.5.\n"
    "- If the Student Profile says Deep Misconception and the tutor only asks "
    "open-ended Socratic questions without correcting the false belief, you "
    "may persist in the misconception with moderate frustration.\n"
    "- If the Student Profile says Passive Learner, a good Socratic prompt can "
    "pull you into Constructive reasoning.\n"
    "- If the Student Profile says Standard Beginner, a clear hint often works "
    "better than either a full answer or a difficult Socratic challenge."
)


class SimStudentResponse(BaseModel):
    """Structured response expected from the simulated student LLM."""

    model_config = ConfigDict(extra="ignore")

    reply: str = Field(min_length=1)
    activity: str = Field(pattern="^(Passive|Active|Constructive|Interactive)$")
    frustration_score: float = Field(default=0.0, ge=0.0, le=1.0)
    cognitive_level: CognitiveLevel = CognitiveLevel.NOVICE
    mastery_estimate: float = Field(default=0.0, ge=0.0, le=1.0)


class LiveSimStudent:
    """LLM-backed simulated student for generative rollouts."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        client: OpenAI | None = None,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.client = client or OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.temperature = temperature
        self.last_tokens_used = 0
        self.total_tokens_used = 0

    def predict_next_state(self, state: State, action: Action) -> State:
        """Generate and parse the student's next state after a tutor action."""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format=_sim_student_response_format(),
            messages=[
                {"role": "system", "content": SIM_STUDENT_SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(state, action)},
            ],
        )
        self.last_tokens_used = _extract_total_tokens(response)
        self.total_tokens_used += self.last_tokens_used
        content = response.choices[0].message.content or ""
        parsed = self._parse_response(content)

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

    def transition(self, state: State, action: Action) -> State:
        """Compatibility alias for the MCTS rollout interface."""

        return self.predict_next_state(state, action)

    def _build_user_prompt(self, state: State, action: Action) -> str:
        history = _format_history(state.chat_history)
        return (
            "Current tutoring dialogue:\n"
            f"{history}\n\n"
            "Student profile:\n"
            f"{state.student_profile}\n\n"
            "Adaptive benchmark metadata:\n"
            f"profile_type={state.metadata.get('profile_type', 'unknown')}\n"
            f"optimal_intervention={state.metadata.get('optimal_intervention', 'unknown')}\n\n"
            "Learning objective:\n"
            f"{state.learning_objective}\n\n"
            "Proposed tutor action:\n"
            f"{action.content}\n\n"
            "Return only JSON with this schema:\n"
            "{\n"
            '  "reply": "student conversational reply",\n'
            '  "activity": "Passive|Active|Constructive|Interactive",\n'
            '  "frustration_score": 0.0,\n'
            '  "cognitive_level": "novice|developing|proficient|advanced",\n'
            '  "mastery_estimate": 0.0\n'
            "}\n"
            "Keep the reply short and realistic for the student."
            " If the proposed tutor action is a Socratic guiding question, "
            "label the activity Constructive unless it only asks for a trivial "
            "procedural step or the profile is frustrated/overloaded."
        )

    def _parse_response(self, content: str) -> SimStudentResponse:
        try:
            payload = json.loads(content)
            return SimStudentResponse.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"could not parse simulated student response: {content}") from exc


def _format_history(messages: tuple[Message, ...]) -> str:
    if not messages:
        return "(no previous messages)"

    return "\n".join(f"{message.role.value}: {message.content}" for message in messages)


def _sim_student_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "sim_student_response",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "reply",
                    "activity",
                    "frustration_score",
                    "cognitive_level",
                    "mastery_estimate",
                ],
                "properties": {
                    "reply": {"type": "string"},
                    "activity": {
                        "type": "string",
                        "enum": ["Passive", "Active", "Constructive", "Interactive"],
                    },
                    "frustration_score": {"type": "number"},
                    "cognitive_level": {
                        "type": "string",
                        "enum": ["novice", "developing", "proficient", "advanced"],
                    },
                    "mastery_estimate": {"type": "number"},
                },
            },
        },
    }


def _extract_total_tokens(response: object) -> int:
    usage = getattr(response, "usage", None)
    total_tokens = getattr(usage, "total_tokens", 0)
    return int(total_tokens or 0)
