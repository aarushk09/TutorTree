from pmcts.sim_student import SIM_STUDENT_SYSTEM_PROMPT, LiveSimStudent
from pmcts.models import Action, InterventionType, State


def test_sim_student_prompt_defines_icap_boundaries() -> None:
    assert "PASSIVE" in SIM_STUDENT_SYSTEM_PROMPT
    assert "ACTIVE" in SIM_STUDENT_SYSTEM_PROMPT
    assert "CONSTRUCTIVE" in SIM_STUDENT_SYSTEM_PROMPT
    assert "Label MUST be: Constructive" in SIM_STUDENT_SYSTEM_PROMPT
    assert "CRITICAL BEHAVIORAL RULE" in SIM_STUDENT_SYSTEM_PROMPT
    assert "LAZY student" in SIM_STUDENT_SYSTEM_PROMPT
    assert "under 10 words" in SIM_STUDENT_SYSTEM_PROMPT
    assert "PASSIVE" in SIM_STUDENT_SYSTEM_PROMPT
    assert "ADAPTIVE PROFILE RULES" in SIM_STUDENT_SYSTEM_PROMPT
    assert "Frustrated/Overloaded Learner" in SIM_STUDENT_SYSTEM_PROMPT
    assert "frustration_score above 0.8" in SIM_STUDENT_SYSTEM_PROMPT


def test_socratic_user_prompt_reminds_model_to_label_constructive() -> None:
    sim_student = LiveSimStudent.__new__(LiveSimStudent)
    state = State(
        learning_objective="Solve a one-step equation",
        metadata={"profile_type": "Passive Learner", "optimal_intervention": "socratic_prompt"},
    )
    action = Action(
        intervention_type=InterventionType.SOCRATIC_PROMPT,
        content="What do you think the first step should be?",
    )

    prompt = sim_student._build_user_prompt(state, action)

    assert "Socratic guiding question" in prompt
    assert "Constructive" in prompt
    assert "profile_type=Passive Learner" in prompt
    assert "frustration_score" in prompt


def test_sim_student_parser_accepts_frustration_score() -> None:
    sim_student = LiveSimStudent.__new__(LiveSimStudent)

    parsed = sim_student._parse_response(
        """
        {
          "reply": "Stop asking me questions.",
          "activity": "Active",
          "frustration_score": 0.95,
          "cognitive_level": "novice",
          "mastery_estimate": 0.1
        }
        """
    )

    assert parsed.frustration_score == 0.95
