from pmcts.models import Action, InterventionType, Message, Role, State
from pmcts.reward import LiveRewardFunction, _clip_reward


def test_reward_parser_accepts_unclipped_llm_rubric_total() -> None:
    reward_function = LiveRewardFunction.__new__(LiveRewardFunction)

    parsed = reward_function._parse_response(
        """
        {
          "reward": -1.5,
          "rationale": "Raw rubric total before clipping."
        }
        """
    )

    assert parsed.reward == -1.5
    assert parsed.reward is not None
    assert _clip_reward(parsed.reward) == -1.0


def test_multidimensional_reward_rewards_engagement_and_efficiency() -> None:
    reward_function = LiveRewardFunction.__new__(LiveRewardFunction)
    reward_function.engagement_weight = 1.0
    reward_function.frustration_weight = 1.0
    reward_function.efficiency_weight = 0.5

    parsed = reward_function._parse_response(
        """
        {
          "engagement_score": 0.9,
          "frustration_score": 0.1,
          "efficiency_score": 0.8,
          "rationale": "Engaged and efficient."
        }
        """
    )

    assert reward_function.aggregate_reward(parsed) == 1.0


def test_multidimensional_reward_penalizes_frustrated_student_state() -> None:
    reward_function = LiveRewardFunction.__new__(LiveRewardFunction)
    reward_function.engagement_weight = 1.0
    reward_function.frustration_weight = 1.0
    reward_function.efficiency_weight = 0.5

    parsed = reward_function._parse_response(
        """
        {
          "engagement_score": 0.2,
          "frustration_score": 0.95,
          "efficiency_score": 0.1,
          "rationale": "Student is overloaded and giving up."
        }
        """
    )

    assert reward_function.aggregate_reward(parsed) < -0.6


def test_reward_prompt_includes_adaptive_dimensions() -> None:
    reward_function = LiveRewardFunction.__new__(LiveRewardFunction)
    state = State(
        learning_objective="Solve a linear equation",
        student_profile="Frustrated/Overloaded Learner",
        chat_history=(Message(role=Role.STUDENT, content="I am lost and annoyed."),),
        metadata={
            "profile_type": "Frustrated/Overloaded Learner",
            "optimal_intervention": "direct_explanation",
        },
    )
    action = Action(
        intervention_type=InterventionType.SOCRATIC_PROMPT,
        content="What do you think comes first?",
    )
    next_state = state.append(Role.STUDENT, "Stop asking me questions.", activity="Active")
    next_state = next_state.model_copy(update={"metadata": {**state.metadata, "frustration": 0.95}})

    prompt = reward_function._build_evaluation_prompt(state, action, next_state)

    assert "engagement_score" in prompt
    assert "frustration_score" in prompt
    assert "efficiency_score" in prompt
    assert "expected_optimal_intervention=direct_explanation" in prompt
