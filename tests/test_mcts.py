from pmcts.mcts import PedagogicalMCTS
from pmcts.models import Action, CognitiveLevel, InterventionType, Message, Node, Role, State


class MockSimStudent:
    """Deterministic student model for MCTS tests."""

    def transition(self, state: State, action: Action) -> State:
        if action.intervention_type == InterventionType.DIRECT_ANSWER:
            activity = "Passive"
            mastery = 0.05
            level = CognitiveLevel.NOVICE
        elif action.intervention_type == InterventionType.SOCRATIC_PROMPT:
            activity = "Constructive"
            mastery = 0.75
            level = CognitiveLevel.DEVELOPING
        else:
            activity = "Active"
            mastery = 0.4
            level = CognitiveLevel.NOVICE

        next_state = state.append(
            Role.STUDENT,
            f"{activity} response to: {action.content}",
            activity=activity,
        )
        return next_state.model_copy(
            update={
                "cognitive_level": level,
                "mastery_estimate": mastery,
                "metadata": {**state.metadata, "activity": activity},
            }
        )


class MockReward:
    """Deterministic reward model based on ICAP-style activity labels."""

    def score(self, state: State, action: Action, next_state: State) -> float:
        activity = next_state.metadata["activity"]
        if activity == "Passive":
            return -1.0
        if activity == "Constructive":
            return 1.0
        return 0.25


def test_mcts_selects_socratic_question_with_mock_rollouts() -> None:
    root_state = State(
        learning_objective="Debug a crashing program",
        chat_history=(
            Message(role=Role.STUDENT, content="Why is my code crashing?"),
        ),
    )
    root = Node(state=root_state)
    mcts = PedagogicalMCTS(
        sim_student=MockSimStudent(),
        reward_function=MockReward(),
        exploration_constant=1.0,
        discount_factor=0.9,
        max_depth=2,
    )

    best = mcts.run(root, iterations=50)

    assert len(root.children) == 3
    assert root.visits == 50
    assert best.action is not None
    assert best.action.intervention_type == InterventionType.SOCRATIC_PROMPT
    assert best.value_estimate > 0
    assert best.visits == max(child.visits for child in root.children)


def test_mcts_scores_unvisited_action_node_before_expanding_below_it() -> None:
    root_state = State(learning_objective="Debug a crashing program")
    root = Node(state=root_state)
    mcts = PedagogicalMCTS(
        sim_student=MockSimStudent(),
        reward_function=MockReward(),
        exploration_constant=1.0,
        discount_factor=0.9,
        max_depth=3,
    )

    best = mcts.run(root, iterations=3)

    assert best.action is not None
    assert best.action.intervention_type == InterventionType.SOCRATIC_PROMPT
    root_scores = {
        child.action.intervention_type: child.value_estimate
        for child in root.children
        if child.action is not None
    }
    assert root_scores[InterventionType.SOCRATIC_PROMPT] > 0
