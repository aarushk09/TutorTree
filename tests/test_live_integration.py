import os

import pytest

from pmcts.mcts import PedagogicalMCTS
from pmcts.models import Message, Node, Role, State
from pmcts.reward import LiveRewardFunction
from pmcts.sim_student import LiveSimStudent


@pytest.mark.live
def test_live_mcts_integration_runs_three_iterations() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for live integration testing")

    root_state = State(
        learning_objective="Solve the linear equation 2x + 5 = 15",
        chat_history=(
            Message(
                role=Role.STUDENT,
                content="I don't understand how to solve 2x + 5 = 15.",
            ),
        ),
    )
    root = Node(state=root_state)
    mcts = PedagogicalMCTS(
        sim_student=LiveSimStudent(),
        reward_function=LiveRewardFunction(),
        exploration_constant=1.0,
        discount_factor=0.9,
        max_depth=1,
    )

    best = mcts.run(root, iterations=3)

    assert best.action is not None
    print(f"Selected optimal action: {best.action.intervention_type.value}")
    print(f"Selected action content: {best.action.content}")
