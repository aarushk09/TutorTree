"""P-MCTS core search algorithm."""

from __future__ import annotations

import math
from typing import Protocol

from pmcts.models import Action, InterventionType, Node, State


class SimStudent(Protocol):
    """Interface for student-transition models used during rollout."""

    def transition(self, state: State, action: Action) -> State:
        """Predict the next pedagogical state after taking an action."""


class RewardFunction(Protocol):
    """Interface for pedagogical reward models used during rollout."""

    def score(self, state: State, action: Action, next_state: State) -> float:
        """Return the pedagogical value of an action and transition."""


class PedagogicalMCTS:
    """Monte Carlo Tree Search over pedagogical interventions."""

    def __init__(
        self,
        sim_student: SimStudent,
        reward_function: RewardFunction,
        *,
        exploration_constant: float = 1.4,
        discount_factor: float = 0.9,
        max_depth: int = 3,
    ) -> None:
        if exploration_constant < 0:
            raise ValueError("exploration_constant must be non-negative")
        if not 0 <= discount_factor <= 1:
            raise ValueError("discount_factor must be between 0 and 1")
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")

        self.sim_student = sim_student
        self.reward_function = reward_function
        self.exploration_constant = exploration_constant
        self.discount_factor = discount_factor
        self.max_depth = max_depth

    def run(self, root: Node, iterations: int) -> Node:
        """Run MCTS from a root node and return the most visited child."""

        if iterations < 1:
            raise ValueError("iterations must be at least 1")

        for _ in range(iterations):
            leaf, path = self._select(root)

            if leaf.action is not None and leaf.visits == 0:
                reward = self._simulate(leaf)
            elif not leaf.terminal and leaf.depth < self.max_depth:
                self._expand(leaf)
                if leaf.children:
                    leaf = self._select_child_for_simulation(leaf)
                    path.append(leaf)
                reward = self._simulate(leaf)
            else:
                reward = self._simulate(leaf)

            self._backpropagate(path, reward)

        return self.best_child(root)

    def best_child(self, node: Node) -> Node:
        """Return the most visited child, using value as a deterministic tie-break."""

        if not node.children:
            raise ValueError("cannot select a best child from an unexpanded node")

        return max(node.children, key=lambda child: (child.visits, child.value_estimate))

    def generate_actions(self, state: State) -> list[Action]:
        """Generate candidate pedagogical actions for a state."""

        return self._branch_actions(state)

    def _select(self, node: Node) -> tuple[Node, list[Node]]:
        path = [node]
        current = node

        while current.children and not current.terminal:
            current = max(current.children, key=lambda child: self._uct(current, child))
            path.append(current)

        return current, path

    def _uct(self, parent: Node, child: Node) -> float:
        if child.visits == 0:
            return math.inf

        exploitation = child.value_estimate
        exploration = self.exploration_constant * math.sqrt(
            math.log(max(parent.visits, 1)) / child.visits
        )
        return exploitation + exploration

    def _expand(self, node: Node) -> None:
        if node.children:
            return

        child_depth = node.depth + 1
        node.children = [
            Node(
                state=node.state,
                action=action,
                parent_id=node.node_id,
                depth=child_depth,
                terminal=child_depth >= self.max_depth,
            )
            for action in self._branch_actions(node.state)
        ]

    def _branch_actions(self, state: State) -> list[Action]:
        objective = state.learning_objective
        return [
            Action(
                intervention_type=InterventionType.DIRECT_ANSWER,
                content=f"Here is the answer for {objective}.",
                rationale="Supplies the solution directly.",
                expected_student_activity="Passive",
            ),
            Action(
                intervention_type=InterventionType.SOCRATIC_PROMPT,
                content=(
                    "What is the smallest observation you can make about the "
                    "problem before choosing a next step?"
                ),
                rationale="Elicits constructive reasoning before revealing information.",
                expected_student_activity="Constructive",
            ),
            Action(
                intervention_type=InterventionType.HINT,
                content="Look for the first place where the expected and actual behavior diverge.",
                rationale="Provides a scaffold while preserving student agency.",
                expected_student_activity="Active",
            ),
        ]

    def _select_child_for_simulation(self, node: Node) -> Node:
        return max(node.children, key=lambda child: self._uct(node, child))

    def _simulate(self, node: Node) -> float:
        if node.action is None:
            return 0.0

        previous_state = node.state
        next_state = self.sim_student.transition(previous_state, node.action)
        reward = self.reward_function.score(previous_state, node.action, next_state)
        node.state = next_state
        return reward

    def _backpropagate(self, path: list[Node], reward: float) -> None:
        discounted_value = reward

        for node in reversed(path):
            node.visits += 1
            node.value_sum += discounted_value
            discounted_value *= self.discount_factor
