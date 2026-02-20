import math
from MuZeroNet import MuZeroNet
import random
import jax.numpy as jnp
import jax
import numpy as np
from mcts_node import MCTSNode
from config import NUM_ACTIONS


class UMCTS:
    """
    The UMCTS class implements the Monte Carlo Tree Search algorithm.
    It uses the MuZeroNet to evaluate game states and guide the search.
    """

    def __init__(self, model: MuZeroNet, params, discount_factor=0.99):
        self.model = model
        self.params = params
        self.num_actions = NUM_ACTIONS
        self.discount_factor = discount_factor

        self.min_value = float("inf")
        self.max_value = -float("inf")

        # JIT compile the model's recurrent and prediction functions for speed
        # This is a lot faster than calling apply with method=model.recurrent_inference every time
        self.recurrent_fn = jax.jit(
            lambda p, s, a: self.model.apply(
                p, s, a, method=self.model.recurrent_inference
            )
        )
        self.prediction_fn = jax.jit(
            lambda p, s: self.model.apply(p, s, method=self.model.prediction)
        )

    def run(self, root_node: MCTSNode, num_simulations=50):
        """
        Runs the full algorithm.
        """

        for _ in range(num_simulations):
            node: MCTSNode = root_node
            search_path = [node]

            # Walk down the search path until we find a leaf node (a node that isn't expanded yet)
            while node.is_expanded():
                action, node = self.select_child(node)

                if node.game_state is None:
                    parent_state = search_path[-1].game_state
                    action = jnp.array([action])

                    # Generate new state + reward
                    state, reward, _, _ = self.recurrent_fn(
                        self.params, parent_state, action
                    )

                    node.game_state = state
                    node.reward = float(reward[0, 0])

                search_path.append(node)

            # node is now a leaf node
            # Use the prediction nn to generate empty children and attatch them to the node
            action_probs, _ = self.prediction_fn(self.params, node.game_state)

            # Convert action probabilities to sum up to 1
            action_probs = jax.nn.softmax(action_probs[0])
            action_probs = np.array(action_probs)

            # Add some noise to encourage exploration
            if len(search_path) == 1:
                noise = np.random.dirichlet([0.3] * self.num_actions)
                action_probs = 0.75 * action_probs + 0.25 * noise

            for i in range(self.num_actions):
                child = MCTSNode(action_probs[i])
                node.children[i] = child

            # Do rollout and get the value
            rollout_value = self.rollout(node)

            # Walk up search_path and update stats
            self.backpropagate(search_path, rollout_value)

    def select_child(self, node):
        """
        Select the child action/node with the highest UCB score.
        """

        best_score = -float("inf")
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            score = self.ucb_score(node, child)

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def ucb_score(self, parent: MCTSNode, child: MCTSNode):
        """
        Computes the UCB score for a (parent, child) edge
        """

        # Constants are taken from the MuZero paper
        c1 = 1.25
        c2 = 19652

        parent_visit_count = parent.visit_count
        child_visit_count = child.visit_count

        explo_rate = c1 + math.log((parent_visit_count + c2 + 1) / c2)

        # P(s, a) -> probability of choosing this action from the parent node
        prior_score = (
            explo_rate
            * child.prior
            * math.sqrt(parent_visit_count)
            / (child_visit_count + 1)
        )

        # Q(s, a) -> the value of this child node (average reward)
        if child_visit_count > 0:
            value_score = self.normalize(child.value())
        else:
            value_score = 0

        return prior_score + value_score

    def update_stats(self, value):
        """
        Update min/max bounds used for value normalization.
        """

        self.max_value = max(self.max_value, value)
        self.min_value = min(self.min_value, value)

    def normalize(self, value):
        """
        Normalize a value to [0, 1] using running min/max bounds.
        """

        if self.max_value > self.min_value:
            return (value - self.min_value) / (self.max_value - self.min_value)
        else:
            return value

    def backpropagate(self, search_path, value):
        """
        Walks the search path in reversed order, updating stats for each node.
        """

        current_value = value

        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += current_value

            # Update stats so we can normalize the values
            self.update_stats(node.value())

            # Prepare current_value for the parent
            current_value = node.reward + (self.discount_factor * current_value)

    def rollout(self, node: MCTSNode):
        """
        Picks a random action, evaluates the predicted game state from this action
        and returns the value.
        """

        action = jnp.array([random.randint(0, self.num_actions - 1)])

        # Generate next state and evaluate it
        _, reward, _, value_next = self.recurrent_fn(
            self.params, node.game_state, action
        )

        reward = float(reward[0, 0])
        value_next = float(value_next[0, 0])

        return reward + (self.discount_factor * value_next)

    def extract_mcts_data(self, root_node, num_actions):
        root_value = root_node.value()

        # Extract child visits and turn them into a probability distribution
        visits = []
        for action in range(num_actions):
            if action in root_node.children:
                visits.append(root_node.children[action].visit_count)
            else:
                visits.append(0)  # In case an action was never explored

        total_visits = sum(visits)

        # Divide each visit count by the total to get a percentage
        policy_distribution = [
            v / total_visits if total_visits > 0 else 0 for v in visits
        ]

        return policy_distribution, root_value
