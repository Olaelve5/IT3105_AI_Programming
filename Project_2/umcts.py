import math
from MuZeroNet import MuZeroNet
import random
import jax.numpy as jnp
import jax
import numpy as np
from mcts_node import MCTSNode
from config import NUM_ACTIONS


class MinMaxStats:
    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return 0.0


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

    def run(self, root_node, legal_actions, num_simulations=50):
        """
        Runs the full algorithm.
        """
        min_max = MinMaxStats()

        for _ in range(num_simulations):
            node = root_node
            search_path = [node]

            while node.is_expanded():
                action, node = self.select_child(node, min_max)

                if node.game_state is None:
                    parent_state = search_path[-1].game_state

                    action_arr = np.array([action], dtype=np.int32)

                    state, reward, discount, _, _ = self.recurrent_fn(
                        self.params, parent_state, action_arr
                    )

                    node.game_state = state
                    node.reward = reward.item()
                    node.discount = discount.item()

                search_path.append(node)

            action_probs_jax, predicted_value_jax = self.prediction_fn(
                self.params, node.game_state
            )

            logits = np.asarray(action_probs_jax)[0]
            predicted_value = predicted_value_jax.item()

            max_logit = np.max(logits)
            exp_logits = np.exp(logits - max_logit)
            action_probs = exp_logits / np.sum(exp_logits)

            # Mask out illegal actions + add noise for exploration
            if len(search_path) == 1:
                mask = np.zeros(self.num_actions, dtype=np.float32)
                mask[legal_actions] = 1.0
                action_probs *= mask

                # Normalize again
                prob_sum = np.sum(action_probs)
                if prob_sum > 0:
                    action_probs /= prob_sum
                else:
                    action_probs = mask / np.sum(mask)

                noise = np.zeros(self.num_actions, dtype=np.float32)
                noise[legal_actions] = np.random.dirichlet([0.3] * len(legal_actions))
                action_probs = 0.75 * action_probs + 0.25 * noise

            for i in range(self.num_actions):
                if action_probs[i] > 0.001:
                    node.children[i] = MCTSNode(action_probs[i])

            self.backpropagate(search_path, predicted_value, min_max)

    def select_child(self, node, min_max):
        """Select the child action/node with the highest UCB score."""
        best_score = -float("inf")
        best_action = -1
        best_child = None

        c1 = 1.25
        c2 = 19652
        parent_visits = node.visit_count
        explo_rate = c1 + math.log((parent_visits + c2 + 1) / c2)
        parent_sqrt = math.sqrt(parent_visits)

        # Pre-calculate parent value for baseline
        parent_value = min_max.normalize(node.value()) if parent_visits > 0 else 0.0

        for action, child in node.children.items():
            score = self.ucb_score(
                child, min_max, explo_rate, parent_sqrt, parent_value
            )

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def ucb_score(
        self,
        child: MCTSNode,
        min_max: MinMaxStats,
        explo_rate,
        parent_sqrt,
        parent_value,
    ):
        """Computes the UCB score for a child edge using pre-calculated parent stats."""
        child_visits = child.visit_count

        # P(s, a) -> probability of choosing this action
        prior_score = explo_rate * child.prior * parent_sqrt / (child_visits + 1)

        # Q(s, a) -> the value of this child node
        if child_visits > 0:
            value_score = min_max.normalize(child.value())
        else:
            value_score = parent_value

        return prior_score + value_score

    def backpropagate(self, search_path, value, min_max: MinMaxStats):
        """
        Walks the search path in reversed order, updating stats for each node.
        """

        current_value = value

        for node in reversed(search_path):
            current_value = node.reward + (node.discount * current_value)
            node.visit_count += 1
            node.value_sum += current_value

            # Update stats so we can normalize the values
            min_max.update(node.value())

    def rollout(self, node: MCTSNode):
        """
        Picks a random action, evaluates the predicted game state from this action
        and returns the value.
        """

        action = jnp.array([random.randint(0, self.num_actions - 1)])

        # Generate next state and evaluate it
        _, reward, discount, value_next = self.recurrent_fn(
            self.params, node.game_state, action
        )

        reward = float(reward[0, 0])
        value_next = float(value_next[0, 0])
        discount = float(discount[0, 0])

        return reward + (discount * value_next)

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
