import random
import numpy as np


class Game:
    """
    Stores the history of a single episode.
    """

    def __init__(self, discount=0.99):
        self.states = []
        self.actions = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.pred_discounts = []
        self.discount = discount

    def store_step(self, state, action, reward, discount, child_visits, root_value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.pred_discounts.append(discount)
        self.child_visits.append(child_visits)
        self.root_values.append(root_value)

    def compute_target_value(self, index, n_steps=10):
        """
        Computes the target value with bootstrapping as described in the
        MuZero paper. It uses actual values for the n-first steps, then use
        the values predicted from the network.
        """

        value = 0.0
        for i in range(n_steps):
            step = index + i
            if step < len(self.rewards):
                value += (self.discount**i) * self.rewards[step]
            else:
                return value

        bootstrap_idx = index + n_steps
        if bootstrap_idx < len(self.root_values):
            value += (self.discount**n_steps) * self.root_values[bootstrap_idx]

        return value

    def __len__(self):
        return len(self.actions)


class ReplayBuffer:
    def __init__(self, capacity=200):
        self.buffer = []
        self.capacity = capacity

    def save_game(self, game: Game):
        if len(self.buffer) >= self.capacity:
            # Remove oldest game if full
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, batch_size, unroll_steps, num_actions=5):
        # We don't filter out short games anymore! Every game matters.
        if not self.buffer:
            return None

        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_values = []
        batch_policies = []
        batch_discounts = []

        for _ in range(batch_size):
            game = random.choice(self.buffer)

            # FIX 1: Allow sampling all the way to the very last frame!
            random_pos = random.randint(0, len(game) - 1)

            batch_obs.append(game.states[random_pos])

            # Get the slices (they might be shorter than unroll_steps if near the end)
            actions = game.actions[random_pos : random_pos + unroll_steps]
            rewards = game.rewards[random_pos : random_pos + unroll_steps]
            policies = game.child_visits[random_pos : random_pos + unroll_steps + 1]
            discounts = game.pred_discounts[random_pos : random_pos + unroll_steps]

            # PADDING LOGIC: Fill the rest with zeros if we hit the end of the game
            while len(actions) < unroll_steps:
                actions.append(0)  # Pad with "Do Nothing" action
                rewards.append(0.0)  # Pad with 0 reward
                discounts.append(0.0)  # Pad with discount of 0 (game over)

            while len(policies) < unroll_steps + 1:
                # Pad policy with uniform distribution
                policies.append([1.0 / num_actions] * num_actions)

            batch_actions.append(actions)
            batch_rewards.append(rewards)

            # The compute_target_value function already handles out-of-bounds safely!
            target_vals = [
                game.compute_target_value(random_pos + t, n_steps=unroll_steps)
                for t in range(unroll_steps + 1)
            ]
            batch_values.append(target_vals)
            batch_policies.append(policies)
            batch_discounts.append(discounts)

        return {
            "observations": np.array(batch_obs),
            "actions": np.array(batch_actions),
            "target_rewards": np.array(batch_rewards),
            "target_discounts": np.array(batch_discounts),
            "target_values": np.array(batch_values),
            "target_policies": np.array(batch_policies),
        }
