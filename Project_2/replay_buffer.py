import random
import numpy as np


class Game:
    """
    Stores the history of a single episode.
    """

    def __init__(self, discount=0.98):
        self.states = []
        self.actions = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.discount = discount

    def store_step(self, state, action, reward, child_visits, root_value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
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
    def __init__(self, capacity=5000):
        self.buffer = []
        self.capacity = capacity

    def save_game(self, game: Game):
        if len(self.buffer) >= self.capacity:
            # Remove oldest game if full
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, batch_size, unroll_steps):
        # Filter out games that are too short to sample from
        valid_games = [g for g in self.buffer if len(g) > unroll_steps]

        if not valid_games:
            return None

        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_values = []
        batch_policies = []

        for _ in range(batch_size):
            game = random.choice(valid_games)

            random_pos = random.randint(0, len(game) - unroll_steps - 1)

            batch_obs.append(game.states[random_pos])
            batch_actions.append(game.actions[random_pos : random_pos + unroll_steps])
            batch_rewards.append(game.rewards[random_pos : random_pos + unroll_steps])

            target_vals = [
                game.compute_target_value(random_pos + t, n_steps=unroll_steps)
                for t in range(unroll_steps + 1)
            ]
            batch_values.append(target_vals)

            batch_policies.append(
                game.child_visits[random_pos : random_pos + unroll_steps + 1]
            )

        return {
            # Add an extra dimension to observations to match the expected input shape of the model
            "observations": np.expand_dims(np.array(batch_obs), axis=-1),
            "actions": np.array(batch_actions),
            "target_rewards": np.array(batch_rewards),
            "target_values": np.array(batch_values),
            "target_policies": np.array(batch_policies),
        }
