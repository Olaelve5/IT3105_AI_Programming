import random
import numpy as np


class Game:
    """
    Stores the history of a single episode.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []

    def store_step(self, state, action, reward, child_visits, root_value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.child_visits.append(child_visits)
        self.root_values.append(root_value)

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

            # The next K actions and rewards
            batch_actions.append(game.actions[random_pos : random_pos + unroll_steps])
            batch_rewards.append(game.rewards[random_pos : random_pos + unroll_steps])

            # The K+1 targets for policy and value
            batch_values.append(
                game.root_values[random_pos : random_pos + unroll_steps + 1]
            )
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
