import random
import numpy as np
import collections
from config import NUM_ACTIONS


class Game:
    def __init__(self, discount=1.0):
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

    def compute_target_value(self, index, n_steps=42):
        """
        Computes the value of a state
        - looks all the way til then end of the game for connect 4 (42 steps)
        """
        value = 0.0
        for i in range(n_steps):
            step = index + i
            if step < len(self.rewards):
                value += (self.discount**i) * self.rewards[step] * ((-1) ** i)
            else:
                return value

        bootstrap_idx = index + n_steps
        if bootstrap_idx < len(self.root_values):
            value += (
                (self.discount**n_steps)
                * self.root_values[bootstrap_idx]
                * ((-1) ** n_steps)
            )
        return value

    def __len__(self):
        return len(self.actions)


class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.buffer = collections.deque(maxlen=capacity)
        self.total_steps = 0

    def save_game(self, game: Game):
        if len(self.buffer) == self.buffer.maxlen:
            self.total_steps -= len(self.buffer[0])

        self.buffer.append(game)
        self.total_steps += len(game)

    def sample_batch(
        self, batch_size, td_steps=30, unroll_steps=5, num_actions=NUM_ACTIONS
    ):
        if not self.buffer:
            return None

        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_values = []
        batch_policies = []
        batch_discounts = []

        # higher weights for longher games to ensure balanced sampling across different game lengths
        game_lengths = [len(g) for g in self.buffer]
        selected_games = random.choices(self.buffer, weights=game_lengths, k=batch_size)

        for game in selected_games:
            random_pos = random.randint(0, len(game) - 1)
            batch_obs.append(game.states[random_pos])

            actions = game.actions[random_pos : random_pos + unroll_steps]
            rewards = game.rewards[random_pos : random_pos + unroll_steps]
            policies = game.child_visits[random_pos : random_pos + unroll_steps + 1]

            discounts = []
            for t in range(unroll_steps):
                step_idx = random_pos + t
                if step_idx < len(game.rewards) - 1:
                    discounts.append(game.discount)
                elif step_idx == len(game.rewards) - 1:
                    discounts.append(0.0)
                else:
                    discounts.append(0.0)

            # padding
            while len(actions) < unroll_steps:
                actions.append(0)
                rewards.append(0.0)

            while len(discounts) < unroll_steps:
                discounts.append(0.0)

            while len(policies) < unroll_steps + 1:
                policies.append([1.0 / num_actions] * num_actions)

            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_discounts.append(discounts)

            target_vals = []
            for t in range(unroll_steps + 1):
                step_idx = random_pos + t
                if step_idx < len(game.states):
                    target_vals.append(
                        game.compute_target_value(step_idx, n_steps=td_steps)
                    )
                else:
                    target_vals.append(0.0)

            while len(target_vals) < unroll_steps + 1:
                target_vals.append(0.0)

            batch_values.append(target_vals)
            batch_policies.append(policies)

        return {
            "observations": np.array(batch_obs),
            "actions": np.array(batch_actions),
            "target_rewards": np.array(batch_rewards),
            "target_discounts": np.array(batch_discounts),
            "target_values": np.array(batch_values),
            "target_policies": np.array(batch_policies),
        }
