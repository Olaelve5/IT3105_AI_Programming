import math

import wandb

from replay_buffer import Game, ReplayBuffer
import numpy as np
from mcts_node import MCTSNode
from tetris.tetris_env import TetrisEnv
from umcts import UMCTS
import jax.numpy as jnp
import jax
from config import NUM_ACTIONS


class GameManager:
    def __init__(self, model, params):
        self.replay_buffer = ReplayBuffer(capacity=5000)
        self.env = TetrisEnv()
        self.model = model
        self.params = params
        self.num_actions = NUM_ACTIONS
        self.mcts = UMCTS(model, params)

        self.representation_fn = jax.jit(
            lambda p, s: self.model.apply(p, s, method=self.model.representation)
        )

    def play_single_episode(self, max_episode_length=1000):
        """
        Simulates one episode and stores it in the replay buffer.
        """

        # Ensure MCTS has the latest parameters
        self.mcts.params = self.params

        total_entropy = 0.0
        steps_taken = 0
        game_state, _ = self.env.reset()
        game = Game()

        done = False

        while not done and steps_taken < max_episode_length:

            # Initialize the root node of the MCTS
            root_node = MCTSNode(prior=1.0)
            state_jnp = jnp.array([game_state])
            abstract_state = self.representation_fn(self.params, state_jnp)
            root_node.game_state = abstract_state

            # Run MCTS to populate the search tree and get action probabilities
            self.mcts.run(root_node, num_simulations=50)
            policy_distribution, root_value = self.mcts.extract_mcts_data(
                root_node, self.num_actions
            )

            # Calculate the entropy of the policy distribution for this step and accumulate it
            step_entropy = -sum(p * math.log(p + 1e-8) for p in policy_distribution)
            total_entropy += step_entropy

            # Sample action and step the environment
            action = np.random.choice(self.num_actions, p=policy_distribution)
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            if terminated or truncated:
                done = True

            # Store the step in the game history
            game.store_step(
                state=game_state,
                action=action,
                reward=reward,
                discount=0.0 if done else 0.99,
                child_visits=policy_distribution,
                root_value=root_value,
            )

            # Move to the next state
            game_state = next_state
            steps_taken += 1

        avg_entropy = total_entropy / steps_taken if steps_taken > 0 else 0
        total_reward = sum(game.rewards)

        wandb.log(
            {
                "Game/Episode_Length": steps_taken,
                "Game/Total_Reward": total_reward,
                "Game/Lines_Cleared": self.env.lines_cleared,
                "MCTS/Average_Entropy": avg_entropy,
            }
        )

        self.replay_buffer.save_game(game)
        print(
            f"Game finished in {steps_taken} steps with total reward {sum(game.rewards):.2f}"
        )

        return sum(game.rewards), steps_taken
