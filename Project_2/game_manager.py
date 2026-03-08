import math
from functools import partial
from replay_buffer import Game, ReplayBuffer
import numpy as np
from mcts_node import MCTSNode
from tron.tron_env import TronEnv
from tron.env_wrapper import TronEnvWrapper
from umcts import UMCTS
import jax.numpy as jnp
import jax
from config import NUM_ACTIONS


@partial(jax.jit, static_argnums=(1,))
def representation_inference_fn(params, model, state):
    return model.apply(params, state, method=model.representation)


class GameManager:
    def __init__(self, model, params, mcts_num_simulations):
        self.replay_buffer = ReplayBuffer()
        self.model = model
        self.env = TronEnvWrapper(TronEnv())
        self.params = params
        self.num_actions = NUM_ACTIONS
        self.mcts = UMCTS(model, params)
        self.mcts_num_simulations = mcts_num_simulations

        self.decay_rate = 0.98

    def play_single_episode(self, max_episode_length):
        """
        Simulates one episode and stores it in the replay buffer.
        """

        # Ensure MCTS has the latest parameters
        self.mcts.params = self.params

        total_entropy = 0.0
        steps_taken = 0
        game_state = self.env.reset()
        game = Game()

        done = False

        while not done and steps_taken < max_episode_length:
            # Initialize the root node of the MCTS
            root_node = MCTSNode(prior=1.0)
            state_jnp = jnp.array([game_state])
            abstract_state = representation_inference_fn(
                self.params, self.model, state_jnp
            )
            root_node.game_state = abstract_state

            # Run MCTS to populate the search tree and get action probabilities
            self.mcts.run(root_node, num_simulations=self.mcts_num_simulations)
            policy_distribution, root_value = self.mcts.extract_mcts_data(
                root_node, self.num_actions
            )

            # Calculate the entropy of the policy distribution for this step and accumulate it
            step_entropy = -sum(p * math.log(p + 1e-8) for p in policy_distribution)
            total_entropy += step_entropy

            # Exploration decays over time
            exploration_rate = max(0.05, 1.0 * (self.decay_rate**steps_taken))
            adjusted_probs = np.power(policy_distribution, 1.0 / exploration_rate)
            adjusted_probs /= np.sum(adjusted_probs)
            action = np.random.choice(self.num_actions, p=adjusted_probs)

            next_state, reward, terminated = self.env.step(action)

            if terminated:
                done = True

            # Store the step in the game history
            game.store_step(
                state=game_state,
                action=action,
                reward=reward,
                child_visits=policy_distribution,
                root_value=root_value,
            )

            # Move to the next state
            game_state = next_state
            steps_taken += 1

        score = self.env.env.score

        avg_entropy = total_entropy / steps_taken if steps_taken > 0 else 0
        total_reward = sum(game.rewards)

        self.replay_buffer.save_game(game)

        num_50s = steps_taken // 50
        flames = "🔥" * num_50s

        print(
            f"{flames} Game finished in {steps_taken} | total reward {total_reward:.2f}"
        )

        return total_reward, steps_taken, score, avg_entropy
