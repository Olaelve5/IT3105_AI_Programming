import math
from functools import partial
from replay_buffer import Game, ReplayBuffer
import numpy as np
from mcts_node import MCTSNode
from connect4.env_wrapper import EnvWrapper
from connect4.env import Connect4Env
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
        self.env = EnvWrapper(Connect4Env())
        self.params = params
        self.num_actions = NUM_ACTIONS
        self.mcts = UMCTS(model, params)
        self.mcts_num_simulations = mcts_num_simulations

        self.decay_rate = 0.85

    def play_single_episode(self, max_episode_length):
        self.mcts.params = self.params

        total_entropy = 0.0
        steps_taken = 0
        game_state = self.env.reset()
        game = Game()

        done = False

        while not done and steps_taken < max_episode_length:
            root_node = MCTSNode(prior=1.0)
            state_jnp = jnp.array([game_state])
            abstract_state = representation_inference_fn(
                self.params, self.model, state_jnp
            )
            root_node.game_state = abstract_state

            valid_actions = self.env.env.get_valid_actions()

            self.mcts.run(
                root_node,
                num_simulations=self.mcts_num_simulations,
                valid_actions=valid_actions,
            )
            policy_distribution, root_value = self.mcts.extract_mcts_data(
                root_node, self.num_actions
            )

            step_entropy = -sum(p * math.log(p + 1e-8) for p in policy_distribution)
            total_entropy += step_entropy

            exploration_rate = max(0.05, 1.0 * (self.decay_rate**steps_taken))
            adjusted_probs = np.power(policy_distribution, 1.0 / exploration_rate)

            # normalize to get a valid probability distribution
            if np.sum(adjusted_probs) > 0:
                adjusted_probs /= np.sum(adjusted_probs)
            else:
                adjusted_probs = np.ones(self.num_actions) / self.num_actions

            action = np.random.choice(self.num_actions, p=adjusted_probs)

            next_state, reward, terminated = self.env.step(action)

            if terminated:
                done = True

            game.store_step(
                state=game_state,
                action=action,
                reward=reward,
                child_visits=policy_distribution,
                root_value=root_value,
            )

            game_state = next_state
            steps_taken += 1

        self.replay_buffer.save_game(game)

        avg_entropy = total_entropy / steps_taken if steps_taken > 0 else 0
        total_reward = sum(game.rewards)

        if total_reward == 1.0:
            winner = 1 if steps_taken % 2 != 0 else 2

            absolute_reward = 1.0 if winner == 1 else -1.0

            print(
                f"Player {winner} WIN after {steps_taken} steps! (Absolute: {absolute_reward})"
            )
        else:
            absolute_reward = 0.0
            print(f"DRAW after {steps_taken} steps!")

        return absolute_reward, steps_taken, avg_entropy
