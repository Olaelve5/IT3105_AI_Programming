import math
from functools import partial
from replay_buffer import Game, ReplayBuffer
import numpy as np
from mcts_node import MCTSNode
from Game2048.env_wrapper import EnvWrapper
from Game2048.env import Game2048Env
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
        self.env = EnvWrapper(Game2048Env())
        self.params = params
        self.num_actions = NUM_ACTIONS
        self.mcts = UMCTS(model, params)
        self.mcts_num_simulations = mcts_num_simulations

        self.decay_rate = 0.995

    def play_single_episode(self, max_episode_length):
        """
        Simulates one episode and stores it in the replay buffer.
        """
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

            # Masking at the root node
            valid_actions = self.env.env.get_valid_actions()

            # Run MCTS to populate the search tree and get action probabilities
            self.mcts.run(
                root_node,
                num_simulations=self.mcts_num_simulations,
                valid_actions=valid_actions,
            )
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

        max_tile = self.env.env.get_max_tile()
        base_msg = f"Game finished after {steps_taken} steps | Max tile: {max_tile} | Reward: {total_reward:.2f} | Score: {score}"

        if max_tile >= 2048:
            print(f"⭐️⭐️⭐️⭐️⭐️ {base_msg}! ⭐️⭐️⭐️⭐️⭐️")
        elif max_tile >= 1024:
            print(f"🔥🔥🔥🔥 {base_msg}! 🔥🔥🔥🔥")
        elif max_tile >= 512:
            print(f"🔥🔥🔥 {base_msg}! 🔥🔥🔥")
        elif max_tile >= 256:
            print(f"🔥🔥 {base_msg}! 🔥🔥")
        elif max_tile >= 128:
            print(f"🔥 {base_msg}! 🔥")
        else:
            print(f"{base_msg}.")

        return total_reward, steps_taken, score, avg_entropy, max_tile
