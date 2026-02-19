from game_manager import GameManager
from tetris_env import TetrisEnv
from MuZeroNet import MuZeroNet
import jax
import jax.numpy as jnp
import numpy as np

if __name__ == "__main__":
    print("--- Starting GameManager Test ---")

    # 1. Setup basic parameters
    # Assuming Tetris has e.g., 4 actions (Left, Right, Rotate, Drop)
    NUM_ACTIONS = 4

    # 2. Instantiate the model
    model = MuZeroNet(num_actions=NUM_ACTIONS)

    # 3. Initialize the JAX Network Weights (params)
    # JAX requires a random key to initialize weights
    rng = jax.random.PRNGKey(42)

    # We need dummy data to tell Flax what shape the inputs will be.
    # IMPORTANT: Convolutional layers usually expect shapes like (Batch, Height, Width, Channels)
    # If your Tetris board is 20x10, your dummy observation should be (1, 20, 10, 1)
    dummy_observation = jnp.ones((1, 20, 10, 1))
    dummy_action = jnp.array([0])

    print("Initializing model parameters...")
    # Use the init_params method you wrote in MuZeroNet!
    params = model.init(rng, dummy_observation, dummy_action, method=model.init_params)

    # 4. Create the Manager and Play
    manager = GameManager(model, params, NUM_ACTIONS)

    print("Playing a test episode (capped at 50 steps for speed)...")
    # Cap it at 50 so you don't have to wait 10 minutes to see if it works
    manager.play_single_episode(max_episode_length=10)

    # 5. Verify the buffer
    print(f"Games in Replay Buffer: {len(manager.replay_buffer.buffer)}")
    if len(manager.replay_buffer.buffer) > 0:
        print("✅ SUCCESS! Data is flowing from the Game into the Memory Bank.")
