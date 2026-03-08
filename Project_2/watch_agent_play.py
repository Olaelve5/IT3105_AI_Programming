import os
import sys
import time
import jax
import jax.numpy as jnp
import flax.serialization
import numpy as np
import pygame
from MuZeroNet import MuZeroNet
from umcts import UMCTS
from mcts_node import MCTSNode

# Import your Tron files
from tron.tron_env import TronEnv
from tron.env_wrapper import TronEnvWrapper
from config import NUM_ACTIONS, BOARD_WIDTH, BOARD_HEIGHT

# Update this path to where your Tron params are saved
PARAMS_FOLDER = "saved_params"


def list_available_params():
    if not os.path.exists(PARAMS_FOLDER):
        print(f"Folder '{PARAMS_FOLDER}' does not exist. Run training first.")
        sys.exit()

    files = [f for f in os.listdir(PARAMS_FOLDER) if f.endswith(".msgpack")]
    if not files:
        print(f"No saved params found in '{PARAMS_FOLDER}'.")
        sys.exit()

    try:
        files.sort(key=lambda x: int(x.split("_")[0]))
    except ValueError:
        files.sort()

    print("\nAvailable params:")
    for i, name in enumerate(files):
        print(f"  [{i}] {name}")

    while True:
        choice = input(f"\nSelect params to load (0-{len(files)-1}): ")
        try:
            idx = int(choice)
            if 0 <= idx < len(files):
                path = os.path.join(PARAMS_FOLDER, files[idx])
                print(f"Loading {path}...")
                return path
        except ValueError:
            pass
        print("Invalid choice, try again.")


def load_params(model, filepath):
    rng = jax.random.PRNGKey(0)
    dummy_obs = jnp.ones((1, BOARD_HEIGHT, BOARD_WIDTH, 5))
    dummy_act = jnp.array([0])
    template = model.init(rng, dummy_obs, dummy_act, method=model.init_params)

    with open(filepath, "rb") as f:
        data = f.read()

    return flax.serialization.from_bytes(template, data)


def watch_game(human_control=False, debug=False):
    filepath = list_available_params()

    # Pass num_actions to the network
    model = MuZeroNet(num_actions=NUM_ACTIONS)
    params = load_params(model, filepath)

    # Initialize Tron wrapper
    env = TronEnvWrapper(TronEnv())
    mcts = UMCTS(model, params)

    representation_fn = jax.jit(
        lambda p, s: model.apply(p, s, method=model.representation)
    )

    # Wrapper reset returns only the observation matrix
    game_state = env.reset()
    done = False
    step_count = 0

    print("▶️ Watching Tron Agent...")

    while not done:
        step_count += 1

        # Render the underlying Tron game
        env.env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Run MCTS
        state_jnp = jnp.array([game_state])
        abstract_state = representation_fn(params, state_jnp)

        root = MCTSNode(prior=1.0)
        root.game_state = abstract_state

        mcts.run(root, num_simulations=50)
        policy, value = mcts.extract_mcts_data(root, NUM_ACTIONS)

        probs = np.asarray(policy, dtype=np.float32)
        if debug:
            print(
                f"DEBUG: Head: {env.env.head_pos} | Value: {value:.3f} | Action Probs: [Left: {probs[0]:.3f}, Right: {probs[1]:.3f}, Forward: {probs[2]:.3f}]"
            )
            current_state = env.get_obs()
            print(f"DEBUG: State Shape: {current_state.shape}")
            trail_sum = jnp.sum(current_state[:, :, 1])
            wall_sum = jnp.sum(current_state[:, :, 2])
            print(f"DEBUG: Trail pixels: {trail_sum} | Wall pixels: {wall_sum}")

        action = int(np.argmax(probs))

        print(f"Step {step_count} | Action: {action}")

        # Wrapper returns (obs, reward, terminated)
        game_state, _, terminated = env.step(action)

        if terminated:
            print(f"💀 Game Over — survived {step_count} steps.")
            game_state = env.reset()
            step_count = 0
            time.sleep(1)

        if human_control:
            # Wait for user input to proceed to the next step
            input("Press Enter to continue...")

    pygame.quit()


if __name__ == "__main__":
    watch_game(human_control=True)
