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

# Import your 2048 files
from Game2048.env import Game2048Env
from Game2048.env_wrapper import EnvWrapper
from config import NUM_ACTIONS

# Update this path to where your 2048 params are saved
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

    # 2048 SHAPE: 4x4 board, 16 channels
    dummy_obs = jnp.ones((1, 4, 4, 16))
    dummy_act = jnp.array([0])

    template = model.init(rng, dummy_obs, dummy_act, method=model.init_params)

    with open(filepath, "rb") as f:
        data = f.read()

    return flax.serialization.from_bytes(template, data)


def watch_game(step_delay=0.15):
    filepath = list_available_params()

    # 1. Initialize Pygame BEFORE creating the environment!
    pygame.init()

    # 2. Initialize Models & Env
    model = MuZeroNet(num_actions=NUM_ACTIONS)
    params = load_params(model, filepath)

    # This will automatically pop open your native Pygame window
    env = EnvWrapper(Game2048Env())
    mcts = UMCTS(model, params)

    representation_fn = jax.jit(
        lambda p, s: model.apply(p, s, method=model.representation)
    )

    game_state = env.reset()
    done = False
    step_count = 0
    action_names = ["Up", "Down", "Left", "Right"]

    print("\n▶️ Watching MuZero play 2048...\n")

    while not done:
        step_count += 1

        # 3. Call your environment's native render function
        env.env.render()

        # 4. Handle Pygame window events so it doesn't freeze or crash
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Run MCTS
        state_jnp = jnp.array([game_state])
        abstract_state = representation_fn(params, state_jnp)

        root = MCTSNode(prior=1.0)
        root.game_state = abstract_state

        valid_actions = env.env.get_valid_actions()
        mcts.run(root, num_simulations=128, inject_noise=False, valid_actions=valid_actions)
        policy, value = mcts.extract_mcts_data(root, NUM_ACTIONS)

        probs = np.asarray(policy, dtype=np.float32)
        val_scalar = value[0] if isinstance(value, (list, np.ndarray)) else value
        probs_str = ", ".join([f"{p:.2f}" for p in probs])

        action = int(np.argmax(probs))

        # Print its "thoughts" to the terminal
        print(
            f"Step {step_count} | Score: {env.env.score} | Max Tile: {env.env.get_max_tile()}"
        )
        print(f"🧠 Value: {val_scalar:.3f} | Probs: [{probs_str}]")
        print(f"👉 Chosen Action: {action} ({action_names[action]})\n")

        # Take the step
        obs, reward, terminated = env.step(action)

        if terminated:
            env.env.render()  # Draw the final death screen
            print(f"💀 Game Over — survived {step_count} steps.")
            print(
                f"🏆 Final Score: {env.env.score} | Max Tile: {env.env.get_max_tile()}"
            )

            # Pause so you can look at the final board before it resets
            time.sleep(3)

            game_state = env.reset()
            step_count = 0

        # Wait before the next frame so it doesn't move at lightspeed
        time.sleep(step_delay)

    pygame.quit()


if __name__ == "__main__":
    watch_game(step_delay=0.05)
