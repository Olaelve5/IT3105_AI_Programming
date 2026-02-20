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
from tetris.tetris_env import TetrisEnv
from config import NUM_ACTIONS, BOARD_WIDTH, BOARD_HEIGHT

PARAMS_FOLDER = "Project_2/saved_params"


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
    dummy_obs = jnp.ones((1, BOARD_HEIGHT, BOARD_WIDTH, 1))
    dummy_act = jnp.array([0])
    template = model.init(rng, dummy_obs, dummy_act, method=model.init_params)

    with open(filepath, "rb") as f:
        data = f.read()

    return flax.serialization.from_bytes(template, data)


def watch_game():
    filepath = list_available_params()

    model = MuZeroNet(num_actions=NUM_ACTIONS)
    params = load_params(model, filepath)

    env = TetrisEnv(board_height=BOARD_HEIGHT, board_width=BOARD_WIDTH)
    mcts = UMCTS(model, params, num_actions=NUM_ACTIONS)

    representation_fn = jax.jit(
        lambda p, s: model.apply(p, s, method=model.representation)
    )

    state, _ = env.reset()
    done = False
    step_count = 0

    while not done:
        step_count += 1
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        state_jnp = jnp.expand_dims(jnp.array([state]), axis=-1)
        abstract_state = representation_fn(params, state_jnp)

        root = MCTSNode(prior=1.0)
        root.game_state = abstract_state

        mcts.run(root, num_simulations=50)
        policy, _ = mcts.extract_mcts_data(root, NUM_ACTIONS)
        action = np.argmax(policy)

        print(f"Step {step_count} | Action: {action}")
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    print(f"Game Over — survived {step_count} steps.")
    time.sleep(2)
    env.close()


if __name__ == "__main__":
    watch_game()
