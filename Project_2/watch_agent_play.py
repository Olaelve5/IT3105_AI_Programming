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
from EnvWrapper import EnvWrapper
from config import NUM_ACTIONS, BOARD_WIDTH, BOARD_HEIGHT
from micro_agent.A_Star import A_Star  # <-- Imported A* here

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_FOLDER = os.path.join(SCRIPT_DIR, "saved_params")

pygame.init()


def list_available_params():
    if not os.path.exists(PARAMS_FOLDER):
        print(f"Folder '{PARAMS_FOLDER}' does not exist. Run training first.")
        sys.exit()

    files = [f for f in os.listdir(PARAMS_FOLDER) if f.endswith(".msgpack")]
    if not files:
        print(f"No saved params found in '{PARAMS_FOLDER}'.")
        sys.exit()

    try:
        files.sort(key=lambda x: int(x.split("_")[0]) if "_" in x else 999999)
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


def load_params(filepath):
    model = MuZeroNet(num_actions=NUM_ACTIONS)
    rng = jax.random.PRNGKey(0)

    dummy_obs = jnp.ones((1, BOARD_HEIGHT, BOARD_WIDTH, 3))
    dummy_act = jnp.array([0])
    template = model.init(rng, dummy_obs, dummy_act, method=model.init_params)

    with open(filepath, "rb") as f:
        data = f.read()

    return model, flax.serialization.from_bytes(template, data)


def watch_game():
    filepath = list_available_params()

    model, params = load_params(filepath)

    raw_env = TetrisEnv(tick_speed=10)
    env = EnvWrapper(raw_env)

    mcts = UMCTS(model, params)

    representation_fn = jax.jit(
        lambda p, s: model.apply(p, s, method=model.representation)
    )

    # --- A* Prompt ---
    print("\n" + "=" * 50)
    ans = input("🌟 Do you want to use A* to animate the pieces falling? (y/n): ")
    use_a_star = ans.strip().lower() == "y"
    print("=" * 50 + "\n")

    game_state = env.reset()
    done = False
    step_count = 0

    print("▶️ Watching Agent Play...")

    while not done:
        step_count += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.env.close()
                sys.exit()

        state_jnp = jnp.array([game_state])
        abstract_state = representation_fn(params, state_jnp)

        root = MCTSNode(prior=1.0)
        root.game_state = abstract_state

        legal_actions = env.get_legal_actions()

        mcts.run(root, legal_actions=legal_actions, num_simulations=30)
        policy, _ = mcts.extract_mcts_data(root, NUM_ACTIONS)

        # Pure exploitation for evaluation
        action = int(np.argmax(policy))
        print(f"Step {step_count} | Action: {action}")

        # --- Corrected A* Animation Logic ---
        if use_a_star:
            target_x, target_rot = env.decode_action(action)
            target_y = env.best_y_for_action.get(action)

            if target_y is not None:
                start_state = (
                    raw_env.active_piece.x,
                    raw_env.active_piece.y,
                    raw_env.active_piece.rotation,
                )
                target_state = (target_x, target_y, target_rot)

                # Instantiate the class properly!
                a_star_planner = A_Star(raw_env)
                path = a_star_planner.find_path(start_state, target_state)

                if path:
                    for micro_action in path:
                        # Skip the final hard drop so we don't accidentally lock the piece early!
                        # We let the Macro Wrapper handle the official lock to keep the math synced.
                        if micro_action == 4:
                            continue

                        raw_env.step(micro_action)
                        raw_env.render()

                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                raw_env.close()
                                sys.exit()

        # Step the wrapper to formally teleport (if needed), lock the piece, and get reward
        game_state, reward, terminated = env.step(action)

        if not use_a_star:
            env.render()
            time.sleep(0.2)

        if terminated:
            env.render()
            print(f"\n💀 Game Over — Survived {step_count} steps.")
            print(f"Total Lines Cleared: {env.env.lines_cleared}")
            time.sleep(1)

            game_state = env.reset()
            step_count = 0

    env.env.close()


if __name__ == "__main__":
    watch_game()
