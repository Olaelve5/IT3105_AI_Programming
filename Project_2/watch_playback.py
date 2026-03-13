import os
import sys
import time
import random
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

# --- CONFIGURATION ---
PARAMS_FOLDER = "saved_params"
NUM_GAMES = 10  # How many games to pre-calculate
PLAYBACK_FPS = 20  # Speed of the playback highlight reel
STARTING_SEED = 42  # Base seed to ensure reproducibility


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
    dummy_obs = jnp.ones((1, BOARD_HEIGHT, BOARD_WIDTH, 3))
    dummy_act = jnp.array([0])
    template = model.init(rng, dummy_obs, dummy_act, method=model.init_params)

    with open(filepath, "rb") as f:
        data = f.read()

    return flax.serialization.from_bytes(template, data)


def precalculate_and_playback():
    filepath = list_available_params()

    model = MuZeroNet(num_actions=NUM_ACTIONS)
    params = load_params(model, filepath)

    env = TronEnvWrapper(TronEnv())
    mcts = UMCTS(model, params)

    representation_fn = jax.jit(
        lambda p, s: model.apply(p, s, method=model.representation)
    )

    # We will store the history of all 10 games here.
    # Format: [(seed, [action1, action2, ...]), ...]
    all_games_history = []

    # --- PHASE 1: THINK IN THE DARK ---
    print(f"\n🧠 AI is thinking... Pre-calculating {NUM_GAMES} games silently.")
    start_time = time.time()

    for game_idx in range(NUM_GAMES):
        # Create a unique seed for this specific game
        game_seed = STARTING_SEED + game_idx
        random.seed(game_seed)
        np.random.seed(game_seed)

        mcts = UMCTS(model, params) 
        
        game_state = env.reset()
        terminated = False
        step_count = 0
        actions_taken = []

        print(f"Simulating Game {game_idx + 1}/{NUM_GAMES}...")

        while not terminated:
            step_count += 1

            # Pure math, no rendering
            state_jnp = jnp.array([game_state])
            abstract_state = representation_fn(params, state_jnp)

            root = MCTSNode(prior=1.0)
            root.game_state = abstract_state

            mcts.run(root, num_simulations=100)
            policy, _ = mcts.extract_mcts_data(root, NUM_ACTIONS)

            probs = np.asarray(policy, dtype=np.float32)
            #action = int(np.argmax(probs))
            action = np.random.choice(NUM_ACTIONS, p=probs)
            
            actions_taken.append(action)
            game_state, _, terminated = env.step(action)

        all_games_history.append((game_seed, actions_taken))
        print(f"   -> Survived {step_count} steps.")

    calc_time = time.time() - start_time
    print(f"\n✅ Calculation complete! Total thinking time: {calc_time:.1f} seconds.")
    print(f"🎬 Starting {PLAYBACK_FPS} FPS Playback in 2 seconds...\n")
    time.sleep(2)

    # --- PHASE 2: PLAYBACK AT HIGH SPEED ---
    clock = pygame.time.Clock()

    for game_idx, (game_seed, actions_taken) in enumerate(all_games_history):
        print(
            f"▶️ Playing Game {game_idx + 1}/{NUM_GAMES} (Length: {len(actions_taken)} steps)"
        )

        # Reset the EXACT SAME SEED used during simulation so the random walls match
        random.seed(game_seed)
        np.random.seed(game_seed)

        env.reset()

        for action in actions_taken:
            # Keep the UI responsive so you can close the window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Draw the screen
            env.env.render()

            # Apply the pre-calculated move
            env.step(action)

            # Control the playback speed
            clock.tick(PLAYBACK_FPS)

        # Draw the final crash frame
        env.env.render()

        # Pause for 1 second between games so you can see how it died
        time.sleep(1)

    print("\n🏁 All replays finished.")
    pygame.quit()


if __name__ == "__main__":
    precalculate_and_playback()
