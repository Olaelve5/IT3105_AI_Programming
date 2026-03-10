import os
import sys
import time
import random

# Force Mac to use CPU for the thinking phase - it's much faster for batch_size=1
os.environ["JAX_PLATFORM_NAME"] = "cpu"

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


def precalculate_and_playback():
    filepath = list_available_params()

    model = MuZeroNet(num_actions=NUM_ACTIONS)
    params = load_params(model, filepath)

    env = TronEnvWrapper(TronEnv())
    mcts = UMCTS(model, params)

    representation_fn = jax.jit(
        lambda p, s: model.apply(p, s, method=model.representation)
    )

    # --- PHASE 1: THINK IN THE DARK ---
    print("\n🧠 AI is thinking... Pre-calculating the entire game silently.")
    print("Please wait. It is running 50 MCTS simulations per move...")
    
    # We freeze the random seed so the environment spawns identically both times
    MASTER_SEED = 42
    random.seed(MASTER_SEED)
    np.random.seed(MASTER_SEED)
    
    # If your environment supports seeding directly, uncomment this:
    # env.env.seed(MASTER_SEED)
    
    game_state = env.reset()
    terminated = False
    step_count = 0
    actions_taken = []

    start_time = time.time()

    while not terminated:
        step_count += 1
        
        # We DO NOT render here. Just pure math.
        state_jnp = jnp.array([game_state])
        abstract_state = representation_fn(params, state_jnp)

        root = MCTSNode(prior=1.0)
        root.game_state = abstract_state

        mcts.run(root, num_simulations=50)
        policy, _ = mcts.extract_mcts_data(root, NUM_ACTIONS)

        probs = np.asarray(policy, dtype=np.float32)
        action = int(np.argmax(probs))
        
        # Save the move for the movie later
        actions_taken.append(action)

        game_state, _, terminated = env.step(action)

        if step_count % 10 == 0:
            print(f"   ...calculated {step_count} steps...")

    calc_time = time.time() - start_time
    print(f"\n✅ Calculation complete! The agent survived {step_count} steps.")
    print(f"⏱️  Thinking took {calc_time:.1f} seconds.")
    print("🎬 Starting 60 FPS Playback in 2 seconds...\n")
    time.sleep(2)


    # --- PHASE 2: PLAYBACK AT 60 FPS ---
    
    # Reset the exact same seeds so the board generates identically
    random.seed(MASTER_SEED)
    np.random.seed(MASTER_SEED)
    # env.env.seed(MASTER_SEED) 
    
    env.reset()
    clock = pygame.time.Clock()

    for i, action in enumerate(actions_taken):
        # Allow user to close the window during playback
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Draw the screen
        env.env.render()
        
        # Apply the pre-calculated move
        env.step(action)

        # Force the game to run at exactly 60 Frames Per Second!
        clock.tick(15)

    # Draw the final crash frame
    env.env.render()
    print("💀 Crash! End of playback.")
    
    # Keep the window open for a few seconds so you can see the final board
    time.sleep(3)
    pygame.quit()


if __name__ == "__main__":
    precalculate_and_playback()