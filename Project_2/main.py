import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

from game_manager import GameManager
from MuZeroNet import MuZeroNet
import jax
import jax.numpy as jnp
from train import perform_training_steps
import optax
import flax.serialization
import os
from utils.lates_checkpoint import find_latest_checkpoint
from config import NUM_ACTIONS, BOARD_WIDTH, BOARD_HEIGHT
import wandb
import time
import sys
import numpy as np

rng = jax.random.PRNGKey(42)
print("🚨 JAX IS USING:", jax.devices())

# ================ Hyperparameters ================
NUM_GENERATIONS = 10000
TARGET_STEPS_PER_GENERATION = 100
TRAINING_STEPS_PER_GENERATION = 100
NUM_SIMULATIONS = 128
LEARNING_RATE = 0.0001
BATCH_SIZE = 256
UNROLL_STEPS = 6
SAVE_PARAMS = True
TD_STEPS = 100
RUN_NAME = "muzero-tron-run-vMessi"


def load_params(params, path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            params = flax.serialization.from_bytes(params, f.read())
        print(f"🧠 Successfully loaded brain from {path}!")
    else:
        print(f"⚠️  No saved brain found at {path}. Aborting...")
        sys.exit(1)

    return params


def main(save_params=SAVE_PARAMS, load_checkpoint=True):
    wandb.init(project="muzero-tron", id=RUN_NAME, config={"run_name": RUN_NAME})
    wandb.config.update(
        {
            "num_generations": NUM_GENERATIONS,
            "target_steps_per_generation": TARGET_STEPS_PER_GENERATION,
            "training_steps_per_generation": TRAINING_STEPS_PER_GENERATION,
            "num_simulations": NUM_SIMULATIONS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "unroll_steps": UNROLL_STEPS,
            "td_steps": TD_STEPS,
            "num_channels": 64,
            "num_res_blocks": 5,
        }
    )
    wandb_starting_gen = 0

    # Model initialization
    model = MuZeroNet(num_actions=NUM_ACTIONS)
    dummy_obs = jnp.ones((1, BOARD_HEIGHT, BOARD_WIDTH, 3))
    dummy_act = jnp.array([0])
    params = model.init(rng, dummy_obs, dummy_act, method=model.init_params)

    if load_checkpoint:
        checkpoint, gen_number = find_latest_checkpoint()
        if checkpoint:
            params = load_params(params, checkpoint)
            wandb_starting_gen = gen_number
        else:
            print("⚠️  No checkpoints found. Starting training from scratch.")

    # Optimizer
    lr_schedule = optax.cosine_decay_schedule(
        init_value=LEARNING_RATE, decay_steps=20000, alpha=0.05
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(5.0), optax.adamw(lr_schedule, weight_decay=1e-4)
    )
    opt_state = optimizer.init(params)

    # Game manager
    game_manager = GameManager(model, params, mcts_num_simulations=NUM_SIMULATIONS)

    # Main training loop
    print("\n========== 🚀 Starting Training ==========")
    print(f"Training for {NUM_GENERATIONS} generations...\n")

    best_avg_reward = -float("inf")

    for gen in range(NUM_GENERATIONS):
        print(f"===== Generation {gen + 1 + wandb_starting_gen} =====")

        results = []
        games_this_generation = []
        start_time = time.time()
        steps_gathered = 0
        games_played = 0

        print(f"Gathering ~{TARGET_STEPS_PER_GENERATION} steps of experience...")
        while steps_gathered < TARGET_STEPS_PER_GENERATION:
            try:
                total_reward, steps, lines, entropy, game = (
                    game_manager.play_single_episode(
                        max_episode_length=400,
                    )
                )
                results.append((total_reward, steps, lines, entropy))
                games_this_generation.append(game)

                steps_gathered += steps
                games_played += 1

            except Exception as e:
                print(f"A game crashed: {e}")

        generation_end_time = time.time()

        if results:
            avg_reward = sum(r[0] for r in results) / len(results)
            avg_steps = sum(r[1] for r in results) / len(results)
            avg_score = sum(r[2] for r in results) / len(results)
            avg_entropy = sum(r[3] for r in results) / len(results)
            max_steps = max(r[1] for r in results)

            print(f"🎮 Played {games_played} games to gather {steps_gathered} steps.")
            print(
                f"🏆 Average Reward: {avg_reward:.2f} | ⏱️  Average Steps: {avg_steps:.0f} | 👑 Max: {max_steps}"
            )
            print(f"⏱️  Generation Time: {generation_end_time - start_time:.2f} seconds")
            wandb.log(
                {
                    "Game/Average_Total_Reward": avg_reward,
                    "Game/Average_Episode_Length": avg_steps,
                    "Game/Average_Score": avg_score,
                    "MCTS/Average_Entropy": avg_entropy,
                }
            )

        # Scale training steps to available data (avoid overfitting small buffers)
        buffer_size = game_manager.replay_buffer.total_steps
        max_training_steps = max(1, buffer_size // BATCH_SIZE)
        training_steps = min(TRAINING_STEPS_PER_GENERATION, max_training_steps)

        # Train on the experience
        params, opt_state, _ = perform_training_steps(
            model,
            params,
            optimizer,
            opt_state,
            training_steps,
            BATCH_SIZE,
            UNROLL_STEPS,
            TD_STEPS,
            game_manager,
        )

        training_end_time = time.time()
        print(
            f"⏱️  Training Time: {training_end_time - generation_end_time:.2f} seconds"
        )

        game_manager.params = params

        if (gen + 1 + wandb_starting_gen) % 25 == 0 or gen == 0:
            if save_params:
                os.makedirs("saved_params", exist_ok=True)
                save_path = (
                    f"saved_params/{gen + 1 + wandb_starting_gen}_generations.msgpack"
                )
                with open(save_path, "wb") as f:
                    f.write(flax.serialization.to_bytes(params))
                print(
                    f"💾 Saved params after {gen + 1 + wandb_starting_gen} generations -> {save_path}"
                )

        if results:
            avg_reward = sum(r[0] for r in results) / len(results)
        else:
            avg_reward = -float("inf")

        if avg_reward > best_avg_reward:
            print(f"🏆 NEW HIGH SCORE! ({best_avg_reward:.2f})")
            best_avg_reward = avg_reward

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
