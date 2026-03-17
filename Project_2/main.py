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
NUM_GENERATIONS = 1000
TARGET_STEPS_PER_GENERATION = 800
TRAINING_STEPS_PER_GENERATION = 100
NUM_SIMULATIONS = 100
LEARNING_RATE = 0.0005
BATCH_SIZE = 128
UNROLL_STEPS = 5
SAVE_PARAMS = True
TD_STEPS = 42
MAX_EPISODE_LENGTH = 42
RUN_NAME = "muzero-connect4-v6"


# ================ Load Params Function ================
def load_params(params, path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            params = flax.serialization.from_bytes(params, f.read())
        print(f"🧠 Successfully loaded brain from {path}!")
    else:
        print(f"⚠️  No saved brain found at {path}. Aborting...")
        sys.exit(1)

    return params


# ================ Run training loop ================
def main(save_params=SAVE_PARAMS, load_checkpoint=True):
    wandb.init(
        project="muzero-connect4",
        id=RUN_NAME,
        resume="allow",
        config={"run_name": RUN_NAME},
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
        init_value=LEARNING_RATE,
        decay_steps=100000,
        alpha=0.1, 
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(5.0), optax.adamw(lr_schedule, weight_decay=1e-4)
    )
    opt_state = optimizer.init(params)

    # Game manager
    game_manager = GameManager(model, params, mcts_num_simulations=NUM_SIMULATIONS)

    # Main training loop
    print("\n========== 🚀 Starting Training (Connect-4) ==========")
    print(f"Training for {NUM_GENERATIONS} generations...\n")

    best_p1_win_rate = -1.0

    for gen in range(NUM_GENERATIONS):
        print(f"===== Generation {gen + 1 + wandb_starting_gen} =====")

        results = []
        start_time = time.time()
        steps_gathered = 0
        games_played = 0

        print(f"Gathering ~{TARGET_STEPS_PER_GENERATION} steps of experience...")
        while steps_gathered < TARGET_STEPS_PER_GENERATION:
            try:
                total_reward, steps, entropy = game_manager.play_single_episode(
                    max_episode_length=MAX_EPISODE_LENGTH,
                )
                results.append((total_reward, steps, entropy))

                steps_gathered += steps
                games_played += 1

            except Exception as e:
                print(f"A game crashed: {e}")

        generation_end_time = time.time()

        if results:
            avg_steps = sum(r[1] for r in results) / len(results)
            avg_entropy = sum(r[2] for r in results) / len(results)

            p1_wins = sum(1 for r in results if r[0] == 1.0)
            p2_wins = sum(1 for r in results if r[0] == -1.0)
            draws = sum(1 for r in results if r[0] == 0.0)

            p1_win_rate = p1_wins / len(results)
            p2_win_rate = p2_wins / len(results)
            draw_rate = draws / len(results)

            print(f"🎮 Played {games_played} games to gather {steps_gathered} steps.")
            print(
                f"🏆 P1 Win Rate: {p1_win_rate:.2%} | P2 Win Rate: {p2_win_rate:.2%} | Draws: {draw_rate:.2%}"
            )
            print(f"⏱️  Avg Steps: {avg_steps:.1f} | 🧠 Entropy: {avg_entropy:.2f}")
            print(f"⏱️  Generation Time: {generation_end_time - start_time:.2f} seconds")

            wandb.log(
                {
                    "Game/Player1_Win_Rate": p1_win_rate,
                    "Game/Player2_Win_Rate": p2_win_rate,
                    "Game/Draw_Rate": draw_rate,
                    "Game/Average_Episode_Length": avg_steps,
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

        if (gen + 1 + wandb_starting_gen) % 10 == 0:
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

        if p1_win_rate > best_p1_win_rate and gen > 10:
            best_p1_win_rate = p1_win_rate
            print(f"📈 NEW HIGH P1 WIN RATE! ({best_p1_win_rate:.2%})")

    print("\nTraining complete!")


if __name__ == "__main__":
    main(load_checkpoint=False)
