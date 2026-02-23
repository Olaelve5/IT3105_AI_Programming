from game_manager import GameManager
from MuZeroNet import MuZeroNet
import jax
import jax.numpy as jnp
from train import perform_training_steps
import optax
import flax.serialization
import os
from config import NUM_ACTIONS, BOARD_WIDTH, BOARD_HEIGHT
import wandb

print("JAX is using:", jax.devices())

rng = jax.random.PRNGKey(42)

# ================ Hyperparameters ================
NUM_GENERATIONS = 1000
STEPS_PER_GENERATION = 1000
GAMES_PER_GENERATION = 30
TRAINING_STEPS_PER_GENERATION = 200
LEARNING_RATE = 0.0002
BATCH_SIZE = 128
UNROLL_STEPS = 5
SAVE_PARAMS = True


# ================ Run training loop ================
def main(save_params=SAVE_PARAMS):
    wandb.init(project="muzero-tetris")

    # Model initialization
    model = MuZeroNet(num_actions=NUM_ACTIONS)
    dummy_obs = jnp.ones((1, BOARD_HEIGHT, BOARD_WIDTH, 2))
    dummy_act = jnp.array([0])
    params = model.init(rng, dummy_obs, dummy_act, method=model.init_params)

    # Optimizer
    optimizer = optax.adamw(LEARNING_RATE, weight_decay=1e-4)
    opt_state = optimizer.init(params)

    # Game manager
    game_manager = GameManager(model, params)

    # Main training loop
    print("\n========== 🚀 Starting Training ==========")
    print(f"Training for {NUM_GENERATIONS} generations...\n")

    loss_history = []
    reward_history = []
    best_steps_record = 0

    for gen in range(NUM_GENERATIONS):
        print(f"===== Generation {gen + 1} =====")
        print(f"Playing {GAMES_PER_GENERATION} games... \n")

        results = []

        # Parallell games to speed up data collection
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:

            futures = [
                executor.submit(
                    game_manager.play_single_episode,
                    max_episode_length=500,
                )
                for _ in range(GAMES_PER_GENERATION)
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    total_reward, steps = future.result()
                    results.append((total_reward, steps))
                except Exception as e:
                    print(f"A game crashed: {e}")

        if results:
            avg_reward = sum(r[0] for r in results) / len(results)
            avg_steps = sum(r[1] for r in results) / len(results)
            max_steps = max(r[1] for r in results)
            print(
                f"🏆 Average Reward: {avg_reward:.2f} | ⏱️  Average Steps: {avg_steps:.0f} | 👑 Max: {max_steps}"
            )

        # Train on the experience
        params, opt_state, avg_loss = perform_training_steps(
            model,
            params,
            optimizer,
            opt_state,
            TRAINING_STEPS_PER_GENERATION,
            BATCH_SIZE,
            UNROLL_STEPS,
            game_manager,
        )

        if avg_loss is not None:
            loss_history.append(avg_loss)

        if save_params:
            if (gen + 1) % 25 == 0:
                os.makedirs("Project_2/saved_params", exist_ok=True)
                save_path = f"Project_2/saved_params/{gen + 1}_generations.msgpack"
                with open(save_path, "wb") as f:
                    f.write(flax.serialization.to_bytes(params))
                print(f"💾 Saved params after {gen + 1} generations -> {save_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
