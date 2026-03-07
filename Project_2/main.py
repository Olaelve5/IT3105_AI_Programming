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

rng = jax.random.PRNGKey(42)

# ================ Hyperparameters ================
NUM_GENERATIONS = 5000
GAMES_PER_GENERATION = 40
TRAINING_STEPS_PER_GENERATION = 150
NUM_SIMULATIONS = 50
LEARNING_RATE = 0.0003
BATCH_SIZE = 64
UNROLL_STEPS = 5
SAVE_PARAMS = True
TD_STEPS = 30


# ================ Run training loop ================
def main(save_params=SAVE_PARAMS):
    wandb.init(project="muzero-tron")

    # Model initialization
    model = MuZeroNet(num_actions=NUM_ACTIONS)
    dummy_obs = jnp.ones((1, BOARD_HEIGHT, BOARD_WIDTH, 3))
    dummy_act = jnp.array([0])
    params = model.init(rng, dummy_obs, dummy_act, method=model.init_params)

    # Optimizer
    optimizer = optax.adamw(LEARNING_RATE, weight_decay=1e-4)
    opt_state = optimizer.init(params)

    # Game manager
    game_manager = GameManager(model, params, mcts_num_simulations=NUM_SIMULATIONS)

    # Main training loop
    print("\n========== 🚀 Starting Training ==========")
    print(f"Training for {NUM_GENERATIONS} generations...\n")

    best_avg_reward = -float("inf")

    for gen in range(NUM_GENERATIONS):
        print(f"===== Generation {gen + 1} =====")
        print(f"Playing {GAMES_PER_GENERATION} games... \n")

        results = []

        for _ in range(GAMES_PER_GENERATION):
            try:
                total_reward, steps, lines, entropy = game_manager.play_single_episode(
                    max_episode_length=400,
                )
                results.append((total_reward, steps, lines, entropy))
            except Exception as e:
                print(f"A game crashed: {e}")

        if results:
            avg_reward = sum(r[0] for r in results) / len(results)
            avg_steps = sum(r[1] for r in results) / len(results)
            avg_score = sum(r[2] for r in results) / len(results)
            avg_entropy = sum(r[3] for r in results) / len(results)
            max_steps = max(r[1] for r in results)
            print(
                f"🏆 Average Reward: {avg_reward:.2f} | ⏱️  Average Steps: {avg_steps:.0f} | 👑 Max: {max_steps}"
            )
            wandb.log(
                {
                    "Game/Average_Total_Reward": avg_reward,
                    "Game/Average_Episode_Length": avg_steps,
                    "Game/Average_Score": avg_score,
                    "MCTS/Average_Entropy": avg_entropy,
                },
                step=gen,
            )

        # Train on the experience
        params, opt_state, _ = perform_training_steps(
            model,
            params,
            optimizer,
            opt_state,
            TRAINING_STEPS_PER_GENERATION,
            BATCH_SIZE,
            UNROLL_STEPS,
            TD_STEPS,
            game_manager,
        )

        game_manager.params = params

        if (gen + 1) % 25 == 0:
            if save_params:
                os.makedirs("Project_2/saved_params", exist_ok=True)
                save_path = f"Project_2/saved_params/{gen + 1}_generations.msgpack"
                with open(save_path, "wb") as f:
                    f.write(flax.serialization.to_bytes(params))
                print(f"💾 Saved params after {gen + 1} generations -> {save_path}")

        if results:
            avg_reward = sum(r[0] for r in results) / len(results)
        else:
            avg_reward = -float("inf")

        # Save the best model based on average reward
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            if save_params:
                os.makedirs("Project_2/saved_params", exist_ok=True)
                best_save_path = "Project_2/saved_params/best_model.msgpack"
                with open(best_save_path, "wb") as f:
                    f.write(flax.serialization.to_bytes(params))
                print(
                    f"🏆 NEW HIGH SCORE! ({best_avg_reward:.2f}) Saved best brain -> {best_save_path}"
                )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
