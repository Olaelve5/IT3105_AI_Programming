from game_manager import GameManager
from MuZeroNet import MuZeroNet
import jax
import jax.numpy as jnp
from train import perform_training_steps
import optax
import flax.serialization
import os
from config import NUM_ACTIONS, BOARD_WIDTH, BOARD_HEIGHT

rng = jax.random.PRNGKey(42)

# ================ Hyperparameters ================
NUM_GENERATIONS = 500
STEPS_PER_GENERATION = 1000
GAMES_PER_GENERATION = 20
TRAINING_STEPS_PER_GENERATION = 50
LEARNING_RATE = 0.005
BATCH_SIZE = 32
UNROLL_STEPS = 10
SAVE_PARAMS = True


# ================ Run training loop ================
def main(save_params=SAVE_PARAMS):

    # Model initialization
    model = MuZeroNet(num_actions=NUM_ACTIONS)
    dummy_obs = jnp.ones((1, BOARD_HEIGHT, BOARD_WIDTH, 1))
    dummy_act = jnp.array([0])
    params = model.init(rng, dummy_obs, dummy_act, method=model.init_params)

    # Optimizer
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)

    # Game manager
    game_manager = GameManager(model, params)

    # Main training loop
    print("\n========== 🚀 Starting Training ==========")
    print(f"Training for {NUM_GENERATIONS} generations...\n")

    loss_history = []
    reward_history = []

    for gen in range(NUM_GENERATIONS):
        print(f"===== Generation {gen + 1} =====")
        print(f"Playing {GAMES_PER_GENERATION} games... \n")

        rewards_this_gen = []
        steps_per_game = []

        # Play games to gather experience
        for _ in range(GAMES_PER_GENERATION):
            game_manager.params = params
            episode_reward, steps_taken = game_manager.play_single_episode(
                max_episode_length=STEPS_PER_GENERATION
            )
            rewards_this_gen.append(episode_reward)
            steps_per_game.append(steps_taken)

        avg_reward = sum(rewards_this_gen) / len(rewards_this_gen)
        avg_steps = sum(steps_per_game) / len(steps_per_game)
        reward_history.append(avg_reward)

        print(f"\n🏆 Average Reward this generation: {avg_reward:.2f}")
        print(f"⏱️  Average steps per game this generation: {avg_steps:.0f} \n")

        if len(reward_history) > 1:
            reward_change = avg_reward - reward_history[-2]
            if reward_change > 0:
                print(
                    f"📈 Reward increased by {reward_change:.2f} from last generation!\n"
                )
            else:
                print(
                    f"📉 Reward decreased by {abs(reward_change):.2f} from last generation.\n"
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
            if (gen + 1) % 5 == 0:
                os.makedirs("Project_2/saved_params", exist_ok=True)
                save_path = f"Project_2/saved_params/{gen + 1}_generations.msgpack"
                with open(save_path, "wb") as f:
                    f.write(flax.serialization.to_bytes(params))
                print(f"💾 Saved params after {gen + 1} generations -> {save_path}")

    print("\nTraining complete!")
    print("Loss history:", loss_history)
    print("Reward history:", reward_history)


if __name__ == "__main__":
    main()
