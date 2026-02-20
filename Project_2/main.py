from game_manager import GameManager
from MuZeroNet import MuZeroNet
import jax
import jax.numpy as jnp
from train import perform_training_steps
import optax
import flax.serialization
import os

rng = jax.random.PRNGKey(42)

# ================ Hyperparameters ================
NUM_GENERATIONS = 200
STEPS_PER_GENERATION = 1000
GAMES_PER_GENERATION = 5
TRAINING_STEPS_PER_GENERATION = 50
LEARNING_RATE = 0.005
BATCH_SIZE = 32
UNROLL_STEPS = 10
NUM_ACTIONS = 4
SAVE_PARAMS = True


# ================ Run training loop ================
def main(save_params=SAVE_PARAMS):

    # Model initialization
    model = MuZeroNet(num_actions=NUM_ACTIONS)
    dummy_obs = jnp.ones((1, 20, 10, 1))
    dummy_act = jnp.array([0])
    params = model.init(rng, dummy_obs, dummy_act, method=model.init_params)

    # Optimizer
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)

    # Game manager
    game_manager = GameManager(model, params, NUM_ACTIONS)

    # Main training loop
    print("\n 🚀 ========== Starting Training ==========")
    print(f"Training for {NUM_GENERATIONS} generations...\n")

    loss_history = []
    reward_history = []

    for gen in range(NUM_GENERATIONS):
        print(f"===== Generation {gen + 1} =====")
        print(f"Playing {GAMES_PER_GENERATION} games... \n")

        rewards_this_gen = []

        # Play games to gather experience
        for i in range(GAMES_PER_GENERATION):
            game_manager.params = params
            episode_reward = game_manager.play_single_episode(
                max_episode_length=STEPS_PER_GENERATION
            )
            rewards_this_gen.append(episode_reward)

        avg_reward = sum(rewards_this_gen) / len(rewards_this_gen)
        reward_history.append(avg_reward)
        print(f"\n🏆 Average Reward this generation: {avg_reward:.2f} \n")

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
