from game_manager import GameManager
from MuZeroNet import MuZeroNet
import jax
import jax.numpy as jnp
from train import perform_training_steps
import optax

rng = jax.random.PRNGKey(42)


# ================ Hyperparameters ================
NUM_GENERATIONS = 100
GAMES_PER_GENERATION = 5
TRAINING_STEPS_PER_GENERATION = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 32
UNROLL_STEPS = 5
NUM_ACTIONS = 4


# ================ Run training loop ================
def main():

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

        # 1. Play games to generate data and store in replay buffer
        for _ in range(GAMES_PER_GENERATION):
            game_manager.params = params
            episode_reward = game_manager.play_single_episode(max_episode_length=100)
            rewards_this_gen.append(episode_reward)

        avg_reward = sum(rewards_this_gen) / len(rewards_this_gen)
        reward_history.append(avg_reward)
        print(f"\n🏆 Average Reward this generation: {avg_reward:.2f} \n")

        # 2. Train the Network
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

    print("\nTraining complete!")
    print("Loss history:", loss_history)
    print("Reward history:", reward_history)


if __name__ == "__main__":
    main()
