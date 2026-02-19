from game_manager import GameManager
from tetris_env import TetrisEnv
from MuZeroNet import MuZeroNet
import jax
import jax.numpy as jnp
import numpy as np
from train import training_loop
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

# ================ Initialize classes ================

# We need dummy data to initialize the shape of the parameters
model = MuZeroNet(num_actions=NUM_ACTIONS)
dummy_obs = jnp.ones((1, 20, 10, 1))
dummy_act = jnp.array([0])
params = model.init(rng, dummy_obs, dummy_act, method=model.init_params)

# Optimizer
optimizer = optax.adam(LEARNING_RATE)
opt_state = optimizer.init(params)


# Game manager
game_manager = GameManager(model, params, NUM_ACTIONS)


def main():
    loss_history = training_loop(
        model,
        params,
        optimizer,
        opt_state,
        NUM_GENERATIONS,
        GAMES_PER_GENERATION,
        TRAINING_STEPS_PER_GENERATION,
        BATCH_SIZE,
        UNROLL_STEPS,
        game_manager,
    )

    print("\nTraining complete!")
    print("Loss history:", loss_history)


if __name__ == "__main__":
    main()
