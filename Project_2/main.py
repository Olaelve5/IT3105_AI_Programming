from tetris_env import TetrisEnv
from MuZeroNet import MuZeroNet
import jax
import jax.numpy as jnp
import numpy as np

key = jax.random.PRNGKey(42)

BOARD_HEIGHT = 20
BOARD_WIDTH = 10
NUM_ACTIONS = 4  # No-op, Left, Right, Rotate

env = TetrisEnv()
observation, info = env.reset()

model = MuZeroNet(num_actions=NUM_ACTIONS)

# Dummy input needed to initialize the model parameters
dummy_input = jnp.zeros((1, BOARD_HEIGHT, BOARD_WIDTH, 1))
dummy_action = jnp.array([0])
params = model.init(key, dummy_input, dummy_action, method=model.init_params)
print("✅ Model initialized with random weights.")


# 6. Test: Run the actual game observation through the network
# We must reshape the real observation to match the dummy data's 4D shape
real_obs_reshaped = jnp.array(observation).reshape(1, BOARD_HEIGHT, BOARD_WIDTH, 1)

# We use 'apply' to pass the weights (params) and the data into the network
state, policy_logits, value = model.apply(
    params, real_obs_reshaped, method=model.initial_inference
)

print(f"Abstract State Shape: {state.shape}")
print(f"Policy (Move Hunches): {policy_logits}")
print(f"Board Value: {value}")


# --- RECURRENT INFERENCE TEST ---
print("\n--- Testing Imagination (Recurrent Inference) ---")

# 1. Pick an action to imagine (e.g., Action 1 = Move Right)
# We need to wrap it in a JAX array.
action_to_imagine = jnp.array([1])

# 2. Run the Recurrent Inference
# Input: The abstract state we got from the previous step + The action we want to test
next_state, projected_reward, next_policy, next_value = model.apply(
    params, state, action_to_imagine, method=model.recurrent_inference
)

print(f"Imagined Next State: {next_state.shape}")  # Should be (1, 10, 5, 64)
print(
    f"Imagined Reward: {projected_reward}"
)  # Should be a single number (e.g., [[0.02]])
print(f"Next Policy: {next_policy}")  # Predictions for the *next* turn
print(f"Next Value: {next_value}")
