import jax
import jax.numpy as jnp
import flax.linen as nn
from config import NUM_ACTIONS

# Increased channel capacity
NUM_CHANNELS = 64
NUM_RES_BLOCKS = 3


def min_max_scale(x, tol=1e-5):
    max_val = jnp.max(x, axis=(1, 2), keepdims=True)
    min_val = jnp.min(x, axis=(1, 2), keepdims=True)
    return (x - min_val) / (max_val - min_val + tol)


class ResBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.LayerNorm()(x)  # CRUCIAL for unrolled stability
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.LayerNorm()(x)  # CRUCIAL
        return nn.relu(x + residual)


class RepresentationNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=NUM_CHANNELS, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # Stack multiple ResBlocks for a wider receptive field
        for _ in range(NUM_RES_BLOCKS):
            x = ResBlock(NUM_CHANNELS)(x)

        return min_max_scale(x)


class DynamicsNet(nn.Module):
    num_actions: int = NUM_ACTIONS

    @nn.compact
    def __call__(self, state, action):
        action_one_hot = jax.nn.one_hot(action, self.num_actions)
        action_embedded = nn.Dense(8)(action_one_hot)
        action_embedded = nn.relu(action_embedded)

        action_plane = jnp.tile(
            action_embedded[:, None, None, :], (1, state.shape[1], state.shape[2], 1)
        )

        x = jnp.concatenate([state, action_plane], axis=-1)
        x = nn.Conv(features=NUM_CHANNELS, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # Stack multiple ResBlocks to understand complex game dynamics
        for _ in range(NUM_RES_BLOCKS):
            x = ResBlock(NUM_CHANNELS)(x)

        next_state = min_max_scale(x)

        # --- THE CHANGES START HERE ---

        # 1. Compress the 64 channels down to 4 using a 1x1 convolution
        # This preserves spatial info but drastically reduces the parameter count before flattening
        flat_state = nn.Conv(features=4, kernel_size=(1, 1))(next_state)
        flat_state = nn.LayerNorm()(flat_state)
        flat_state = nn.relu(flat_state)

        # 2. Flatten the spatial dimensions: (Batch, Height, Width, Channels) -> (Batch, H * W * C)
        flat_state = flat_state.reshape((flat_state.shape[0], -1))

        # 3. Pass the flattened spatial data to the dense layers
        hidden = nn.Dense(128)(flat_state)
        hidden = nn.relu(hidden)

        # 4. Zero-initialize the final layers to prevent wild gradients at the start of training
        reward = nn.Dense(1, kernel_init=nn.initializers.zeros)(hidden)
        discount_logits = nn.Dense(1, kernel_init=nn.initializers.zeros)(hidden)
        discount = nn.sigmoid(discount_logits)

        # --- THE CHANGES END HERE ---

        return next_state, reward, discount


class PredictionNet(nn.Module):
    num_actions: int = NUM_ACTIONS

    @nn.compact
    def __call__(self, state):
        # 1. Compress the 64 channels down to 2 or 4 to save parameters
        x = nn.Conv(features=4, kernel_size=(1, 1))(state)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # 2. Flatten the spatial dimensions
        flat = x.reshape((x.shape[0], -1))

        # 3. Pass to dense layers
        hidden = nn.Dense(128)(flat)
        hidden = nn.relu(hidden)

        raw_policy_scores = nn.Dense(self.num_actions)(hidden)
        value = nn.Dense(1)(hidden)

        return raw_policy_scores, value


class MuZeroNet(nn.Module):
    num_actions: int = NUM_ACTIONS

    def setup(self):
        self._representation = RepresentationNet()
        self._dynamics = DynamicsNet()
        self._prediction = PredictionNet()

    def representation(self, observation):
        return self._representation(observation)

    def prediction(self, state):
        return self._prediction(state)

    def dynamics(self, state, action):
        return self._dynamics(state, action)

    def init_params(self, observation, action):
        state, _, _ = self.initial_inference(observation)
        self.recurrent_inference(state, action)

    def initial_inference(self, observation):
        state = self.representation(observation)
        raw_policy_scores, value = self.prediction(state)
        return state, raw_policy_scores, value

    def recurrent_inference(self, state, action):
        next_state, reward, discount = self.dynamics(state, action)
        raw_policy_scores, value = self.prediction(next_state)
        return next_state, reward, discount, raw_policy_scores, value
