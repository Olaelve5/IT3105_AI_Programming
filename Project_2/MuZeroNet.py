import jax
import jax.numpy as jnp
import flax.linen as nn
from config import NUM_ACTIONS

NUM_CHANNELS = 32
NUM_RES_BLOCKS = 5


def min_max_scale(x, tol=1e-5):
    max_val = jnp.max(x, axis=(1, 2, 3), keepdims=True)
    min_val = jnp.min(x, axis=(1, 2, 3), keepdims=True)
    return (x - min_val) / (max_val - min_val + tol)


class ResBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.LayerNorm()(x)
        return nn.relu(x + residual)


class RepresentationNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=NUM_CHANNELS, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

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

        for _ in range(NUM_RES_BLOCKS):
            x = ResBlock(NUM_CHANNELS)(x)

        next_state = min_max_scale(x)
        batch_size = next_state.shape[0]

        rd_conv = nn.Conv(features=16, kernel_size=(1, 1))(next_state)
        rd_conv = nn.relu(rd_conv)
        rd_flat = rd_conv.reshape((batch_size, -1))

        hidden = nn.Dense(256)(rd_flat)
        hidden = nn.relu(hidden)

        reward = nn.Dense(1)(hidden)
        discount_logits = nn.Dense(1)(hidden)
        discount = nn.sigmoid(discount_logits)

        return next_state, reward, discount


class PredictionNet(nn.Module):
    num_actions: int = NUM_ACTIONS

    @nn.compact
    def __call__(self, state):
        batch_size = state.shape[0]

        # --- POLICY HEAD ---
        p_conv = nn.Conv(features=32, kernel_size=(1, 1))(state)
        p_conv = nn.relu(p_conv)
        p_flat = p_conv.reshape((batch_size, -1))

        p_hidden = nn.Dense(256)(p_flat)
        p_hidden = nn.relu(p_hidden)
        raw_policy_scores = nn.Dense(self.num_actions)(p_hidden)

        # --- VALUE HEAD ---
        v_conv = nn.Conv(features=16, kernel_size=(1, 1))(state)
        v_conv = nn.relu(v_conv)
        v_flat = v_conv.reshape((batch_size, -1))

        v_hidden = nn.Dense(256)(v_flat)
        v_hidden = nn.relu(v_hidden)
        value = nn.Dense(1)(v_hidden)

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
