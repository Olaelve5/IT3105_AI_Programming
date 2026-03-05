import jax
import jax.numpy as jnp
import flax.linen as nn
from config import NUM_ACTIONS

# Doubled to give the network more "memory"
NUM_CHANNELS = 64


class ResBlock(nn.Module):
    """A standard Residual Block to help the network think deeper without losing information."""

    features: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        return nn.relu(x + residual)


class RepresentationNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=NUM_CHANNELS, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)

        # Two residual blocks gives a much wider receptive field to understand the board
        x = ResBlock(NUM_CHANNELS)(x)
        x = ResBlock(NUM_CHANNELS)(x)

        return x


class DynamicsNet(nn.Module):
    num_actions: int = NUM_ACTIONS

    @nn.compact
    def __call__(self, state, action):
        action_one_hot = jax.nn.one_hot(action, self.num_actions)
        action_embedded = nn.Dense(16)(action_one_hot)
        action_embedded = nn.relu(action_embedded)

        action_plane = jnp.tile(
            action_embedded[:, None, None, :], (1, state.shape[1], state.shape[2], 1)
        )

        x = jnp.concatenate([state, action_plane], axis=-1)

        # Compress the concatenated state+action back to NUM_CHANNELS
        x = nn.Conv(features=NUM_CHANNELS, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)

        # Use ResBlocks to calculate the complex physics of the next state!
        next_state = ResBlock(NUM_CHANNELS)(x)
        next_state = ResBlock(NUM_CHANNELS)(next_state)

        # Flatten the next state for the dense reward heads
        flat_x = next_state.reshape((next_state.shape[0], -1))

        hidden = nn.Dense(256)(flat_x)
        hidden = nn.relu(hidden)

        reward = nn.Dense(1)(hidden)
        discount_logits = nn.Dense(1)(hidden)
        discount = nn.sigmoid(discount_logits)

        return next_state, reward, discount


class PredictionNet(nn.Module):
    num_actions: int = NUM_ACTIONS

    @nn.compact
    def __call__(self, state):
        flat = state.reshape((state.shape[0], -1))

        # Added a bit more capacity to the policy head
        hidden = nn.Dense(512)(flat)
        hidden = nn.relu(hidden)
        hidden = nn.Dense(256)(hidden)
        hidden = nn.relu(hidden)
        raw_policy_scores = nn.Dense(self.num_actions)(hidden)

        # Value head
        value_hidden = nn.Dense(256)(flat)
        value_hidden = nn.relu(value_hidden)
        value = nn.Dense(1)(value_hidden)

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
