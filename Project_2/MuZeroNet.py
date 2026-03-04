import jax
import jax.numpy as jnp
import flax.linen as nn
from config import NUM_ACTIONS

"""
This file defines the neural network architecture for MuZero.

@nn.compact is a Flax decorator that allows us to define layers inline without a separate setup() method.
This way we don't have to initialize layers in a constructor -> we just pass the input through the layers directly.
"""


# A hyperparameter to control the size of our abstract state representation
NUM_CHANNELS = 32


class RepresentationNet(nn.Module):
    """
    The Representation Network takes the raw game state (20x10 grid) and
    transforms it into an abstract "state" representation.

    It uses the ResNet architecture with convolutional layers to ensure information is not lost.

    Returns a tensor of shape (Batch, 10, 5, NUM_CHANNELS) which is a compressed version of the game state.
    """

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=NUM_CHANNELS, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)

        # Store the original game state as a residual
        residual = x

        x = nn.Conv(features=NUM_CHANNELS, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=NUM_CHANNELS, kernel_size=(3, 3), padding="SAME")(x)
        x = x + residual
        x = nn.relu(x)

        return x


class DynamicsNet(nn.Module):
    """
    The Dynamics Network takes the current abstract state and an action, and predicts the next abstract state and the reward.
    """

    num_actions: int = NUM_ACTIONS

    @nn.compact
    def __call__(self, state, action):
        action_one_hot = jax.nn.one_hot(action, self.num_actions)

        action_embedded = nn.Dense(16)(action_one_hot)
        action_embedded = nn.relu(action_embedded)

        # Transform the action into a plane so it can be used in layers
        action_plane = jnp.tile(
            action_embedded[:, None, None, :], (1, state.shape[1], state.shape[2], 1)
        )

        # Concatenate the state and action along the channel dimension
        x = jnp.concatenate([state, action_plane], axis=-1)

        # Next state prediction: use NUM_CHANNELS filters to maintain the same abstract state size
        next_state = nn.Conv(features=NUM_CHANNELS, kernel_size=(3, 3), padding="SAME")(
            x
        )
        next_state = nn.relu(next_state)

        # Flatten the next state and pass through a dense layer to predict the reward
        flat_x = x.reshape((x.shape[0], -1))
        hidden = nn.Dense(256)(flat_x)
        hidden = nn.relu(hidden)

        # Predict the reward
        reward = nn.Dense(1)(reward)

        # Predict the discount (essentially whether the game is over)
        discount_logits = nn.Dense(1)(hidden)
        discount = nn.sigmoid(discount_logits)

        return next_state, reward, discount


class PredictionNet(nn.Module):
    """
    The Prediction Network takes the abstract state and predicts both the
    action probabilities and the value (expected reward).
    """

    num_actions: int = NUM_ACTIONS

    @nn.compact
    def __call__(self, state):
        flat = state.reshape((state.shape[0], -1))

        hidden = nn.Dense(1024)(flat)
        hidden = nn.relu(hidden)

        # Head for policy scores
        raw_policy_scores = nn.Dense(self.num_actions)(hidden)

        # Head for values
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

    # Wrappers to make the networks available to Flax's apply method
    def representation(self, observation):
        return self._representation(observation)

    def prediction(self, state):
        return self._prediction(state)

    def dynamics(self, state, action):
        return self._dynamics(state, action)

    def init_params(self, observation, action):
        # Used for initializing the model parameters with dummy data
        state, _, _ = self.initial_inference(observation)
        self.recurrent_inference(state, action)

    def initial_inference(self, observation):
        # Used once, at the root of the search tree, to get the initial state and predictions
        state = self.representation(observation)
        raw_policy_scores, value = self.prediction(state)
        return state, raw_policy_scores, value

    def recurrent_inference(self, state, action):
        # Used when simulating future steps in the search tree
        next_state, reward, discount = self.dynamics(state, action)
        raw_policy_scores, value = self.prediction(next_state)
        return next_state, reward, discount, raw_policy_scores, value
