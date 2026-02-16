import jax
import jax.numpy as jnp
import flax.linen as nn

"""
This file defines the neural network architecture for MuZero.

@nn.compact is a Flax decorator that allows us to define layers inline without a separate setup() method.
This way we don't have to initialize layers in a constructor -> we just pass the input through the layers directly.
"""


# A hyperparameter to control the size of our abstract state representation
NUM_CHANNELS = 64


class RepresentationNet(nn.Module):
    """
    The Representation Network takes the raw game state (20x10 grid) and
    transforms it into an abstract "state" representation.

    It uses the ResNet architecture with convolutional layers to ensure information is not lost.

    Returns a tensor of shape (Batch, 10, 5, NUM_CHANNELS) which is a compressed version of the game state.
    """

    @nn.compact
    def __call__(self, x):
        # Downsample: strides=(2, 2) reduces 20x10 to 10x5
        x = nn.Conv(
            features=NUM_CHANNELS, kernel_size=(3, 3), strides=(2, 2), padding="SAME"
        )(x)
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

    num_actions: int

    @nn.compact
    def __call__(self, state, action):
        action_one_hot = jax.nn.one_hot(action, self.num_actions)

        # Transform the action into a plane so it can be used in layers
        action_plane = jnp.tile(
            action_one_hot[:, None, None, :], (1, state.shape[1], state.shape[2], 1)
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
        reward = nn.Dense(1)(flat_x)

        return next_state, reward


class PredictionNet(nn.Module):
    """
    The Prediction Network takes the abstract state and predicts both the
    action probabilities and the value (expected reward).
    """

    num_actions: int

    @nn.compact
    def __call__(self, state):
        flat = state.reshape((state.shape[0], -1))

        # Policy (Move Probabilities)
        action_probs = nn.Dense(self.num_actions)(flat)

        # Value (Win probability or Score estimate)
        value = nn.Dense(1)(flat)

        return action_probs, value


class MuZeroNet(nn.Module):
    num_actions: int

    def setup(self):
        self.representation = RepresentationNet()
        self.dynamics = DynamicsNet(self.num_actions)
        self.prediction = PredictionNet(self.num_actions)

    def initial_inference(self, observation):
        # Used once, at the root of the search tree, to get the initial state and predictions
        state = self.representation(observation)
        action_probs, value = self.prediction(state)
        return state, action_probs, value

    def recurrent_inference(self, state, action):
        # Used when simulating future steps in the search tree
        next_state, reward = self.dynamics(state, action)
        action_probs, value = self.prediction(next_state)
        return next_state, reward, action_probs, value
