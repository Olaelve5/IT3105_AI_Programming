import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial
from flax.serialization import to_bytes, from_bytes


class MicroAgentNetwork(nn.Module):
    """A CNN that outputs Q-values for each of the 5 Tetris actions."""

    action_dim: int = 5

    @nn.compact
    def __call__(self, x):
        # x shape expects: (batch_size, 20, 10, 3)

        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)

        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        q_values = nn.Dense(features=self.action_dim)(x)

        return q_values


class MicroAgent:
    def __init__(self, action_dim=5, learning_rate=1e-4):
        self.action_dim = action_dim
        self.network = MicroAgentNetwork(action_dim=self.action_dim)

        self.rng = jax.random.PRNGKey(42)
        self.rng, init_rng = jax.random.split(self.rng)

        # 2 sets of parameters, one for the online network and one for the target network
        dummy_obs = jnp.zeros((1, 20, 10, 3))
        self.online_params = self.network.init(init_rng, dummy_obs)
        self.target_params = self.online_params

        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.online_params)

        # Hyperparameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.tau = 0.005

    def choose_action(self, observation, evaluate=False):
        """Choose an action using epsilon-greedy policy. If evaluate is True, always choose the best action."""

        if not evaluate and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        obs_batched = jnp.expand_dims(observation, axis=0)

        q_values = self.network.apply(self.online_params, obs_batched)
        best_action = jnp.argmax(q_values, axis=-1)[0]

        return int(best_action)

    def update_target(self):
        """
        Update the target network using a soft update (Polyak averaging) of the online network parameters.
        """

        self.target_params = jax.tree_util.tree_map(
            lambda target_weight, online_weight: self.tau * online_weight
            + (1.0 - self.tau) * target_weight,
            self.target_params,
            self.online_params,
        )

    def learn(self, replay_buffer, batch_size=64):
        """Samples a batch from memory and updates the neural network."""
        if len(replay_buffer) < batch_size:
            return None

        batch = replay_buffer.sample(batch_size)

        self.online_params, self.opt_state, loss = train_step(
            self.network,
            self.optimizer,
            self.online_params,
            self.target_params,
            self.opt_state,
            batch,
            self.gamma,
        )

        return loss

    def save_model(
        self, filepath="micro_agent/saved_params/micro_agent_weights.msgpack"
    ):
        """Serializes the online network parameters and saves them to a file."""
        bytes_output = to_bytes(self.online_params)
        with open(filepath, "wb") as f:
            f.write(bytes_output)
        print(f"💾 Model weights saved to {filepath}")

    def load_model(
        self, filepath="micro_agent/saved_params/micro_agent_weights.msgpack"
    ):
        """Loads the weights from a file and syncs both networks."""
        with open(filepath, "rb") as f:
            bytes_input = f.read()

        self.online_params = from_bytes(self.online_params, bytes_input)
        self.target_params = self.online_params
        print(f"📂 Model weights loaded from {filepath}")


@partial(jax.jit, static_argnums=(0, 1))
def train_step(
    network, optimizer, online_params, target_params, opt_state, batch, discount
):
    states, actions, rewards, next_states, dones = batch

    def loss_fn(params):
        # Ask the network for the value it gave its chosen action
        q_values = network.apply(params, states)
        q_action = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze()

        # Online network
        next_q_online = network.apply(params, next_states)
        next_actions = jnp.argmax(next_q_online, axis=-1)

        # Target network
        next_q_target = network.apply(target_params, next_states)
        next_q_val = jnp.take_along_axis(
            next_q_target, next_actions[:, None], axis=-1
        ).squeeze()

        # Bellman target
        target_q = rewards + discount * next_q_val * (1.0 - dones)

        loss = jnp.mean(optax.huber_loss(q_action, target_q, delta=1.0))
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(online_params)

    updates, new_opt_state = optimizer.update(grads, opt_state, online_params)
    new_online_params = optax.apply_updates(online_params, updates)

    return new_online_params, new_opt_state, loss
