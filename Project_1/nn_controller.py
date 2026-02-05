import jax.random as jrandom
import jax.numpy as jnp
import jax.nn as jnn


class NN_Controller:
    def __init__(
        self,
        layers=[(None, 3), ("relu", 10), (None, 1)],
        input_scale=[10.0, 1200.0, 1.0],
    ):
        self.layers = layers
        self.input_scale = jnp.array(input_scale)

        self.act_funcs = {
            "relu": jnn.relu,
            "tanh": jnp.tanh,
            "sigmoid": jnn.sigmoid,
            None: lambda x: x,
        }

    def init_params(self, key):
        """
        Initializes params based on layers.
        Params should be initialized and updated in CONSYS.
        """

        params = []
        keys = jrandom.split(key, len(self.layers) - 1)

        for i in range(len(self.layers) - 1):
            in_dim = self.layers[i][1]
            out_dim = self.layers[i + 1][1]

            scale = jnp.sqrt(1.0 / in_dim)
            weights = jrandom.uniform(
                keys[i], shape=(in_dim, out_dim), minval=-scale, maxval=scale
            )
            biases = jnp.zeros((out_dim,))

            params.append({"w": weights, "b": biases})

        return params

    def decision(self, error_history, params, timestep):
        """
        Performs a forward pass and returns the decision (U) of the network
        - a single float number
        """

        error = error_history[timestep]
        prev_error = jnp.where(timestep > 0, error_history[timestep - 1], error)
        error_sum = jnp.sum(error_history[: timestep + 1])
        error_change = error - prev_error

        # Scale input to be in range = [-1.0, 1.0] (normalization)
        raw_input = jnp.array([error, error_sum, error_change])
        normalized_input = raw_input / self.input_scale
        x = normalized_input

        # Apply activation
        for i in range(len(params)):
            x = jnp.dot(x, params[i]["w"]) + params[i]["b"]
            act_name = self.layers[i + 1][0]
            x = self.act_funcs[act_name](x)

        return x[0]
