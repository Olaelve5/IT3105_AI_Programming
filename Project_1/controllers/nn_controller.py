import jax.random as jrandom
import jax.numpy as jnp
import jax.nn as jnn


class NN_Controller:
    def __init__(
        self,
        layers=[(None, 3), ("relu", 10), (None, 1)],
        input_scale=[1.0, 1.0, 1.0],
        weight_init_range=None,
        bias_init_range=None,
    ):
        self.layers = layers

        if input_scale is None:
            self.input_scale = jnp.array([1.0, 1.0, 1.0])
        else:
            self.input_scale = jnp.array(input_scale)

        self.weight_init_range = weight_init_range
        self.bias_init_range = bias_init_range

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

            if self.weight_init_range is None:
                # Xavier initialization
                # Initialize weights in range [-sqrt(1/in_dim), sqrt(1/in_dim)]
                scale = jnp.sqrt(1.0 / in_dim)
                w_min, w_max = -scale, scale
            else:
                # If provided, use the specified weight initialization range
                w_min, w_max = self.weight_init_range

            weights = jrandom.uniform(
                keys[i], shape=(in_dim, out_dim), minval=w_min, maxval=w_max
            )

            # Random initialization if provided, otherwise zero
            if self.bias_init_range is None:
                biases = jnp.zeros((out_dim,))
            else:
                # Need to split key again for bias initialization
                key_b = jrandom.split(keys[i])[0]
                biases = jrandom.uniform(
                    key_b,
                    shape=(out_dim,),
                    minval=self.bias_init_range[0],
                    maxval=self.bias_init_range[1],
                )

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

        # Scale input if input_scale is provided, otherwise use raw error values
        raw_input = jnp.array([error, error_sum, error_change])
        normalized_input = raw_input / self.input_scale
        x = normalized_input

        # Apply activation
        for i in range(len(params)):
            x = jnp.dot(x, params[i]["w"]) + params[i]["b"]
            act_name = self.layers[i + 1][0]
            x = self.act_funcs[act_name](x)

        return x[0]
