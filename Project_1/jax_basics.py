import jax.numpy as jnp
import jax.nn as jnn
from jax import grad
from jax import jit
from jax import vmap


def test(x: int):
    return 3 * (x**2) + 2 * x + 1


derivative_test = grad(test)


def test_2(state: int, input: int):
    return state + input


compiled_test_2 = jit(test_2)


apply_noise = vmap(test_2, in_axes=(0, None))
noises = jnp.array([0.1, -0.2, 0.3, -0.1, 0.05])


import jax.random as jrandom

key = jrandom.PRNGKey(10)
key, subkey1, subkey2 = jrandom.split(key, num=3)
networkkey_1, networkkey_2 = jrandom.split(subkey1, num=2)


weights_1 = jrandom.uniform(subkey1, shape=(3, 8), minval=-0.1, maxval=0.1)
biases_1 = jrandom.uniform(subkey2, shape=(8,), minval=-0.1, maxval=0.1)
weights_2 = jrandom.uniform(subkey1, shape=(8, 1), minval=-0.1, maxval=0.1)
biases_2 = jrandom.uniform(subkey2, shape=(8,), minval=-0.1, maxval=0.1)

params = {
    "layer_1": {"w": weights_1, "b": biases_1},
    "layer_2": {"w": weights_2, "b": biases_2},
}

activations = {"sigmoid": jnn.sigmoid, "tanh": jnp.tanh, "relu": jnn.relu}


def forward_pass(params: dict, input_data: jnp.array, act_key: str):
    x = input_data  # Start with the initial 3 error terms [cite: 73]

    # Get a list of layers (e.g., ['layer1', 'layer2', 'output_layer'])
    # We sort them to ensure they run in the correct order
    layer_names = sorted(params.keys())

    for i, name in enumerate(layer_names):
        layer_params = params[name]

        # 1. Matrix Multiplication and Bias
        # Using dot(W, x) or dot(x, W) depends on how you shaped your weights! [cite: 26]
        x = jnp.dot(x, layer_params["w"]) + layer_params["b"]

        # 2. Activation (Only for HIDDEN layers)
        # The very last layer (the output) should NOT have an activation
        # because the control signal U needs to be a raw value. [cite: 19, 26]
        if i < len(layer_names) - 1:
            x = activations[act_key](x)

    return x  # This is now your Control Signal U


def my_loss_fn(params, x_input):
    prediction = forward_pass(params, x_input, "relu")
    return jnp.mean(prediction**2)  # This is your MSE


grad_fn = grad(my_loss_fn)

input_data = jnp.array([2.0, 1.3, -1.2])

gradients = grad_fn(params, input_data)

print(gradients)
