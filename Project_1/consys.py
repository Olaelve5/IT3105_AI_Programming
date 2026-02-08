import jax
import jax.numpy as jnp
import jax.random as jrandom
from controllers.nn_controller import NN_Controller
from controllers.pid_controller import PID_Controller
from plants.plant import Plant


class CONSYS:
    def __init__(
        self,
        plant: Plant,
        nn_controller: NN_Controller,
        pid_controller: PID_Controller,
        pid_params={"kp": 0.1, "ki": 0.01, "kd": 0.01},
        controller_type="PID",
        epochs=100,
        learning_rate=0.01,
        timesteps=100,
    ):
        self.controller_type = controller_type
        self.plant = plant
        self.random_key = jrandom.PRNGKey(27)

        self.loss_history = []
        self.params_history = []

        if controller_type == "PID":
            self.controller = pid_controller
            self.params = pid_params
        else:
            self.controller = nn_controller
            self.params = self.controller.init_params(self.random_key)

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.timesteps = timesteps

    def run_simulation(self, params, controller, noise, plant, timesteps):
        """
        Function to loop over timesteps and calculate the average MSE.
        This is one epoch of the run.

        For each timestep, it calculates the error, gets the controller's decision,
        updates the plant state, and stores the error for MSE calculation.
        """

        error_history = jnp.zeros(timesteps)
        current_state = self.plant.get_initial_state()
        target = plant.get_target()

        for t in range(timesteps):
            state_value = plant.get_state_value(current_state)

            error = target - state_value
            error_history = error_history.at[t].set(error)

            U = controller.decision(error_history, params, t)
            U = jnp.clip(U, -1.0, 1.0)

            current_state = plant.update(U, noise[t], current_state)

        return jnp.mean(jnp.square(error_history))

    def train(self):
        """
        Function to loop over epochs to calculate gradients and update params.
        It uses JAX's value_and_grad to compute the loss and its gradients.

        The gradients are clipped to prevent exploding gradients, and then the parameters are updated using gradient descent.
        """

        # value_and_grad works like grad, but also returns the value of the function (loss).
        grad_func = jax.value_and_grad(self.run_simulation, argnums=0)

        for i in range(self.epochs):
            self.random_key, subkey = jrandom.split(self.random_key)

            # Generate a fresh set of noise for an entire epoch
            noise_vector = jrandom.uniform(
                subkey,
                shape=(self.timesteps,),
                minval=self.plant.noise_range[0],
                maxval=self.plant.noise_range[1],
            )

            # Compute loss and gradients
            current_loss, gradients = grad_func(
                self.params,
                self.controller,
                noise_vector,
                self.plant,
                self.timesteps,
            )

            clipped_grads = jax.tree_util.tree_map(
                lambda g: jnp.clip(g, -1.0, 1.0), gradients
            )

            # This is the learning step. It updates the parameters by applying the function
            # p - learning_rate * g for each parameter p and its corresponding gradient g.
            self.params = jax.tree_util.tree_map(
                lambda p, g: p - self.learning_rate * g, self.params, clipped_grads
            )

            # Print status every 10 epoch
            if i % 10 == 0:
                print(f"Epoch {i}: MSE = {current_loss:.8f}")

            self.loss_history.append(current_loss)

            if self.controller_type == "PID":
                self.params_history.append(self.params)
