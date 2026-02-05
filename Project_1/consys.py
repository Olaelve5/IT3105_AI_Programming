import jax
import jax.numpy as jnp
import jax.random as jrandom
import pickle
from bathtub_plant import Bathtub_Plant
from cournot_plant import Cournot_Plant
from cruise_control_plant import Cruise_Control_Plant
from pid_controller import PID_Controller
from nn_controller import NN_Controller
from plot_utils import plot_mse, plot_params


class CONSYS:
    def __init__(
        self,
        plant,
        nn_controller,
        pid_controller,
        controller_type="PID",
        epochs=100,
        learning_rate=0.01,
        timesteps=60,
    ):
        self.controller_type = controller_type
        self.plant = plant
        self.random_key = jrandom.PRNGKey(27)

        self.loss_history = []
        self.params_history = []

        if controller_type == "PID":
            self.controller = pid_controller
            self.params = {"kp": 0.1, "ki": 0.01, "kd": 0.01}
        else:
            self.controller = nn_controller
            self.params = self.controller.init_params(self.random_key)

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.timesteps = timesteps

    def run_simulation(self, params, controller, noise, plant, timesteps):
        """
        Function to loop over timesteps and calculate the average MSE
        """

        error_history = jnp.zeros(timesteps)
        current_state = self.plant.get_initial_state()
        target = plant.get_target()

        for t in range(timesteps):
            state_value = plant.get_state_value(current_state)

            error = target - state_value
            error_history = error_history.at[t].set(error)

            U = controller.decision(error_history, params, t)
            current_state = plant.update(U, noise[t], current_state)

        return jnp.mean(jnp.square(error_history))

    def train(self):
        """
        Function to loop over epochs to calculate gradients and update params.
        """

        grad_func = jax.value_and_grad(self.run_simulation, argnums=0)

        for i in range(self.epochs):
            self.random_key, subkey = jrandom.split(self.random_key)
            noise_vector = jrandom.uniform(
                subkey,
                shape=(self.timesteps,),
                minval=self.plant.noise_range[0],
                maxval=self.plant.noise_range[1],
            )

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

            self.params = jax.tree_util.tree_map(
                lambda p, g: p - self.learning_rate * g, self.params, clipped_grads
            )

            # Print status every 10 epoch
            if i % 10 == 0:
                print(f"Epoch {i}: MSE = {current_loss:.4f}")

            self.loss_history.append(current_loss)

            if self.controller_type == "PID":
                self.params_history.append(self.params)

        # Save parameters
        if self.controller_type == "PID":
            with open("Project_1/saved_params/pid_params.pkl", "wb") as f:
                pickle.dump(self.params, f)
        else:
            with open("Project_1/saved_params/nn_params.pkl", "wb") as f:
                pickle.dump(self.params, f)

    def test_trained_model(self, test_timesteps=100):
        try:
            if self.controller_type == "PID":
                with open("Project_1/saved_params/pid_params.pkl", "rb") as f:
                    loaded_params = pickle.load(f)
            else:
                with open("Project_1/saved_params/nn_params.pkl", "rb") as f:
                    loaded_params = pickle.load(f)
        except:
            print("No saved params found!")
            return

        test_key = jrandom.PRNGKey(1)
        noise = jrandom.uniform(test_key, (test_timesteps,), minval=-0.01, maxval=0.01)

        loss = self.run_simulation(
            loaded_params,
            self.controller,
            noise,
            self.plant,
            test_timesteps,
        )
        print(f"Test MSE: {loss:.6f}")
