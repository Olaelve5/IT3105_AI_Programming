import jax.numpy as jnp


class PID_Controller:
    def decision(self, error_history, params, timestep):
        error = error_history[timestep]

        # Handles t = 0 by setting prev_error = error, instead of error_history[-1]
        prev_error = jnp.where(timestep > 0, error_history[timestep - 1], error)

        error_sum = jnp.sum(error_history[: timestep + 1])
        error_change = error - prev_error

        U = (
            (params["kp"] * error)
            + (params["ki"] * error_sum)
            + (params["kd"] * error_change)
        )

        return U
