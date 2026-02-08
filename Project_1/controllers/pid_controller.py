import jax.numpy as jnp


class PID_Controller:
    """
    The PID controller only neeeds to implement the decision function,
    as the CONSYS class handles the rest of the training loop and parameter updates.
    """

    def decision(self, error_history, params, timestep):
        """
        Decision function for the PID controller.
        Calculates the control variable U based on the error history and PID parameters.
        """

        error = error_history[timestep]

        # Handles t = 0 by setting prev_error = error, instead of error_history[-1]
        prev_error = jnp.where(timestep > 0, error_history[timestep - 1], error)

        error_sum = jnp.sum(error_history[: timestep + 1])
        error_change = error - prev_error

        # U = Kp * error + Ki * error_sum + Kd * error_change
        U = (
            (params["kp"] * error)
            + (params["ki"] * error_sum)
            + (params["kd"] * error_change)
        )

        return U
