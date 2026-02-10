from abc import ABC, abstractmethod
import jax.numpy as jnp


class Plant(ABC):
    """
    Abstract Base Class for all Control Systems Plants.
    All plants should inherit from this class and implement the required methods.
    """

    @abstractmethod
    def update(self, U, D, state):
        """
        Calculates the next state based on current state, control signal (U), and noise (D).

        Args:
            U (float): Control signal from the controller.
            D (float): Disturbance/Noise signal.
            state (jnp.array): The current state vector of the system.

        Returns:
            jnp.array: The updated state vector for the next timestep.
        """
        pass

    @abstractmethod
    def get_initial_state(self):
        """
        Returns the starting state vector for the simulation.

        Returns:
            jnp.array: A JAX array representing the initial state.
            For example in Cournot, it could be jnp.array([q1, q2]).
        """
        pass

    @abstractmethod
    def get_target(self):
        """
        Returns the target value that the controller is trying to achieve.

        Returns:
            float: The target value (e.g., target velocity, target water level).
        """
        pass

    @abstractmethod
    def get_state_value(self, state):
        """
        Args:
            state (jnp.array): The full state vector.

        Returns:
            float: The value used for error calculation (e.g., current velocity).
        """
        pass
