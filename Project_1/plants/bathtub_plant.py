import jax.numpy as jnp
from plants.plant import Plant


class Bathtub_Plant(Plant):
    def __init__(self, A=10.0, C=0.1, H0=1.0, G=9.8, noise_range=(-0.01, 0.01)):
        self.A = A
        self.C = C
        self.H0 = H0
        self.G = G
        self.target = H0
        self.noise_range = noise_range

    def update(self, U, D, state):
        """
        This function takes a single update step and returns the
        new water level value for the plant.

        U is controller output
        C is the cross-sectional area of the drain
        D is random noise
        V is velocity
        Q is flow rate
        B is change in bathtub volume
        """

        current_h = state[0]

        V = jnp.sqrt(2 * self.G * current_h)
        Q = self.C * V
        B = U + D - Q

        new_h = current_h + (B / self.A)

        # This is needed to prevent the water level from going negative,
        # which would cause issues with the square root in the velocity calculation.
        new_h = jnp.maximum(new_h, 0.0001)

        return jnp.array([new_h])

    def get_initial_state(self):
        return jnp.array([self.H0])

    def get_state_value(self, state):
        return state[0]

    def get_target(self):
        return self.target
