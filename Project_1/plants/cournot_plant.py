import jax.numpy as jnp
from plants.plant import Plant


class Cournot_Plant(Plant):
    def __init__(
        self,
        initial_state=[0.0, 1.0],
        cost_margin=0.1,
        max_price=5.0,
        noise_range=(-0.01, 0.01),
        target=3.0,
    ):
        self.initial_state = initial_state  # [q1, q2]
        self.cost_margin = cost_margin
        self.max_price = max_price
        self.noise_range = noise_range
        self.target = target

    def update(self, U, D, state):
        """
        This function takes a single update step and returns the
        new state for the plant.

        U is controller output (change in q1)
        D is random noise (change in q2)
        """

        new_q1 = jnp.clip((state[0] + U), 0.0, 1.0)
        new_q2 = jnp.clip((state[1] + D), 0.0, 1.0)
        new_state = jnp.array([new_q1, new_q2])

        return new_state

    def get_initial_state(self):
        return jnp.array(self.initial_state)

    def get_state_value(self, state):
        """
        The state value is the profit, which is what we want to maximize.
        """

        price = self.max_price - (state[0] + state[1])
        profit = state[0] * (price - self.cost_margin)
        return profit

    def get_target(self):
        return self.target
