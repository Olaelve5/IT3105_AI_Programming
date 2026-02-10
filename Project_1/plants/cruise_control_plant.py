import jax.numpy as jnp
from plants.plant import Plant


class Cruise_Control_Plant(Plant):
    def __init__(
        self,
        m=1000.0,
        b=50.0,
        target=10.0,
        starting_velocity=0.0,
        force_multiplier=2000.0,
        noise_range=(-0.1, 0.1),
    ):
        """
        m = 1000 kg -> mass of the car
        b -> the damping factor
            - (b is proportional to the velocity of the car, and represents friction and air resistance)

        target = 10 m/s -> target speed

        Force multiplier is necessary to make the accelerate
        and noise affect the velocity of the car.
        """

        self.m = m
        self.b = b
        self.target = target
        self.starting_velocity = starting_velocity
        self.force_multiplier = force_multiplier
        self.noise_range = noise_range

    def update(self, U, D, state):
        """
        This function takes a single update step and returns the
        new velocity of the car.
        """

        current_velocity = state[0]

        # friction (bv)
        friction_force = self.b * current_velocity

        # scale force from engine
        # U is in range [-1.0, 1.0], so maximum force from engine is 1.0 * 1000 = 1000 newtons,
        # if the force multiplier is 1000.0
        engine_force = U * self.force_multiplier

        # Scale noise, otherwise the car wont be affected
        # Maximum force from noise is 0.1 * 1000 = 100.0 newtons,
        # which is 100.0/1000.0 kg = 0.1 m/s of force each second
        noise_force = D * self.force_multiplier

        # Total force on the car
        force = engine_force - friction_force + noise_force

        # Acceleration is equal to force divided by mass (F = ma -> a = F/m)
        acceleration = force / self.m

        new_velocity = current_velocity + acceleration

        return jnp.array([new_velocity])

    def get_initial_state(self):
        return jnp.array([self.starting_velocity])

    def get_state_value(self, state):
        return state[0]

    def get_target(self):
        return self.target
