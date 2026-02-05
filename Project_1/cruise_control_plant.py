import jax.numpy as jnp


class Cruise_Control_Plant:
    def __init__(
        self,
        m=1000.0,
        b=50.0,
        target=10.0,
        force_multiplier=1000.0,
        noise_range=(-0.1, 0.1),
        timesteps=100,
    ):
        """
        m = 1000 kg -> mass of the car
        b = 50 Nsec/m -> the damping factor
        target = 10 m/s -> target speed

        Force multiplier is necessary to make the acceleration
        and noise affect the velocity of the car.

        nn_input_scale is used to normalize input values
        for the neural net controller
        """

        self.m = m
        self.b = b
        self.target = target
        self.force_multiplier = force_multiplier
        self.noise_range = noise_range
        self.nn_input_scale = [self.target, self.target * timesteps, 1.0]

    def update(self, U, D, state):
        current_velocity = state[0]

        # friction (bv)
        friction_force = self.b * current_velocity

        # scale force from engine
        engine_force = U * self.force_multiplier

        # Scale noise, otherwise the car wont be affected
        # Maximum force from noise is 0.1 * 1000 = 100.0 newtons,
        # which is 100.0/1000.0 kg = 0.1 m/s of force each second
        noise_force = D * self.force_multiplier

        # Total force on the car
        force = engine_force - friction_force + noise_force

        # Acceleration
        acceleration = force / self.m

        new_velocity = current_velocity + acceleration

        return jnp.array([new_velocity])

    def get_initial_state(self):
        return jnp.array([0.0])

    def get_state_value(self, state):
        return state[0]

    def get_target(self):
        return self.target
