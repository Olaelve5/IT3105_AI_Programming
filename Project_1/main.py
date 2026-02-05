from bathtub_plant import Bathtub_Plant
from cournot_plant import Cournot_Plant
from cruise_control_plant import Cruise_Control_Plant
from pid_controller import PID_Controller
from nn_controller import NN_Controller
from consys import CONSYS
from plot_utils import plot_mse, plot_params


def main():
    EPOCHS = 100
    TIMESTEPS = 120
    LEARNING_RATE = 0.01

    # Plants
    cournot_plant = Cournot_Plant(timesteps=TIMESTEPS)
    bathtub_plant = Bathtub_Plant(timesteps=TIMESTEPS)
    cruise_control_plant = Cruise_Control_Plant(timesteps=TIMESTEPS)

    PLANT_TO_USE = cruise_control_plant

    # Controllers
    PID_CONTROLLER = PID_Controller()

    NN_CONTROLLER = NN_Controller(
        layers=[(None, 3), ("tanh", 10), (None, 1)],
        input_scale=PLANT_TO_USE.nn_input_scale,
    )

    # Controller type should be either "PID" or "NN"
    sim = CONSYS(
        plant=PLANT_TO_USE,
        nn_controller=NN_CONTROLLER,
        pid_controller=PID_CONTROLLER,
        controller_type="NN",
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        timesteps=TIMESTEPS,
    )

    print("Starting training...")
    sim.train()
    print("Training complete!")

    plot_mse(sim.loss_history)
    plot_params(sim.params_history)


if __name__ == "__main__":
    main()
