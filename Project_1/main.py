from plants.bathtub_plant import Bathtub_Plant
from plants.cournot_plant import Cournot_Plant
from plants.cruise_control_plant import Cruise_Control_Plant
from controllers.pid_controller import PID_Controller
from controllers.nn_controller import NN_Controller
from consys import CONSYS
from plot_utils import plot_mse, plot_params


# ==================================================================
# Config parameters
# ==================================================================

# Bathtub Plant -------------------------------------------------->
config_bathtub_pid = {
    "name": "Bathtub_PID",
    "plant_class": Bathtub_Plant,
    "plant_params": {"A": 10.0, "C": 0.1, "H0": 5.0, "noise_range": (-0.01, 0.01)},
    "controller_type": "PID",
    "pid_params": {"kp": 0.1, "ki": 0.01, "kd": 0.01},
    "nn_layers": None,
    "input_scale": None,
    "weight_init_range": None,
    "bias_init_range": None,
    "epochs": 100,
    "timesteps": 100,
    "learning_rate": 0.01,
}

config_bathtub_nn = {
    "name": "Bathtub_NN",
    "plant_class": Bathtub_Plant,
    "plant_params": {"A": 10.0, "C": 0.1, "H0": 5.0, "noise_range": (-0.01, 0.01)},
    "controller_type": "NN",
    "nn_layers": [(None, 3), ("tanh", 10), (None, 1)],
    "input_scale": [10.0, 500.0, 1.0],
    "weight_init_range": None,
    "bias_init_range": None,
    "epochs": 100,
    "timesteps": 100,
    "learning_rate": 0.01,
}


# Cournot Plant -------------------------------------------------->
config_cournot_pid = {
    "name": "Cournot_PID",
    "plant_class": Cournot_Plant,
    "plant_params": {
        "max_price": 10.0,
        "cost_margin": 0.1,
        "target": 3.0,
        "initial_state": [0.1, 0.5],
        "noise_range": (-0.01, 0.01),
    },
    "controller_type": "PID",
    "pid_params": {"kp": 0.0, "ki": 0.0, "kd": 0.0},
    "nn_layers": None,
    "input_scale": None,
    "weight_init_range": None,
    "bias_init_range": None,
    "epochs": 100,
    "timesteps": 100,
    "learning_rate": 0.001,
}

config_cournot_nn = {
    "name": "Cournot_NN",
    "plant_class": Cournot_Plant,
    "plant_params": {
        "max_price": 10.0,
        "cost_margin": 0.1,
        "target": 3.0,
        "initial_state": [0.1, 0.5],
        "noise_range": (-0.01, 0.01),
    },
    "controller_type": "NN",
    "input_scale": [1.0, 1.0, 1.0],
    "nn_layers": [(None, 3), ("tanh", 10), ("tanh", 10), (None, 1)],
    "weight_init_range": None,  # [-0.5, 0.5] works well too
    "bias_init_range": None,
    "epochs": 250,
    "timesteps": 100,
    "learning_rate": 0.01,
}


# Cruise Control Plant -------------------------------------------------->
config_cruise_control_pid = {
    "name": "Cruise_Control_PID",
    "plant_class": Cruise_Control_Plant,
    "plant_params": {
        "m": 1000.0,
        "b": 50.0,
        "target": 10.0,
        "starting_velocity": 20.0,
        "force_multiplier": 1000.0,
        "noise_range": (-0.1, 0.1),
    },
    "controller_type": "PID",
    "pid_params": {"kp": 0.1, "ki": 0.01, "kd": 0.01},
    "nn_layers": None,
    "input_scale": None,
    "weight_init_range": None,
    "bias_init_range": None,
    "epochs": 100,
    "timesteps": 100,
    "learning_rate": 0.005,
}

config_cruise_control_nn = {
    "name": "Cruise_Control_NN",
    "plant_class": Cruise_Control_Plant,
    "plant_params": {
        "m": 1000.0,
        "b": 50.0,
        "target": 10.0,
        "starting_velocity": 20.0,
        "force_multiplier": 1000.0,
        "noise_range": (-0.1, 0.1),
    },
    "controller_type": "NN",
    "input_scale": [10.0, 1000.0, 1.0],
    "nn_layers": [(None, 3), ("tanh", 10), (None, 1)],
    "weight_init_range": None,
    "bias_init_range": None,
    "epochs": 100,
    "timesteps": 100,
    "learning_rate": 0.005,
}


# ==================================================================
# Main function to run the experiment
# ==================================================================


def main():
    ACTIVE_CONFIG = config_cruise_control_nn
    print(f"\n========== Running Experiment: {ACTIVE_CONFIG['name']} ===========")

    plant = ACTIVE_CONFIG["plant_class"](**ACTIVE_CONFIG["plant_params"])

    EPOCHS = ACTIVE_CONFIG.get("epochs")
    LEARNING_RATE = ACTIVE_CONFIG.get("learning_rate")

    # Controller type should be either "PID" or "NN"
    sim = CONSYS(
        plant=plant,
        nn_controller=NN_Controller(
            layers=ACTIVE_CONFIG.get("nn_layers"),
            input_scale=ACTIVE_CONFIG.get("input_scale"),
            weight_init_range=ACTIVE_CONFIG.get("weight_init_range"),
            bias_init_range=ACTIVE_CONFIG.get("bias_init_range"),
        ),
        pid_controller=PID_Controller(),
        pid_params=ACTIVE_CONFIG.get("pid_params"),
        controller_type=ACTIVE_CONFIG.get("controller_type"),
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        timesteps=ACTIVE_CONFIG.get("timesteps"),
    )

    print("\nStarting training...")
    sim.train()
    print("\nTraining complete!")

    plot_mse(sim.loss_history)
    plot_params(sim.params_history)


if __name__ == "__main__":
    main()
