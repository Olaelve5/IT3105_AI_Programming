import matplotlib.pyplot as plt


def plot_mse(error_history):
    plt.figure(figsize=(10, 6))
    plt.plot(error_history, label="Error", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Learning progression")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_params(params_history):
    if not params_history:
        print("No params history to plot!")
        return

    kp_values = [params["kp"] for params in params_history]
    ki_values = [params["ki"] for params in params_history]
    kd_values = [params["kd"] for params in params_history]

    plt.figure(figsize=(10, 6))
    plt.plot(kp_values, label="Kp", color="red", linewidth=2)
    plt.plot(ki_values, label="Ki", color="green", linewidth=2)
    plt.plot(kd_values, label="Kd", color="blue", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Parameter Value")
    plt.title("PID Parameter Progression During Training")
    plt.grid(True)
    plt.legend()
    plt.show()
