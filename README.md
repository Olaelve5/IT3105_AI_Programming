# IT3105 AI Programming

This repository contains projects and assignments for the **IT3105 AI Programming** course. It features implementations ranging from foundational control systems (PID and Neural Network controllers) to advanced Deep Reinforcement Learning (a custom MuZero implementation in JAX).

## 📁 Repository Structure

```text
.
├── Animations/           # Manim-based educational animations explaining RL and MuZero
├── Project_1/            # Control Systems: PID vs. Neural Network Controllers
└── Project_2/            # Deep Reinforcement Learning: MuZero in JAX
```

---

## 🚀 Project 1: Control Systems

This project explores continuous control by comparing traditional **PID Controllers** with **Neural Network (NN) Controllers** across three distinct environments.

### Environments (Plants)
1. **Bathtub Plant:** Regulating the water level in a bathtub with noise and volume constraints.
2. **Cournot Plant:** An economics-inspired environment simulating price and production competition.
3. **Cruise Control Plant:** Managing the velocity of a vehicle subjected to physics, mass, and friction constraints.

### How to Run
Navigate to `Project_1` and execute the main simulation script. You can switch the active configuration in `main.py` (e.g., `config_cruise_control_nn`, `config_bathtub_pid`) to test different plants and controllers.

```bash
cd Project_1
python main.py
```

*The script outputs training metrics and displays plots of the Mean Squared Error (MSE) and parameter history.*

---

## 🧠 Project 2: MuZero (JAX/Flax)

This project features a high-performance implementation of DeepMind's **MuZero** algorithm built from scratch using **JAX** and **Flax**. It is configured to play grid-based environments like **Tron** (and Tetris).

### Features
* **JAX/Flax Architecture:** Fully vectorized and JIT-compiled for rapid training and inference.
* **Monte Carlo Tree Search (MCTS):** Integrated with the learned dynamics model for planning.
* **WandB Integration:** Real-time logging of rewards, episode lengths, and MCTS entropy.
* **Checkpointing:** Automatic saving and loading of model parameters using `flax.serialization`.

### Configuration
Key hyperparameters are located in `Project_2/main.py` and environment configs in `Project_2/config.py`:
* **Board Size:** 22x22 (Grid Size: 20)
* **Actions:** 3 (Tron mechanics)
* **Batch Size:** 256
* **Unroll Steps:** 6

### How to Run
First, ensure you have the required JAX environment installed (see Installation). 

To start the training loop:

```bash
cd Project_2
python main.py
```

*Note: The script automatically sets `SDL_VIDEODRIVER="dummy"` for headless training. Checkpoints are automatically saved to the `saved_params/` directory.*

---

## 🛠️ Installation & Requirements

This project relies heavily on the JAX ecosystem for high-performance computing. 

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/IT3105_AI_Programming.git](https://github.com/your-username/IT3105_AI_Programming.git)
cd IT3105_AI_Programming
```

**2. Create a virtual environment (Recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

**3. Install dependencies:**
Navigate to `Project_2` and install the requirements.
```bash
cd Project_2
pip install -r requirements.txt
```

### Key Dependencies
* `jax` & `jaxlib`
* `flax`
* `optax`
* `wandb`
* `numpy` & `scipy`

---

## 🎥 Animations
The `Animations/` folder contains Python scripts using the `Manim` library to visually explain concepts like Reinforcement Learning and the inner workings of MuZero. 

*(To render these, you will need to have [Manim](https://www.manim.community/) installed on your system).*
