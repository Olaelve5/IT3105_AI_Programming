import numpy as np


class SnakeEnvWrapper:
    def __init__(self, env):
        self.env = env

    def reset(self):
        self.env.reset()
        return self.get_obs()

    def step(self, action):
        _, terminated, ate_fruit = self.env.step(action)

        if ate_fruit:
            print("🍎 Ate a fruit! Current length:", self.env.body_length)

        if terminated:
            reward = -1.0
        else:
            reward = 1.0 if ate_fruit else -0.01

        if self.env.body_length == self.env.max_body_length:
            reward += 10.0
            terminated = True

        return self.get_obs(), reward, terminated

    def get_obs(self):
        """Translates the screen into a multi-channel matrix for the Neural Network."""
        grid_w = self.env.width // self.env.grid_size
        grid_h = self.env.height // self.env.grid_size
        obs = np.zeros((grid_h, grid_w, 3), dtype=np.float32)

        for pos in self.env.body_positions:
            x_idx = int(pos[0] // self.env.grid_size)
            y_idx = int(pos[1] // self.env.grid_size)
            if 0 <= x_idx < grid_w and 0 <= y_idx < grid_h:
                obs[y_idx, x_idx, 0] = 1.0

        head_x = int(self.env.head_pos[0] // self.env.grid_size)
        head_y = int(self.env.head_pos[1] // self.env.grid_size)
        if 0 <= head_x < grid_w and 0 <= head_y < grid_h:
            obs[head_y, head_x, 1] = 1.0

        fruit_x = int(self.env.fruit_pos[0] // self.env.grid_size)
        fruit_y = int(self.env.fruit_pos[1] // self.env.grid_size)
        if 0 <= fruit_x < grid_w and 0 <= fruit_y < grid_h:
            obs[fruit_y, fruit_x, 2] = 1.0

        return obs
