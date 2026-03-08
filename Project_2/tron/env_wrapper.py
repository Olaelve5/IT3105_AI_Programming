import numpy as np


class TronEnvWrapper:
    def __init__(self, env):
        self.env = env

    def reset(self):
        self.env.reset()
        return self.get_obs()

    def step(self, action):
        _, _, terminated = self.env.step(action)

        if terminated:
            reward = -1.0
        else:
            reward = 0.05

        return self.get_obs(), reward, terminated

    def get_obs(self):
        """Translates the screen into a multi-channel matrix for the Neural Network."""
        grid_w = self.env.width // self.env.grid_size
        grid_h = self.env.height // self.env.grid_size

        obs = np.zeros((grid_h, grid_w, 3), dtype=np.float32)

        # Channel 0: Head
        head_x = int(self.env.head_pos[0] // self.env.grid_size)
        head_y = int(self.env.head_pos[1] // self.env.grid_size)
        if 0 <= head_x < grid_w and 0 <= head_y < grid_h:
            obs[head_y, head_x, 0] = 1.0

        # Channel 1: Trail
        for pos in self.env.body_positions:
            x_idx = int(pos[0] // self.env.grid_size)
            y_idx = int(pos[1] // self.env.grid_size)
            if 0 <= x_idx < grid_w and 0 <= y_idx < grid_h:
                obs[y_idx, x_idx, 1] = 1.0

        # Channel 2: Walls
        obs[0, :, 2] = 1.0
        obs[-1, :, 2] = 1.0
        obs[:, 0, 2] = 1.0
        obs[:, -1, 2] = 1.0

        if hasattr(self.env, "walls"):
            for wall_pos in self.env.walls:
                wx = int(wall_pos[0] // self.env.grid_size)
                wy = int(wall_pos[1] // self.env.grid_size)
                if 0 <= wx < grid_w and 0 <= wy < grid_h:
                    obs[wy, wx, 2] = 1.0

        return obs
