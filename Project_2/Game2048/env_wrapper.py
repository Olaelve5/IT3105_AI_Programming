import numpy as np


class EnvWrapper:
    def __init__(self, env):
        self.env = env

    def reset(self):
        self.env.reset()
        return self.get_obs()

    def step(self, action):
        valid_move, step_score, done = self.env.step(action)

        if not valid_move:
                return self.get_obs(), -1.0, True

        reward = step_score / 2048.0 if step_score > 0 else 0.0

        if done:
            reward = -1.0

        return self.get_obs(), reward, done

    def get_obs(self):
        """1 channel for each possible tile value (2^1 to 2^15) and 1 channel for empty tiles."""
        board = self.env.board
        obs = np.zeros((4, 4, 16), dtype=np.float32)

        obs[:, :, 0] = (board == 0).astype(np.float32)

        for power in range(1, 16):
            target_value = 2**power
            obs[:, :, power] = (board == target_value).astype(np.float32)

        return obs
