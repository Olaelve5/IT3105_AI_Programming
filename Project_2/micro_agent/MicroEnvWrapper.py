import numpy as np


class MicroEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.target_pos = None
        self.target_rot = None
        self.target_shape = None
        self.prev_distance = 0

    def reset(self, scenario):
        self.env.load_training_scenario(
            board_state=scenario["board"],
            target_pos=scenario["target_pos"],
            target_rotation=scenario["target_rot"],
        )

        self.env.active_piece = self.env.generate_new_piece(id=scenario["piece_id"])
        self.target_pos = scenario["target_pos"]
        self.target_rot = scenario["target_rot"]
        self.target_shape = self.env.active_piece.shapes[self.target_rot]

        self.prev_distance = self._get_distance()

        return self._get_obs()

    def step(self, action):
        """
        Takes a step in the environment. We can't use the normal step function, we need other rewards
        """

        _, _, done, _, _ = self.env.step(
            action, spawn_new_piece=False, clear_lines=False
        )

        reward = 0.0

        if done:
            if (
                self.env.active_piece.x == self.target_pos[0]
                and self.env.active_piece.y == self.target_pos[1]
                and self.env.active_piece.rotation == self.target_rot
            ):
                # Hitting the target
                reward = 10.0
            else:
                # Did not hit target
                reward = -1.0
        else:
            # Tiny reward for getting closer, tiny penalty for moving away
            current_distance = self._get_distance()
            if current_distance < self.prev_distance:
                reward = 0.1
            elif current_distance > self.prev_distance:
                reward = -0.1

            self.prev_distance = current_distance

        return self._get_obs(), reward, done

    def load_scenario(self, s):
        self.env.load_traning_scenario(
            board_state=s["board"],
            target_pos=s["target_pos"],
            target_rot=s["target_rot"],
        )

        self.target_pos = tuple(s["target_pos"])
        self.target_rot = int(s["target_rot"])

        target_piece = self.env.generate_new_piece(int(s["piece_id"]))
        target_piece.rotation = self.target_rot
        target_piece.active_shape = target_piece.shapes[target_piece.rotation]
        self.target_shape = target_piece.active_shape
        self.prev_distance = self._get_distance()

        return self._get_obs()

    def _get_distance(self):
        """Calculates Manhattan distance between current piece and target."""
        return abs(self.env.active_piece.x - self.target_pos[0]) + abs(
            self.env.active_piece.y - self.target_pos[1]
        )

    def _get_obs(self):
        """Custom obs function, as the DDQN network expect a different structure than the MuZero agent"""

        obs_board = np.where(self.env.board > 0, 1.0, 0.0)

        # Add the falling piece to the obs
        obs_active = np.zeros((self.env.height, self.env.width), dtype=float)
        shape = self.env.active_piece.active_shape
        for i in range(4):
            for j in range(4):
                if i * 4 + j in shape:
                    by = self.env.active_piece.y + i
                    bx = self.env.active_piece.x + j
                    if 0 <= by < self.env.height and 0 <= bx < self.env.width:
                        obs_active[by, bx] = 1.0

        # Add the target to the obs
        obs_target = np.zeros((self.env.height, self.env.width), dtype=float)
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.target_shape:
                    by = self.target_pos[1] + i
                    bx = self.target_pos[0] + j
                    if 0 <= by < self.env.height and 0 <= bx < self.env.width:
                        obs_target[by, bx] = 1.0

        # Stack and return them as the obs
        # Shape is (Board Height, Board Width, 3)
        return np.stack((obs_board, obs_active, obs_target), axis=-1)
