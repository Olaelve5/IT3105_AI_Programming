import numpy as np
from valid_move_generator import generate_valid_moves
from tetris.tetris_env import TetrisEnv
from tetris.tetris_shape import FIGURES_BY_ID
from config import NUM_ACTIONS


class EnvWrapper:
    def __init__(self, env: TetrisEnv):
        self.env = env
        self.num_actions = NUM_ACTIONS

    def reset(self):
        """Resets the base environment and returns the initial observation."""
        self.env.reset()
        return self._get_obs()

    def encode_action(self, x, rot):
        """Maps (x, rot) to a single integer 0-47."""
        shifted_x = x + 2  # Absorb the -2 bounding box overhang
        return shifted_x * 4 + rot

    def decode_action(self, action_id):
        """Maps a single integer 0-47 back to (x, rot)."""
        rot = action_id % 4
        shifted_x = action_id // 4
        return shifted_x - 2, rot

    def get_legal_actions(self):
        """
        Runs BFS, but filters it to only return the lowest possible resting
        point (highest y) for each (x, rot) combination.
        """
        start_state = (
            self.env.active_piece.x,
            self.env.active_piece.y,
            self.env.active_piece.rotation,
        )

        valid_states = generate_valid_moves(
            start_state, self.env.active_piece.id, self.env
        )

        # Dictionary to store the lowest y for each (x, rot)
        self.best_y_for_action = {}
        legal_actions = []

        for x, y, rot in valid_states:
            action_id = self.encode_action(x, rot)

            # If we haven't seen this action, or this 'y' is lower (higher value), update it!
            if (
                action_id not in self.best_y_for_action
                or y > self.best_y_for_action[action_id]
            ):
                self.best_y_for_action[action_id] = y

        # Return just the unique action IDs (0 to 47)
        return list(self.best_y_for_action.keys())

    def step(self, action_id):
        """Teleports the piece to the chosen column, drops it, and locks it."""
        x, rot = self.decode_action(action_id)

        # Look up the resting y-coordinate we calculated during get_legal_actions!
        # (Fallback to get_drop_position just in case the AI picks an illegal move during exploration)
        y = self.best_y_for_action.get(action_id, None)

        self.env.active_piece.x = x
        self.env.active_piece.rotation = rot
        self.env.active_piece.active_shape = self.env.active_piece.shapes[rot]

        if y is not None:
            self.env.active_piece.y = y
        else:
            self.env.active_piece.y = self.env.get_drop_position()

        # Force a "Down" action to trigger the lock and respawn
        _, lines_cleared, terminated, _, _ = self.env.step(0)

        # Reward function
        if terminated:
            # Massive punishment for dying!
            reward = -1.0
        elif lines_cleared > 0:
            # Massive reward for doing the right thing!
            reward = float(lines_cleared**2)
        else:
            # Tiny breadcrumb reward just for staying alive and placing a piece
            reward = 0.01

        return self._get_obs(), reward, terminated

    def _get_obs(self):
        """
        Returns a 3-channel array for the MuZero Representation Net.
        """

        # Board channel
        board_channel = (self.env.board > 0).astype(np.float32)

        # Active piece channel
        piece_channel = np.zeros((self.env.height, self.env.width), dtype=np.float32)
        piece = self.env.active_piece
        if piece is not None:
            for i in range(4):
                for j in range(4):
                    if i * 4 + j in piece.active_shape:
                        y = piece.y + i
                        x = piece.x + j
                        if 0 <= y < self.env.height and 0 <= x < self.env.width:
                            piece_channel[y, x] = 1.0

        # Next piece channel
        next_piece_channel = np.zeros(
            (self.env.height, self.env.width), dtype=np.float32
        )
        if self.env.next_piece is not None:
            # Normalize piece ids
            normalized_id = self.env.next_piece.id / 7.0
            next_piece_channel.fill(normalized_id)

        return np.stack((board_channel, piece_channel, next_piece_channel), axis=-1)

    def render(self):
        self.env.render(tick=False)
