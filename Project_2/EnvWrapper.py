import numpy as np
from valid_move_generator import generate_valid_moves
from tetris.tetris_env import TetrisEnv


class EnvWrapper:
    def __init__(self, env: TetrisEnv):
        self.env = env
        self.num_actions = 800

    def reset(self):
        """Resets the base environment and returns the initial observation."""
        self.env.reset()
        return self._get_obs()

    def get_legal_actions(self):
        """Returns a list of valid action IDs (0-799) using the BFS generator."""
        piece = self.env.active_piece
        current_state = (piece.x, piece.y, piece.rotation)

        valid_actions = generate_valid_moves(current_state)
        move_ids = []

        for a in valid_actions:
            id = self.encode_action(a[0], a[1], a[2])
            move_ids.append(id)

        return id

    def encode_action(self, x, y, rot):
        """Maps (x, y, rot) to a single integer 0-799."""
        return (y * 10 + x) * 4 + rot

    def decode_action(self, action_id):
        """Maps a single integer 0-799 back to (x, y, rot)."""
        rot = action_id % 4
        x = (action_id // 4) % 10
        y = action_id // 40
        return int(x), int(y), int(rot)

    def step(self, action_id):
        """Teleports the piece to the chosen action, locks it, and returns the result."""

        action = self.decode_action(action_id)
        self.env.active_piece.x = action[0]
        self.env.active_piece.y = action[1]
        self.env.active_piece.rotation = action[2]

        # Force a locking of piece by moving it down
        _, lines_cleared, terminated, _, _ = self.env.step(0)

        reward = lines_cleared**2

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
