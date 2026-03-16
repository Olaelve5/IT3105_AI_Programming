import numpy as np


class EnvWrapper:
    def __init__(self, env):
        self.env = env

    def reset(self):
        self.env.reset()
        return self.get_obs()

    def step(self, action, is_training=True):
        valid_move, winning_move, done = self.env.step(action)

        if winning_move:
            reward = 1.0
        else:
            reward = 0.0

        if not valid_move:
            if is_training:
                return self.get_obs(), -1.0, True
            else:
                return self.get_obs(), 0.0, False

        return self.get_obs(), reward, done

    def get_obs(self):
        """
        Converts the raw board into a 3-Channel Canonical Observation.
        Shape: (6, 7, 3)
        Channel 0: Current player's pieces
        Channel 1: Opponent's pieces
        Channel 2: Turn indicator (All 1s for Player 1, All 0s for Player 2)
        """
        board = self.env.board
        current_player = self.env.current_player
        opponent = 2 if current_player == 1 else 1

        # Channel 0: 1 where the current player has a piece, 0 elsewhere
        my_pieces = (board == current_player).astype(np.float32)

        # Channel 1: 1 where the opponent has a piece, 0 elsewhere
        opponent_pieces = (board == opponent).astype(np.float32)

        # Channel 2: Player turn indicator
        if current_player == 1:
            turn_channel = np.ones_like(board, dtype=np.float32)
        else:
            turn_channel = np.zeros_like(board, dtype=np.float32)

        obs = np.stack([my_pieces, opponent_pieces, turn_channel], axis=-1)
        return obs
