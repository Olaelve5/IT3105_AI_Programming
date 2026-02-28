import numpy as np
import random
import json
from tetris.tetris_env import TetrisEnv
from config import BOARD_HEIGHT, BOARD_WIDTH


class ScenarioGenerator:
    def __init__(self, board_height=BOARD_HEIGHT, board_width=BOARD_WIDTH):
        self.h = board_height
        self.w = board_width
        self.env = TetrisEnv()

    # ---------------------------------------->
    # Normal random scenario generator (guaranteed reachable, but not necessarily tricky)
    def generate_normal_scenario(self):
        """
        Generates a guaranteed-reachable sandbox scenario.
        """
        self.env.reset()

        # Build a random landscape
        num_garbage_pieces = random.randint(3, 15)
        for _ in range(num_garbage_pieces):
            direction = random.choice([1, 2])
            moves = random.randint(0, 5)

            for _ in range(moves):
                self.env.step(direction)

            for _ in range(random.randint(1, 5)):
                action = random.choice([0, 1, 2, 3])
                self.env.step(action)

            if np.any(self.env.board[:6, :] != 0):
                break

            self.env.step(4)

        # Wipe the top 4 rows to ensure the new piece can spawn
        self.env.board[:4, :] = 0

        base_board = self.env.board.copy()

        # change all non-zero values to 1, so we can ignore the colors
        base_board[base_board != 0] = 1

        test_piece = self.env.generate_new_piece()
        self.env.active_piece = test_piece

        direction = random.choice([1, 2])
        moves = random.randint(0, 3)
        for _ in range(moves):
            self.env.step(direction, spawn_new_piece=False, clear_lines=False)

        # Generate the target position
        for _ in range(random.randint(0, 10)):
            self.env.step(
                random.choice([0, 1, 2, 3]), spawn_new_piece=False, clear_lines=False
            )

        self.env.step(4, spawn_new_piece=False, clear_lines=False)

        target_pos = (test_piece.x, test_piece.y)
        target_rot = test_piece.rotation

        return {
            "name": "normal_random",
            "board": base_board,
            "piece_id": test_piece.id,
            "target_pos": target_pos,
            "target_rot": target_rot,
        }

    # ---------------------------------------->
    # Tricky scenario generators with specific patterns that require non-trivial maneuvers to solve
    # Except for the deep well, but the agent has to know how to solve it
    def _get_noisy_base(self, stack_height, keep_clear_cols):
        """Build a filled base and sprinkle some random garbage on the row above it."""
        board = np.zeros((self.h, self.w), dtype=int)
        board[self.h - stack_height :, :] = 1

        noise_row = self.h - stack_height - 1
        if noise_row >= 0:
            for c in range(self.w):
                if c not in keep_clear_cols and random.random() > 0.5:
                    board[noise_row, c] = 1
        return board

    def get_dynamic_well(self):
        """Create a deep 1-wide well that only an I-piece can fill."""
        stack_height = random.randint(3, 8)
        well_col = random.randint(0, self.w - 1)

        board = self._get_noisy_base(stack_height, keep_clear_cols=[well_col])
        board[self.h - stack_height :, well_col] = 0

        return {
            "name": "dynamic_deep_well",
            "board": board,
            "piece_id": 1,
            "target_pos": (well_col - 1, self.h - 4),
            "target_rot": 0,
        }

    def get_dynamic_overhang(self):
        """Generates a slide-tuck cave scenario for an L or J piece."""
        stack_height = random.randint(3, 6)
        start_col = random.randint(0, self.w - 4)

        is_l_piece = random.choice([True, False])

        if is_l_piece:
            piece_id = 4
            target_rot = 1
            target_x = start_col
            overhang_col = start_col
            shaft_cols = [start_col + 1, start_col + 2, start_col + 3]
        else:
            piece_id = 5
            target_rot = 3
            target_x = start_col
            overhang_col = start_col + 3
            shaft_cols = [start_col, start_col + 1, start_col + 2]

        board = self._get_noisy_base(stack_height, keep_clear_cols=shaft_cols)

        cave_top_y = self.h - stack_height
        cave_bot_y = self.h - stack_height + 1

        board[cave_top_y, start_col : start_col + 4] = 0
        board[cave_bot_y, start_col : start_col + 4] = 0

        board[cave_top_y - 1, overhang_col] = 1
        board[cave_top_y - 2, overhang_col] = 1

        return {
            "name": f"dynamic_overhang_{'L' if is_l_piece else 'J'}",
            "board": board,
            "piece_id": piece_id,
            "target_pos": (target_x, cave_top_y),
            "target_rot": target_rot,
        }

    def get_dynamic_t_spin(self):
        """Generates a classic T-slot with an overhang for a T-spin double."""
        stack_height = random.randint(4, 7)
        t_col = random.randint(1, self.w - 3)

        board = self._get_noisy_base(stack_height, keep_clear_cols=[t_col, t_col + 1])
        t_y = self.h - stack_height + 2

        board[t_y, t_col + 1] = 0
        board[t_y - 1, t_col : t_col + 3] = 0

        board[t_y - 2, t_col : t_col + 2] = 0

        return {
            "name": "dynamic_t_spin",
            "board": board,
            "piece_id": 6,
            "target_pos": (t_col, t_y - 2),
            "target_rot": 2,
        }

    def get_dynamic_z_spin(self):
        """Generates a dogleg pattern that can be solved by a Z piece with the right setup moves."""
        stack_height = 8
        start_col = random.randint(1, self.w - 5)

        is_z_piece = random.choice([True, False])
        shift_dir = random.choice([-1, 1])

        if shift_dir == 1:
            keep_cols = [start_col + 1, start_col + 2, start_col + 3]
        else:
            keep_cols = [start_col - 1, start_col, start_col + 1]

        board = self._get_noisy_base(stack_height, keep_clear_cols=keep_cols)

        board[self.h - 3 :, start_col : start_col + 3] = 0

        if shift_dir == 1:
            board[self.h - 6 : self.h - 3, start_col : start_col + 4] = 0
        else:
            board[self.h - 6 : self.h - 3, start_col - 1 : start_col + 3] = 0

        if shift_dir == 1:
            board[self.h - stack_height : self.h - 6, start_col + 1 : start_col + 4] = 0
        else:
            board[self.h - stack_height : self.h - 6, start_col - 1 : start_col + 2] = 0

        if is_z_piece:
            board[self.h - 1, start_col + 2] = 1
            piece_id = 3
            target_x = start_col - 1
        else:
            board[self.h - 1, start_col] = 1
            piece_id = 2
            target_x = start_col

        return {
            "name": f"dynamic_dogleg_{'Z' if is_z_piece else 'S'}",
            "board": board,
            "piece_id": piece_id,
            "target_pos": (target_x, self.h - 3),
            "target_rot": 0,
        }

    def generate_tricky_scenario(self):
        """Pick one of the tricky scenarios at random and return it."""
        scenarios = [
            self.get_dynamic_well,
            self.get_dynamic_overhang,
            self.get_dynamic_t_spin,
            self.get_dynamic_z_spin,
        ]
        return random.choice(scenarios)()

    def get_random_scenario(self, tricky=False):
        if tricky:
            return self.generate_tricky_scenario()
        else:
            return self.generate_normal_scenario()


if __name__ == "__main__":
    generator = ScenarioGenerator()
    generator.generate_test_set(750)
