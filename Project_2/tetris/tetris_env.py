import numpy as np
import pygame
from tetris.tetris_shape import FIGURES, TetrisPiece
import random
from config import BOARD_WIDTH, BOARD_HEIGHT, NUM_ACTIONS, GRID_SIZE

BACKGROUND_COLOR = (15, 15, 20)


class TetrisEnv:
    def __init__(self):
        """
        Initialize the environment's parameters.
        """
        self.height = BOARD_HEIGHT
        self.width = BOARD_WIDTH
        self.grid_size = GRID_SIZE
        self.window_width = self.width * self.grid_size
        self.window_height = self.height * self.grid_size

        self.figure_pool = FIGURES

        self.screen = None
        self.clock = None
        self.font = None

        self.board = None
        self.active_piece = None
        self.score = 0
        self.state = None

        self.reset()

    def reset(self):
        self.score = 0
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.active_piece = None
        self.state = "start"

        self.spawn_new_piece()

        return self._get_observation(), {}

    def spawn_new_piece(self):
        shape_name = random.choice(list(self.figure_pool.keys()))
        piece = self.figure_pool[shape_name]
        start_x = (self.width // 2) - 2
        self.active_piece = TetrisPiece(piece, [start_x, 0])

    def step(self, action):
        old_height, old_holes, old_bumpiness = self.get_board_metrics()
        reward = 0.0

        self.handle_action(action)

        # Apply gravity
        shape_locked = False

        if not self.check_collision(
            self.active_piece.active_shape, self.active_piece.x, self.active_piece.y + 1
        ):
            self.active_piece.y += 1
        else:
            self.freeze_shape()
            shape_locked = True

        # Handle rewards / game over
        if shape_locked:
            if self.check_collision(
                self.active_piece.active_shape, self.active_piece.x, self.active_piece.y
            ):
                self.state = "gameover"
                reward = -1.0
                reward = round(reward, 2)
                return self._get_observation(), reward, True, False, {}
            else:
                new_height, new_holes, new_bumpiness = self.get_board_metrics()

                # Check for cleared lines
                lines_cleared = self.clear_lines()

                # Big reward for clearing lines
                if lines_cleared > 0:
                    clear_reward = (lines_cleared**2) * 10
                    print(
                        f"{'🔥' * lines_cleared} Cleared {lines_cleared} line{'s' if lines_cleared > 1 else ''}! Reward: {clear_reward}"
                    )
                    reward += clear_reward
                    self.score += lines_cleared**2

                # Reward for less bumpiness, penalty for more
                bumpiness_diff = old_bumpiness - new_bumpiness
                reward += bumpiness_diff * 0.01

                # Reward for less holes, penalty for more
                holes_diff = old_holes - new_holes
                reward += holes_diff * 0.1

                # Small reward for locking a shape (not dying)
                reward += 0.1

        terminated = self.state == "gameover"

        return self._get_observation(), round(reward, 2), terminated, False, {}

    def handle_action(self, action):
        """Processes horizontal movement and rotation."""
        if action == 1:  # Left
            if not self.check_collision(
                self.active_piece.active_shape,
                self.active_piece.x - 1,
                self.active_piece.y,
            ):
                self.active_piece.x -= 1

        elif action == 2:  # Right
            if not self.check_collision(
                self.active_piece.active_shape,
                self.active_piece.x + 1,
                self.active_piece.y,
            ):
                self.active_piece.x += 1

        elif action == 3:  # Rotate
            self.active_piece.rotate(self.can_rotate)

        else:
            # action 0 is do nothing
            return

    def _get_observation(self):
        """Creates a temporary obs of the board + the falling piece for the AI."""
        obs = self.board.copy()

        # Draw the active piece onto the copy
        if self.active_piece is not None:
            shape = self.active_piece.active_shape
            for i in range(4):
                for j in range(4):
                    if i * 4 + j in shape:
                        board_y = self.active_piece.y + i
                        board_x = self.active_piece.x + j

                        if 0 <= board_y < self.height and 0 <= board_x < self.width:
                            obs[board_y, board_x] = self.active_piece.id

        return obs

    def can_rotate(self, new_shape):
        return not self.check_collision(
            new_shape, self.active_piece.x, self.active_piece.y
        )

    def check_collision(self, shape, offset_x, offset_y):
        """
        Checks if a given shape at a specific x,y position hits walls or placed blocks.
        """
        for i in range(4):
            for j in range(4):
                if i * 4 + j in shape:
                    board_y = offset_y + i
                    board_x = offset_x + j

                    # 1. Check out of bounds (Walls and Floor)
                    if board_x < 0 or board_x >= self.width or board_y >= self.height:
                        return True

                    # 2. Check placed blocks (Ignore if piece is still spawning above the board)
                    if board_y >= 0 and self.board[board_y, board_x] > 0:
                        return True
        return False

    def freeze_shape(self):
        """
        Saves the active shape into the board and spawns a new one.
        """

        shape = self.active_piece.active_shape
        for i in range(4):
            for j in range(4):
                if i * 4 + j in shape:
                    board_y = self.active_piece.y + i
                    board_x = self.active_piece.x + j

                    if board_y >= 0:
                        self.board[board_y, board_x] = self.active_piece.id

        self.spawn_new_piece()

    def clear_lines(self):
        # Rows with no 0s are full
        full_rows = np.all(self.board > 0, axis=1)
        num_cleared = np.sum(full_rows)

        if num_cleared > 0:
            # Remove the empty rows and shift the other rows down
            remaining_board = self.board[~full_rows]
            empty_rows = np.zeros((num_cleared, self.width), dtype=int)
            self.board = np.vstack((empty_rows, remaining_board))

        return num_cleared

    def render(self):
        # Initialize Pygame on the first render call
        if self.screen is None:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
            pygame.display.set_caption("MuZero Tetris")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 24, bold=True)

            self.id_to_color = {
                details["id"]: details["color"] for details in self.figure_pool.values()
            }

        # Clear the screen with your background color
        self.screen.fill(BACKGROUND_COLOR)

        # Draw the locked blocks
        for i in range(self.height):
            for j in range(self.width):
                block_id = self.board[i, j]
                if block_id > 0:
                    color = self.id_to_color[block_id]

                    rect = [
                        j * self.grid_size,
                        i * self.grid_size,
                        self.grid_size - 1,
                        self.grid_size - 1,
                    ]
                    pygame.draw.rect(self.screen, color, rect)

        # Draw the falling piece
        if self.active_piece is not None:
            shape = self.active_piece.active_shape
            color = self.id_to_color[self.active_piece.id]

            for i in range(4):
                for j in range(4):
                    if i * 4 + j in shape:
                        board_y = self.active_piece.y + i
                        board_x = self.active_piece.x + j

                        if board_y >= 0:
                            rect = [
                                board_x * self.grid_size,
                                board_y * self.grid_size,
                                self.grid_size - 1,
                                self.grid_size - 1,
                            ]
                            pygame.draw.rect(self.screen, color, rect)

        score_text = self.font.render(f"Score: {self.score:.2f}", True, (255, 255, 255))
        self.screen.blit(score_text, [10, 10])

        pygame.display.flip()
        self.clock.tick(5)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def get_board_metrics(self):
        """
        Finds height, holes and bumpiness of the board state.
        To be used in reward function.
        """
        heights = []
        holes = 0

        for col in range(self.width):
            col_data = self.board[:, col]
            non_zeros = np.where(col_data > 0)[0]

            if len(non_zeros) > 0:
                top_row = non_zeros[0]
                heights.append(self.height - top_row)

                # Any 0 that is below something else in the same column is a hole
                holes += np.sum(col_data[top_row:] == 0)
            else:
                heights.append(0)

        heights = np.array(heights)
        sum_height = np.sum(heights)
        bumpiness = np.sum(np.abs(heights[:-1] - heights[1:]))

        return sum_height, holes, bumpiness
