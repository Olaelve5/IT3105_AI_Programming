import numpy as np
import pygame
from tetris.tetris_shape import FIGURES, TetrisPiece
import random
from config import BOARD_WIDTH, BOARD_HEIGHT, GRID_SIZE

BACKGROUND_COLOR = (80, 106, 110)
BORDER_COLOR = (50, 50, 60)
PADDING_LEFT = 175
PADDING_RIGHT = 25
PADDING_BOTTOM = 25
PADDING_TOP = 25


class TetrisEnv:
    def __init__(self, tick_speed=5):
        self.height = BOARD_HEIGHT
        self.width = BOARD_WIDTH
        self.grid_size = GRID_SIZE
        self.window_width = self.width * self.grid_size + PADDING_LEFT + PADDING_RIGHT
        self.window_height = self.height * self.grid_size + PADDING_TOP + PADDING_BOTTOM
        self.tick_speed = tick_speed
        self.lines_cleared = 0

        self.figure_pool = FIGURES

        self.screen = None
        self.clock = None
        self.font = None

        self.board = None
        self.active_piece = None
        self.next_piece = None
        self.score = 0
        self.state = None
        self.step_counter = 0
        self.lines_cleared = 0

        self.reset()

    def reset(self):
        self.score = 0
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.active_piece = None
        self.next_piece = None
        self.state = "start"
        self.step_counter = 0
        self.lines_cleared = 0

        self.spawn_new_piece()

        return self._get_observation(), {}

    def generate_new_piece(self):
        shape_name = random.choice(list(self.figure_pool.keys()))
        piece = self.figure_pool[shape_name]
        start_x = (self.width // 2) - 2
        return TetrisPiece(piece, [start_x, 0])

    def spawn_new_piece(self):
        if self.next_piece is None:
            self.active_piece = self.generate_new_piece()
        else:
            self.active_piece = self.next_piece

        self.next_piece = self.generate_new_piece()

    def step(self, action):
        reward = 0.0
        self.step_counter += 1

        drop_distance = self.handle_action(action)

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
                reward = 0.0
                return self._get_observation(), reward, True, False, {}
            else:
                new_max_height, new_sum_height, new_holes, new_bumpiness = (
                    self.get_board_metrics()
                )

                # Large base reward
                base_reward = 0.05

                # Penalize holes, bumpiness and height
                board_penalty = (
                    (new_holes * 0.02)
                    + (new_bumpiness * 0.002)
                    + (new_max_height * 0.002)
                )

                step_reward = base_reward - board_penalty

                # Guarantee a non-negative reward
                reward = max(0.001, step_reward)

                # Small reward for fast dropping a piece
                # Only if the drop resulted in a positive reward
                if action == 4 and step_reward > 0:
                    reward += 0.002 * drop_distance

                # Check for line clears and add bonuses
                lines_cleared = self.clear_lines()
                if lines_cleared > 0:
                    clear_reward = lines_cleared * 0.5
                    print(
                        f"{'🔥' * lines_cleared} Cleared {lines_cleared} line{'s'}! Reward: {clear_reward}"
                    )
                    reward += clear_reward

                self.score += reward

        terminated = self.state == "gameover"

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            {"drop_distance": drop_distance, "step_counter": self.step_counter},
        )

    def handle_action(self, action):
        """Processes horizontal movement and rotation."""
        # if action == 0:  # No action

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

        elif action == 4:  # Fast Drop
            drop_y = self.get_drop_position()
            drop_distance = drop_y - self.active_piece.y
            self.active_piece.y = drop_y
            return drop_distance

        return 0.0

    def get_drop_position(self):
        """Returns the y position where the piece would land if dropped."""
        drop_y = self.active_piece.y
        while not self.check_collision(
            self.active_piece.active_shape, self.active_piece.x, drop_y + 1
        ):
            drop_y += 1
        return drop_y

    def _get_observation(self):
        """
        Creates a temporary obs of the board
        + the falling piece for the AI
        + the next piece
        """
        obs_board = np.where(self.board > 0, 1.0, 0.0)

        # Draw the active piece onto the copy
        if self.active_piece is not None:
            shape = self.active_piece.active_shape
            for i in range(4):
                for j in range(4):
                    if i * 4 + j in shape:
                        board_y = self.active_piece.y + i
                        board_x = self.active_piece.x + j

                        if 0 <= board_y < self.height and 0 <= board_x < self.width:
                            obs_board[board_y, board_x] = 0.5

        obs_next_piece = np.zeros((self.height, self.width), dtype=float)

        if self.next_piece is not None:
            # Normalize the ID to a value between 0 and 1 for the observation
            max_id = max(details["id"] for details in self.figure_pool.values())
            obs_next_piece.fill(self.next_piece.id / max_id)

        obs = np.stack((obs_board, obs_next_piece), axis=-1)

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

            self.lines_cleared += num_cleared
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
            self.font = pygame.font.Font("Project_2/assets/Jersey20-Regular.ttf", 25)

            self.id_to_color = {
                details["id"]: details["color"] for details in self.figure_pool.values()
            }

        # Clear the screen with your background color
        self.screen.fill(BACKGROUND_COLOR)

        # Fill the inner area
        inner_rect = [
            PADDING_LEFT,
            PADDING_TOP,
            self.width * self.grid_size,
            self.height * self.grid_size,
        ]
        pygame.draw.rect(self.screen, (20, 20, 20), inner_rect)

        # Draw the locked blocks
        for i in range(self.height):
            for j in range(self.width):
                block_id = self.board[i, j]
                if block_id > 0:
                    color = self.id_to_color[block_id]

                    rect = [
                        j * self.grid_size + PADDING_LEFT,
                        i * self.grid_size + PADDING_TOP,
                        self.grid_size - 1,
                        self.grid_size - 1,
                    ]
                    pygame.draw.rect(self.screen, color, rect)

        # Draw the falling piece
        if self.active_piece is not None:
            # Draw the shadow of where the piece would land
            drop_y = self.get_drop_position()
            self.draw_piece(
                self.active_piece.active_shape,
                self.active_piece.x,
                drop_y,
                (100, 100, 100),
            )

            self.draw_piece(
                self.active_piece.active_shape,
                self.active_piece.x,
                self.active_piece.y,
                self.id_to_color[self.active_piece.id],
            )

        # Draw the next piece preview
        self.draw_next_piece()

        # Draw text info
        score_text = self.font.render(
            f"Reward: {self.score:.2f}", True, (255, 255, 255)
        )
        self.screen.blit(score_text, [15, self.window_height - 50])

        step_text = self.font.render(
            f"Steps: {self.step_counter}", True, (255, 255, 255)
        )
        self.screen.blit(step_text, [15, self.window_height - 80])

        lines_text = self.font.render(
            f"Lines: {self.lines_cleared}", True, (255, 255, 255)
        )
        self.screen.blit(lines_text, [15, self.window_height - 110])

        pygame.display.flip()
        self.clock.tick(self.tick_speed)

    def draw_piece(self, shape, offset_x, offset_y, color):
        """Helper function to draw a piece at a given position with a given color."""
        for i in range(4):
            for j in range(4):
                if i * 4 + j in shape:
                    board_y = offset_y + i
                    board_x = offset_x + j

                    if board_y >= 0:
                        rect = [
                            board_x * self.grid_size + PADDING_LEFT,
                            board_y * self.grid_size + PADDING_TOP,
                            self.grid_size - 1,
                            self.grid_size - 1,
                        ]
                        pygame.draw.rect(self.screen, color, rect)

    def draw_next_piece(self):
        """Draws the next piece in a small preview box."""
        if self.next_piece is None:
            return

        # Scale down the grid size for the preview
        preview_grid_size = int(self.grid_size * 0.7)
        box_size = preview_grid_size * 5

        center_x = PADDING_LEFT // 2
        box_x = center_x - (box_size // 2)

        # Draw "Next Piece" title at the original top position
        title_y = PADDING_TOP
        title_text = self.font.render("Next Piece", True, (255, 255, 255))
        title_rect = title_text.get_rect(centerx=center_x, top=title_y)
        self.screen.blit(title_text, title_rect)

        # Shift the preview box below the title
        box_y = title_y + title_text.get_height() + 8

        pygame.draw.rect(
            self.screen,
            (20, 20, 20),
            [box_x, box_y, box_size, box_size],
        )

        shape = self.next_piece.active_shape
        cols = [idx % 4 for idx in shape]
        rows = [idx // 4 for idx in shape]

        min_col, max_col = min(cols), max(cols)
        min_row, max_row = min(rows), max(rows)

        piece_width_px = (max_col - min_col + 1) * preview_grid_size
        piece_height_px = (max_row - min_row + 1) * preview_grid_size

        offset_x = (
            box_x + (box_size - piece_width_px) // 2 - (min_col * preview_grid_size)
        )
        offset_y = (
            box_y + (box_size - piece_height_px) // 2 - (min_row * preview_grid_size)
        )

        color = self.id_to_color[self.next_piece.id]

        for i in range(4):
            for j in range(4):
                if i * 4 + j in shape:
                    rect = [
                        offset_x + j * preview_grid_size,
                        offset_y + i * preview_grid_size,
                        preview_grid_size - 1,
                        preview_grid_size - 1,
                    ]
                    pygame.draw.rect(self.screen, color, rect)

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
        max_height = np.max(heights)
        bumpiness = np.sum(np.abs(heights[:-1] - heights[1:]))

        return max_height, sum_height, holes, bumpiness

    def set_active_pieces(self, piece_names):
        """Updates the pool of figures the environment is allowed to spawn."""
        self.figure_pool = {name: FIGURES[name] for name in piece_names}
