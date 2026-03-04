import os
import numpy as np
import pygame
from tetris.tetris_shape import FIGURES, FIGURES_BY_ID, TetrisPiece
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

        self.figure_pool = FIGURES

        self.screen = None
        self.clock = None
        self.font = None

        self.board = None
        self.active_piece = None
        self.next_piece = None
        self.state = None
        self.step_counter = 0
        self.lines_cleared = 0

        self.reset()

    def reset(self):
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.active_piece = None
        self.next_piece = None
        self.state = "start"
        self.step_counter = 0
        self.lines_cleared = 0

        self.spawn_new_piece()

        return self.get_state(), {}

    def generate_new_piece(self, id=None):
        if id is not None:
            shape_name = next(
                name
                for name, details in self.figure_pool.items()
                if details["id"] == id
            )
        else:
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

        # If the newly spawned piece immediately collides, the game is over
        if self.check_collision(
            self.active_piece.active_shape, self.active_piece.x, self.active_piece.y
        ):
            self.state = "gameover"

    def step(self, action):
        """
        Steps the environment
        """
        self.step_counter += 1
        lines_cleared_this_step = 0

        # Player Action
        self.handle_action(action)

        # Gravity / Locking
        if not self.check_collision(
            self.active_piece.active_shape, self.active_piece.x, self.active_piece.y + 1
        ):
            self.active_piece.y += 1
        else:
            self.freeze_shape()
            lines_cleared_this_step = self.clear_lines()
            self.spawn_new_piece()

        terminated = self.state == "gameover"

        return (
            self.get_state(),
            lines_cleared_this_step,
            terminated,
            False,
            {"step_counter": self.step_counter},
        )

    def handle_action(self, action):
        """Processes horizontal movement, rotation, and hard drop."""
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
            self.active_piece.y = self.get_drop_position()

    def get_drop_position(self, piece=None):
        if piece is None:
            piece = self.active_piece

        drop_y = piece.y
        while not self.check_collision(piece.active_shape, piece.x, drop_y + 1):
            drop_y += 1
        return drop_y

    def get_state(self):
        """Returns a clean dictionary of the raw game state."""
        return {
            "board": self.board.copy(),
            "active_piece": self.active_piece,
            "next_piece": self.next_piece,
        }

    def can_rotate(self, new_shape):
        return not self.check_collision(
            new_shape, self.active_piece.x, self.active_piece.y
        )

    def check_collision(self, shape, offset_x, offset_y, board=None):
        if board is None:
            board = self.board

        for i in range(4):
            for j in range(4):
                if i * 4 + j in shape:
                    board_y = offset_y + i
                    board_x = offset_x + j

                    if board_x < 0 or board_x >= self.width or board_y >= self.height:
                        return True
                    if board_y >= 0 and board[board_y, board_x] > 0:
                        return True
        return False

    def is_valid_state(self, state, piece_id, board=None):
        if board is None:
            board = self.board

        figure = FIGURES_BY_ID[int(piece_id)]
        shape = figure["shape"][state[2]]
        return not self.check_collision(shape, state[0], state[1])

    def freeze_shape(self):
        shape = self.active_piece.active_shape
        for i in range(4):
            for j in range(4):
                if i * 4 + j in shape:
                    board_y = self.active_piece.y + i
                    board_x = self.active_piece.x + j

                    if board_y >= 0:
                        self.board[board_y, board_x] = self.active_piece.id

    def clear_lines(self):
        full_rows = np.all(self.board > 0, axis=1)
        num_cleared = np.sum(full_rows)

        if num_cleared > 0:
            remaining_board = self.board[~full_rows]
            empty_rows = np.zeros((num_cleared, self.width), dtype=int)
            self.board = np.vstack((empty_rows, remaining_board))
            self.lines_cleared += num_cleared

        return num_cleared

    def render(self, ghost_piece=False, tick=True):
        if self.screen is None:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
            pygame.display.set_caption("MuZero Tetris")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(
                os.path.join(
                    os.path.dirname(__file__), "..", "assets", "Jersey20-Regular.ttf"
                ),
                25,
            )
            self.id_to_color = {
                details["id"]: details["color"] for details in self.figure_pool.values()
            }

        self.screen.fill(BACKGROUND_COLOR)

        inner_rect = [
            PADDING_LEFT,
            PADDING_TOP,
            self.width * self.grid_size,
            self.height * self.grid_size,
        ]
        pygame.draw.rect(self.screen, (20, 20, 20), inner_rect)

        for i in range(self.height):
            for j in range(self.width):
                block_id = self.board[i, j]
                if block_id > 0:
                    rect = [
                        j * self.grid_size + PADDING_LEFT,
                        i * self.grid_size + PADDING_TOP,
                        self.grid_size - 1,
                        self.grid_size - 1,
                    ]
                    pygame.draw.rect(self.screen, self.id_to_color[block_id], rect)

        if self.active_piece is not None:
            if not ghost_piece:
                drop_y = self.get_drop_position()
                self.draw_piece(
                    self.active_piece.active_shape,
                    self.active_piece.x,
                    drop_y,
                    (100, 100, 100),
                )
            else:
                self.draw_piece(
                    ghost_piece.active_shape,
                    ghost_piece.x,
                    ghost_piece.y,
                    (100, 100, 100),
                )

            self.draw_piece(
                self.active_piece.active_shape,
                self.active_piece.x,
                self.active_piece.y,
                self.id_to_color[self.active_piece.id],
            )

        self.draw_next_piece()

        step_text = self.font.render(
            f"Steps: {self.step_counter}", True, (255, 255, 255)
        )
        self.screen.blit(step_text, [15, self.window_height - 80])

        lines_text = self.font.render(
            f"Lines: {self.lines_cleared}", True, (255, 255, 255)
        )
        self.screen.blit(lines_text, [15, self.window_height - 110])

        pygame.display.flip()

        if tick:
            self.clock.tick(self.tick_speed)

    def draw_piece(self, shape, offset_x, offset_y, color):
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
        if self.next_piece is None:
            return

        preview_grid_size = int(self.grid_size * 0.7)
        box_size = preview_grid_size * 5
        center_x = PADDING_LEFT // 2
        box_x = center_x - (box_size // 2)
        title_y = PADDING_TOP

        title_text = self.font.render("Next Piece", True, (255, 255, 255))
        title_rect = title_text.get_rect(centerx=center_x, top=title_y)
        self.screen.blit(title_text, title_rect)

        box_y = title_y + title_text.get_height() + 8
        pygame.draw.rect(self.screen, (20, 20, 20), [box_x, box_y, box_size, box_size])

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
