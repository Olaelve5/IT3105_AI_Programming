from config import BOARD_WIDTH, BOARD_HEIGHT, GRID_SIZE
import pygame
import numpy as np

BACKGROUND_COLOR = (30, 30, 36)

TILE_COLORS = {
    0: (43, 43, 54),
    2: (76, 86, 106),
    4: (94, 129, 172),
    8: (129, 161, 193),
    16: (136, 192, 208),
    32: (143, 188, 187),
    64: (163, 190, 140),
    128: (235, 203, 139),
    256: (208, 135, 112),
    512: (191, 97, 106),
    1024: (180, 142, 173),
    2048: (229, 233, 240),
    "other": (216, 222, 233),
}

TEXT_COLORS = {
    16: (43, 43, 54),
    32: (43, 43, 54),
    64: (43, 43, 54),
    128: (43, 43, 54),
    256: (43, 43, 54),
    2048: (43, 43, 54),
    "other": (236, 239, 244),
}


class Game2048Env:
    def __init__(self, width=BOARD_WIDTH, height=BOARD_HEIGHT, grid_size=GRID_SIZE):
        self.width = width * grid_size
        self.height = height * grid_size
        self.grid_size = grid_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.font = pygame.font.SysFont("arial", 40, bold=True)

        self.reset()

    def reset(self):
        self.screen.fill(BACKGROUND_COLOR)
        self.score = 0
        self.game_over = False
        self.board = None
        self.initialize_board()

    def step(self, action):
        if not self.can_move():
            self.game_over = True
            return

        old_score = self.score
        valid_move = self.handle_action(action)
        step_score = self.score - old_score

        return valid_move, step_score, self.game_over

    def render(self):
        self.screen.fill(BACKGROUND_COLOR)

        rows, cols = self.board.shape
        margin = 5

        for row in range(rows):
            for col in range(cols):
                val = self.board[row, col]

                rect_color = TILE_COLORS.get(val, TILE_COLORS["other"])
                text_color = TEXT_COLORS.get(val, TEXT_COLORS["other"])

                x = col * self.grid_size + margin
                y = row * self.grid_size + margin
                w = self.grid_size - (2 * margin)
                h = self.grid_size - (2 * margin)

                pygame.draw.rect(self.screen, rect_color, (x, y, w, h), border_radius=8)

                if val > 0:
                    text_surface = self.font.render(str(val), True, text_color)
                    text_rect = text_surface.get_rect(center=(x + w / 2, y + h / 2))
                    self.screen.blit(text_surface, text_rect)

        pygame.display.flip()

    def handle_action(self, action):
        """0: Up, 1: Down, 2: Left, 3: Right"""
        original_board = self.board.copy()

        rotations = {2: 0, 0: 1, 3: 2, 1: 3}
        k = rotations[action]

        rotated_board = np.rot90(self.board, k=k)

        new_board = np.zeros_like(rotated_board)
        for i in range(4):
            new_board[i] = self.slide_and_merge_row(rotated_board[i])

        self.board = np.rot90(new_board, k=-k)

        if np.array_equal(self.board, original_board):
            return False
        else:
            self.spawn_new_tile()
            return True

    def slide_and_merge_row(self, row):
        non_zero = [val for val in row if val != 0]

        merged = []
        skip = False
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i < len(non_zero) - 1 and non_zero[i] == non_zero[i + 1]:
                merged.append(non_zero[i] * 2)
                self.score += non_zero[i] * 2
                skip = True
            else:
                merged.append(non_zero[i])

        return merged + [0] * (len(row) - len(merged))

    def initialize_board(self):
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.spawn_new_tile()
        self.spawn_new_tile()

    def spawn_new_tile(self):
        empty_cells = np.argwhere(self.board == 0)
        if len(empty_cells) > 0:
            x, y = empty_cells[np.random.choice(len(empty_cells))]
            self.board[x, y] = 2 if np.random.rand() < 0.9 else 4

    def can_move(self):
        if np.any(self.board == 0):
            return True

        for i in range(4):
            for j in range(3):
                if self.board[i, j] == self.board[i, j + 1]:
                    return True
                if self.board[j, i] == self.board[j + 1, i]:
                    return True

        return False


if __name__ == "__main__":
    pygame.init()
    env = Game2048Env()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    env.step(0)
                elif event.key == pygame.K_DOWN:
                    env.step(1)
                elif event.key == pygame.K_LEFT:
                    env.step(2)
                elif event.key == pygame.K_RIGHT:
                    env.step(3)

        env.render()
        env.clock.tick(60)

    pygame.quit()
