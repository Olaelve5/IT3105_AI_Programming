from config import BOARD_WIDTH, BOARD_HEIGHT, GRID_SIZE
import pygame
import numpy as np
import random

BACKGROUND_COLOR = (30, 30, 36)
PLAYER1_COLOR = (255, 0, 0)
PLAYER2_COLOR = (255, 255, 0)
EMPTY_COLOR = (43, 43, 54)


class Connect4Env:
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
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.current_player = 1
        self.game_over = False

    def step(self, action):
        """Returns (valid_move, win, done)"""
        if self.game_over:
            return False, False, True

        if not self.is_valid_action(action):
            return False, False, False

        row = self.get_next_open_row(action)
        self.board[row][action] = self.current_player

        if self.check_win(self.current_player):
            self.game_over = True
            return True, True, True

        if np.all(self.board != 0):
            self.game_over = True
            return True, False, True

        self.current_player = 2 if self.current_player == 1 else 1

        return True, False, False

    def is_valid_action(self, action):
        return self.board[0][action] == 0

    def get_next_open_row(self, action):
        for r in range(BOARD_HEIGHT - 1, -1, -1):
            if self.board[r][action] == 0:
                return r
        return None

    def check_win(self, player):
        # horizontal
        for c in range(BOARD_WIDTH - 3):
            for r in range(BOARD_HEIGHT):
                if all(self.board[r][c + i] == player for i in range(4)):
                    return True

        # vertical
        for c in range(BOARD_WIDTH):
            for r in range(BOARD_HEIGHT - 3):
                if all(self.board[r + i][c] == player for i in range(4)):
                    return True

        # diagonal up
        for c in range(BOARD_WIDTH - 3):
            for r in range(BOARD_HEIGHT - 3):
                if all(self.board[r + i][c + i] == player for i in range(4)):
                    return True

        # diagonal down
        for c in range(BOARD_WIDTH - 3):
            for r in range(3, BOARD_HEIGHT):
                if all(self.board[r - i][c + i] == player for i in range(4)):
                    return True

        return False

    def render(self):
        self.screen.fill(BACKGROUND_COLOR)

        for c in range(BOARD_WIDTH):
            for r in range(BOARD_HEIGHT):
                color = EMPTY_COLOR
                if self.board[r][c] == 1:
                    color = PLAYER1_COLOR
                elif self.board[r][c] == 2:
                    color = PLAYER2_COLOR

                pygame.draw.circle(
                    self.screen,
                    color,
                    (
                        c * self.grid_size + self.grid_size // 2,
                        r * self.grid_size + self.grid_size // 2,
                    ),
                    self.grid_size // 2 - 5,
                )

        pygame.display.flip()

    def get_valid_actions(self):
        return [a for a in range(BOARD_WIDTH) if self.board[0][a] == 0]


if __name__ == "__main__":

    env = Connect4Env()
    env.render()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not env.game_over and running:
            valid_actions = env.get_valid_actions()

            if valid_actions:
                action = random.choice(valid_actions)
                player_who_moved = env.current_player
                valid_move, reward, done = env.step(action)
                env.render()
                pygame.time.delay(500)

                if done:
                    if reward == 1:
                        print(f"Player {player_who_moved} wins!")
                    else:
                        print("It's a draw!")
                    pygame.time.delay(2000)
                    env.reset()
                    env.render()

    pygame.quit()
