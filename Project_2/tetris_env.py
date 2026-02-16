import pygame
import random
import numpy as np

# Colors
COLORS = [
    (0, 0, 0),
    (120, 37, 179),
    (100, 179, 179),
    (80, 34, 22),
    (80, 134, 22),
    (180, 34, 22),
    (180, 34, 178),
    (24, 60, 28),
]


class TetrisEnv:
    def __init__(self, height=20, width=10, grid_size=30):
        self.height = height
        self.width = width
        self.grid_size = grid_size
        self.window_width = self.width * self.grid_size
        self.window_height = self.height * self.grid_size

        # Standard Tetris shapes
        self.figures = [
            [[1, 5, 9, 13], [4, 5, 6, 7]],
            [[4, 5, 9, 10], [2, 6, 5, 9]],
            [[6, 7, 9, 10], [1, 5, 6, 10]],
            [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],
            [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],
            [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],
            [[1, 2, 5, 6]],
        ]

        self.reset()
        self.screen = None
        self.clock = None
        self.font = None

    def reset(self):
        self.field = [[0] * self.width for _ in range(self.height)]
        self.score = 0
        self.state = "start"
        self.figure = None
        self.x = 0
        self.y = 0
        self.rotation = 0
        self.current_color = 1
        self.new_figure()
        return self._get_observation(), {}

    def new_figure(self):
        choice = random.randint(0, 6)
        self.figure = self.figures[choice]
        self.current_color = choice + 1
        self.x = 3
        self.y = 0
        self.rotation = 0

    def intersects(self):
        intersection = False
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure[self.rotation]:
                    if (
                        i + self.y > self.height - 1
                        or j + self.x > self.width - 1
                        or j + self.x < 0
                        or self.field[i + self.y][j + self.x] > 0
                    ):
                        intersection = True
        return intersection

    def freeze(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure[self.rotation]:
                    self.field[i + self.y][j + self.x] = self.current_color
        self.break_lines()
        self.new_figure()
        if self.intersects():
            self.state = "gameover"

    def break_lines(self):
        lines = 0
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                lines += 1
                for k in range(i, 1, -1):
                    for j in range(self.width):
                        self.field[k][j] = self.field[k - 1][j]
        self.score += lines**2

    def step(self, action):
        reward = 0
        prev_score = self.score

        if action == 1:  # Left
            self.x -= 1
            if self.intersects():
                self.x += 1
        elif action == 2:  # Right
            self.x += 1
            if self.intersects():
                self.x -= 1
        elif action == 3:  # Rotate
            old_rotation = self.rotation
            self.rotation = (self.rotation + 1) % len(self.figure)
            if self.intersects():
                self.rotation = old_rotation

        if action == 4:  # Drop fast
            self.y += 1
            if self.intersects():
                self.y -= 1
                self.freeze()
        else:
            self.y += 1
            if self.intersects():
                self.y -= 1
                self.freeze()

        reward = self.score - prev_score
        terminated = self.state == "gameover"

        if terminated:
            reward = -10

        return self._get_observation(), reward, terminated, False, {}

    def _get_observation(self):
        obs = np.array(self.field)
        temp_field = [row[:] for row in self.field]
        if self.figure is not None:
            for i in range(4):
                for j in range(4):
                    if i * 4 + j in self.figure[self.rotation]:
                        if (
                            0 <= i + self.y < self.height
                            and 0 <= j + self.x < self.width
                        ):
                            # Mark falling piece as 1 (or self.current_color if you want color in inputs)
                            temp_field[i + self.y][j + self.x] = 1
        return np.array(temp_field)

    def render(self):
        if self.screen is None:
            pygame.init()
            pygame.font.init()  # Ensure font module is ready
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 25, bold=True)

        self.screen.fill((0, 0, 0))

        # Draw Field
        for i in range(self.height):
            for j in range(self.width):
                if self.field[i][j] > 0:
                    pygame.draw.rect(
                        self.screen,
                        COLORS[self.field[i][j]],
                        [
                            j * self.grid_size,
                            i * self.grid_size,
                            self.grid_size - 1,
                            self.grid_size - 1,
                        ],
                    )

        # Draw Figure
        if self.figure is not None:
            for i in range(4):
                for j in range(4):
                    if i * 4 + j in self.figure[self.rotation]:
                        pygame.draw.rect(
                            self.screen,
                            COLORS[self.current_color],
                            [
                                (j + self.x) * self.grid_size,
                                (i + self.y) * self.grid_size,
                                self.grid_size - 1,
                                self.grid_size - 1,
                            ],
                        )

        text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(text, [10, 10])

        pygame.display.flip()
        self.clock.tick(5)

    def close(self):
        if self.screen is not None:
            pygame.quit()
