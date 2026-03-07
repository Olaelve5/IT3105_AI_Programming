import pygame
import random
from config import BOARD_WIDTH, BOARD_HEIGHT, GRID_SIZE

BACKGROUND_COLOR = (34, 45, 61)
HEAD_COLOR = (255, 0, 93)
BODY_COLOR = (0, 247, 255)
FRUIT_COLOR = (0, 255, 0)
GRID_SIZE = 18


class SnakeEnv:
    def __init__(self, width=BOARD_WIDTH, height=BOARD_HEIGHT, grid_size=GRID_SIZE):
        self.scale = 2
        self.width = width * grid_size * self.scale
        self.height = height * grid_size * self.scale
        self.grid_size = grid_size * self.scale
        self.max_body_length = (width * height) - 1

        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.screen.fill(BACKGROUND_COLOR)
        self.head_pos = self.width // 2, self.height // 2
        self.body_positions = [self.head_pos]
        self.body_length = 0
        self.direction = (0, -1)
        self.fruit_pos = self.spawn_fruit()
        self.game_over = False

    def step(self, action):
        self.handle_actions(action)

        new_head_pos = (
            self.head_pos[0] + self.direction[0] * self.grid_size,
            self.head_pos[1] + self.direction[1] * self.grid_size,
        )

        if self.check_collision(new_head_pos):
            self.game_over = True

        if self.check_fruit_collision():
            self.body_length += 1
            self.fruit_pos = self.spawn_fruit()
            ate_fruit = True
        else:
            ate_fruit = False

        self.head_pos = new_head_pos
        self.body_positions.append(self.head_pos)
        self.update_body_positions()

        return self.head_pos, self.game_over, ate_fruit

    def handle_actions(self, action):
        """
        There are three actions:
        - Change direction to left
        - Change direction to right
        - Do nothing, continue forwards
        """
        x, y = self.direction

        if action == 0:
            # Turn Left
            self.direction = (y, -x)

        elif action == 1:
            # Turn Right
            self.direction = (-y, x)

        elif action == 2:
            pass

    def check_collision(self, pos):
        if pos in self.body_positions:
            return True
        if not (0 <= pos[0] < self.width and 0 <= pos[1] < self.height):
            return True
        return False

    def update_body_positions(self):
        if len(self.body_positions) > self.body_length + 1:
            self.body_positions.pop(0)

    def render(self):
        self.screen.fill(BACKGROUND_COLOR)

        pygame.draw.rect(
            self.screen,
            FRUIT_COLOR,
            (self.fruit_pos[0], self.fruit_pos[1], self.grid_size, self.grid_size),
        )

        for pos in self.body_positions[:-1]:
            pygame.draw.rect(
                self.screen,
                BODY_COLOR,
                (pos[0], pos[1], self.grid_size, self.grid_size),
            )

        pygame.draw.rect(
            self.screen,
            HEAD_COLOR,
            (self.head_pos[0], self.head_pos[1], self.grid_size, self.grid_size),
        )

        pygame.display.flip()
        self.clock.tick(5)

    def spawn_fruit(self):
        grid_w = self.width // self.grid_size
        grid_h = self.height // self.grid_size

        occupied = set(self.body_positions)
        if hasattr(self, "head_pos"):
            occupied.add(self.head_pos)

        empty_positions = []
        for x in range(grid_w):
            for y in range(grid_h):
                pos = (x * self.grid_size, y * self.grid_size)
                if pos not in occupied:
                    empty_positions.append(pos)

        if not empty_positions:
            return None

        return random.choice(empty_positions)

    def check_fruit_collision(self):
        return self.head_pos == self.fruit_pos


if __name__ == "__main__":
    pygame.init()
    env = SnakeEnv()

    running = True
    while running:
        action = 2

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            # Determine if left is a left turn or right turn relative to current direction
            dx, dy = env.direction
            left_dir = (dy, -dx)
            right_dir = (-dy, dx)
            target = (-1, 0)
            if target == left_dir:
                action = 0
            elif target == right_dir:
                action = 1
            # else: opposite direction, ignore
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx, dy = env.direction
            left_dir = (dy, -dx)
            right_dir = (-dy, dx)
            target = (1, 0)
            if target == left_dir:
                action = 0
            elif target == right_dir:
                action = 1
        elif keys[pygame.K_UP] or keys[pygame.K_w]:
            dx, dy = env.direction
            left_dir = (dy, -dx)
            right_dir = (-dy, dx)
            target = (0, -1)
            if target == left_dir:
                action = 0
            elif target == right_dir:
                action = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dx, dy = env.direction
            left_dir = (dy, -dx)
            right_dir = (-dy, dx)
            target = (0, 1)
            if target == left_dir:
                action = 0
            elif target == right_dir:
                action = 1

        if not env.game_over:
            env.step(action)
            env.render()
        else:
            # Press R to restart
            if keys[pygame.K_r]:
                env.reset()

    pygame.quit()
