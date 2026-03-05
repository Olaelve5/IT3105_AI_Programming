import pygame
import random

BACKGROUND_COLOR = (34, 45, 61)
HEAD_COLOR = (255, 255, 255)
BODY_COLOR = (0, 247, 255)
GRID_SIZE = 18


class TronEnv:
    def __init__(self, width=600, height=600):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.screen.fill(BACKGROUND_COLOR)
        self.head_pos = self.get_random_start_position()
        self.body_positions = [self.head_pos]
        self.direction = (0, -1)
        self.score = 0
        self.game_over = False

    def step(self, action):
        self.handle_actions(action)

        new_head_pos = (
            self.head_pos[0] + self.direction[0] * GRID_SIZE,
            self.head_pos[1] + self.direction[1] * GRID_SIZE,
        )

        if self.check_collision(new_head_pos):
            self.game_over = True

        self.head_pos = new_head_pos
        self.body_positions.append(self.head_pos)

        self.score += 1

        return self.head_pos, self.score, self.game_over

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

    def get_random_start_position(self):
        x = random.randint(0, self.width // GRID_SIZE - 1) * GRID_SIZE
        y = random.randint(0, self.height // GRID_SIZE - 1) * GRID_SIZE
        return (x, y)

    def check_collision(self, pos):
        if pos in self.body_positions:
            return True
        if not (0 <= pos[0] < self.width and 0 <= pos[1] < self.height):
            return True
        return False

    def render(self):
        self.screen.fill(BACKGROUND_COLOR)

        for pos in self.body_positions[:-1]:
            pygame.draw.rect(
                self.screen, BODY_COLOR, (pos[0], pos[1], GRID_SIZE, GRID_SIZE)
            )

        pygame.draw.rect(
            self.screen,
            HEAD_COLOR,
            (self.head_pos[0], self.head_pos[1], GRID_SIZE, GRID_SIZE),
        )

        pygame.display.flip()
        self.clock.tick(10)


if __name__ == "__main__":
    pygame.init()
    env = TronEnv()

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
