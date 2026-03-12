import pygame
import random
from config import BOARD_WIDTH, BOARD_HEIGHT, GRID_SIZE

BACKGROUND_COLOR = (34, 45, 61)
HEAD_COLOR = (252, 186, 3)
BODY_COLOR = (0, 247, 255)
WALL_COLOR = (255, 255, 255)


class TronEnv:
    def __init__(self, width=BOARD_WIDTH, height=BOARD_HEIGHT, grid_size=GRID_SIZE):
        self.width = width * grid_size
        self.height = height * grid_size
        self.grid_size = grid_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.screen.fill(BACKGROUND_COLOR)
        self.head_pos = (self.width // 2, self.height // 2)
        self.walls = []
        self.body_positions = [self.head_pos]
        self.direction = (0, -1)
        self.score = 0
        self.game_over = False
        self.place_random_walls(num_walls=random.randint(1, 3))

    def step(self, action):
        self.handle_actions(action)

        new_head_pos = (
            self.head_pos[0] + self.direction[0] * self.grid_size,
            self.head_pos[1] + self.direction[1] * self.grid_size,
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
            self.direction = (y, -x)

        elif action == 1:
            self.direction = (-y, x)

        elif action == 2:
            pass

    def check_collision(self, pos):
        if pos in self.body_positions:
            return True
        if not (0 <= pos[0] < self.width and 0 <= pos[1] < self.height):
            return True
        if pos in self.walls:
            return True
        return False

    def render(self):
        if not pygame.get_init():
            pygame.init()

        padding_top = 40
        total_height = self.height + padding_top

        if self.screen.get_height() != total_height:
            self.screen = pygame.display.set_mode((self.width, total_height))

        self.screen.fill(BACKGROUND_COLOR)

        self.draw_header(padding_top)
        self.draw_board(padding_top)

        pygame.display.flip()
        self.clock.tick(30)

    def draw_header(self, padding_top):
        header_color = (255, 255, 255)
        pygame.draw.rect(self.screen, header_color, (0, 0, self.width, padding_top))

        font = pygame.font.SysFont("monospace", 24, bold=True)
        score_text = font.render(f"Length: {len(self.body_positions)}", True, (0, 0, 0))
        text_rect = score_text.get_rect(center=(self.width // 2, padding_top // 2))
        self.screen.blit(score_text, text_rect)

    def draw_board(self, padding_top):
        for pos in self.body_positions[:-1]:
            pygame.draw.rect(
                self.screen,
                BODY_COLOR,
                (pos[0], pos[1] + padding_top, self.grid_size, self.grid_size),
            )

        pygame.draw.rect(
            self.screen,
            HEAD_COLOR,
            (
                self.head_pos[0],
                self.head_pos[1] + padding_top,
                self.grid_size,
                self.grid_size,
            ),
        )

        for wall in self.walls:
            pygame.draw.rect(
                self.screen,
                WALL_COLOR,
                (wall[0], wall[1] + padding_top, self.grid_size, self.grid_size),
            )

        for x in range(0, self.width + 1, self.grid_size):
            pygame.draw.line(
                self.screen,
                (50, 50, 50),
                (x, padding_top),
                (x, self.height + padding_top),
                1,
            )
        for y in range(0, self.height + 1, self.grid_size):
            pygame.draw.line(
                self.screen,
                (50, 50, 50),
                (0, y + padding_top),
                (self.width, y + padding_top),
                1,
            )

    def place_random_walls(self, num_walls):
        min_size = 4
        max_size = self.width // self.grid_size // 3
        safe_zone = set()
        hx, hy = self.head_pos

        for dist in range(1, max_size + 2):
            for offset in range(-1, 2):
                safe_pos = (
                    hx + offset * self.grid_size,
                    hy + self.direction[1] * dist * self.grid_size,
                )
                safe_zone.add(safe_pos)
        safe_zone.add(self.head_pos)

        for _ in range(num_walls):
            wall_length = random.randint(min_size, max_size)
            wall_width = random.randint(1, 2)
            wall_direction = random.choice(["horizontal", "vertical"])

            max_attempts = 50
            for _ in range(max_attempts):
                wall_positions = []

                if wall_direction == "horizontal":
                    wall_x = (
                        random.randint(0, self.width // self.grid_size - wall_length)
                        * self.grid_size
                    )
                    wall_y = (
                        random.randint(0, self.height // self.grid_size - wall_width)
                        * self.grid_size
                    )
                    for i in range(wall_length):
                        for j in range(wall_width):
                            wall_pos = (
                                wall_x + i * self.grid_size,
                                wall_y + j * self.grid_size,
                            )
                            wall_positions.append(wall_pos)
                else:
                    wall_x = (
                        random.randint(0, self.width // self.grid_size - wall_width)
                        * self.grid_size
                    )
                    wall_y = (
                        random.randint(0, self.height // self.grid_size - wall_length)
                        * self.grid_size
                    )
                    for i in range(wall_length):
                        for j in range(wall_width):
                            wall_pos = (
                                wall_x + j * self.grid_size,
                                wall_y + i * self.grid_size,
                            )
                            wall_positions.append(wall_pos)

                if not any(pos in safe_zone for pos in wall_positions):
                    self.walls.extend(wall_positions)
                    break

    def draw_walls(self):
        for wall in self.walls:
            pygame.draw.rect(
                self.screen,
                WALL_COLOR,
                (wall[0], wall[1], self.grid_size, self.grid_size),
            )


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
            dx, dy = env.direction
            left_dir = (dy, -dx)
            right_dir = (-dy, dx)
            target = (-1, 0)
            if target == left_dir:
                action = 0
            elif target == right_dir:
                action = 1
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
            if keys[pygame.K_r]:
                env.reset()

    pygame.quit()
