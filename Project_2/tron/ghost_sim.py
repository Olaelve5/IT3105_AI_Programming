import pygame
from tron.tron_env import TronEnv  # Adjust import to your file structure

# Colors
GHOST_HEAD_COLOR = (150, 150, 150)
GHOST_BODY_COLOR = (70, 70, 70)


class TronController:
    def __init__(self):
        pygame.init()
        self.env = TronEnv()
        # We store ghost state variables separately instead of a whole new Env
        self.is_ghost_mode = False
        self.ghost_head = None
        self.ghost_body = []
        self.ghost_dir = None

    def toggle_ghost_mode(self):
        self.is_ghost_mode = not self.is_ghost_mode
        if self.is_ghost_mode:
            # Sync ghost starting point to current reality
            self.ghost_head = self.env.head_pos
            self.ghost_body = list(self.env.body_positions)
            self.ghost_dir = self.env.direction
        else:
            # Clear ghost data
            self.ghost_head = None
            self.ghost_body = []

    def handle_ghost_move(self, action):
        # Logic to move the ghost one step (simplified version of env.step)
        x, y = self.ghost_dir
        if action == 0:
            self.ghost_dir = (y, -x)  # Left
        elif action == 1:
            self.ghost_dir = (-y, x)  # Right

        # Calculate new pos
        new_pos = (
            self.ghost_head[0] + self.ghost_dir[0] * self.env.grid_size,
            self.ghost_head[1] + self.ghost_dir[1] * self.env.grid_size,
        )

        # We don't worry about "dying" in ghost mode for the demo
        self.ghost_head = new_pos
        self.ghost_body.append(new_pos)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.toggle_ghost_mode()

                    action = None
                    if event.key in [pygame.K_LEFT, pygame.K_a]:
                        action = 0
                    elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                        action = 1
                    elif event.key in [pygame.K_UP, pygame.K_w]:
                        action = 2

                    if action is not None:
                        if self.is_ghost_mode:
                            self.handle_ghost_move(action)
                        elif not self.env.game_over:
                            self.env.step(action)

            self.render_combined()

    def render_combined(self):
        padding_top = 40

        # 1. Draw the actual game state, but DON'T flip the display yet
        self.env.render(do_flip=False)

        # 2. Draw the Ghost data ON TOP of the same screen
        if self.is_ghost_mode:

            # --- THE FIX IS HERE ---
            # Get the length of the real body
            real_len = len(self.env.body_positions)

            # Slice the list: Start AFTER the real body ends, and stop before the ghost head
            for pos in self.ghost_body[real_len:-1]:
                pygame.draw.rect(
                    self.env.screen,
                    GHOST_BODY_COLOR,
                    (
                        pos[0],
                        pos[1] + padding_top,
                        self.env.grid_size,
                        self.env.grid_size,
                    ),
                )

            # Only draw the ghost head if it has stepped away from the real head
            if self.ghost_head != self.env.head_pos:
                pygame.draw.rect(
                    self.env.screen,
                    GHOST_HEAD_COLOR,
                    (
                        self.ghost_head[0],
                        self.ghost_head[1] + padding_top,
                        self.env.grid_size,
                        self.env.grid_size,
                    ),
                )

        # 3. NOW update the full display and tick the clock
        pygame.display.flip()
        self.env.clock.tick(20)


if __name__ == "__main__":
    controller = TronController()
    controller.run()
