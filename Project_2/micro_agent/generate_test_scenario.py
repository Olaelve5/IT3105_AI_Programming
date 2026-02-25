import random
import numpy as np
from tetris.tetris_env import TetrisEnv
import pygame


def generate_test_scenario(env):
    """
    Generates a guaranteed-reachable sandbox scenario.
    """
    env.reset()

    # Build a random landscape
    num_garbage_pieces = random.randint(3, 15)
    for _ in range(num_garbage_pieces):
        direction = random.choice([1, 2])
        moves = random.randint(0, 5)

        for _ in range(moves):
            env.step(direction)

        for _ in range(random.randint(1, 5)):
            action = random.choice([0, 1, 2, 3])
            env.step(action)

        if np.any(env.board[:6, :] != 0):
            break

        env.step(4)

    # Wipe the top 4 rows to ensure the new piece can spawn
    env.board[:4, :] = 0

    base_board = env.board.copy()

    # change all non-zero values to 1, so we can ignore the colors
    base_board[base_board != 0] = 1

    test_piece = env.generate_new_piece()
    env.active_piece = test_piece

    direction = random.choice([1, 2])
    moves = random.randint(0, 3)
    for _ in range(moves):
        env.step(direction, spawn_new_piece=False, clear_lines=False)

    # Generate the target position
    for _ in range(random.randint(0, 10)):
        env.step(random.choice([0, 1, 2, 3]), spawn_new_piece=False, clear_lines=False)

    env.step(4, spawn_new_piece=False, clear_lines=False)

    target_pos = (test_piece.x, test_piece.y)
    target_rot = test_piece.rotation

    return base_board, test_piece.id, target_pos, target_rot


def load_scenario(env):
    scenario = generate_test_scenario(env)
    env.board = scenario[0]
    env.active_piece = env.generate_new_piece(scenario[1])
    ghost_piece = env.generate_new_piece(scenario[1])
    ghost_piece.x, ghost_piece.y = scenario[2]
    ghost_piece.rotation = scenario[3]
    ghost_piece.active_shape = ghost_piece.shapes[ghost_piece.rotation]
    return ghost_piece


if __name__ == "__main__":

    env = TetrisEnv()
    ghost_piece = load_scenario(env)
    env.render(tick=False, ghost_piece=ghost_piece)

    REFRESH_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(REFRESH_EVENT, 300)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == REFRESH_EVENT:
                ghost_piece = load_scenario(env)
                env.render(tick=False, ghost_piece=ghost_piece)

    env.close()
