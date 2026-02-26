import pygame
from tetris.tetris_env import TetrisEnv
from Project_2.micro_agent.ScenarioGenerator import ScenarioGenerator

env = TetrisEnv(tick_speed=5)
observation, info = env.reset()

print("Controls: Left, Right, Up (Rotate), Down (Fast Drop). Q to quit.")


def load_scenario(env, scenario):
    env.board = scenario["board"]
    env.active_piece = env.generate_new_piece(scenario["piece_id"])
    ghost_piece = env.generate_new_piece(scenario["piece_id"])
    ghost_piece.x, ghost_piece.y = scenario["target_pos"]
    ghost_piece.rotation = scenario["target_rot"]
    ghost_piece.active_shape = ghost_piece.shapes[ghost_piece.rotation]
    return ghost_piece


running = True
total_reward = 0
test_scenario = ScenarioGenerator().get_random_tricky_scenario()
load_scenario(env, test_scenario)

while running:
    env.render()

    action = 0  # 0 = no action

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_LEFT:
                action = 1
            elif event.key == pygame.K_RIGHT:
                action = 2
            elif event.key == pygame.K_UP:
                action = 3
            elif event.key == pygame.K_SPACE:
                action = 4

    # Step the environment
    next_state, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    if reward > 0:
        print(f"Scored! Reward: {reward}")

    if terminated:
        print(f"Game Over! Total Score: {total_reward:.2f}")
        env.reset()
        test_scenario = ScenarioGenerator().get_random_tricky_scenario()
        load_scenario(env, test_scenario)
        total_reward = 0

env.close()
