import pygame
from tetris.tetris_env import TetrisEnv
from config import BOARD_WIDTH, BOARD_HEIGHT

env = TetrisEnv()
observation, info = env.reset()

print("Controls: Left, Right, Up (Rotate), Down (Fast Drop). Q to quit.")

running = True
total_reward = 0

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
        total_reward = 0

env.close()
