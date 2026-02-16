import pygame
from tetris_env import TetrisEnv  # Import your new local class

# 1. Initialize custom environment
env = TetrisEnv(height=20, width=10)
observation, info = env.reset()

print("Controls: Left, Right, Up (Rotate), Down (Fast Drop). Q to quit.")

running = True
total_reward = 0

while running:
    env.render()

    action = 0  # 0 = No-op (just fall)

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
            elif event.key == pygame.K_DOWN:
                action = 4

    # Step the environment
    # Note: Our custom class has gravity built into step(),
    # so even action=0 moves the piece down.
    next_state, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    if reward > 0:
        print(f"Scored! Reward: {reward}")

    if terminated:
        print(f"Game Over! Total Score: {total_reward}")
        env.reset()
        total_reward = 0

env.close()
