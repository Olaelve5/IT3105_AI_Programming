import time
import pygame
import random
import json
from micro_agent.A_Star import A_Star
from micro_agent.MicroEnvWrapper import MicroEnvWrapper
from tetris.tetris_env import TetrisEnv

pygame.init()


def watch_agent_play(agent, env_wrapper, scenarios, num_games=5):
    for i in range(num_games):
        scenario = random.choice(scenarios)
        _ = env_wrapper.load_scenario(scenario)

        done = False
        print(f"\n▶️ Starting Game {i+1}...")

        # Extract target state ONCE outside the loop and ensure they are integers
        target_state = (
            int(scenario["target_pos"][0]),
            int(scenario["target_pos"][1]),
            int(scenario["target_rot"]),
        )

        while not done:
            # 1. THE MAC OS FIX: Pump the Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # 2. Get current state of the piece
            start_state = (
                env_wrapper.env.active_piece.x,
                env_wrapper.env.active_piece.y,
                env_wrapper.env.active_piece.rotation,
            )

            # 3. Choose action using A*
            action = agent.choose_action(start_state, target_state)

            # 4. If A* returns None, we are at the target! Hard drop to lock it.
            if action is None:
                action = 4

            # 5. Take the step
            _, reward, done = env_wrapper.step(action)

            # 6. Render and pause
            env_wrapper.render()
            time.sleep(0.05)

        print(f"Game Over! Final Reward: {reward}")
        time.sleep(1)


if __name__ == "__main__":
    # Initialize raw env
    env = TetrisEnv(tick_speed=10)

    # Initialize wrapper
    env_wrapper = MicroEnvWrapper(env)

    # CRITICAL FIX: Pass the RAW env to the A* agent, not the wrapper!
    agent = A_Star(env)

    with open("micro_agent/evaluation_set.json", "r", encoding="utf-8") as f:
        eval_scenarios = json.load(f)

    watch_agent_play(agent, env_wrapper, eval_scenarios, num_games=40)
