import time
import pygame
import random
import json
from micro_agent.A_Star import A_Star
from tetris.tetris_env import TetrisEnv
import numpy as np

pygame.init()


def watch_agent_play(agent, env, scenarios, num_games=5):
    for i in range(num_games):
        scenario = random.choice(scenarios)

        env.board = np.array(scenario["board"], dtype=int, copy=True)
        env.active_piece = env.generate_new_piece(id=int(scenario["piece_id"]))
        env.next_piece = env.generate_new_piece()
        env.state = "start"

        target_state = (
            int(scenario["target_pos"][0]),
            int(scenario["target_pos"][1]),
            int(scenario["target_rot"]),
        )

        done = False
        print(f"\n▶️ Starting Game {i+1}...")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            start_state = (
                env.active_piece.x,
                env.active_piece.y,
                env.active_piece.rotation,
            )

            action = agent.choose_action(start_state, target_state)

            if action is None:
                action = 4

            _, _, done, _, _ = env.step(action)
            env.render(tick=True)
            time.sleep(0.15)

        time.sleep(1)


if __name__ == "__main__":
    env = TetrisEnv(tick_speed=10)
    agent = A_Star(env)

    with open("micro_agent/evaluation_set.json", "r", encoding="utf-8") as f:
        eval_scenarios = json.load(f)

    watch_agent_play(agent, env, eval_scenarios, num_games=40)
