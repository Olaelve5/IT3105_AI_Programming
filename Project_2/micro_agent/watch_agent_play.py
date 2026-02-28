import time
import random
from micro_agent.MicroAgent import MicroAgent
from micro_agent.MicroEnvWrapper import MicroEnvWrapper
from tetris.tetris_env import TetrisEnv
import json


import time
import pygame  # Make sure this is imported!

pygame.init()  


def watch_agent_play(agent, env_wrapper, scenarios, num_games=5):
    for i in range(num_games):
        scenario = random.choice(scenarios)
        obs = env_wrapper.load_scenario(scenario)

        done = False
        print(f"\n▶️ Starting Game {i+1}...")

        while not done:
            # 1. THE MAC OS FIX: Pump the Pygame events so the window actually draws
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return  # Exit completely if you close the window

            # 2. Choose action and step
            action = agent.choose_action(obs, evaluate=True)
            obs, reward, done = env_wrapper.step(action)

            # 3. Render and pause
            env_wrapper.render()
            time.sleep(0.05)

        print(f"Game Over! Final Reward: {reward}")
        time.sleep(1)


if __name__ == "__main__":
    agent = MicroAgent()
    agent.load_model("micro_agent/best_micro_agent.msgpack")
    env_wrapper = MicroEnvWrapper(TetrisEnv())

    with open("micro_agent/evaluation_set.json", "r", encoding="utf-8") as f:
        eval_scenarios = json.load(f)

    watch_agent_play(agent, env_wrapper, eval_scenarios[200:300], num_games=20)
