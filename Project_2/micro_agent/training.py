from micro_agent.MicroAgent import MicroAgent
from micro_agent.MicroEnvWrapper import MicroEnvWrapper
from micro_agent.ReplayBuffer import ReplayBuffer
from micro_agent.ScenarioGenerator import ScenarioGenerator
import json
import random
from tetris.tetris_env import TetrisEnv
import wandb
from collections import deque


def load_eval_set():
    with open("micro_agent/evaluation_set.json", "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    return scenarios


def evaluate_agent(agent, env_wrapper, scenarios):
    success = []

    for s in scenarios:
        obs = env_wrapper.load_scenario(s)
        done = False
        reward = 0.0

        steps = 0
        while not done and steps < 50:
            action = agent.choose_action(obs, evaluate=True)
            obs, reward, done = env_wrapper.step(action)
            steps += 1

        success.append(1 if reward == 10 else 0)

    ones = sum(success)
    return ones / len(success) if success else 0.0


def train():
    env = TetrisEnv()
    env_wrapper = MicroEnvWrapper(env=env)
    agent = MicroAgent()
    buffer = ReplayBuffer()
    generator = ScenarioGenerator()
    eval_scenarios = load_eval_set()

    wandb.init(
        project="tetris-micro-agent",
        name="ddqn-run-01",
        config={
            "num_episodes": 50000,
            "batch_size": 64,
            "gamma": 0.99,
            "tau": 0.005,
            "epsilon_decay": 0.995,
        },
    )

    # Training settings
    num_episodes = 50000
    batch_size = 64
    eval_frequency = 500
    eval_highscore = 0.0

    # Trackers for logging
    recent_rewards = deque(maxlen=50)
    recent_losses = deque(maxlen=50)
    print_frequency = 50

    # Main training loop
    for episode in range(num_episodes):
        if random.random() > 0.7:
            s = generator.generate_tricky_scenario()
        else:
            s = generator.generate_normal_scenario()

        obs = env_wrapper.load_scenario(s)
        done = False
        episode_reward = 0
        episode_losses = []

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done = env_wrapper.step(action)
            buffer.push(obs, action, reward, next_obs, done)

            loss = agent.learn(buffer, batch_size=batch_size)
            agent.update_target()

            if loss is not None:
                episode_losses.append(float(loss))

            obs = next_obs
            episode_reward += reward

        # Decay exploration after each episode
        exploration_episodes = 25000
        agent.epsilon = max(0.05, 1.0 - (episode / exploration_episodes))

        ep_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0.0

        recent_rewards.append(episode_reward)
        recent_losses.append(ep_loss)

        smoothed_reward = sum(recent_rewards) / len(recent_rewards)
        smoothed_loss = sum(recent_losses) / len(recent_losses)

        wandb.log(
            {
                "train/raw_reward": episode_reward,
                "train/smoothed_reward": smoothed_reward,
                "train/avg_loss": smoothed_loss,
                "train/epsilon": agent.epsilon,
                "episode": episode,
            }
        )

        if episode > 0 and episode % print_frequency == 0:
            print(
                f"Episode: {episode:5d} | Epsilon: {agent.epsilon:.3f} | "
                f"Avg. Reward: {smoothed_reward:5.1f} | Avg. Loss: {smoothed_loss:.4f}"
            )

        # Perform evaluation and save params
        if episode > 0 and episode % eval_frequency == 0:
            print(f"--- Evaluating at Episode {episode} ---")

            success_rate = evaluate_agent(agent, env_wrapper, eval_scenarios[:100])
            print(f"Success Rate: {success_rate * 100:.2f}%")

            wandb.log({"eval/success_rate": success_rate, "episode": episode})

            if success_rate > eval_highscore:
                eval_highscore = success_rate
                print("⭐️ New High Score! Saving model...")
                agent.save_model(filepath="best_micro_agent.msgpack")


if __name__ == "__main__":
    train()
