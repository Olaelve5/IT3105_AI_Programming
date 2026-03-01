from micro_agent.MicroAgent import MicroAgent
from micro_agent.MicroEnvWrapper import MicroEnvWrapper
from micro_agent.ReplayBuffer import ReplayBuffer
from micro_agent.PrioritizedReplayBuffer import PrioritizedReplayBuffer
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
    # agent.load_model("micro_agent/best_micro_agent.msgpack")
    # agent.epsilon = 0.05
    buffer = PrioritizedReplayBuffer(capacity=100000)
    generator = ScenarioGenerator()
    eval_scenarios = load_eval_set()

    wandb.init(
        project="tetris-micro-agent",
        name="ddqn-run-01",
        # id="uvz82cgd",
        # resume="must",
        config={
            "num_episodes": 300000,
            "batch_size": 128,
            "gamma": 0.99,
            "tau": 0.01,
            "epsilon_decay": 0.995,
        },
    )

    # Training settings
    num_episodes = 500000
    batch_size = 128
    eval_frequency = 1000
    eval_highscore = 0.0

    # Trackers for logging
    recent_rewards = deque(maxlen=50)
    recent_losses = deque(maxlen=50)
    print_frequency = 50

    # Main training loop
    for i in range(num_episodes):
        episode = i

        if episode < 30000:
            tricky = False
        else:
            tricky = random.random() < 0.15

        s = generator.get_random_scenario(tricky=tricky)

        obs = env_wrapper.load_scenario(s)
        done = False
        episode_reward = 0
        episode_losses = []

        tricky_reward = 0.0
        normal_reward = 0.0

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done = env_wrapper.step(action)
            buffer.push(obs, action, reward, next_obs, done)

            if len(buffer) > batch_size:
                beta = min(1.0, 0.4 + episode * (1.0 - 0.4) / num_episodes)

                states, actions, rewards, next_states, dones, indices, weights = (
                    buffer.sample(batch_size, beta)
                )

                loss, td_errors = agent.learn(
                    states, actions, rewards, next_states, dones, weights
                )

                buffer.update_priorities(indices, td_errors)
            else:
                loss = None

            if loss is not None:
                episode_losses.append(float(loss))

            obs = next_obs
            episode_reward += reward

        agent.update_target()

        # Decay exploration after each episode
        exploration_episodes = 250000
        agent.epsilon = max(0.05, 1.0 - (episode / exploration_episodes))

        ep_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0.0

        recent_rewards.append(episode_reward)
        recent_losses.append(ep_loss)

        if tricky:
            tricky_reward = episode_reward
        else:
            normal_reward = episode_reward

        smoothed_reward = sum(recent_rewards) / len(recent_rewards)
        smoothed_loss = sum(recent_losses) / len(recent_losses)

        wandb.log(
            {
                "train/smoothed_reward": smoothed_reward,
                "train/avg_loss": smoothed_loss,
                "train/epsilon": agent.epsilon,
                "train/tricky_reward": tricky_reward,
                "train/normal_reward": normal_reward,
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

            success_rate = evaluate_agent(agent, env_wrapper, eval_scenarios[:150])
            print(f"Success Rate: {success_rate * 100:.2f}%")

            wandb.log({"eval/success_rate": success_rate, "episode": episode})

            if success_rate > eval_highscore:
                eval_highscore = success_rate
                print("⭐️ New High Score! Saving model...")
                agent.save_model(filepath="micro_agent/best_micro_agent.msgpack")


if __name__ == "__main__":
    train()
