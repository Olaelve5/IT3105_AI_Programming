import pygame
import jax
import jax.numpy as jnp
import numpy as np
import flax.serialization
import os
from functools import partial
from config import NUM_ACTIONS, BOARD_HEIGHT, BOARD_WIDTH
from MuZeroNet import MuZeroNet
from connect4.env_wrapper import EnvWrapper
from connect4.env import Connect4Env
from umcts import UMCTS
from mcts_node import MCTSNode


@partial(jax.jit, static_argnums=(1,))
def representation_inference_fn(params, model, state):
    return model.apply(params, state, method=model.representation)


def load_params(params_struct, path):
    if not os.path.exists(path):
        print(f"⚠️ Could not find checkpoint at {path}. Using random untrained weights!")
        return params_struct
    with open(path, "rb") as f:
        print(f"🧠 Loaded brain from {path}")
        return flax.serialization.from_bytes(params_struct, f.read())


def get_ai_move(env_wrapper, mcts_agent, model, num_simulations=128):
    """Runs the MCTS to find the absolutely best move."""
    obs = env_wrapper.get_obs()
    state_jnp = jnp.array([obs])
    abstract_state = representation_inference_fn(mcts_agent.params, model, state_jnp)

    root_node = MCTSNode(prior=1.0)
    root_node.game_state = abstract_state
    valid_actions = env_wrapper.env.get_valid_actions()

    mcts_agent.run(
        root_node,
        num_simulations=num_simulations,
        inject_noise=False,
        valid_actions=valid_actions,
    )

    policy_distribution, _ = mcts_agent.extract_mcts_data(root_node, NUM_ACTIONS)

    return int(np.argmax(policy_distribution))


def main():
    print("\n🎮 WELCOME TO THE CONNECT-4 ARENA 🎮")
    print("1. Watch AI vs AI")
    print("2. Play as Player 1 (Red) vs AI")
    print("3. Play as Player 2 (Yellow) vs AI")

    choice = input("\nSelect game mode (1/2/3): ")

    model = MuZeroNet(num_actions=NUM_ACTIONS)
    dummy_obs = jnp.ones((1, BOARD_HEIGHT, BOARD_WIDTH, 3))
    dummy_act = jnp.array([0])
    rng = jax.random.PRNGKey(0)
    base_params = model.init(rng, dummy_obs, dummy_act, method=model.init_params)

    PATH_A = "saved_params/1_generations.msgpack"
    PATH_B = "saved_params/1_generations.msgpack"

    is_p1_human = False
    is_p2_human = False

    if choice == "1":
        mcts_p1 = UMCTS(model, load_params(base_params, PATH_A))
        mcts_p2 = UMCTS(model, load_params(base_params, PATH_B))
    elif choice == "2":
        is_p1_human = True
        mcts_p2 = UMCTS(model, load_params(base_params, PATH_B))
    elif choice == "3":
        mcts_p1 = UMCTS(model, load_params(base_params, PATH_B))
        is_p2_human = True
    else:
        print("Invalid choice. Exiting.")
        return

    raw_env = Connect4Env()
    env = EnvWrapper(raw_env)
    env.reset()
    raw_env.render()

    pygame.font.init()
    title_font = pygame.font.SysFont("arial", 60, bold=True)
    sub_font = pygame.font.SysFont("arial", 30, bold=True)

    running = True
    done = False
    win_msg = ""

    while running:
        current_player = raw_env.current_player
        is_human_turn = (current_player == 1 and is_p1_human) or (
            current_player == 2 and is_p2_human
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN and done:
                if event.key == pygame.K_r:
                    print("🔄 Restarting game...")
                    env.reset()
                    raw_env.render()
                    done = False
                    win_msg = ""
                    continue

            # human clicking logic
            if event.type == pygame.MOUSEBUTTONDOWN and is_human_turn and not done:
                x_pos = event.pos[0]
                col = int(math.floor(x_pos / raw_env.grid_size))

                if raw_env.is_valid_action(col):
                    _, reward, done = env.step(col, is_training=False)
                    raw_env.render()

                    if done:
                        if reward == 1.0:
                            win_msg = f"Player {current_player} (HUMAN) WINS!"
                        else:
                            win_msg = "IT'S A DRAW!"
                        print(win_msg)

        # AI logic
        if not is_human_turn and not done and running:
            pygame.time.delay(300)

            print(f"🤔 AI (Player {current_player}) is thinking...")
            current_mcts = mcts_p1 if current_player == 1 else mcts_p2

            action = get_ai_move(env, current_mcts, model)

            _, reward, done = env.step(action, is_training=False)
            raw_env.render()

            if done:
                if reward == 1.0:
                    win_msg = f"Player {current_player} (AI) WINS!"
                else:
                    win_msg = "IT'S A DRAW!"
                print(win_msg)

        # victory screen
        if done and win_msg != "":
            overlay = pygame.Surface((raw_env.width, raw_env.height))
            overlay.set_alpha(150)
            overlay.fill((0, 0, 0))
            raw_env.screen.blit(overlay, (0, 0))

            text_title = title_font.render(win_msg, True, (255, 255, 255))
            text_sub = sub_font.render("Press 'R' to Restart", True, (200, 200, 200))

            title_rect = text_title.get_rect(
                center=(raw_env.width / 2, raw_env.height / 2 - 30)
            )
            sub_rect = text_sub.get_rect(
                center=(raw_env.width / 2, raw_env.height / 2 + 40)
            )

            raw_env.screen.blit(text_title, title_rect)
            raw_env.screen.blit(text_sub, sub_rect)

            pygame.display.flip()
            win_msg = ""

    pygame.quit()


if __name__ == "__main__":
    import math

    main()
