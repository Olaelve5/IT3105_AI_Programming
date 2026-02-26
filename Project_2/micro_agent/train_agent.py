from Project_2.micro_agent.ScenarioGenerator import TrainingScenarioGenerator
import pygame
import random

training_scenario_generator = TrainingScenarioGenerator()


def load_scenario(env, scenario):
    env.board = scenario["board"]
    env.active_piece = env.generate_new_piece(scenario["piece_id"])
    ghost_piece = env.generate_new_piece(scenario["piece_id"])
    ghost_piece.x, ghost_piece.y = scenario["target_pos"]
    ghost_piece.rotation = scenario["target_rot"]
    ghost_piece.active_shape = ghost_piece.shapes[ghost_piece.rotation]
    return ghost_piece


if __name__ == "__main__":
    pygame.init()
    env = TrainingScenarioGenerator().env
    scenario = training_scenario_generator.get_random_scenario(tricky=True)
    ghost_piece = load_scenario(env, scenario)
    scenario_interval_ms = 1000
    last_scenario_change = pygame.time.get_ticks()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        now = pygame.time.get_ticks()
        if now - last_scenario_change >= scenario_interval_ms:
            if random.random() < 0.0:
                scenario = training_scenario_generator.get_random_scenario(tricky=False)
            else:
                scenario = training_scenario_generator.get_random_scenario(tricky=True)
            ghost_piece = load_scenario(env, scenario)
            last_scenario_change = now

        env.render(tick=False, ghost_piece=ghost_piece)
