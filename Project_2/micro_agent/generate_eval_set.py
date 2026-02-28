import json
from random import random
import numpy as np
from micro_agent.ScenarioGenerator import ScenarioGenerator


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def generate_eval_set(n, generator):
    """
    Generates a set of test scenarios, one of each type, for evaluation purposes.
    Saves it to a JSON file that can be loaded by the agent during evaluation.
    """
    scenarios = []

    for _ in range(n):
        tricky = random() < 0.15
        scenario = generator.get_random_scenario(tricky=tricky)
        scenarios.append(scenario)

    clean_scenarios = _to_jsonable(scenarios)

    with open("micro_agent/evaluation_set.json", "w") as f:
        json.dump(clean_scenarios, f, indent=2)


if __name__ == "__main__":
    generator = ScenarioGenerator()
    generate_eval_set(750, generator)
