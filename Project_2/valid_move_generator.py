from tetris.tetris_shape import FIGURES_BY_ID
from tetris.tetris_env import TetrisEnv
from collections import deque


def generate_valid_moves(starting_state, piece_id, env: TetrisEnv):
    """
    Use BFS to find all possible landing spots (x, y, rotation)
    for a given Tetris piece.

    State is (x, y, rotation) for a piece
    """
    valid_spots = set()
    visited = set()

    # Initialize the queue with our spawn point
    queue = deque([starting_state])
    visited.add(starting_state)

    while len(queue) > 0:
        current_state = queue.pop()

        # Add current state as a valid resting spot if it collides when being pushed one block down
        current_rot = current_state[2]
        shape = FIGURES_BY_ID[piece_id]["shape"][current_rot]
        resting_spot = env.check_collision(
            shape, current_state[0], current_state[1] + 1
        )

        if resting_spot:
            valid_spots.add(current_state)

        # Loop through possible actions
        for action in range(4):
            new_state = generate_new_state(action, current_state, piece_id)

            new_rot = new_state[2]
            new_shape = FIGURES_BY_ID[piece_id]["shape"][new_rot]

            if new_state in visited:
                continue

            if env.check_collision(new_shape, new_state[0], new_state[1]):
                continue

            queue.append(new_state)
            visited.add(new_state)

    # Convert the set back to a list before returning
    return list(valid_spots)


def generate_new_state(action, state, piece_id):
    """
    Returns the new (x, y, rotation) tuple after applying the action.
    Action mapping: 0: Down, 1: Left, 2: Right, 3: Rotate.
    """
    if action == 0:
        return (state[0], state[1] + 1, state[2])
    elif action == 1:
        return (state[0] - 1, state[1], state[2])
    elif action == 2:
        return (state[0] + 1, state[1], state[2])
    elif action == 3:
        return (
            state[0],
            state[1],
            (state[2] + 1) % len(FIGURES_BY_ID[piece_id]["shape"]),
        )
    else:
        raise ValueError(f"Invalid action: {action}")
