from tetris.tetris_shape import TetrisPiece, FIGURES_BY_ID
from config import NUM_ACTIONS
import heapq


class A_Star:
    def __init__(self, env):
        self.env = env
        self.path = []

    def choose_action(self, start_state, target_state):
        if len(self.path) <= 0:
            self.path = self.find_path(start_state, target_state)

        if len(self.path) <= 0:
            return 0

        return self.path.pop(0)

    def find_path(self, start_state, target_state):
        """State is (x, y, rotation) of the active piece. Returns the best action according to A* search."""
        queue = []
        heapq.heappush(queue, (0, 0, start_state, []))

        visited = set()

        while len(queue) > 0:
            f_score, g_score, current_state, path = heapq.heappop(queue)

            if tuple(current_state) == tuple(target_state):
                return path

            if current_state in visited:
                continue

            visited.add(current_state)

            for action in range(5):
                new_state = self.get_new_state(
                    action, current_state, self.env.active_piece.id, target_state
                )

                if action == 4 and new_state != target_state:
                    continue

                if not self.env.is_valid_state(new_state, self.env.active_piece.id):
                    continue

                if new_state not in visited:
                    new_g = g_score + 1
                    h_score = self._get_distance(new_state, target_state)
                    new_f = new_g + h_score
                    new_path = path + [action]
                    heapq.heappush(queue, (new_f, new_g, new_state, new_path))
        return []

    def get_new_state(self, action, state, piece_id, target_state):
        gravity = 0 if target_state[1] == state[1] else 1

        if action == 0:
            return (state[0], state[1] + gravity, state[2])
        elif action == 1:
            return (state[0] - 1, state[1] + gravity, state[2])
        elif action == 2:
            return (state[0] + 1, state[1] + gravity, state[2])
        elif action == 3:
            return (
                state[0],
                state[1] + gravity,
                (state[2] + 1) % len(FIGURES_BY_ID[piece_id]["shape"]),
            )
        elif action == 4:
            piece = TetrisPiece(
                FIGURES_BY_ID[piece_id], (state[0], state[1]), rotation=state[2]
            )
            drop_position = self.env.get_drop_position(piece)
            return (piece.x, drop_position, state[2])
        else:
            raise ValueError(f"Invalid action: {action}")

    def _get_distance(self, state, target_state):
        """Calculates Manhattan distance between current piece and target."""
        manhattan_distance = abs(state[0] - target_state[0]) + abs(
            state[1] - target_state[1]
        )
        num_shapes = len(FIGURES_BY_ID[self.env.active_piece.id]["shape"])

        # Calculate forward rotations needed to reach target
        rotation_distance = (target_state[2] - state[2]) % num_shapes

        return manhattan_distance + rotation_distance
