import random


class Game:
    """
    Stores the history of a single episode.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []

    def store_step(self, state, action, reward, child_visits, root_value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.child_visits.append(child_visits)
        self.root_values.append(root_value)

    def __len__(self):
        return len(self.actions)


class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = []
        self.capacity = capacity

    def save_game(self, game: Game):
        if len(self.buffer) >= self.capacity:
            # Remove oldest game if full
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, batch_size, unroll_steps):
        games = []
        game_pos = []

        # Pick batch_size random games
        for _ in range(batch_size):
            game = random.choice(self.buffer)

            # Pick a random starting position 'k' in that game
            # -> subtract unroll_steps so we don't overflow the length of the game
            pos = random.randint(0, len(game) - unroll_steps - 1)

            games.append(game)
            game_pos.append(pos)

        # You would then pack these into a (batch, unroll_steps, ...) array
        # to return to the trainer.
        # For now, just returning the raw list is fine.
        return games, game_pos
