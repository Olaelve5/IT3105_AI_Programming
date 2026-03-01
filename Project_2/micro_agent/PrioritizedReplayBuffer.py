import numpy as np
import random


class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6):
        self.capacity = capacity
        self.alpha = (
            alpha  # How much prioritization to use (0 = random, 1 = strict priority)
        )
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        # New experiences get absolute top priority to guarantee they are seen
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]

        # Calculate probabilities based on priorities
        probs = prios**self.alpha
        probs /= probs.sum()

        # Sample based on those probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Calculate Importance Sampling (IS) weights to correct for the bias
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability

        states, actions, rewards, next_states, dones = zip(*samples)

        # Notice we return the indices and weights now too!
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, td_errors):
        # Add a tiny constant (1e-5) so priorities never reach absolute zero
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + 1e-5

    def __len__(self):
        return len(self.buffer)
