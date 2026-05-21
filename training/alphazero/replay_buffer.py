"""A fixed-size replay buffer of self-play training samples.

Each sample is a tuple (encoded_state, pi, z):
  encoded_state : list[float]  — the network input
  pi            : list[float]  — MCTS visit-count distribution (policy target)
  z             : float        — game outcome from that state's perspective
"""

from collections import deque

import config


class ReplayBuffer:
    def __init__(self, capacity=config.BUFFER_SIZE):
        self.samples = deque(maxlen=capacity)

    def add_game(self, samples):
        """Append all samples from one finished game."""
        self.samples.extend(samples)

    def __len__(self):
        return len(self.samples)

    def sample_batch(self, batch_size, rng):
        """Return up to `batch_size` random samples as three parallel lists.

        `rng` is a random.Random instance, kept explicit for reproducibility.
        """
        n = min(batch_size, len(self.samples))
        batch = rng.sample(self.samples, n)
        states = [s for s, _, _ in batch]
        pis = [p for _, p, _ in batch]
        zs = [z for _, _, z in batch]
        return states, pis, zs
