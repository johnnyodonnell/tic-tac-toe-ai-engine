"""A fixed-size replay buffer of DQN transitions.

Each transition is (state, action, reward, next_state, done):
  state, next_state : canonical 9-int board tuples
  action            : the action index taken from `state`
  reward            : immediate reward — 0 for a non-terminal move, otherwise
                      the game result for the player who moved
  done              : True if `action` ended the game
"""

from collections import deque

from dqn import config


class ReplayBuffer:
    def __init__(self, capacity=config.BUFFER_SIZE):
        self.transitions = deque(maxlen=capacity)

    def add_many(self, transitions):
        """Append all transitions from one finished game."""
        self.transitions.extend(transitions)

    def __len__(self):
        return len(self.transitions)

    def sample(self, batch_size, rng):
        """Return up to `batch_size` random transitions as five parallel lists.

        `rng` is a random.Random instance, kept explicit for reproducibility.
        """
        n = min(batch_size, len(self.transitions))
        batch = rng.sample(self.transitions, n)
        states = [t[0] for t in batch]
        actions = [t[1] for t in batch]
        rewards = [t[2] for t in batch]
        next_states = [t[3] for t in batch]
        dones = [t[4] for t in batch]
        return states, actions, rewards, next_states, dones
