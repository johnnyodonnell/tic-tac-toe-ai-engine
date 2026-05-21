"""The game-agnostic contract the training pipelines depend on.

Both the AlphaZero and DQN cores talk only to this interface — never a concrete
game. A new game is added by subclassing `Game`; nothing else changes.
tic-tac-toe (games/tictactoe.py) is the first plugin.

State convention
----------------
A *state* is an immutable, hashable value (a tuple) describing a position.
It is always canonicalized to the perspective of the player to move: the
network sees "my pieces vs. opponent's pieces", never "X vs. O". This is what
lets a single network play both sides and keeps self-play data consistent.
"""

from abc import ABC, abstractmethod


class Game(ABC):
    # Shape information, read by the network to size its layers.
    input_size: int   # length of the encode() vector
    action_size: int  # number of distinct actions

    @abstractmethod
    def initial_state(self):
        """Return the canonical starting state."""

    @abstractmethod
    def current_player(self, state):
        """Return +1 or -1 — the absolute identity of the player to move.

        +1 is whoever moved first in the game. Used only to attribute wins
        (e.g. in the arena); the network never sees this.
        """

    @abstractmethod
    def legal_actions(self, state):
        """Return a list of legal action indices in [0, action_size)."""

    @abstractmethod
    def apply_action(self, state, action):
        """Return the state after `action`, re-canonicalized so the result is
        again from the (new) moving player's perspective."""

    @abstractmethod
    def is_terminal(self, state):
        """Return True if `state` is an ended game."""

    @abstractmethod
    def outcome(self, state):
        """Terminal result from the perspective of the player to move at
        `state`: +1 win, 0 draw, -1 loss. Defined only when is_terminal()."""

    @abstractmethod
    def encode(self, state):
        """Return a list[float] of length input_size — the network input."""

    def render(self, state):
        """Human-readable rendering, for debugging. Optional to override."""
        return str(state)
