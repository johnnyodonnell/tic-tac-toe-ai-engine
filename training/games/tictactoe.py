"""Tic-tac-toe as a Game plugin, shared by the AlphaZero and DQN pipelines.

State: a 9-tuple of ints, row-major (index 0..8), from the moving player's
perspective: +1 = my mark, -1 = opponent's mark, 0 = empty. After every move
the board is negated so the opponent becomes "+1" — keeping every state in
the mover's frame.

The rules mirror src/engine/game.js exactly; WIN_LINES is the same constant.
"""

from game_interface import Game

# The 8 winning triples — identical to WIN_LINES in src/engine/game.js.
WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
)


class TicTacToe(Game):
    input_size = 18   # two 9-cell planes
    action_size = 9

    def initial_state(self):
        return (0,) * 9

    def current_player(self, state):
        # The first player moves whenever an even number of cells are filled.
        filled = sum(1 for c in state if c != 0)
        return 1 if filled % 2 == 0 else -1

    def legal_actions(self, state):
        return [i for i, c in enumerate(state) if c == 0]

    def apply_action(self, state, action):
        board = list(state)
        board[action] = 1                   # the current player plays
        return tuple(-c for c in board)     # flip to the opponent's perspective

    def _winner(self, state):
        """Return the mark value (+1 / -1) of a completed line, else 0."""
        for a, b, c in WIN_LINES:
            if state[a] != 0 and state[a] == state[b] == state[c]:
                return state[a]
        return 0

    def is_terminal(self, state):
        return self._winner(state) != 0 or all(c != 0 for c in state)

    def outcome(self, state):
        # _winner is in the mover's frame: a completed line of -1 marks means
        # the player who just moved made it, so the player to move has lost.
        return self._winner(state)

    def encode(self, state):
        mine = [1.0 if c == 1 else 0.0 for c in state]
        theirs = [1.0 if c == -1 else 0.0 for c in state]
        return mine + theirs

    def render(self, state):
        glyph = {1: "X", -1: "O", 0: "."}
        return "\n".join(
            " ".join(glyph[state[r * 3 + col]] for col in range(3))
            for r in range(3)
        )
