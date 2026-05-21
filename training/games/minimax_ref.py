"""A perfect minimax (negamax) reference player.

Used ONLY to evaluate the trained network — never as a training signal. It
plays through the Game interface, so it is game-agnostic in principle, though
full-tree search is only tractable for tiny games like tic-tac-toe.

`_CACHE` memoizes negamax values by canonical state. Because states are in the
mover's frame, the value of a state is independent of which absolute player is
moving — so the cache is correct. It assumes a single game type per process
(true for this project).
"""

_CACHE = {}


def best_action(game, state):
    """Return an optimal action for the player to move at `state`.

    Ties are broken toward the lowest action index (legal_actions is ascending
    and the comparison below is strict), matching config.py's tie-break rule.
    """
    best_a, best_v = None, -2.0   # real values are in [-1, 1]
    for action in game.legal_actions(state):
        value = -_value(game, game.apply_action(state, action))
        if best_a is None or value > best_v:
            best_a, best_v = action, value
    return best_a


def _value(game, state):
    """Negamax value of `state` from the perspective of its player to move."""
    cached = _CACHE.get(state)
    if cached is not None:
        return cached

    if game.is_terminal(state):
        result = float(game.outcome(state))
    else:
        result = -2.0
        for action in game.legal_actions(state):
            value = -_value(game, game.apply_action(state, action))
            if value > result:
                result = value

    _CACHE[state] = result
    return result
