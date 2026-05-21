"""Head-to-head evaluation between two networks — the promotion gate.

Both nets pick moves with greedy MCTS (most-visited, no Dirichlet noise). To
keep games from being identical, the opening `TEMPERATURE_MOVES` plies are
sampled from the visit distribution; this spreads games across openings so a
40-game match is informative rather than 40 copies of one game.
"""

import config
from alphazero import mcts


def _play_match(game, net_a, net_b, a_first, rng):
    """Play one game; net_a and net_b alternate moves.

    Returns the result for net_a: +1 win, 0 draw, -1 loss.
    """
    state = game.initial_state()
    a_to_move = a_first
    move_number = 0

    while not game.is_terminal(state):
        net = net_a if a_to_move else net_b
        root = mcts.run_mcts(game, net, state, config.NUM_SIMULATIONS, add_noise=False)
        if move_number < config.TEMPERATURE_MOVES:
            probs = mcts.action_probs(root, game.action_size, temperature=1.0)
            action = int(rng.choice(len(probs), p=probs))
        else:
            action = mcts.best_action(root)
        state = game.apply_action(state, action)
        a_to_move = not a_to_move
        move_number += 1

    # `a_to_move` now marks whose turn it is at the terminal state — the player
    # outcome() is relative to. Convert that to net_a's perspective.
    value = game.outcome(state)
    return value if a_to_move else -value


def compare(game, challenger, champion, rng):
    """Play config.ARENA_GAMES games, alternating who moves first.

    Returns (score, wins, draws, losses) for the challenger, where
    score = (wins + 0.5 * draws) / games.
    """
    wins = draws = losses = 0
    for i in range(config.ARENA_GAMES):
        result = _play_match(game, challenger, champion, a_first=(i % 2 == 0), rng=rng)
        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            draws += 1
    score = (wins + 0.5 * draws) / config.ARENA_GAMES
    return score, wins, draws, losses
