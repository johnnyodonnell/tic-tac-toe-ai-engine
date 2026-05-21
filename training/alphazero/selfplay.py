"""Self-play game generation — the only source of training data.

The current network plays both sides of a game against itself, guided by MCTS.
Every position visited becomes a training sample once the game ends.

Some games start from a random position rather than the empty board (see
`random_start`). This is what gives the network training signal on positions
that optimal play would never reach — without it, off-principal states stay
unlearned and the bot blunders when a human steers into one.
"""

import config
from alphazero import mcts


def _sample(probs, rng):
    """Sample an action index from a probability list (numpy Generator rng)."""
    return int(rng.choice(len(probs), p=probs))


def random_start(game, rng, max_plies):
    """Return a start state reached by a short random walk from the start.

    Game-agnostic: uses only the Game interface, so it ports to any game.
    The returned state may occasionally be terminal; play_game handles that
    by simply producing no samples.
    """
    state = game.initial_state()
    plies = int(rng.integers(0, max_plies + 1))
    for _ in range(plies):
        if game.is_terminal(state):
            break
        actions = game.legal_actions(state)
        state = game.apply_action(state, actions[int(rng.integers(len(actions)))])
    return state


def play_game(game, net, rng, start_state=None):
    """Play one self-play game; return a list of (encoded_state, pi, z) samples.

    `pi` is the MCTS visit-count distribution (the policy target). `z` is the
    final outcome seen from that state's player's perspective. `rng` is a numpy
    Generator, used for Dirichlet root noise and early-move sampling.
    `start_state` defaults to the empty board.
    """
    state = game.initial_state() if start_state is None else start_state
    history = []          # (encoded_state, pi, player_to_move)
    move_number = 0

    while not game.is_terminal(state):
        root = mcts.run_mcts(
            game, net, state, config.NUM_SIMULATIONS,
            add_noise=True, rng=rng,
        )
        pi = mcts.action_probs(root, game.action_size, temperature=1.0)
        history.append((game.encode(state), pi, game.current_player(state)))

        # Exploratory sampling early, greedy thereafter.
        if move_number < config.TEMPERATURE_MOVES:
            action = _sample(pi, rng)
        else:
            action = mcts.best_action(root)

        state = game.apply_action(state, action)
        move_number += 1

    # Assign the outcome to every recorded state from ITS player's perspective.
    # Players strictly alternate, so a state's result is the terminal result
    # if its mover matches the terminal mover, else the negation.
    terminal_value = game.outcome(state)
    terminal_player = game.current_player(state)
    samples = []
    for encoded, pi, player in history:
        z = terminal_value if player == terminal_player else -terminal_value
        samples.append((encoded, pi, float(z)))
    return samples
