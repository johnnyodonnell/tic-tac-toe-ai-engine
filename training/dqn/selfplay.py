"""Self-play experience generation for DQN.

One Q-network plays both sides of a game with epsilon-greedy exploration; every
move becomes a transition. States are canonical (the mover's perspective), so a
single network and a single stream of transitions cover both players.
"""

from dqn.qnetwork import q_values


def random_start(game, rng, max_plies):
    """A start state from a short random walk — diversifies coverage.

    Game-agnostic (uses only the Game interface). May return a terminal state;
    play_game then simply produces no transitions.
    """
    state = game.initial_state()
    plies = int(rng.integers(0, max_plies + 1))
    for _ in range(plies):
        if game.is_terminal(state):
            break
        actions = game.legal_actions(state)
        state = game.apply_action(state, actions[int(rng.integers(len(actions)))])
    return state


def epsilon_greedy_action(net, game, state, epsilon, rng):
    """Pick a legal action: random with probability epsilon, else argmax Q.

    Ties in the argmax go to the lowest action index (legal is ascending and
    the comparison is strict) — the same rule the JS engine uses.
    """
    legal = game.legal_actions(state)
    if rng.random() < epsilon:
        return legal[int(rng.integers(len(legal)))]

    qs = q_values(net, game, state)
    best_a = legal[0]
    best_q = qs[best_a]
    for a in legal:
        if qs[a] > best_q:
            best_q, best_a = qs[a], a
    return best_a


def play_game(game, net, rng, epsilon, start_state=None):
    """Play one epsilon-greedy self-play game.

    Returns a list of (state, action, reward, next_state, done) transitions.
    `rng` is a numpy Generator.
    """
    state = game.initial_state() if start_state is None else start_state
    transitions = []

    while not game.is_terminal(state):
        action = epsilon_greedy_action(net, game, state, epsilon, rng)
        next_state = game.apply_action(state, action)

        if game.is_terminal(next_state):
            # The move ended the game. outcome(next_state) is from the next
            # mover's perspective; negate it for the player who just moved.
            reward = float(-game.outcome(next_state))
            transitions.append((state, action, reward, next_state, True))
        else:
            transitions.append((state, action, 0.0, next_state, False))

        state = next_state

    return transitions
