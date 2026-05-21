"""PUCT Monte Carlo Tree Search guided by the policy/value network.

Game-agnostic: depends only on the Game interface and the network's infer().
This is ported almost line-for-line to src/engine/mcts.js — keep the two in
step (PUCT formula, expansion order, tie-breaking).

Tie-breaking: ties in PUCT child selection and in argmax over visit counts go
to the LOWEST action index. legal_actions() returns actions ascending and
children are inserted in that order, so ascending iteration plus a strict `>`
comparison yields exactly that.
"""

import math

import numpy as np

import config
from alphazero.network import infer


class Node:
    __slots__ = ("prior", "state", "visit_count", "value_sum", "children", "is_terminal")

    def __init__(self, prior, state):
        self.prior = prior
        self.state = state
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}        # action -> Node, inserted ascending by action
        self.is_terminal = False

    def value(self):
        """Mean value, from this node's own perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _masked_softmax(logits, legal):
    """Softmax of `logits` restricted to `legal` actions -> {action: prob}."""
    m = max(logits[a] for a in legal)
    exps = {a: math.exp(logits[a] - m) for a in legal}
    z = sum(exps.values())
    return {a: e / z for a, e in exps.items()}


def _expand(node, game, net):
    """Create children for `node` and return the network's value estimate.

    If `node` is terminal, mark it and return the true outcome instead.
    """
    if game.is_terminal(node.state):
        node.is_terminal = True
        return float(game.outcome(node.state))

    logits, value = infer(net, game, node.state)
    legal = game.legal_actions(node.state)            # ascending
    priors = _masked_softmax(logits, legal)
    for action in legal:
        child_state = game.apply_action(node.state, action)
        node.children[action] = Node(priors[action], child_state)
    return value


def _select_child(node):
    """Pick the child maximizing the PUCT score (lowest index breaks ties)."""
    sqrt_total = math.sqrt(node.visit_count)
    best_score = -float("inf")
    best = None
    for action, child in node.children.items():
        q = -child.value()        # child.value() is in the child's frame
        u = config.C_PUCT * child.prior * sqrt_total / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best = (action, child)
    return best


def _add_dirichlet_noise(root, rng):
    """Mix Dirichlet noise into the root priors — self-play exploration only."""
    if rng is None:
        rng = np.random.default_rng()
    actions = list(root.children.keys())
    noise = rng.dirichlet([config.DIRICHLET_ALPHA] * len(actions))
    eps = config.DIRICHLET_EPSILON
    for action, n in zip(actions, noise):
        child = root.children[action]
        child.prior = child.prior * (1 - eps) + float(n) * eps


def run_mcts(game, net, root_state, num_simulations, add_noise=False, rng=None):
    """Run `num_simulations` simulations and return the populated root Node."""
    root = Node(prior=0.0, state=root_state)
    _expand(root, game, net)

    if add_noise and root.children:
        _add_dirichlet_noise(root, rng)

    for _ in range(num_simulations):
        node = root
        path = [node]
        while node.children:                       # descend until a leaf
            _, node = _select_child(node)
            path.append(node)

        # `node` is a leaf — score it (terminal) or expand it (network value).
        if node.is_terminal:
            value = float(game.outcome(node.state))
        else:
            value = _expand(node, game, net)

        # Backup: `value` is from the leaf's perspective; flip each ply up.
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

    return root


def _argmax(values):
    """Index of the largest value; lowest index breaks ties."""
    best_i, best_v = 0, values[0]
    for i, v in enumerate(values):
        if v > best_v:
            best_v, best_i = v, i
    return best_i


def action_probs(root, action_size, temperature=1.0):
    """Distribution over actions from root visit counts (0 for unvisited).

    temperature == 0 -> one-hot on the most-visited action.
    temperature  > 0 -> proportional to count ** (1 / temperature).
    """
    counts = [0] * action_size
    for action, child in root.children.items():
        counts[action] = child.visit_count

    if temperature == 0:
        probs = [0.0] * action_size
        probs[_argmax(counts)] = 1.0
        return probs

    weighted = [c ** (1.0 / temperature) for c in counts]
    total = sum(weighted)
    if total == 0:                                  # no visits — stay safe
        return [1.0 / action_size] * action_size
    return [w / total for w in weighted]


def best_action(root):
    """Most-visited action; lowest index breaks ties."""
    best_a, best_n = None, -1
    for action, child in root.children.items():
        if child.visit_count > best_n:
            best_n, best_a = child.visit_count, action
    return best_a
