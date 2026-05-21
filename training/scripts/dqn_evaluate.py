"""Evaluate the trained DQN difficulty models against minimax and random play.

For each difficulty (easy / medium / hard) it reports:
  - exhaustive optimality — losing blunders across every reachable position;
  - full-game outcomes vs perfect minimax (which varies among optimal moves so
    games differ);
  - full-game outcomes vs a random opponent.

The DQN bot picks moves with a single forward pass — argmax Q over legal
actions, no search.

Run:  python training/scripts/dqn_evaluate.py
"""

import os
import random
import sys

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING_DIR)

import torch

from dqn import config
from dqn.qnetwork import QNetwork, q_values
from games import minimax_ref
from games.tictactoe import TicTacToe

CHECKPOINT_DIR = os.path.join(TRAINING_DIR, "dqn", "checkpoints")
MATCH_GAMES = 400


def all_states(game):
    """Every reachable canonical state, found by expanding all legal moves."""
    seen, stack = set(), [game.initial_state()]
    while stack:
        state = stack.pop()
        if state in seen:
            continue
        seen.add(state)
        if not game.is_terminal(state):
            for action in game.legal_actions(state):
                stack.append(game.apply_action(state, action))
    return seen


def net_move(game, net, state):
    """The DQN bot's move: argmax Q over legal actions (lowest index breaks ties)."""
    qs = q_values(net, game, state)
    legal = game.legal_actions(state)
    best_a = legal[0]
    for a in legal:
        if qs[a] > qs[best_a]:
            best_a = a
    return best_a


def optimality(game, net):
    """Returns (positions checked, suboptimal moves, losing blunders).

    A losing blunder turns a non-losing position into a losing one.
    """
    states = sorted(s for s in all_states(game) if not game.is_terminal(s))
    suboptimal = losing_blunders = 0
    for state in states:
        v_before = minimax_ref._value(game, state)
        action = net_move(game, net, state)
        v_after = -minimax_ref._value(game, game.apply_action(state, action))
        if v_after < v_before - 1e-9:
            suboptimal += 1
            if v_before >= 0 and v_after < 0:
                losing_blunders += 1
    return len(states), suboptimal, losing_blunders


def optimal_move(game, state, rng):
    """A randomly chosen optimal move — perfect play that varies game to game."""
    position_value = minimax_ref._value(game, state)
    best = [
        a for a in game.legal_actions(state)
        if -minimax_ref._value(game, game.apply_action(state, a)) == position_value
    ]
    return rng.choice(best)


def random_move(game, state, rng):
    """A uniformly random legal move."""
    return rng.choice(game.legal_actions(state))


def play_match(game, net, opponent_move, net_first):
    """Play one game; return the net's result: +1 win, 0 draw, -1 loss."""
    state = game.initial_state()
    net_to_move = net_first
    while not game.is_terminal(state):
        if net_to_move:
            action = net_move(game, net, state)
        else:
            action = opponent_move(game, state)
        state = game.apply_action(state, action)
        net_to_move = not net_to_move
    value = game.outcome(state)
    return value if net_to_move else -value


def measure(game, net, opponent_move, games):
    """Play `games` games, alternating who moves first; return (wins, draws, losses)."""
    wins = draws = losses = 0
    for i in range(games):
        result = play_match(game, net, opponent_move, net_first=(i % 2 == 0))
        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            draws += 1
    return wins, draws, losses


def main():
    game = TicTacToe()
    rng = random.Random(config.SEED)

    for difficulty in config.DIFFICULTY_TARGETS:
        checkpoint = os.path.join(CHECKPOINT_DIR, f"{difficulty}.pt")
        net = QNetwork(game.input_size, game.action_size)
        net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        net.eval()

        total, suboptimal, blunders = optimality(game, net)
        vs_perfect = measure(game, net, lambda g, s: optimal_move(g, s, rng), MATCH_GAMES)
        vs_random = measure(game, net, lambda g, s: random_move(g, s, rng), MATCH_GAMES)

        print(f"[{difficulty.upper()}]  {os.path.basename(checkpoint)}")
        print(f"  optimality      : {blunders} losing blunders, "
              f"{suboptimal} suboptimal  (of {total} positions)")
        w, d, l = vs_perfect
        print(f"  vs perfect play : {w} win / {d} draw / {l} loss  (of {MATCH_GAMES})")
        w, d, l = vs_random
        print(f"  vs random play  : {w} win / {d} draw / {l} loss  (of {MATCH_GAMES})")
        print()


if __name__ == "__main__":
    main()
