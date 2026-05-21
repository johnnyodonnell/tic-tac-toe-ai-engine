"""Evaluate the trained network against perfect minimax.

Two checks:
  1. Head-to-head — the net vs minimax, net moving first and second.
  2. Exhaustive optimality — for every reachable position, verify the net's
     greedy move never worsens the game-theoretic value. Zero "losing
     blunders" means the bot can never be beaten.

Run:  python training/scripts/evaluate_minimax.py
"""

import os
import sys

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING_DIR)

import torch

import config
from alphazero import mcts
from alphazero.network import PolicyValueNet
from games import minimax_ref
from games.tictactoe import TicTacToe

CHECKPOINT = os.path.join(TRAINING_DIR, "checkpoints", "best.pt")


def all_states(game):
    """Every reachable canonical state, found by expanding all legal moves."""
    seen = set()
    stack = [game.initial_state()]
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
    """The net's greedy (temperature-0) move via MCTS — at play strength."""
    root = mcts.run_mcts(game, net, state, config.PLAY_SIMULATIONS, add_noise=False)
    return mcts.best_action(root)


def head_to_head(game, net):
    """Play the net vs minimax, net moving first then second."""
    results = []
    for net_first in (True, False):
        state = game.initial_state()
        net_to_move = net_first
        while not game.is_terminal(state):
            if net_to_move:
                action = net_move(game, net, state)
            else:
                action = minimax_ref.best_action(game, state)
            state = game.apply_action(state, action)
            net_to_move = not net_to_move
        value = game.outcome(state)
        net_value = value if net_to_move else -value
        outcome = "draw" if net_value == 0 else ("WIN" if net_value > 0 else "LOSS")
        results.append(("net first" if net_first else "net second", outcome))
    return results


def optimality(game, net):
    """Check the net's move at every decision position against minimax.

    Returns (positions checked, suboptimal moves, losing blunders). A losing
    blunder is a move that turns a non-losing position into a losing one.
    """
    states = sorted(s for s in all_states(game) if not game.is_terminal(s))
    suboptimal = 0
    losing_blunders = 0
    for state in states:
        v_before = minimax_ref._value(game, state)
        action = net_move(game, net, state)
        v_after = -minimax_ref._value(game, game.apply_action(state, action))
        if v_after < v_before - 1e-9:
            suboptimal += 1
            if v_before >= 0 and v_after < 0:
                losing_blunders += 1
    return len(states), suboptimal, losing_blunders


def main():
    game = TicTacToe()
    net = PolicyValueNet(game.input_size, game.action_size)
    net.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    net.eval()

    print("head-to-head vs perfect minimax:")
    for label, outcome in head_to_head(game, net):
        print(f"  {label:11s}: {outcome}")

    print("exhaustive optimality check:")
    total, suboptimal, blunders = optimality(game, net)
    print(f"  decision positions checked : {total}")
    print(f"  suboptimal moves           : {suboptimal}")
    print(f"  losing blunders            : {blunders}")
    if blunders == 0:
        print("  => the bot can never be beaten (no losing blunders).")
    else:
        print("  => WARNING: the bot has positions where it can be beaten.")


if __name__ == "__main__":
    main()
