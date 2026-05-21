"""Parity check (Python side): dump reference outputs for a set of fixed boards.

For each board it records the network's policy logits and value, plus the MCTS
visit counts and chosen move. parity_check.mjs then runs the JS engine on the
same boards and compares.

The network is run in float64 (`.double()`) so it matches the JS engine's
float64 arithmetic — any large mismatch then points to a real implementation
bug rather than a precision difference.

Run:  python training/scripts/parity_check.py
"""

import json
import os
import sys

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING_DIR)

import torch

import config
from alphazero import mcts
from alphazero.network import PolicyValueNet, infer
from games.tictactoe import TicTacToe

CHECKPOINT = os.path.join(TRAINING_DIR, "checkpoints", "best.pt")
OUT_PATH = os.path.join(TRAINING_DIR, "parity_expected.json")

# Fixed canonical states spanning empty, early, mid, near-terminal and tactical
# (forced win / forced block) positions. The 8th is the position that needed a
# deep search to play correctly — a good stress case.
FIXED_STATES = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],     # empty board
    [1, 0, 0, 0, 0, 0, 0, 0, 0],     # one move played
    [0, 0, 0, 0, 1, 0, 0, 0, 0],     # centre taken
    [1, -1, 0, 0, 0, 0, 0, 0, 0],    # two moves
    [1, 1, 0, -1, -1, 0, 0, 0, 0],   # forced win available at cell 2
    [-1, -1, 0, 0, 1, 0, 0, 0, 0],   # forced block required at cell 2
    [1, -1, 1, -1, 1, -1, 0, 0, 0],  # crowded mid-game
    [0, -1, 0, 0, 0, -1, 0, 1, 0],   # position that needs a deep search
    [1, -1, 1, -1, -1, 1, -1, 1, 0], # one empty cell left
]


def main():
    game = TicTacToe()
    net = PolicyValueNet(game.input_size, game.action_size)
    net.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    net.double()        # float64, to match the JS engine
    net.eval()

    cases = []
    for state in FIXED_STATES:
        s = tuple(state)
        logits, value = infer(net, game, s)
        record = {"state": list(state), "policyLogits": logits, "value": value}

        if game.is_terminal(s):
            record["visitCounts"] = None
            record["bestMove"] = None
        else:
            root = mcts.run_mcts(game, net, s, config.PLAY_SIMULATIONS,
                                 add_noise=False)
            record["visitCounts"] = [
                root.children[a].visit_count if a in root.children else 0
                for a in range(game.action_size)
            ]
            record["bestMove"] = mcts.best_action(root)
        cases.append(record)

    payload = {
        "meta": {
            "playSimulations": config.PLAY_SIMULATIONS,
            "cPuct": config.C_PUCT,
        },
        "cases": cases,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {len(cases)} reference cases -> {OUT_PATH}")


if __name__ == "__main__":
    main()
