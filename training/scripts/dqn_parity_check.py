"""Parity check (Python side) for the DQN bot.

Dumps the Q-network's outputs for a set of fixed boards — the Q-values and the
greedy move — as dqn_parity_expected.json. dqn_parity_check.mjs then runs the JS
engine on the same boards and compares.

The network is run in float64 (`.double()`) to match the JS engine's float64
arithmetic, so any large mismatch points to a real implementation bug.

Run:  python training/scripts/dqn_parity_check.py
"""

import json
import os
import sys

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING_DIR)

import torch

from dqn.qnetwork import QNetwork, q_values
from games.tictactoe import TicTacToe

CHECKPOINT = os.path.join(TRAINING_DIR, "dqn", "checkpoints", "best.pt")
OUT_PATH = os.path.join(TRAINING_DIR, "dqn_parity_expected.json")

# Fixed canonical states spanning empty, early, mid, near-terminal and tactical
# positions (mirrors the AlphaZero parity check).
FIXED_STATES = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],     # empty board
    [1, 0, 0, 0, 0, 0, 0, 0, 0],     # one move played
    [0, 0, 0, 0, 1, 0, 0, 0, 0],     # centre taken
    [1, -1, 0, 0, 0, 0, 0, 0, 0],    # two moves
    [1, 1, 0, -1, -1, 0, 0, 0, 0],   # forced win available at cell 2
    [-1, -1, 0, 0, 1, 0, 0, 0, 0],   # forced block required at cell 2
    [1, -1, 1, -1, 1, -1, 0, 0, 0],  # crowded mid-game
    [0, -1, 0, 0, 0, -1, 0, 1, 0],   # a position needing care
    [1, -1, 1, -1, -1, 1, -1, 1, 0], # one empty cell left
]


def main():
    game = TicTacToe()
    net = QNetwork(game.input_size, game.action_size)
    net.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    net.double()        # float64, to match the JS engine
    net.eval()

    cases = []
    for state in FIXED_STATES:
        s = tuple(state)
        qs = q_values(net, game, s)
        legal = game.legal_actions(s)
        best = legal[0] if legal else None
        for a in legal:
            if qs[a] > qs[best]:
                best = a
        cases.append({"state": list(state), "qValues": qs, "bestMove": best})

    with open(OUT_PATH, "w") as f:
        json.dump({"cases": cases}, f, indent=2)
    print(f"wrote {len(cases)} reference cases -> {OUT_PATH}")


if __name__ == "__main__":
    main()
