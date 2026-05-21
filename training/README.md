# Training — AlphaZero tic-tac-toe bot

This folder trains the neural network behind the app's bot. It learns entirely
by **self-play** — no minimax teacher, no human games — using an AlphaZero-style
loop: a policy/value network guided by MCTS, improved by playing itself.

The trained weights are exported to `../src/engine/weights.json`, which the app
loads. Training is a manual offline step; it is **not** part of `npm run build`.

## Setup

`python3-venv` / `python3-pip` must be installed (`sudo apt install
python3-venv python3-pip`). Then, from this `training/` directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the CPU build of PyTorch — tic-tac-toe trains on a CPU
in a couple of minutes.

## Workflow

All commands assume the venv is active and are run from `training/`.

```sh
python scripts/run_training.py        # self-play -> train -> arena gate, looped
python scripts/export_weights.py      # best checkpoint -> ../src/engine/weights.json
python scripts/evaluate_minimax.py    # check the bot against perfect minimax
python scripts/parity_check.py        # dump JS reference outputs (parity_expected.json)
node   scripts/parity_check.mjs       # confirm the JS engine matches Python
```

`run_training.py` accepts `--iterations` and `--games-per-iter` for quick smoke
tests, e.g. `python scripts/run_training.py --iterations 3 --games-per-iter 15`.

After retraining, re-run `export_weights.py` and commit the updated
`weights.json`.

## How it works

| Module | Role |
| --- | --- |
| `alphazero/game_interface.py` | the game-agnostic contract the core depends on |
| `alphazero/network.py` | policy/value MLP (game-agnostic, sized from config) |
| `alphazero/mcts.py` | PUCT search guided by the network |
| `alphazero/selfplay.py` | generates self-play games (the only training data) |
| `alphazero/train.py` | one training step (policy + value loss) |
| `alphazero/arena.py` | head-to-head gate — a new net is kept only if better |
| `games/tictactoe.py` | tic-tac-toe as the first `Game` plugin |
| `games/minimax_ref.py` | perfect minimax — used **only** to evaluate, never to train |
| `config.py` | every hyperparameter, in one place |

The AlphaZero core (`alphazero/`) talks only to the `Game` interface, so a new
game is added by writing one plugin under `games/` — nothing else changes.

State is always canonicalized to the moving player's perspective (`+1` = my
mark, `-1` = opponent's, `0` = empty), so one network plays both sides.

## Results

With the settings in `config.py`, the exported bot is **unbeatable**:

```
head-to-head vs perfect minimax:  draw (net first), draw (net second)
exhaustive optimality check:      4520 positions, 0 losing blunders
```

Every reachable position was checked; the bot never makes a move that turns a
non-losing position into a losing one. tic-tac-toe is a forced draw under
perfect play, so a flawless bot draws — which is exactly what it does.

## Parity with the browser

The browser runs its own JavaScript port of the forward pass and MCTS
(`../src/engine/nn.js`, `mcts.js`). `parity_check.py` + `parity_check.mjs`
confirm the two implementations agree: the network is run in float64 on both
sides, so forward outputs match to ~1e-15 and the chosen moves are identical.
