# Training — neural tic-tac-toe bots

This folder trains the project's neural bots. Both learn tic-tac-toe entirely by
**self-play** — no minimax teacher, no human games — but with two different
reinforcement-learning methods:

- **AlphaZero** (`alphazero/`) — a policy/value network guided by Monte Carlo
  Tree Search. It *searches* at play time.
- **DQN** (`dqn/`) — a model-free Q-network. It does **no search**: one forward
  pass scores every move and the best is played.

Each pipeline exports weights into `../src/engine/`, which the browser engines
load. Training is a manual offline step — it is **not** part of `npm run build`.

## Setup

`python3-venv` / `python3-pip` must be installed (`sudo apt install
python3-venv python3-pip`). Then, from this `training/` directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the CPU build of PyTorch — tic-tac-toe trains on a CPU
in a few minutes.

## Workflow

All commands assume the venv is active and are run from `training/`.

**AlphaZero bot:**
```sh
python scripts/run_training.py        # self-play -> train -> arena gate, looped
python scripts/export_weights.py      # best checkpoint -> ../src/engine/weights.json
python scripts/evaluate_minimax.py    # check the bot against perfect minimax
python scripts/parity_check.py && node scripts/parity_check.mjs   # JS == Python
```

**DQN bot:**
```sh
python scripts/dqn_train.py           # self-play -> train -> sync target, looped
python scripts/dqn_export.py          # best checkpoint -> ../src/engine/dqnWeights.json
python scripts/dqn_evaluate.py        # check the bot against perfect minimax
python scripts/dqn_parity_check.py && node scripts/dqn_parity_check.mjs   # JS == Python
```

Both training scripts accept `--iterations` and `--games-per-iter` for quick
smoke tests. After retraining, re-run the matching export script and commit the
updated weights JSON.

## How it works

**Shared infrastructure:**

| Module | Role |
| --- | --- |
| `game_interface.py` | the game-agnostic `Game` contract both pipelines depend on |
| `games/tictactoe.py` | tic-tac-toe as the first `Game` plugin |
| `games/minimax_ref.py` | perfect minimax — used **only** to evaluate, never to train |

A new game is added by writing one plugin under `games/` — neither algorithm's
core changes. State is always canonicalized to the moving player's perspective
(`+1` = my mark, `-1` = opponent's, `0` = empty), so one network plays both sides.

**`alphazero/`** — `network.py` (policy/value MLP), `mcts.py` (PUCT search),
`selfplay.py`, `train.py`, `arena.py` (promotion gate). Config: `config.py`.

**`dqn/`** — `qnetwork.py` (Q-value MLP), `replay_buffer.py`, `selfplay.py`
(ε-greedy), `train.py` (temporal-difference loss with a target network). Config:
`dqn/config.py`. The DQN bot learns `Q(state, action)` and plays `argmax` Q over
legal moves — no tree, no lookahead.

## Results

All three bots were checked against perfect minimax on every one of the 4520
reachable decision positions. A *losing blunder* is a move that turns a
non-losing position into a losing one; **0 means the bot can never be beaten.**

| Bot | Method | Search at play time | Result |
| --- | --- | --- | --- |
| minimax | full game-tree search | yes (exhaustive) | perfect |
| AlphaZero | policy/value net + MCTS | yes (800-sim MCTS) | 0 losing blunders |
| DQN | Q-network | **none** (one forward pass) | 0 losing blunders |

Both neural bots are **unbeatable**. The DQN result is the notable one: with no
search at all, the network alone plays flawlessly. tic-tac-toe is a forced draw
under perfect play, so both bots draw minimax as first and second player.

## Parity with the browser

Each bot has a JavaScript engine (`../src/engine/`) that re-implements the
forward pass — and, for AlphaZero, the MCTS. The `*_parity_check.py` /
`*_parity_check.mjs` pairs confirm the JS and Python implementations agree: the
networks are run in float64 on both sides, so outputs match to ~1e-15 and the
chosen moves are identical.
