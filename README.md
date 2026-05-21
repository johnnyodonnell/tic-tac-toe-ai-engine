# Tic-Tac-Toe AI Engine

A tic-tac-toe web app (React + Vite) with **three interchangeable AI engines** —
one classic search algorithm and two neural networks that taught themselves the
game through self-play. All three play perfectly; the interesting part is *how*.

## Running locally

```sh
npm install
npm run dev        # or: ./run-local.sh
```

Then open the printed URL. `npm run build` produces a production build in `dist/`.

## The engines

Every engine lives in `src/engine/` and exposes the exact same function:

```js
bestMove(board)  // board: 9-element array of 'X' | 'O' | null  ->  cell index
```

Because the signature is identical, the engines are drop-in interchangeable. The
app picks one with a single import in `src/App.jsx`.

| Engine | Approach | Learns? | Searches at play time? | Result |
| --- | --- | --- | --- | --- |
| Minimax | full game-tree search | no | yes — exhaustive | perfect |
| AlphaZero | policy/value net + MCTS | self-play | yes — 800-sim MCTS | unbeatable |
| DQN | Q-network | self-play | no — one forward pass | unbeatable |

Both neural engines were checked against minimax on all 4520 reachable positions
and never make a move that turns a non-losing position into a loss.

### 1. Minimax — `src/engine/minimax.js`

The classic approach: a full-depth game-tree search. On every move it explores
*every* possible continuation to the end of the game and plays a provably
optimal move. It has no network and no training — it's a pure algorithm, and
tic-tac-toe's game tree is small enough to search exhaustively in an instant.

It is perfect by construction: it cannot be beaten, and the best a human can do
is force a draw.

### 2. AlphaZero — `src/engine/neural.js`

A neural network guided by **Monte Carlo Tree Search (MCTS)**, in the style of
DeepMind's AlphaZero. The network has two heads — a *policy* (which moves look
promising) and a *value* (who's winning) — and it learned entirely by
**self-play reinforcement learning**: it played itself thousands of times with
no minimax teacher and no human games.

At play time it still *searches*: MCTS runs 800 simulated continuations per
move, using the network to steer the search toward promising lines. The network
makes the search smart; the search makes the network's play flawless.

Supporting files: `nn.js` (the forward pass), `mcts.js` (the search),
`nnGame.js` (board encoding), and `weights.json` (the trained weights).

### 3. DQN — `src/engine/dqn.js`  *(currently used by the app)*

A model-free **Deep Q-Network**. The network learned a *Q-value* for every move
— an estimate of the eventual game result if that move is played — again purely
through self-play.

The contrast with AlphaZero is the point: DQN does **no search at all**. Picking
a move is a single forward pass — board in, nine Q-values out, play the best
legal one. There is no tree, no lookahead; all of the skill lives in the network
weights themselves. At full strength it reaches the same unbeatable standard as
AlphaZero without ever simulating a future move.

**Difficulty levels.** The app's selector — Easy / Medium / Hard — chooses
between *three separately trained models*, each a snapshot from a different
point in self-play training. Hard is the unbeatable net; Medium and Easy are
weaker earlier snapshots that make systematic mistakes. All three are still
deterministic single-forward-pass players — only the network differs. Roughly:
Easy loses almost every game to good play, Medium is a fair fight, Hard cannot
be beaten.

Supporting files: `dqnNet.js` (the forward pass), `nnGame.js` (board encoding),
and `dqnWeights{Easy,Medium,Hard}.json` (the three trained models).

> **Model-based vs. model-free** — AlphaZero *uses the rules to plan ahead*
> (model-based); DQN never simulates the future, it just reacts (model-free).

## Switching the app's engine

`src/App.jsx` imports `bestMove` from one engine. To change which bot the app
plays against, change that one line:

```js
import { bestMove } from './engine/dqn.js'      // DQN        (current)
// import { bestMove } from './engine/neural.js'  // AlphaZero
// import { bestMove } from './engine/minimax.js' // Minimax
```

## Project layout

```
src/
  App.jsx              the game UI and turn logic
  components/          Board, Cell, Status, DifficultySelector
  engine/
    game.js            shared rules (win lines, legal moves, ...)
    minimax.js         engine 1 — minimax
    neural.js  nn.js  mcts.js  nnGame.js  weights.json      engine 2 — AlphaZero
    dqn.js  dqnNet.js  nnGame.js  dqnWeights{Easy,Medium,Hard}.json   engine 3 — DQN
training/              offline Python pipelines that trained the neural engines
```

## Training the neural engines

The AlphaZero and DQN networks are trained offline in Python; the browser only
runs the finished weights. The weight files (`weights.json`, `dqnWeights.json`)
are committed, so training is **not** needed to run the app — it is a manual
step, separate from `npm run build`.

See [`training/README.md`](training/README.md) for how the networks learn,
how to retrain them, and how the Python and JavaScript implementations are kept
in exact agreement.
