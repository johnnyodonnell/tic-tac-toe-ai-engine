// Model-free DQN tic-tac-toe bot. Exports bestMove(board, difficulty).
//
// Three separately trained models back the three difficulties — each a snapshot
// from a different point in self-play training. Hard is the unbeatable
// 0-blunder net; Medium and Easy are weaker (earlier) snapshots that make
// systematic mistakes. All play deterministically: one forward pass through the
// Q-network, argmax over legal moves, no search.
//
// Weights are produced offline by training/scripts/dqn_export.py.

import dqnWeightsEasy from './dqnWeightsEasy.json'
import dqnWeightsMedium from './dqnWeightsMedium.json'
import dqnWeightsHard from './dqnWeightsHard.json'
import { HUMAN, BOT, availableMoves } from './game.js'
import { encode } from './nnGame.js'
import { qForward } from './dqnNet.js'

// The selectable difficulty levels, weakest to strongest.
export const DIFFICULTIES = ['easy', 'medium', 'hard']

const WEIGHTS = {
  easy: dqnWeightsEasy,
  medium: dqnWeightsMedium,
  hard: dqnWeightsHard,
}

// Convert the app board ('X' | 'O' | null) to a canonical state: +1 for the
// player to move, -1 for the opponent, 0 empty. 'X' moves first.
function toCanonical(board) {
  let xs = 0
  let os = 0
  for (const cell of board) {
    if (cell === HUMAN) xs++
    else if (cell === BOT) os++
  }
  const mover = xs === os ? HUMAN : BOT
  return board.map((cell) => {
    if (cell === null) return 0
    return cell === mover ? 1 : -1
  })
}

// Return the bot's move for `board` at the given difficulty: the legal cell
// with the highest Q-value, or null if the board is full. `difficulty`
// defaults to 'hard', keeping bestMove(board) a drop-in for the minimax engine.
export function bestMove(board, difficulty = 'hard') {
  const moves = availableMoves(board)
  if (moves.length === 0) return null

  const weights = WEIGHTS[difficulty] ?? WEIGHTS.hard
  const q = qForward(weights, encode(toCanonical(board)))
  let best = moves[0]
  for (const move of moves) {
    if (q[move] > q[best]) best = move
  }
  return best
}
