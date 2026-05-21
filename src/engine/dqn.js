// Model-free DQN tic-tac-toe bot. Exports bestMove(board) — a drop-in
// replacement for the minimax engine: identical signature and return contract
// (a cell index, or null when the board is full).
//
// Unlike the AlphaZero bot, this does NO search. One forward pass through the
// Q-network scores every move, and the best legal move is played. Weights come
// from dqnWeights.json, produced offline by training/scripts/dqn_export.py.
//
// Built for completeness and comparison; the app's UI uses neural.js. To try
// this bot instead, swap the import in App.jsx.

import dqnWeights from './dqnWeights.json'
import { HUMAN, BOT, availableMoves } from './game.js'
import { encode } from './nnGame.js'
import { qForward } from './dqnNet.js'

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

// Return the bot's move for `board`: the legal cell with the highest Q-value,
// or null if the board is full. Drop-in replacement for bestMove() in minimax.js.
export function bestMove(board) {
  const moves = availableMoves(board)
  if (moves.length === 0) return null

  const q = qForward(dqnWeights, encode(toCanonical(board)))
  let best = moves[0]
  for (const move of moves) {
    if (q[move] > q[best]) best = move
  }
  return best
}
