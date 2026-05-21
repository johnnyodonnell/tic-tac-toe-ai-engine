// Neural-network tic-tac-toe bot. Exports bestMove(board) — a drop-in
// replacement for the minimax engine: identical signature and return
// contract (a cell index, or null when the board is full).
//
// A trained policy/value network guided by MCTS chooses the move. The weights
// are produced offline by training/scripts/export_weights.py; see training/
// for how the network learned to play through self-play.

import weights from './weights.json'
import { HUMAN, BOT, availableMoves } from './game.js'
import { forward } from './nn.js'
import { game, encode } from './nnGame.js'
import { runMcts, bestAction } from './mcts.js'

// MCTS evaluation callback: a canonical state -> { logits, value }.
function evaluate(state) {
  const { policyLogits, value } = forward(weights, encode(state))
  return { logits: policyLogits, value }
}

// Convert the app board ('X' | 'O' | null) to a canonical state. The player to
// move is whoever has played no more marks than the other; 'X' goes first.
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

// Return the bot's move for `board`: a cell index, or null if the board is
// full. Drop-in replacement for bestMove() in minimax.js.
export function bestMove(board) {
  const moves = availableMoves(board)
  if (moves.length === 0) return null

  const root = runMcts(
    game,
    evaluate,
    toCanonical(board),
    weights.meta.numSimulations,
    weights.meta.cPuct,
  )
  const action = bestAction(root)
  return action === null ? moves[0] : action
}
