// The AI engine: a full-depth minimax search. Because tic-tac-toe's game
// tree is tiny, this explores every reachable position and plays perfectly —
// the bot can never be beaten; the human's best outcome is a draw.

import { BOT, HUMAN, availableMoves, getWinner, isFull } from './game.js'

// Scores a board from the bot's perspective. `depth` is the number of plies
// searched to reach this position. Subtracting it makes the bot prefer
// winning sooner and losing later, rather than playing "lazily".
function minimax(board, isBotTurn, depth) {
  const winner = getWinner(board)
  if (winner === BOT) return 10 - depth
  if (winner === HUMAN) return depth - 10
  if (isFull(board)) return 0

  const mark = isBotTurn ? BOT : HUMAN
  let best = isBotTurn ? -Infinity : Infinity

  for (const move of availableMoves(board)) {
    board[move] = mark
    const score = minimax(board, !isBotTurn, depth + 1)
    board[move] = null // restore — keeps the search allocation-free
    best = isBotTurn ? Math.max(best, score) : Math.min(best, score)
  }
  return best
}

// Returns the index of the optimal move for the bot ('O'),
// or null if the board is full.
export function bestMove(board) {
  const working = [...board]
  let bestScore = -Infinity
  let move = null

  for (const candidate of availableMoves(working)) {
    working[candidate] = BOT
    const score = minimax(working, false, 1)
    working[candidate] = null
    if (score > bestScore) {
      bestScore = score
      move = candidate
    }
  }
  return move
}
