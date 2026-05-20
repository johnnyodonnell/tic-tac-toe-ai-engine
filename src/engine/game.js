// Pure tic-tac-toe rules — no React, no side effects.
// A board is a 9-element array; each slot is 'X', 'O', or null.

export const HUMAN = 'X'
export const BOT = 'O'

// The 8 ways to win: 3 rows, 3 columns, 2 diagonals.
export const WIN_LINES = [
  [0, 1, 2],
  [3, 4, 5],
  [6, 7, 8],
  [0, 3, 6],
  [1, 4, 7],
  [2, 5, 8],
  [0, 4, 8],
  [2, 4, 6],
]

export function createBoard() {
  return Array(9).fill(null)
}

// Returns the winning triple of indices, or null if nobody has won.
export function getWinningLine(board) {
  for (const line of WIN_LINES) {
    const [a, b, c] = line
    if (board[a] && board[a] === board[b] && board[a] === board[c]) {
      return line
    }
  }
  return null
}

// Returns 'X', 'O', or null.
export function getWinner(board) {
  const line = getWinningLine(board)
  return line ? board[line[0]] : null
}

export function isFull(board) {
  return board.every((cell) => cell !== null)
}

export function isGameOver(board) {
  return getWinner(board) !== null || isFull(board)
}

// Indices of all empty cells.
export function availableMoves(board) {
  const moves = []
  board.forEach((cell, i) => {
    if (cell === null) moves.push(i)
  })
  return moves
}
