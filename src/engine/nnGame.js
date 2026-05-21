// Tic-tac-toe in the representation the neural network was trained on — a
// mirror of training/games/tictactoe.py. A state is a 9-int array from the
// player-to-move's perspective: +1 = my mark, -1 = opponent's, 0 = empty.
// After every move the board is negated so the opponent becomes "+1".

import { WIN_LINES } from './game.js'

// Winner of a canonical state: +1, -1, or 0 if none.
function winner(state) {
  for (const [a, b, c] of WIN_LINES) {
    if (state[a] !== 0 && state[a] === state[b] && state[a] === state[c]) {
      return state[a]
    }
  }
  return 0
}

// The Game interface the generic MCTS talks to.
export const game = {
  legalActions(state) {
    const actions = []
    for (let i = 0; i < 9; i++) {
      if (state[i] === 0) actions.push(i)
    }
    return actions
  },
  applyAction(state, action) {
    const next = state.slice()
    next[action] = 1
    for (let i = 0; i < 9; i++) {
      next[i] = -next[i] // flip to the opponent's perspective
    }
    return next
  },
  isTerminal(state) {
    return winner(state) !== 0 || state.every((cell) => cell !== 0)
  },
  outcome(state) {
    return winner(state)
  },
}

// Encode a canonical state as the network input: two 9-cell planes
// (current player's marks, then the opponent's). Matches TicTacToe.encode.
export function encode(state) {
  const input = new Array(18).fill(0)
  for (let i = 0; i < 9; i++) {
    if (state[i] === 1) input[i] = 1
    else if (state[i] === -1) input[i + 9] = 1
  }
  return input
}
