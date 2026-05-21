import { useState } from 'react'
import Board from './components/Board.jsx'
import Status from './components/Status.jsx'
import DifficultySelector from './components/DifficultySelector.jsx'
import { bestMove } from './engine/dqn.js'
import {
  BOT,
  HUMAN,
  createBoard,
  getWinner,
  getWinningLine,
  isGameOver,
} from './engine/game.js'

function statusMessage(board) {
  const winner = getWinner(board)
  if (winner === HUMAN) return 'You win!'
  if (winner === BOT) return 'Bot wins'
  if (isGameOver(board)) return "It's a draw"
  return 'Your turn'
}

export default function App() {
  const [board, setBoard] = useState(createBoard)
  const [difficulty, setDifficulty] = useState('medium')

  function handleCellClick(index) {
    if (board[index] !== null || isGameOver(board)) return

    // Apply the human's move.
    const afterHuman = [...board]
    afterHuman[index] = HUMAN

    // If that ended the game, stop — otherwise let the bot respond.
    if (isGameOver(afterHuman)) {
      setBoard(afterHuman)
      return
    }

    const afterBot = [...afterHuman]
    afterBot[bestMove(afterHuman, difficulty)] = BOT
    setBoard(afterBot)
  }

  function newGame() {
    setBoard(createBoard())
  }

  return (
    <main className="app">
      <h1>Tic-Tac-Toe</h1>
      <Status message={statusMessage(board)} />
      <Board
        board={board}
        winningLine={getWinningLine(board)}
        onCellClick={handleCellClick}
        frozen={isGameOver(board)}
      />
      <button className="new-game" onClick={newGame}>
        New Game
      </button>
      <DifficultySelector difficulty={difficulty} onChange={setDifficulty} />
    </main>
  )
}
