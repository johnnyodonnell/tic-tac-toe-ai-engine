import Cell from './Cell.jsx'

export default function Board({ board, winningLine, onCellClick, frozen }) {
  return (
    <div className="board">
      {board.map((value, i) => (
        <Cell
          key={i}
          value={value}
          highlight={winningLine?.includes(i) ?? false}
          disabled={frozen || value !== null}
          onClick={() => onCellClick(i)}
        />
      ))}
    </div>
  )
}
