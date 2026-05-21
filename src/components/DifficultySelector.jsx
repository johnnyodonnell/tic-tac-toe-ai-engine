import { DIFFICULTIES } from '../engine/dqn.js'

const LABELS = { easy: 'Easy', medium: 'Medium', hard: 'Hard' }

export default function DifficultySelector({ difficulty, onChange }) {
  return (
    <div className="difficulty">
      <span className="difficulty__label">Difficulty</span>
      <div className="difficulty__options" role="group" aria-label="Difficulty">
        {DIFFICULTIES.map((level) => (
          <button
            key={level}
            type="button"
            className={
              'difficulty__btn' +
              (level === difficulty ? ' difficulty__btn--active' : '')
            }
            aria-pressed={level === difficulty}
            onClick={() => onChange(level)}
          >
            {LABELS[level]}
          </button>
        ))}
      </div>
    </div>
  )
}
