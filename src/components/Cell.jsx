export default function Cell({ value, onClick, disabled, highlight }) {
  const classes = ['cell']
  if (value) classes.push(`cell--${value.toLowerCase()}`)
  if (highlight) classes.push('cell--win')

  return (
    <button
      className={classes.join(' ')}
      onClick={onClick}
      disabled={disabled}
      aria-label={value ? `Cell taken by ${value}` : 'Empty cell'}
    >
      {value}
    </button>
  )
}
