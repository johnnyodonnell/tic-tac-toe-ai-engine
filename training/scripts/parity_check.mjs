// Parity check (JS side): run the fixed boards from parity_check.py through
// the browser engine and compare against parity_expected.json.
//
// Confirms src/engine/nn.js and src/engine/mcts.js reproduce the Python
// network.py and mcts.py. A real implementation bug shows up as a forward-pass
// difference of ~0.01 or more; cross-language float noise stays far below the
// tolerance. The chosen move must match exactly.
//
// Run (after parity_check.py):  node training/scripts/parity_check.mjs

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { forward } from '../../src/engine/nn.js'
import { game, encode } from '../../src/engine/nnGame.js'
import { runMcts, bestAction, visitCounts } from '../../src/engine/mcts.js'

const here = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(here, '..', '..')

const weights = JSON.parse(
  fs.readFileSync(path.join(repoRoot, 'src/engine/weights.json'), 'utf8'),
)
const expected = JSON.parse(
  fs.readFileSync(path.join(here, '..', 'parity_expected.json'), 'utf8'),
)

const FORWARD_TOL = 1e-5

function maxAbsDiff(a, b) {
  let m = 0
  for (let i = 0; i < a.length; i++) {
    m = Math.max(m, Math.abs(a[i] - b[i]))
  }
  return m
}

function evaluate(state) {
  const { policyLogits, value } = forward(weights, encode(state))
  return { logits: policyLogits, value }
}

let failures = 0

for (const c of expected.cases) {
  const { policyLogits, value } = forward(weights, encode(c.state))
  const logitDiff = maxAbsDiff(policyLogits, c.policyLogits)
  const valueDiff = Math.abs(value - c.value)
  let ok = logitDiff < FORWARD_TOL && valueDiff < FORWARD_TOL

  let detail = `forward(Δlogit=${logitDiff.toExponential(1)}, Δvalue=${valueDiff.toExponential(1)})`

  if (c.visitCounts !== null) {
    const root = runMcts(
      game, evaluate, c.state,
      expected.meta.playSimulations, expected.meta.cPuct,
    )
    const counts = visitCounts(root, 9)
    const move = bestAction(root)
    const moveMatch = move === c.bestMove
    const countsMatch = counts.every((n, i) => n === c.visitCounts[i])
    ok = ok && moveMatch
    detail += `  move ${move}${moveMatch ? ' == ' : ' != '}${c.bestMove}`
    detail += `  visitCounts ${countsMatch ? 'exact' : 'differ (float noise)'}`
  } else {
    detail += '  [terminal]'
  }

  if (!ok) failures++
  console.log(`${ok ? 'OK  ' : 'FAIL'}  state ${JSON.stringify(c.state)}  ${detail}`)
}

console.log(
  failures === 0
    ? '\nPARITY OK — the JS engine matches the Python reference.'
    : `\nPARITY FAILED — ${failures} case(s) mismatched.`,
)
process.exit(failures === 0 ? 0 : 1)
