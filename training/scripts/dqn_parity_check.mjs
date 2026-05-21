// Parity check (JS side) for the DQN difficulty models: run the fixed boards
// from dqn_parity_check.py through the JS engine and compare against
// dqn_parity_expected.json, for every difficulty.
//
// Confirms src/engine/dqnNet.js reproduces training/dqn/qnetwork.py. A real bug
// shows up as a Q-value difference of ~0.01 or more; cross-language float noise
// stays far below the tolerance. The chosen move must match exactly.
//
// Run (after dqn_parity_check.py):  node training/scripts/dqn_parity_check.mjs

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { encode } from '../../src/engine/nnGame.js'
import { qForward } from '../../src/engine/dqnNet.js'

const here = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(here, '..', '..')

const expected = JSON.parse(
  fs.readFileSync(path.join(here, '..', 'dqn_parity_expected.json'), 'utf8'),
)

const TOL = 1e-5

function maxAbsDiff(a, b) {
  let m = 0
  for (let i = 0; i < a.length; i++) {
    m = Math.max(m, Math.abs(a[i] - b[i]))
  }
  return m
}

function weightsFor(difficulty) {
  const name = `dqnWeights${difficulty[0].toUpperCase()}${difficulty.slice(1)}.json`
  return JSON.parse(fs.readFileSync(path.join(repoRoot, 'src/engine', name), 'utf8'))
}

let failures = 0

for (const [difficulty, cases] of Object.entries(expected.models)) {
  const weights = weightsFor(difficulty)
  for (const c of cases) {
    const q = qForward(weights, encode(c.state))
    const qDiff = maxAbsDiff(q, c.qValues)

    const legal = []
    for (let i = 0; i < 9; i++) {
      if (c.state[i] === 0) legal.push(i)
    }
    let best = legal[0]
    for (const a of legal) {
      if (q[a] > q[best]) best = a
    }
    const moveMatch = best === c.bestMove
    const ok = qDiff < TOL && moveMatch
    if (!ok) failures++

    console.log(
      `${ok ? 'OK  ' : 'FAIL'}  [${difficulty.padEnd(6)}] state ` +
      `${JSON.stringify(c.state)}  Δq=${qDiff.toExponential(1)}  ` +
      `move ${best}${moveMatch ? ' == ' : ' != '}${c.bestMove}`,
    )
  }
}

console.log(
  failures === 0
    ? '\nPARITY OK — the JS DQN engine matches the Python reference for all models.'
    : `\nPARITY FAILED — ${failures} case(s) mismatched.`,
)
process.exit(failures === 0 ? 0 : 1)
