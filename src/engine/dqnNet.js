// Forward pass for the DQN Q-network. Mirrors training/dqn/qnetwork.py: each
// trunk layer is Linear + ReLU, and the Q-head is Linear + tanh.
//
// `weights` is passed in rather than imported, so this module stays pure and
// loads in any environment — the browser app and the Node parity check alike.

import { linear } from './nn.js'

// Run the Q-network on an encoded board. `weights` is the parsed dqnWeights.json.
// Returns an array of Q-values, one per action, each in [-1, 1].
export function qForward(weights, input) {
  let x = input
  for (const layer of weights.trunk) {
    x = linear(x, layer.w, layer.b).map((v) => (v > 0 ? v : 0)) // ReLU
  }
  return linear(x, weights.qHead.w, weights.qHead.b).map((q) => Math.tanh(q))
}
