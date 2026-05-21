// Hand-written forward pass for the AlphaZero policy/value network.
// Mirrors training/alphazero/network.py exactly: each trunk layer is a linear
// map followed by ReLU; the policy head is linear (raw logits) and the value
// head is linear followed by tanh.
//
// `weights` is passed in rather than imported, so this module stays pure and
// loads in any environment — the browser app and the Node parity check alike.

// Linear layer: `w` is row-major [out][in], `b` is [out]. Returns a length-out
// array. The loop order matches export_weights.py's serialization. Exported so
// the DQN engine (dqnNet.js) can reuse it.
export function linear(input, w, b) {
  const out = new Array(w.length)
  for (let o = 0; o < w.length; o++) {
    const row = w[o]
    let sum = b[o]
    for (let i = 0; i < row.length; i++) {
      sum += row[i] * input[i]
    }
    out[o] = sum
  }
  return out
}

// Run the network on an encoded board. `weights` is the parsed weights.json.
// Returns { policyLogits: number[], value: number }.
export function forward(weights, input) {
  let x = input
  for (const layer of weights.trunk) {
    x = linear(x, layer.w, layer.b).map((v) => (v > 0 ? v : 0)) // ReLU
  }
  const policyLogits = linear(x, weights.policyHead.w, weights.policyHead.b)
  const valueRaw = linear(x, weights.valueHead.w, weights.valueHead.b)[0]
  return { policyLogits, value: Math.tanh(valueRaw) }
}
