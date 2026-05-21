// PUCT Monte Carlo Tree Search — a JS port of training/alphazero/mcts.py.
// Game-agnostic: it talks only to a `game` object (legalActions, applyAction,
// isTerminal, outcome) and an `evaluate(state)` callback returning the
// network's { logits, value }.
//
// Tie-breaking matches the Python side: ties in PUCT selection and in the
// visit-count argmax go to the LOWEST action index. Children are inserted in
// ascending action order and every comparison below is a strict `>`.

class Node {
  constructor(prior, state) {
    this.prior = prior
    this.state = state
    this.visitCount = 0
    this.valueSum = 0
    this.children = new Map() // action -> Node, inserted ascending by action
    this.isTerminal = false
  }

  // Mean value, from this node's own perspective.
  value() {
    return this.visitCount === 0 ? 0 : this.valueSum / this.visitCount
  }
}

// Softmax of `logits` restricted to `legal` actions -> Map(action -> prob).
function maskedSoftmax(logits, legal) {
  let max = -Infinity
  for (const a of legal) {
    if (logits[a] > max) max = logits[a]
  }
  let total = 0
  const exps = new Map()
  for (const a of legal) {
    const e = Math.exp(logits[a] - max)
    exps.set(a, e)
    total += e
  }
  const probs = new Map()
  for (const a of legal) {
    probs.set(a, exps.get(a) / total)
  }
  return probs
}

// Create children for `node`; return the value estimate to back up.
function expand(node, game, evaluate) {
  if (game.isTerminal(node.state)) {
    node.isTerminal = true
    return game.outcome(node.state)
  }
  const { logits, value } = evaluate(node.state)
  const legal = game.legalActions(node.state) // ascending
  const priors = maskedSoftmax(logits, legal)
  for (const action of legal) {
    const child = new Node(priors.get(action), game.applyAction(node.state, action))
    node.children.set(action, child)
  }
  return value
}

// Pick the child with the highest PUCT score (lowest index breaks ties).
function selectChild(node, cPuct) {
  const sqrtTotal = Math.sqrt(node.visitCount)
  let bestScore = -Infinity
  let best = null
  for (const [, child] of node.children) {
    const q = -child.value() // child.value() is in the child's frame
    const u = (cPuct * child.prior * sqrtTotal) / (1 + child.visitCount)
    const score = q + u
    if (score > bestScore) {
      bestScore = score
      best = child
    }
  }
  return best
}

// Run `numSimulations` simulations and return the populated root node.
export function runMcts(game, evaluate, rootState, numSimulations, cPuct) {
  const root = new Node(0, rootState)
  expand(root, game, evaluate)

  for (let sim = 0; sim < numSimulations; sim++) {
    let node = root
    const path = [node]
    while (node.children.size > 0) {
      node = selectChild(node, cPuct)
      path.push(node)
    }

    // `node` is a leaf — score it (terminal) or expand it (network value).
    let value = node.isTerminal
      ? game.outcome(node.state)
      : expand(node, game, evaluate)

    // Backup: `value` is from the leaf's perspective; flip each ply up.
    for (let i = path.length - 1; i >= 0; i--) {
      path[i].visitCount += 1
      path[i].valueSum += value
      value = -value
    }
  }
  return root
}

// Most-visited action; lowest index breaks ties.
export function bestAction(root) {
  let bestActionIndex = null
  let bestVisits = -1
  for (const [action, child] of root.children) {
    if (child.visitCount > bestVisits) {
      bestVisits = child.visitCount
      bestActionIndex = action
    }
  }
  return bestActionIndex
}

// Visit counts for every action 0..actionSize-1 (0 where unexplored).
// Used by the parity check to compare against the Python search.
export function visitCounts(root, actionSize) {
  const counts = new Array(actionSize).fill(0)
  for (const [action, child] of root.children) {
    counts[action] = child.visitCount
  }
  return counts
}
