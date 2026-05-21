"""The policy/value network — a small MLP, game-agnostic.

Layer sizes come from config (HIDDEN_SIZE, TRUNK_LAYERS) and from the Game's
input_size / action_size, so the same class serves any game plugin.

forward() returns *raw* policy logits — softmax and illegal-move masking are
done by the caller (MCTS). This keeps the network identical to the hand-written
JS forward pass in src/engine/nn.js, which the parity check relies on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class PolicyValueNet(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()
        hidden = config.HIDDEN_SIZE

        trunk = []
        prev = input_size
        for _ in range(config.TRUNK_LAYERS):
            trunk.append(nn.Linear(prev, hidden))
            prev = hidden
        self.trunk = nn.ModuleList(trunk)

        self.policy_head = nn.Linear(hidden, action_size)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        """x: [batch, input_size] float tensor.

        Returns (policy_logits [batch, action_size], value [batch]).
        """
        for layer in self.trunk:
            x = F.relu(layer(x))
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x)).squeeze(-1)
        return policy_logits, value


@torch.no_grad()
def infer(net, game, state):
    """Single-state inference for MCTS.

    Returns (policy_logits: list[action_size], value: float) — raw logits with
    no masking; the caller restricts them to legal actions.

    The input matches the network's own dtype, so a net moved to float64
    (`.double()`, used by the parity check) runs the whole forward pass in
    float64 to match the JS engine.
    """
    dtype = next(net.parameters()).dtype
    x = torch.tensor([game.encode(state)], dtype=dtype)
    logits, value = net(x)
    return logits[0].tolist(), float(value[0])
