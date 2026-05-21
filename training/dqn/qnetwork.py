"""The Q-network for the DQN bot — a small MLP, game-agnostic.

It maps a board to one Q-value per action: Q(state, action) is an estimate of
the eventual game result (-1..+1, from the moving player's perspective) of
playing that action and then playing on. `tanh` keeps the outputs in the valid
[-1, 1] range and curbs DQN's tendency to overestimate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dqn import config


class QNetwork(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()
        hidden = config.HIDDEN_SIZE

        trunk = []
        prev = input_size
        for _ in range(config.TRUNK_LAYERS):
            trunk.append(nn.Linear(prev, hidden))
            prev = hidden
        self.trunk = nn.ModuleList(trunk)

        self.q_head = nn.Linear(hidden, action_size)

    def forward(self, x):
        """x: [batch, input_size] -> Q-values [batch, action_size] in [-1, 1]."""
        for layer in self.trunk:
            x = F.relu(layer(x))
        return torch.tanh(self.q_head(x))


@torch.no_grad()
def q_values(net, game, state):
    """Q-values for a single state, as a plain list (one per action).

    The input matches the network's dtype, so a net moved to float64
    (`.double()`, used by the parity check) runs the forward pass in float64.
    """
    dtype = next(net.parameters()).dtype
    x = torch.tensor([game.encode(state)], dtype=dtype)
    return net(x)[0].tolist()
