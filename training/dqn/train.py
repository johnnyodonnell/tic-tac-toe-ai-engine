"""One DQN training step: temporal-difference targets + MSE loss.

The target for a transition (s, a, r, s', done):
  done       -> r                              (the game ended; r is the result)
  otherwise  -> gamma * ( -max_{a'} Q_target(s', a') )   over legal a'

The negation is the two-player negamax twist: s' is the opponent's turn, so the
opponent's best reply is the worst case for the player who just moved. The
bootstrap is computed with the frozen target network for stability.
"""

import torch
import torch.nn.functional as F

from dqn import config


def train_step(qnet, target_qnet, optimizer, game, batch):
    """Run one optimizer step on a batch; return the loss as a float.

    `batch` is (states, actions, rewards, next_states, dones).
    """
    states, actions, rewards, next_states, dones = batch

    x = torch.tensor([game.encode(s) for s in states], dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.long)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    dones_t = torch.tensor(dones, dtype=torch.bool)

    # Predicted Q for the action actually taken (online network, has gradient).
    predicted = qnet(x).gather(1, actions_t.unsqueeze(1)).squeeze(1)

    # Bootstrap target from the frozen target network.
    with torch.no_grad():
        next_x = torch.tensor([game.encode(s) for s in next_states],
                              dtype=torch.float32)
        next_q = target_qnet(next_x)

        # Mask illegal actions in s' to -inf so they can never be the max.
        mask_rows = []
        for s in next_states:
            row = [float("-inf")] * game.action_size
            for a in game.legal_actions(s):
                row[a] = 0.0
            mask_rows.append(row)
        mask = torch.tensor(mask_rows, dtype=torch.float32)

        best_next = (next_q + mask).max(dim=1).values
        bootstrap = config.GAMMA * (-best_next)        # negamax
        # For terminal moves the bootstrap is meaningless (s' has no legal
        # actions); torch.where discards it in favour of the real reward.
        target = torch.where(dones_t, rewards_t, bootstrap)

    loss = F.mse_loss(predicted, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
