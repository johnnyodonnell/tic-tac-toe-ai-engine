"""One AlphaZero training step.

Loss = policy cross-entropy + value MSE. L2 regularization is applied via the
optimizer's `weight_decay` (set when the optimizer is constructed), so it does
not appear explicitly here.
"""

import torch
import torch.nn.functional as F


def train_step(net, optimizer, states, pis, zs):
    """Run a single optimizer step on one batch.

    Returns (total_loss, policy_loss, value_loss) as floats.
    """
    x = torch.tensor(states, dtype=torch.float32)
    target_pi = torch.tensor(pis, dtype=torch.float32)
    target_z = torch.tensor(zs, dtype=torch.float32)

    policy_logits, value = net(x)

    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(target_pi * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(value, target_z)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), policy_loss.item(), value_loss.item()
