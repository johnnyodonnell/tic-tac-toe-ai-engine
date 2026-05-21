"""Orchestrate DQN training: self-play -> train -> sync target -> evaluate.

Run from anywhere:
    python training/scripts/dqn_train.py
Optional smoke-test overrides:
    python training/scripts/dqn_train.py --iterations 3 --games-per-iter 20

One run snapshots three checkpoints — easy.pt, medium.pt, hard.pt — into
training/dqn/checkpoints/. For each difficulty, the checkpoint kept is the one
whose losing-blunder count is closest to that difficulty's target (see
dqn/config.py DIFFICULTY_TARGETS). Export them with scripts/dqn_export.py.
"""

import argparse
import copy
import os
import random
import sys

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING_DIR)

import numpy as np
import torch

from dqn import config
from dqn.qnetwork import QNetwork
from dqn.replay_buffer import ReplayBuffer
from dqn.selfplay import play_game, random_start
from dqn.train import train_step
from games import minimax_ref
from games.tictactoe import TicTacToe

CHECKPOINT_DIR = os.path.join(TRAINING_DIR, "dqn", "checkpoints")


def epsilon_for(iteration):
    """Linear decay from EPSILON_START to EPSILON_END over EPSILON_DECAY_ITERS."""
    frac = min(1.0, (iteration - 1) / max(1, config.EPSILON_DECAY_ITERS))
    return config.EPSILON_START + frac * (config.EPSILON_END - config.EPSILON_START)


def all_decision_states(game):
    """Every reachable non-terminal canonical state (the bot's decision points)."""
    seen, stack = set(), [game.initial_state()]
    while stack:
        s = stack.pop()
        if s in seen:
            continue
        seen.add(s)
        if not game.is_terminal(s):
            for a in game.legal_actions(s):
                stack.append(game.apply_action(s, a))
    return [s for s in seen if not game.is_terminal(s)]


def count_losing_blunders(game, net, states):
    """How many decision states the net's greedy move turns from non-losing into
    losing, judged by minimax. 0 == the bot can never be beaten.

    Cheap because DQN has no search — one batched forward pass scores everything.
    """
    x = torch.tensor([game.encode(s) for s in states], dtype=torch.float32)
    with torch.no_grad():
        q = net(x).tolist()

    blunders = 0
    for qs, s in zip(q, states):
        legal = game.legal_actions(s)
        best_a = legal[0]
        for a in legal:
            if qs[a] > qs[best_a]:
                best_a = a
        v_before = minimax_ref._value(game, s)
        v_after = -minimax_ref._value(game, game.apply_action(s, best_a))
        if v_before >= 0 and v_after < 0:
            blunders += 1
    return blunders


def main():
    parser = argparse.ArgumentParser(description="DQN tic-tac-toe training")
    parser.add_argument("--iterations", type=int, default=config.NUM_ITERATIONS)
    parser.add_argument("--games-per-iter", type=int, default=config.GAMES_PER_ITER)
    args = parser.parse_args()

    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    rng_np = np.random.default_rng(config.SEED)
    rng_py = random.Random(config.SEED)

    game = TicTacToe()
    qnet = QNetwork(game.input_size, game.action_size)
    qnet.eval()
    target_qnet = copy.deepcopy(qnet)
    target_qnet.eval()
    optimizer = torch.optim.Adam(qnet.parameters(), lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)
    buffer = ReplayBuffer()
    decision_states = all_decision_states(game)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    targets = config.DIFFICULTY_TARGETS
    # For each difficulty, track the checkpoint whose blunder count is closest
    # to its target: snapshots[d] = (best_distance, blunders_when_saved).
    snapshots = {d: None for d in targets}

    print(f"training: {args.iterations} iterations x {args.games_per_iter} games "
          f"({len(decision_states)} decision states)")
    print(f"difficulty targets (losing blunders): {targets}")
    for iteration in range(1, args.iterations + 1):
        epsilon = epsilon_for(iteration)

        # 1. Self-play — fill the replay buffer (some games from random starts).
        for _ in range(args.games_per_iter):
            if rng_np.random() < config.RANDOM_START_FRACTION:
                start = random_start(game, rng_np, config.RANDOM_START_MAX_PLIES)
            else:
                start = None
            buffer.add_many(play_game(game, qnet, rng_np, epsilon, start))

        # 2. Train the online network.
        qnet.train()
        last_loss = 0.0
        for _ in range(config.TRAIN_STEPS_PER_ITER):
            batch = buffer.sample(config.BATCH_SIZE, rng_py)
            last_loss = train_step(qnet, target_qnet, optimizer, game, batch)
        qnet.eval()

        # 3. Sync the target network.
        if iteration % config.TARGET_SYNC_EVERY == 0:
            target_qnet.load_state_dict(qnet.state_dict())

        # 4. Exhaustive evaluation; snapshot the closest net for each difficulty.
        blunders = count_losing_blunders(game, qnet, decision_states)
        saved = []
        for difficulty, target in targets.items():
            distance = abs(blunders - target)
            if snapshots[difficulty] is None or distance < snapshots[difficulty][0]:
                snapshots[difficulty] = (distance, blunders)
                torch.save(qnet.state_dict(),
                           os.path.join(CHECKPOINT_DIR, f"{difficulty}.pt"))
                saved.append(difficulty)
        marker = ("  <- " + ", ".join(saved)) if saved else ""
        print(f"iter {iteration:2d}  eps={epsilon:.2f}  loss={last_loss:.4f}  "
              f"losing-blunders={blunders}{marker}")

    print("done - difficulty snapshots:")
    for difficulty, target in targets.items():
        _, blunders_at = snapshots[difficulty]
        print(f"  {difficulty:7s} (target {target:4d})  "
              f"saved at {blunders_at} losing blunders  -> {difficulty}.pt")


if __name__ == "__main__":
    main()
