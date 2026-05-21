"""Orchestrate AlphaZero training: self-play -> train -> arena gate -> repeat.

Run from anywhere:
    python training/scripts/run_training.py
Optional smoke-test overrides:
    python training/scripts/run_training.py --iterations 3 --games-per-iter 15

The best network is checkpointed to training/checkpoints/best.pt; export it to
the app with scripts/export_weights.py.
"""

import argparse
import copy
import os
import random
import sys

# Make training/ importable (config, alphazero, games) when run as a script.
TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING_DIR)

import numpy as np
import torch

import config
from alphazero import arena, mcts
from alphazero.network import PolicyValueNet
from alphazero.replay_buffer import ReplayBuffer
from alphazero.selfplay import play_game, random_start
from alphazero.train import train_step
from games import minimax_ref
from games.tictactoe import TicTacToe

CHECKPOINT_DIR = os.path.join(TRAINING_DIR, "checkpoints")


def count_minimax_losses(game, net):
    """Play the net vs perfect minimax as both first and second player.

    Returns how many of the 2 games the net loses (0 == flawless play).
    """
    losses = 0
    for net_first in (True, False):
        state = game.initial_state()
        net_to_move = net_first
        while not game.is_terminal(state):
            if net_to_move:
                root = mcts.run_mcts(game, net, state, config.NUM_SIMULATIONS,
                                     add_noise=False)
                action = mcts.best_action(root)
            else:
                action = minimax_ref.best_action(game, state)
            state = game.apply_action(state, action)
            net_to_move = not net_to_move
        value = game.outcome(state)
        net_value = value if net_to_move else -value
        if net_value < 0:
            losses += 1
    return losses


def main():
    parser = argparse.ArgumentParser(description="AlphaZero tic-tac-toe training")
    parser.add_argument("--iterations", type=int, default=config.NUM_ITERATIONS)
    parser.add_argument("--games-per-iter", type=int, default=config.GAMES_PER_ITER)
    args = parser.parse_args()

    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    rng_np = np.random.default_rng(config.SEED)
    rng_py = random.Random(config.SEED)

    game = TicTacToe()
    best_net = PolicyValueNet(game.input_size, game.action_size)
    best_net.eval()
    buffer = ReplayBuffer()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_path = os.path.join(CHECKPOINT_DIR, "best.pt")
    torch.save(best_net.state_dict(), best_path)

    print(f"training: {args.iterations} iterations x {args.games_per_iter} games")
    for iteration in range(1, args.iterations + 1):
        # 1. Self-play with the current best net. Most games start from a
        #    random position so off-principal states also get training signal.
        for _ in range(args.games_per_iter):
            if rng_np.random() < config.RANDOM_START_FRACTION:
                start = random_start(game, rng_np, config.RANDOM_START_MAX_PLIES)
            else:
                start = None
            buffer.add_game(play_game(game, best_net, rng_np, start_state=start))

        # 2. Train a challenger (a copy of the best net) on the buffer.
        challenger = copy.deepcopy(best_net)
        optimizer = torch.optim.Adam(
            challenger.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        challenger.train()
        last_loss = (0.0, 0.0, 0.0)
        for _ in range(config.TRAIN_STEPS_PER_ITER):
            batch = buffer.sample_batch(config.BATCH_SIZE, rng_py)
            last_loss = train_step(challenger, optimizer, *batch)
        challenger.eval()

        # 3. Arena gate: promote only if clearly better than the current best.
        score, w, d, l = arena.compare(game, challenger, best_net, rng_np)
        promoted = score >= config.ARENA_WIN_RATE
        if promoted:
            best_net = challenger
            torch.save(best_net.state_dict(), best_path)

        mm_losses = count_minimax_losses(game, best_net)
        print(
            f"iter {iteration:2d}  loss={last_loss[0]:.3f}  "
            f"arena={score:.2f} (W{w}/D{d}/L{l})  "
            f"{'PROMOTED' if promoted else 'kept    '}  "
            f"minimax-losses={mm_losses}/2"
        )

    print(f"done - best net saved to {best_path}")


if __name__ == "__main__":
    main()
