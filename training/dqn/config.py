"""Configuration for the DQN training pipeline.

Separate from the AlphaZero config (training/config.py) — each algorithm owns
its own hyperparameters.
"""

# -- Reproducibility -------------------------------------------------------
SEED = 1234

# -- Network ---------------------------------------------------------------
HIDDEN_SIZE = 128
TRUNK_LAYERS = 2          # number of Linear+ReLU layers before the Q-head

# -- Q-learning ------------------------------------------------------------
GAMMA = 1.0               # discount; 1.0 is fine — tic-tac-toe games are short

# -- Exploration (epsilon-greedy) ------------------------------------------
EPSILON_START = 1.0       # fully random at the start of training
EPSILON_END = 0.1         # floor — keep some exploration throughout
EPSILON_DECAY_ITERS = 45  # iterations over which epsilon decays to the floor

# -- Training --------------------------------------------------------------
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4       # L2 regularization, applied via the optimizer
BATCH_SIZE = 256
TRAIN_STEPS_PER_ITER = 200
TARGET_SYNC_EVERY = 2     # copy the online net into the target net every N iters

# -- Replay buffer ---------------------------------------------------------
BUFFER_SIZE = 100000

# -- Training loop ---------------------------------------------------------
NUM_ITERATIONS = 60
GAMES_PER_ITER = 200      # self-play games are cheap — no search

# -- Coverage --------------------------------------------------------------
# A fraction of self-play games start from a random position (a short random
# walk from the empty board) so off-principal positions also get trained.
RANDOM_START_FRACTION = 0.6
RANDOM_START_MAX_PLIES = 6
