"""Central configuration for the AlphaZero training pipeline.

Every tunable lives here so runs are reproducible and the JS side can be kept
in sync. Values the browser also needs (c_puct, simulation count) are
re-exported into weights.json by scripts/export_weights.py.
"""

# -- Reproducibility -------------------------------------------------------
SEED = 1234

# -- Network ---------------------------------------------------------------
HIDDEN_SIZE = 128
TRUNK_LAYERS = 2          # number of Linear+ReLU layers before the heads

# -- MCTS ------------------------------------------------------------------
C_PUCT = 1.5
NUM_SIMULATIONS = 200     # simulations per move during self-play and arena
PLAY_SIMULATIONS = 800    # simulations per move for the exported/evaluated bot;
                          # a deeper search than training uses — strictly
                          # stronger play, and instant for a game this small
DIRICHLET_ALPHA = 0.3     # root-noise concentration (self-play only)
DIRICHLET_EPSILON = 0.25  # root-noise mixing weight (self-play only)

# Tie-breaking: whenever two actions score equally — in PUCT child selection
# and in argmax over visit counts — the LOWEST action index wins. This rule
# is mirrored exactly in src/engine/mcts.js so Python and JS stay in parity.

# -- Self-play -------------------------------------------------------------
TEMPERATURE_MOVES = 4     # first N plies sampled with temperature 1, then argmax
# A fraction of self-play games start from a random position (a short random
# walk from the empty board) so off-principal positions also get trained.
RANDOM_START_FRACTION = 0.6
RANDOM_START_MAX_PLIES = 6

# -- Training --------------------------------------------------------------
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4       # L2 regularization, applied via the optimizer
BATCH_SIZE = 128
TRAIN_STEPS_PER_ITER = 100

# -- Replay buffer ---------------------------------------------------------
BUFFER_SIZE = 40000

# -- Training loop ---------------------------------------------------------
NUM_ITERATIONS = 40
GAMES_PER_ITER = 80

# -- Arena (promotion gate) ------------------------------------------------
ARENA_GAMES = 40
ARENA_WIN_RATE = 0.55     # challenger must score at least this to be promoted
