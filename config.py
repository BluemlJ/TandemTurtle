
# This is where you set the key parameters that influence the algorithm.

# Initialise
INITIAL_RUN_NUMBER = None
INITIAL_MODEL_VERSION = None
INITIAL_MEMORY_VERSION = None
INITIAL_MODEL_PATH = "/run/models/15M"


# Main / Gamemode Options
GAME_AGENT_THREADS = 4  # 0 for selfplay, 4 for playing against itself, 2 for sjeng, 1 for single-player
SERVER_AUTOSTART = 0
SERVER_ADDRESS = "ws://localhost:8080/websocketclient"
GAMEID = "gameid"
TOURNAMENTID = "tournamentid"

# MCTS
RUN_ON_NN_ONLY = False

# disable logging
LOGGER_DISABLED = {
    'main': False, 'memory': False, 'tourney': False, 'mcts': False, 'model': False}

# Random Agent Sleep in seconds
DELAY_FOR_RANDOM = 3

# SELF PLAY
EPISODES = 30
MCTS_SIMS = 25
PARALLEL_READOUTS = 8  # Number of searches to execute in parallel. This is also the batch size for neural network evaluation
MEMORY_SIZE = 30000
TURNS_WITH_HIGH_NOISE = 10  # turn when the agent starts playing with less noise (less exploration)
CPUCT = 1.41
CPUCT_BASE = 19652  # Exploration constants balancing priors vs. value net output

DIRICHLET_ALPHA = 0.03
DIRICHLET_WEIGHT = 0.25  # 'How much to weight the priors vs. dirichlet noise when mixing'

# RETRAINING
BATCH_SIZE = 256
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 10

# EVALUATION
EVAL_EPISODES = 20
SCORING_THRESHOLD = 1.3

run_folder = './run/'
run_archive_folder = './run/archive/'
