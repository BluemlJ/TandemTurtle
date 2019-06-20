
# This is where you set the key parameters that influence the algorithm.

# Initialise
INITIAL_RUN_NUMBER = None
INITIAL_MODEL_VERSION = None
INITIAL_MEMORY_VERSION = None
INITIAL_MODEL_PATH = "/run/models/simple_model_v1"

# SELF PLAY
EPISODES = 30
MCTS_SIMS = 100
MEMORY_SIZE = 30000
TURNS_UNTIL_TAU0 = 10  # turn when the agent starts playing with less noise (less exploration)
CPUCT = 1.41

"""

#### RETRAINING
BATCH_SIZE = 256
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 10

HIDDEN_CNN_LAYERS = [
	{'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	]

#### EVALUATION
EVAL_EPISODES = 20
SCORING_THRESHOLD = 1.3

"""

run_folder = './run/'
run_archive_folder = './run/archive/'