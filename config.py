
# This is where you set the key parameters that influence the algorithm.

# f.e

# SELF PLAY
EPISODES = 30
MCTS_SIMS = 100
MEMORY_SIZE = 30000
TURNS_UNTIL_TAU0 = 10  # turn on which it starts playing deterministically
CPUCT = 1

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
run_archive_folder = './run_archive/'
