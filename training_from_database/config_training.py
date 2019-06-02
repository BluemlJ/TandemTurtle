
BATCH_SIZE = 256
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 10
INPUT_DIM = [34,8,8]  # TODO: adjust to real input dim
OUTPUT_DIM_VALUE_HEAD = 1
NR_RESIDUAL_LAYERS = 40
KERNEL_SIZE_CONVOLUTION = 3
NR_CONV_FILTERS = 256
SIZE_VALUE_HEAD_HIDDEN = 256

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