
BATCH_SIZE = 256
EPOCHS = 20
# Number of samples to take from whole set (1_342_846_339)
N_SAMPLES = 5_120
SHUFFLE_BUFFER_SIZE = 5000

REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
NR_RESIDUAL_LAYERS = 10     # TODO change, was 40 moved down for testing on low memory
KERNEL_SIZE_CONVOLUTION = 3
NR_CONV_FILTERS = 256
NR_CONV_FILTERS_POLICY_HEAD = 8
NR_CONV_FILTERS_VALUE_HEAD = 1
SIZE_VALUE_HEAD_HIDDEN = 256
TEST_MODE = False
RESTORE_CHECKPOINT = False
# Set to "" if not in google colab
GDRIVE_FOLDER = ""  # "drive/My Drive/KI_Praktikum/pretraining/"
INPUT_SHAPE = (34, 8, 8)
INPUT_SHAPE_CHANNELS_LAST = (8, 8, 34)

print("\n\n----------------------------------------")
print("Training Configurations:")
print("Epochs: ", EPOCHS)
print("Nr Residual Layers: ", NR_RESIDUAL_LAYERS)
print("Restore checkpoint: ", RESTORE_CHECKPOINT)
print("Test mode: ", TEST_MODE)
print("Num samples from database: ", N_SAMPLES)
print("----------------------------------------\n\n")
