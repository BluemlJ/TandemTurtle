
BATCH_SIZE = 256
EPOCHS = 20
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

print("\n\n----------------------------------------")
print("Training Configurations:")
print("Epochs: ", EPOCHS)
print("Nr Residual Layers: ", NR_RESIDUAL_LAYERS)
print("Restore checkpoint: ", RESTORE_CHECKPOINT)
print("Test mode: ", TEST_MODE)
print("----------------------------------------\n\n")
