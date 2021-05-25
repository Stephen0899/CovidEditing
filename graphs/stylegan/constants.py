BATCH_SIZE = 8

DIM_Z = 512  # used only for setting default arg value
# resolution = 256
resolution = 128
useGPU = True
NUM_CHANNELS = 3

# Not using
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for
