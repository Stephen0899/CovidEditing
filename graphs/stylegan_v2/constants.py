BATCH_SIZE = 4
DIM_Z = 512  # used only for setting default arg value
resolution = 256
# resolution = 128
useGPU = True
NUM_CHANNELS = 3

#Xray
# REG
reg_json = '/home/diwenxu2/logdir_ori/cfg.json'
reg_path = '/home/diwenxu2/logdir_ori/best.ckpt'
g_path = '/home/diwenxu2/stylegan2-pytorch/checkpoint/640000.pt'


# REG-Scene
# reg_json = None
# reg_path = '/home/peiye/ImageEditing/scene_regressor/checkpoint_256/500_dict.model'

# StyleGAN - scene
# g_path = '/home/peiye/ImageEditing/stylegan2-pytorch/checkpoint/190000.pt'

# REG-ffhq
# g_path = '/shared/rsaas/zpy/2nd_year/stylegan2_celeba/pretrained_ffhq/550000.pt'


# Not used
# CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
# LAMBDA = 10 # Gradient penalty lambda hyperparameter
# ITERS = 200000 # How many generator iterations to train for
