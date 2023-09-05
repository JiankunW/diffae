from templates import *
from templates_latent import *

if __name__ == '__main__':
    # train the autoenc moodel
    # 4 gpus by default
    gpus = [0, 1, 2, 3]
    conf = ffhq128_autoenc_130M_pdae_encoder()
    train(conf, gpus=gpus)
