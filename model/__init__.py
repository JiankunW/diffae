from typing import Union
from .unet import BeatGANsUNetModel, BeatGANsUNetConfig
from .unet_autoenc import BeatGANsAutoencConfig, BeatGANsAutoencModel
from .pdae_encoder import FFHQ128Encoder, CELEBA64Encoder

Model = Union[BeatGANsUNetModel, BeatGANsAutoencModel]
ModelConfig = Union[BeatGANsUNetConfig, BeatGANsAutoencConfig]
