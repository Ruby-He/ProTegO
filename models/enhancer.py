import torch.nn as nn
from .enhancement_layers.combined import Combined
from .enhancement_layers.identity import Identity
from .enhancement_layers.transform import Resize, Translate, D_Binarization, R_Binarization

class Enhancer(nn.Module):
    """
    This module allows to combine different enhancement layers into a sequential module.
    """
    def __init__(self, layers):
        super(Enhancer, self).__init__()
        for i in range(len(layers)):
            layers[i] = eval(layers[i])
        self.enhance = nn.Sequential(*layers)

    def forward(self, adv_image):
        enhance_adv = self.enhance(adv_image)
        return enhance_adv
