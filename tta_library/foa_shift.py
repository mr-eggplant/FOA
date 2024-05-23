"""
Copyright to FOA Authors ICML 2024
"""

from argparse import ArgumentDefaultsHelpFormatter
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

from torch.autograd import Variable
from models.vpt import PromptViT
import cma
import numpy as np
import time
import math

from utils.cli_utils import accuracy, AverageMeter
from calibration_library.metrics import ECELoss
from queue import PriorityQueue
from quant_library.quant_layers.matmul import *
from timm.models.vision_transformer import VisionTransformer

class Shift(nn.Module):
    """back-to-source activation shifting in FOA
    It directly tunes the activations for feature alignment, as efficient as standard forward.
    """
    def __init__(self, model:PromptViT):
        super().__init__()

        self.model = model
        self.hist_stat = None

    def _update_hist(self, batch_mean):
        if self.hist_stat is None:
            self.hist_stat = batch_mean
        else:
            self.hist_stat = 0.9 * self.hist_stat + 0.1 * batch_mean
            
    def _get_shift_vector(self):
        if self.hist_stat is None:
            return None
        else:
            return self.train_info[1][-768:] - self.hist_stat

    def forward(self, x):
        shift_vector = self._get_shift_vector()
        outputs, batch_mean = forward_with_shift(x, self.model, shift_vector)
        self._update_hist(batch_mean[-768:])
        return outputs
    
    def obtain_origin_stat(self, train_loader):
        print('===> begin calculating mean and variance')
        features = []
        with torch.no_grad():
            for _, dl in enumerate(train_loader):
                images = dl[0].cuda()
                feature = self.model.layers_cls_features(images)
                features.append(feature)
            features = torch.cat(features, dim=0)
            self.train_info = torch.std_mean(features, dim=0)
        print('===> calculating mean and variance end')

    def reset(self):
        self.hist_stat = None

@torch.no_grad()
def forward_with_shift(images, model:PromptViT, shift_vector):
    features = model.layers_cls_features_with_prompts(images)
    _, batch_mean = torch.std_mean(features, dim=0)

    cls_features = features[:, -768:]
    if shift_vector is not None:
        cls_features += 1. * shift_vector

    output = model.vit.head(cls_features)
    return output, batch_mean
