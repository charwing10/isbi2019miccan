#!/usr/bin/env python
"""main function for reconstruction"""

__author__ = "Qiaoying Huang"
__date__ = "04/08/2019"
__institute__ = "Rutgers University"


import torch
import torch.nn as nn
from torchvision import models


class Percetual(nn.Module):
    def __init__(self):
        super(Percetual, self).__init__()
        self.select = ['3', '6', '8', '11']
        self.vgg = models.vgg16(pretrained=True).features.cuda().eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, output, target):
        l1loss = self.l1(output, target)
        threeoutput = torch.cat((output, output, output), 1)
        threetarget = torch.cat((target, target, target), 1)

        features = []
        truefeatures = []
        for name, layer in self.vgg._modules.items():
            threeoutput = layer(threeoutput)
            threetarget = layer(threetarget)
            if name in self.select:
                features.append(threeoutput)
                truefeatures.append(threetarget)

        perloss = 0
        weights = [0.5, 0.5, 0.5, 0.5]
        for i in range(len(features)):
            perloss += weights[i]*self.l2(features[i], truefeatures[i])
        loss = perloss + 10*l1loss
        return loss
