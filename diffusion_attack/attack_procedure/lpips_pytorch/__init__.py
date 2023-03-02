from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from .models import dist_model


class PerceptualLoss(torch.nn.Module):
    def __init__(self, model='net-lin', net='vgg',
                 use_gpu=True):  # VGG using our perceptually-learned weights (LPIPS metric)
        print('Setting up Perceptual loss...')
        self.model = dist_model.DistModel()
        self.model.initialize(model=model, net=net, use_gpu=True)
        print('...Done')

    def forward(self, pred, target, normalize=False, grayscale=False):
        """
        Pred and target are Variables.
        If normalize is on, assumes the images are between [0,1] and then scales thembetween [-1, 1]
        If normalize is false, assumes the images are already between [-1,+1]

        Inputs pred and target are Nx3xHxW
        Output pytorch Variable N long
        """
        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1

        # Idea from https://github.com/ad12/meddlr/blob/a500dd2e145d14ce9a42ec43fc634c15610b1dc0/meddlr/metrics/lpip.py
        if grayscale:
            target = target.reshape(target.shape[0] * target.shape[1], 1, target.shape[2], target.shape[3])
            target = target.repeat(1, 3, 1, 1)
            pred = pred.reshape(pred.shape[0] * pred.shape[1], 1, pred.shape[2], pred.shape[3])
            pred = pred.repeat(1, 3, 1, 1)

        dist = self.model.forward_pair(target, pred)

        return dist
