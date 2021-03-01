"""

@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from theconf import Config as C

def adjust_learning_rate_resnet(optimizer):
    epoch = C.get()['epoch']

    if epoch == 90:
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 80])
    elif epoch == 270:
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, [90, 180, 240])
    else:
        raise ValueError('Invalid epoch={} for resnet scheduler.'.format(epoch))
