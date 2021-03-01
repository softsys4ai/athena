"""
Adapted from https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import torch

from torch.nn import DataParallel
import torch.backends.cudnn as cudnn

from models.networks.wideresnet import WideResNet


def get_model(model_type, num_class=10, data_parallel=False, device=None):
    if model_type == 'wresnet40_2':
        model = WideResNet(40, 2, dropout_rate=0.0, num_classes=num_class)
    elif model_type == 'wresnet28_10':
        model = WideResNet(28, 10, dropout_rate=0.0, num_classes=num_class)
    else:
        raise NameError('No such model name: {}.'.format(model_type))

    # data_parallel = False
    if data_parallel:
        model = model.cuda()
        model = DataParallel(model)
    else:
        # import horovod.torch as hvd
        # device = torch.device('cuda', hvd.local_rank())
        model = model.to(device if device else torch.device('cpu'))
    cudnn.benchmark = True
    return model


def num_class(dataset):
    return {
        'mnist': 10,
        'cifar10': 10,
        'reduced_cifar10': 10,
        'cifar10.1': 10,
        'cifar100': 100,
        'svhn': 10,
        'reduced_svhn': 10,
        'imagenet': 1000,
        'reduced_imagenet': 120,
    }[dataset]