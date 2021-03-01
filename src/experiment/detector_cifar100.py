"""
Implement an adversarial detector on top of IBM Trusted-AI ART (1.2.0)
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
sys.path.append("../")

import itertools
import logging
import math
from collections import OrderedDict
import os
import argparse
import time
import json
import torch
from torch import nn, optim
from torch.nn.parallel.data_parallel import DataParallel
import keras
import numpy as np

from art.attacks import FastGradientMethod, ProjectedGradientDescent
from art.data_generators import KerasDataGenerator
from art.classifiers import KerasClassifier
from art.detection import BinaryInputDetector

from models.cifar100_utils import load_model, load_pool
from models.networks import get_model, num_class
from utils.file import load_from_json
from models.utils.estimator import error_rate
from attacks.attacker_art import generate
from models.athena import Ensemble, ENSEMBLE_STRATEGY


def train_detector(model_configs):
    # step 1. loading prereqs and data
    # load data
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar100.load_data()
    print(X_train.shape, np.max(X_train))
    X_train /= 255.
    X_test /= 255.

    num_sample_train = 5000 #X_train.shape[0]
    num_sample_test = X_test.shape[0]
    X_train = X_train[:num_sample_train]
    Y_train = Y_train[:num_sample_train]
    X_test = X_test[:num_sample_test]
    Y_test = Y_test[:num_sample_test]

    class_descr = [i for i in range(10)]

    # load the underlying model
    network_configs = model_configs.get("network_configs")
    model = get_model(
        model_type=network_configs.get("model_type"),
        num_class=2,
        data_parallel=torch.cuda.is_available(),
        device="cuda:0"
    )

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=network_configs.get("lr"),
        momentum=network_configs.get("optimizer_momentum"),
        weight_decay=network_configs.get("optimizer_decay"),
        nesterov=network_configs.get("optimizer_nesterov")
    )

    is_master = True
    lr_scheduler_type = network_configs.get()['lr_schedule'].get('type', 'cosine')
    if lr_scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=model_configs.get()['epoch'], eta_min=0.)
    elif lr_scheduler_type == 'resnet':
        scheduler = adjust_learning_rate_resnet(optimizer)
    else:
        raise ValueError('invalid lr_schduler={}'.format(lr_scheduler_type))

    if C.get()['lr_schedule'].get('warmup', None):
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=C.get()['lr_schedule']['warmup']['multiplier'],
            total_epoch=C.get()['lr_schedule']['warmup']['epoch'],
            after_scheduler=scheduler
        )

    if not tag or not is_master:
        from models.utils.estimator import SummaryWriterDummy as SummaryWriter
    else:
        from tensorboardX import SummaryWriter
    writers = [SummaryWriter(log_dir='./logs/{}/{}'.format(tag, x)) for x in ['train', 'valid', 'test']]

    result = OrderedDict()
    epoch_start = 1
    if save_path and os.path.exists(save_path):
        logger.info('Found file [{}]. Loading...'.format(save_path))
        data = torch.load(save_path)
        if 'model' in data or 'state_dict' in data:
            key = 'model' if 'model' in data else 'state_dict'
            logger.info('checkpoint epoch@{}'.format(data['epoch']))
            if not isinstance(model, DataParallel):
                model.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
            else:
                model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
            optimizer.load_state_dict(data['optimizer'])
            if data['epoch'] < C.get()['epoch']:
                epoch_start = data['epoch']
            else:
                only_eval = True
        else:
            model.load_state_dict({k: v for k, v in data.items()})
        del data
    else:
        logger.info('[{}] file not found. Skip to pretrain weights...'.format(save_path))
        if only_eval:
            logger.warning('model checkpoint not found. only-evaluation mode is off.')
        only_eval = False

    if only_eval:
        logger.info('evaluation only+')
        model.eval()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, criterion, None, desc_default='train', epoch=0, writer=writers[0])
        rs['valid'] = run_epoch(model, validloader, criterion, None, desc_default='valid', epoch=0, writer=writers[1])
        rs['test'] = run_epoch(model, testloader_, criterion, None, desc_default='*test', epoch=0, writer=writers[2])
        for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
            if setname not in rs:
                continue
            result['{}_{}'.format(key, setname)] = rs[setname][key]
        result['epoch'] = 0
        return result

    # train loop
    best_top1 = 0
    for epoch in range(epoch_start, max_epoch + 1):
        model.train()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, criterion, optimizer, desc_default='train',
                                epoch=epoch, writer=writers[0], verbose=is_master, scheduler=scheduler)
        model.eval()

        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

        if epoch % 5 == 0 or epoch == max_epoch:
            rs['valid'] = run_epoch(model, validloader, criterion, None, desc_default='valid',
                                    epoch=epoch, writer=writers[1], verbose=is_master)
            rs['test'] = run_epoch(model, testloader_, criterion, None, desc_default='*test',
                                   epoch=epoch, writer=writers[2], verbose=is_master)

            if metric == 'last' or rs[metric]['top1'] > best_top1:
                if metric != 'last':
                    best_top1 = rs[metric]['top1']
                for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
                    result['{}_{}'.format(key, setname)] = rs[setname][key]
                result['epoch'] = epoch

                writers[1].add_scalar('valid_top1/best', rs['valid']['top1'], epoch)
                writers[2].add_scalar('test_top1/best', rs['test']['top1'], epoch)

                reporter(
                    loss_valid=rs['valid']['loss'], top1_valid=rs['valid']['top1'],
                    loss_test=rs['test']['loss'], top1_test=rs['test']['top1']
                )

                # save checkpoint
                if is_master and save_path:
                    logger.info('save model@%d to %s' % (epoch, save_path))
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'valid': rs['valid'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()
                    }, save_path)
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'valid': rs['valid'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()
                    }, save_path.replace('.pth', '_e%d_top1_%.3f_%.3f' % (epoch, rs['train']['top1'], rs['test']['top1']) + '.pth'))

    del model
    torch.cuda.empty_cache()

    result['top1_test'] = best_top1
    result['trans_cost'] = trans_cost

    return result


if __name__=="__main__":
    train_detector(None)