"""
A neural network classifier on CIFAR-100.
Adapted from https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/train.py
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import itertools
import json
import logging
import math
import os
from collections import OrderedDict

import torch
from theconf import Config as C, ConfigArgumentParser
from torch import nn, optim
from torch.nn.parallel.data_parallel import DataParallel
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

from data.data import get_dataloaders
from models.networks import get_model, num_class
from utils.config import *
from utils.logger import get_logger
from models.utils.lr_scheduler import adjust_learning_rate_resnet
from models.utils.estimator import accuracy, Accumulator

logger = get_logger('Athena')
logger.setLevel(logging.INFO)

def run_epoch(model, loader, loss_fn, optimizer, desc_default='', epoch=0, writer=None, verbose=1, scheduler=None):
    if verbose:
        loader = tqdm(loader, disable=False)
        loader.set_description('[{} {}/{}]'.format(desc_default, epoch, C.get()['epoch']))

    metrics = Accumulator()
    cnt = 0
    total_steps = len(loader)
    steps = 0
    for data, label in loader:
        steps += 1
        data, label = data.cuda(), label.cuda()

        if optimizer:
            optimizer.zero_grad()

        preds = model(data)
        loss = loss_fn(preds, label)

        if optimizer:
            loss.backward()
            if C.get()['optimizer'].get('clip', 5) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), C.get()['optimizer'].get('clip', 5))
            optimizer.step()

        top1, top5 = accuracy(preds, label, (1, 5))
        metrics.add_dict({
            'loss': loss.item() * len(data),
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
        })
        cnt += len(data)
        if verbose:
            postfix = metrics / cnt
            if optimizer:
                postfix['lr'] = optimizer.param_groups[0]['lr']
            loader.set_postfix(postfix)

        if scheduler is not None:
            scheduler.step(epoch - 1 + float(steps) / total_steps)

        del preds, loss, top1, top5, data, label

    metrics /= cnt
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    if verbose:
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)
    return metrics


def train_and_eval(tag, dataroot, trans_type=TRANSFORMATION.clean, test_ratio=0.0, cv_fold=0, reporter=None,
                   metric='last', save_path=None, only_eval=False):
    print('----------------------------')
    print('Augments for model training')
    print('>>> tag:', tag)
    print('>>> dataroot:', dataroot)
    print('>>> save_path:', save_path)
    print('>>> eval:', only_eval)
    print('----------------------------')

    if not reporter:
        reporter = lambda **kwargs: 0

    max_epoch = C.get()['epoch']

    start = time.monotonic()
    trainsampler, trainloader, validloader, testloader_ = get_dataloaders(C.get()['dataset'], C.get()['batch'], dataroot,
                                                                          trans_type=trans_type, split=test_ratio,
                                                                          split_idx=cv_fold)
    trans_cost = time.monotonic() - start
    print('Cost for transformation:', round(trans_cost / 60., 6))

    # create a model & an optimizer
    model = get_model(C.get()['model'], num_class(C.get()['dataset']), data_parallel=True)

    criterion = nn.CrossEntropyLoss()
    if C.get()['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=C.get()['lr'],
            momentum=C.get()['optimizer'].get('momentum', 0.9),
            weight_decay=C.get()['optimizer']['decay'],
            nesterov=C.get()['optimizer']['nesterov']
        )
    else:
        raise ValueError('Optimizer type [{}] is not yet supported, SGD is the only optimizer supported.'.format(C.get()['optimizer']['type']))

    is_master = True
    logger.debug('is_master={}'.format(is_master))

    lr_scheduler_type = C.get()['lr_schedule'].get('type', 'cosine')
    if lr_scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.get()['epoch'], eta_min=0.)
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
        logger.warning('tag not provided, no tensorboard log.')
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

def get_translist_for_usenix():
    # the following list is for Athena (USENIX Security 2020)
    trans_list = []
    trans_list.append(TRANSFORMATION.clean)
    trans_list.append(TRANSFORMATION.denoise_wavelet)
    trans_list.append(TRANSFORMATION.compress_png_compression_5)
    trans_list.append(TRANSFORMATION.flip_horizontal)
    trans_list.append(TRANSFORMATION.compress_png_compression_1)
    trans_list.append(TRANSFORMATION.compress_png_compression_8)
    trans_list.append(TRANSFORMATION.feature_std_norm)
    trans_list.append(TRANSFORMATION.morph_closing)
    trans_list.append(TRANSFORMATION.morph_opening)
    trans_list.append(TRANSFORMATION.samplewise_std_norm)
    trans_list.append(TRANSFORMATION.morph_dilation)
    trans_list.append(TRANSFORMATION.morph_erosion)
    trans_list.append(TRANSFORMATION.shift_left)
    trans_list.append(TRANSFORMATION.shift_down)
    trans_list.append(TRANSFORMATION.cartoon_mean_type3)
    trans_list.append(TRANSFORMATION.compress_jpeg_quality_80)
    trans_list.append(TRANSFORMATION.denoise_nl_fast)
    trans_list.append(TRANSFORMATION.affine_horizontal_stretch)
    trans_list.append(TRANSFORMATION.denoise_tv_bregman)
    trans_list.append(TRANSFORMATION.noise_poisson)
    trans_list.append(TRANSFORMATION.geo_swirl)
    trans_list.append(TRANSFORMATION.filter_rank)
    trans_list.append(TRANSFORMATION.filter_median)

    return trans_list

if __name__ == '__main__':
    # command:
    # python train.py -c confs/<config_file> --aug <augmentation> --dataroot=<folder stores dataset> --dataset <dataset> --save <model_file>
    # if evaluate an existing model, using --save and --only-eval
    # e.g.,
    # python train.py -c confs/wresnet28x10_cifar10_b128.yaml --aug fa_reduced_cifar10 --dataroot=data --dataset cifar100 --save cifar100_wres28x10.pth --only-eval
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--dataroot', type=str, default='/data/private/pretrainedmodels', help='torchvision data folder')
    parser.add_argument('--save', type=str, default='test.pth')
    parser.add_argument('--cv-ratio', type=float, default=0.0)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--only-eval', action='store_true')
    args = parser.parse_args()

    assert (args.only_eval and args.save) or not args.only_eval, 'checkpoint path not provided in evaluation mode.'

    if not args.only_eval:
        if args.save:
            logger.info('checkpoint will be saved at %s' % args.save)
        else:
            logger.warning('Provide --save argument to save the checkpoint. Without it, training result will not be saved!')

    translist = get_translist_for_usenix()

    import time
    print('*** args.save:', args.save)
    for trans_type in translist:
        t = time.time()
        save_path = trans_type + '_' + args.save
        print('*** save_path:', save_path)
        result = train_and_eval(args.tag, args.dataroot, trans_type=trans_type,  test_ratio=args.cv_ratio,
                                cv_fold=args.cv, save_path=save_path, only_eval=args.only_eval, metric='test')
        elapsed = time.time() - t

        logger.info('done.')
        logger.info('model: %s' % C.get()['model'])
        logger.info('augmentation: %s' % C.get()['aug'])
        logger.info('\n' + json.dumps(result, indent=4))
        logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
        logger.info('top1 error in testset: %.4f' % (1. - result['top1_test']))
        logger.info(save_path)