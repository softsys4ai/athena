"""
Implement measures for estimating model's effectiveness.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import copy

import torch
import numpy as np
from collections import defaultdict

from torch import nn


def accuracy(y_pred, y_true, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = y_true.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_true.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1. / batch_size))
    return res


def error_rate(y_pred, y_true, correct_on_bs=None):
    '''
    Compute the error rate
    :param y_pred: predictions
    :param y_true: ground truth
    :param correct_on_bs: indices of corresponding benign samples that
            are correctly classified by the undefended model.
    :return: the error rate
    '''
    if len(y_pred.shape) > 1:
        y_pred = np.asarray([np.argmax(p) for p in y_pred])

    if len(y_true.shape) > 1:
        y_true = np.asarray([np.argmax(p) for p in y_true])

    amount = y_pred.shape[0] if correct_on_bs is None else len(correct_on_bs)

    # Count the number of inputs which successfully fool the model.
    # that is f(x') != f(x).
    if correct_on_bs is not None:
        num_fooled = np.sum([1. for i in range(amount) if (i in correct_on_bs) and (y_pred[i] != y_true[i])])
    else:
        num_fooled = np.sum([1. for i in range(amount) if (y_pred[i] != y_true[i])])

    score = float(num_fooled / amount)
    return score


def get_corrections(y_pred, y_true):
    """
    Collect the indices of the images that are miscalssified.
    :param y_pred: predictions.
    :param y_true: ground truth
    :return:
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_pred.shape) > 1:
        y_pred = np.asarray([np.argmax(p) for p in y_pred])
    if len(y_true.shape) > 1:
        y_true = np.asarray([np.argmax(p) for p in y_true])

    corrections = [i for i in range(y_true.shape[0]) if y_pred[i]==y_true[i]]

    return corrections


def cross_entropy_smooth(input, target, size_average=True, label_smoothing=0.1):
    y = torch.eye(10).cuda()
    lb_oh = y[target]

    target = lb_oh * (1 - label_smoothing) + 0.5 * label_smoothing

    logsoftmax = nn.LogSoftmax()
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


class Accumulator(object):
    def __init__(self):
        self.metrics = defaultdict(lambda: 0.)

    def add(self, key, value):
        self.metrics[key] += value

    def add_dict(self, dict):
        for key, value in dict.items():
            self.add(key, value)

    def __getitem__(self, item):
        return self.metrics[item]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def get_dict(self):
        return copy.deepcopy(dict(self.metrics))

    def items(self):
        return self.metrics.items()

    def __str__(self):
        return str(dict(self.metrics))

    def __truediv__(self, other):
        newone = Accumulator()
        for key, value in self.items():
            if isinstance(other, str):
                if other != key:
                    newone[key] = value / self[other]
                else:
                    newone[key] = value
            else:
                newone[key] = value / other
        return newone


class SummaryWriterDummy(object):
    def __init__(self, log_dir):
        pass

    def add_scalar(self, *args, **kwargs):
        pass