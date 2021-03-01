"""

@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pickle
from art.classifiers import SklearnClassifier, PyTorchClassifier, KerasClassifier

from utils import file
from utils.config import *
from utils.logger import get_logger

target_model_types = [
    'sklearnclassifier',
    'pytorchclassifier',
]

def wrap(model, target_type, **config):
    assert model is not None

    if target_type == 'sklearnclassifier':
        nb_classes = config.get('nb_classes', 10)

        return SklearnClassifier(model=model, clip_values=(0, nb_classes))

    elif target_type == 'pytorchclassifier':
        loss = config.get('loss', None)
        optimizer = config.get('optimizer', None)
        input_shape = config.get('input_shape', None)
        nb_classes = config.get('nb_classes', None)
        clip_values = config.get('clip_values', (0., 1.))

        return PyTorchClassifier(model=model, loss=loss, optimizer=optimizer, input_shape=input_shape,
                                 nb_classes=nb_classes, clip_values=clip_values)

    else:
        raise NotImplementedError('Wrapper for {} is not implemented.'.format(target_type))


def save(model, filename):
    assert model is not None
    assert filename is not None

    print('Saving model to [{}].'.format(filename))
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def load(filename):
    assert filename is not None

    model = None
    print('Loading model from [{}].'.format(filename))
    with open(filename, 'rb') as file:
        model = pickle.load(file)

    return model


