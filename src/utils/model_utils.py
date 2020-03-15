"""

@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import os
import pickle

from utils import file
from utils.config import *
from utils.logger import get_logger

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


