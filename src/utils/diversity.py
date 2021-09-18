"""

@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

from __future__ import print_function

import numpy as np
from keras.utils import to_categorical
from scipy import linalg

def edit_distance_error(y_corrections, num_samples):
    '''
    Compute the average ensemble diversity over a dataset.

    :param y_corrections: a matrix of indices of samples that are correctly classified
        by individual base classifiers in the ensemble.
    :param num_samples: the number of test samples.
    :return: the edit-distance-error based diversity.
    '''
    num_corrections = [len(bc_corrections) for bc_corrections in y_corrections]
    min_phi = min(num_corrections)
    inter_corrections = set.intersection(*[set(bc_corrections) for bc_corrections in y_corrections])
    inter_phi = len(inter_corrections)

    diversity = (min_phi - inter_phi) / num_samples
    return diversity
