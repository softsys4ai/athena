"""

@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import numpy as np


def channels_last(data):
    """
    Check if the image is in the shape of (?, img_rows, img_cols, nb_channels).
    :param data:
    :return: True if channel info is at the last dimension, False otherwise.
    """
    # the images can be color images or gray-scales.
    assert data is not None

    if len(data.shape) > 4 or len(data.shape) < 3:
        raise ValueError('Incorrect dimensions of data (expected 3 or 4): {}'.format(data.shape))
    else:
        return (data.shape[-1] == 3 or data.shape[-1] == 1)


def channels_first(data):
    """
        Check if the image is in the shape of (?, nb_channels, img_rows, img_cols).
        :param data:
        :return: True if channel info is at the first dimension, False otherwise.
        """
    # the images can be color images or gray-scales.
    assert data is not None

    if len(data.shape) > 4 or len(data.shape) < 3:
        raise ValueError('Incorrect dimensions of data (expected 3 or 4): {}'.format(data.shape))
    elif len(data.shape) > 3:
        # the first dimension is the number of samples
        return (data.shape[1] == 3 or data.shape[1] == 1)
    else:
        # 3 dimensional data
        return (data.shape[0] == 3 or data.shape[0] == 1)


def set_channels_first(data):
    if channels_last(data):
        if len(data.shape) == 4:
            data = np.transpose(data, (0, 3, 1, 2))
        else:
            data = np.transpose(data, (2, 0, 1))

    return data


def set_channels_last(data):
    if channels_first(data):
        if len(data.shape) == 4:
            data = np.transpose(data, (0, 2, 3, 1))
        else:
            data = np.transpose(data, (1, 2, 0))

    return data


def rescale(data, range=(0., 1.)):
    """
    Normalize the data to range [0., 1.].
    :param data:
    :return: the normalized data.
    """
    l_bound, u_bound = range
    # normalize to (0., 1.)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    # rescale into specific range
    data = data * (u_bound - l_bound) + l_bound

    return data


def probs2labels(y):
    if len(y.shape) > 1:
        y = [np.argmax(p) for p in y]

    return y
