"""
Utilities for data manipulations.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import random
import time

import keras
import numpy as np
from keras.datasets import mnist as MNIST
import torch
from torch.utils.data import DataLoader, TensorDataset

random.seed(1000)


def load_mnist():
    """
    Dataset of 60,000 28x28 grayscale images of the 10 digits,
    along with a test set of 10,000 images.
    """
    (X_train, Y_train), (X_test, Y_test) = MNIST.load_data()
    _, img_rows, img_cols = X_test.shape
    nb_channels = 1
    nb_classes = 10

    X_train = X_train.reshape(-1, img_rows, img_cols, nb_channels)
    X_test = X_test.reshape(-1, img_rows, img_cols, nb_channels)

    """
    cast pixels to floats, normalize to [0, 1] range
    """
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.

    """
    one-hot-encode the labels
    """
    Y_train = keras.utils.to_categorical(Y_train, nb_classes)
    Y_test = keras.utils.to_categorical(Y_test, nb_classes)

    """
    summarize data set
    """
    print('Dataset({}) Summary:'.format("MNIST"))
    print('Train set: {}, {}'.format(X_train.shape, Y_train.shape))
    print('Test set: {}, {}'.format(X_test.shape, Y_test.shape))
    return (X_train, Y_train), (X_test, Y_test)


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
    # print("[DEBUG][data.set_channels_first]")
    if channels_last(data):
        if len(data.shape) == 4:
            data = np.transpose(data, (0, 3, 1, 2))
        else:
            data = np.transpose(data, (2, 0, 1))
    return data


def set_channels_last(data):
    # print("[DEBUG][data.set_channels_last]")
    if channels_first(data):
        if len(data.shape) == 4:
            data = np.transpose(data, (0, 2, 3, 1))
        else:
            data = np.transpose(data, (1, 2, 0))
    return data


def get_dataloader(data, labels, batch_size=128, shuffle=False, **kwargs):
    dataset = TensorDataset(torch.Tensor(data), torch.LongTensor(labels))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    return dataloader


def subsampling(data, labels, predictions=None,
                num_classes=10, ratio=0.1,
                filepath=None, filename=None):
    """
    Subsampling dataset.
    :param data: numpy array. the population dataset to sample from.
    :param labels: numpy array. the corresponding true labels of the population dataset.
    :param num_classes: integer. the number of classes in the dataset.
    :param ratio: float. the ratio to sample.
    :param output: string or path. the path to save subsampled data and labels.
    :return:
    """
    if data is None or labels is None:
        raise ValueError("`data` and `labels` cannot be None.")

    if num_classes is None or num_classes <= 0:
        raise ValueError("`num_classes` must be a positive number, but found {}.".format(num_classes))

    if ratio <= 0 or ratio > 1:
        raise ValueError("Expect a ratio greater than `0` and no more `1`, but found {}.".format(ratio))

    # prepare sampling
    pool_size = data.shape[0]
    class_size = int(pool_size / num_classes)
    num_samples_per_class = int(class_size * ratio)

    # convert to labels
    if len(labels.shape) > 1:
        labels = [np.argmax(p) for p in labels]

    class_ids = [i for i in range(num_classes)]
    if num_samples_per_class <= 0:
        class_ids = random.sample(population=class_ids, k=int(pool_size*ratio))
        num_samples_per_class = 1

    # sample equal number of samples from each class
    sample_ids = []
    for c_id in class_ids:
        if predictions:
            # get all the samples belong to the c_id-th class and
            # the sample is correctly predicted
            print('sampling only the correctly classified samples.')
            ids = [i for i in range(pool_size) if labels[i] == c_id and predictions[i] == c_id]
        else:
            # do not care about the predictions
            # get all the samples belong to the c_id-th class
            ids = [i for i in range(pool_size) if labels[i] == c_id]

        selected = random.sample(population=ids, k=num_samples_per_class)
        print(">>> Draw {} samples from the {}-th class.".format(len(selected), c_id))
        sample_ids.extend(selected)

    print(">>> Drawn {} random samples.".format(len(sample_ids)))

    # shuffle the selected sample ids
    random.shuffle(sample_ids)
    # get sampled data and labels
    subsamples = np.asarray([data[i] for i in sample_ids])
    sublabels = np.asarray([labels[i] for i in sample_ids])

    if filepath is not None:
        # save the subsamples
        rand_idx = time.monotonic()
        # -cc stands for correctly classified
        file = os.path.join(filepath, 'subsamples-{}-ratio_{}-{}-cc.npy'.format(filename, ratio, rand_idx))
        np.save(file=file, arr=subsamples)
        file = os.path.join(filepath, 'sublabels-{}-ratio_{}-{}-cc.npy'.format(filename, ratio, rand_idx))
        np.save(file=file, arr=sublabels)

    return subsamples, sublabels
