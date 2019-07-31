"""
Implement operations related to data (dataset).
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
from __future__ import division, absolute_import, print_function

import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils

from config import DATA

# set random seed for replication
tf.set_random_seed(1000)

def load_mnist():
    """
    Load and process training set and test set.
    :return:
    """
    dataset = DATA()
    dataset.set_dataset('mnist')
    dataset.set_img_size(28, 28)
    dataset.set_number_classes(10)

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, dataset.IMG_ROW, dataset.IMG_COL, 1)
    X_test = X_test.reshape(-1, dataset.IMG_ROW, dataset.IMG_COL, 1)

    # cast pixels to floats, normalize to [0, 1] range
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # one-hot-encode the labels
    Y_train = np_utils.to_categorical(Y_train, dataset.NB_CLASSES)
    Y_test = np_utils.to_categorical(Y_test, dataset.NB_CLASSES)

    # summarize mnist data set
    print('Dataset(MNIST) Summary:')
    print('Train set: {}, {}'.format(X_train.shape, Y_train.shape))
    print('Test set: {}, {}'.format(X_test.shape, Y_test.shape))

    return (X_train, Y_train), (X_test, Y_test)