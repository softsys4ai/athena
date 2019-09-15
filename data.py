"""
Implement operations related to data (dataset).
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import keras
import tensorflow as tf
from keras.datasets import mnist, cifar10, fashion_mnist, cifar100

from config import DATA, MODE

"""
set random seed for replication
"""
np.random.seed(1000)
tf.set_random_seed(1000)

def load_data(dataset):
    #assert dataset in DATA.get_supported_datasets()
    X_train = None
    Y_train = None
    X_test = None
    Y_test = None
    img_rows = 0
    img_cols = 0
    nb_channels = 0
    nb_classes = 0

    if DATA.mnist == dataset:
        """
        Dataset of 60,000 28x28 grayscale images of the 10 digits,
        along with a test set of 10,000 images.
        """
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        if MODE.DEBUG:
            print('shapes:\ntrain - {}, {}'.format(X_train.shape, Y_train.shape))
            print('test - {}, {}'.format(X_test.shape, Y_test.shape))
        nb_examples, img_rows, img_cols = X_test.shape
        nb_channels = 1
        nb_classes = 10
    elif DATA.fation_mnist == dataset:
        """
        Dataset of 60,000 28x28 grayscale images of 10 fashion categories,
        along with a test set of 10,000 images. The class labels are:
        Label   Description
        0       T-shirt/top
        1       Trouser
        2       Pullover
        3       Dress
        4       Coat
        5       Sandal
        6       Shirt
        7       Sneaker
        8       Bag
        9       Ankle boot
        """
        (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
        if MODE.DEBUG:
            print('shapes:\ntrain - {}, {}'.format(X_train.shape, Y_train.shape))
            print('test - {}, {}'.format(X_test.shape, Y_test.shape))
        nb_examples, img_rows, img_cols = X_test.shape
        nb_channels = 1
        nb_classes = 10
    elif DATA.cifar_10 == dataset:
        """
        Dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images.
        """
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        if MODE.DEBUG:
            print('shapes:\ntrain - {}, {}'.format(X_train.shape, Y_train.shape))
            print('test - {}, {}'.format(X_test.shape, Y_test.shape))
        nb_examples, img_rows, img_cols, nb_channels = X_test.shape
        nb_classes = 10
    elif DATA.cifar_100 == dataset:
        (X_train, Y_train), (X_test, Y_test) = cifar100.load_data()
        if MODE.DEBUG:
            print('shapes:\ntrain - {}, {}'.format(X_train.shape, Y_train.shape))
            print('test - {}, {}'.format(X_test.shape, Y_test.shape))
        nb_examples, img_rows, img_cols, nb_channels = X_test.shape
        nb_classes = 100

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
    mean-std normalization
    """
    # if (DATA.cifar_10 == dataset):
    #     X_train = normalize(X_train)
    #     X_test = normalize(X_test)
    """
    one-hot-encode the labels
    """
    Y_train = keras.utils.to_categorical(Y_train, nb_classes)
    Y_test = keras.utils.to_categorical(Y_test, nb_classes)

    """
    summarize data set
    """
    print('Dataset({}) Summary:'.format(dataset.upper()))
    print('Train set: {}, {}'.format(X_train.shape, Y_train.shape))
    print('Test set: {}, {}'.format(X_test.shape, Y_test.shape))
    return (X_train, Y_train), (X_test, Y_test)

def normalize(X):
    """
    Normalize the given dataset X.
    :param X:
    :return: normalized dataset.
    """
    # z-score
    mean = np.mean(X, axis=(0, 1, 2, 3))
    std = np.std(X, axis=(0, 1, 2, 3))
    # avoid dividing zero by adding a very small number
    X = (X - mean) / (std + 1e-7)

    return X

"""
for testing
"""
def main(args):
    load_data(args)

if __name__ == "__main__":
    main(DATA.cifar_10)