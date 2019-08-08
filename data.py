"""
Implement operations related to data (dataset).
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
from __future__ import division, absolute_import, print_function

import tensorflow as tf
from keras.datasets import mnist, cifar10, fashion_mnist, cifar100
from keras.utils import np_utils

from config import DATA

# set random seed for replication
tf.set_random_seed(1000)

def load_mnist():
    """
    Dataset of 60,000 28x28 grayscale images of the 10 digits,
    along with a test set of 10,000 images.
    :return:
    """
    dataset = DATA()
    dataset.set_dataset('mnist')
    dataset.set_img_size(28, 28)
    dataset.set_number_classes(10)

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, dataset.IMG_ROW, dataset.IMG_COL, 1)
    X_test = X_test.reshape(-1, dataset.IMG_ROW, dataset.IMG_COL, 1)

    # cast pixels to floats, normalize to [-0.5, 0.5] range
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = (X_train / 255) - 0.5
    X_test = (X_test / 255) - 0.5

    # one-hot-encode the labels
    Y_train = np_utils.to_categorical(Y_train, dataset.NB_CLASSES)
    Y_test = np_utils.to_categorical(Y_test, dataset.NB_CLASSES)

    # summarize mnist data set
    print('Dataset(MNIST) Summary:')
    print('Train set: {}, {}'.format(X_train.shape, Y_train.shape))
    print('Test set: {}, {}'.format(X_test.shape, Y_test.shape))

    return (X_train, Y_train), (X_test, Y_test)

def load_fashion_mnist():
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
    :return:
    """
    dataset = DATA()
    dataset.set_dataset('f-mnist')
    dataset.set_img_size(28, 28)
    dataset.set_number_classes(10)

    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    X_train = X_train.reshape(-1, dataset.IMG_ROW, dataset.IMG_COL, 1)
    X_test = X_test.reshape(-1, dataset.IMG_ROW, dataset.IMG_COL, 1)

    # cast pixels to floats, normalize to [-0.5, 0.5] range
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = (X_train / 255) - 0.5
    X_test = (X_test / 255) - 0.5

    # one-hot-encode the labels
    Y_train = np_utils.to_categorical(Y_train, dataset.NB_CLASSES)
    Y_test = np_utils.to_categorical(Y_test, dataset.NB_CLASSES)

    # summarize mnist data set
    print('Dataset(MNIST) Summary:')
    print('Train set: {}, {}'.format(X_train.shape, Y_train.shape))
    print('Test set: {}, {}'.format(X_test.shape, Y_test.shape))

    return (X_train, Y_train), (X_test, Y_test)