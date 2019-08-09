"""
Implement operations related to data (dataset).
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
from __future__ import division, absolute_import, print_function

import keras
import tensorflow as tf
from keras.datasets import mnist, cifar10, fashion_mnist, cifar100

from config import DATA

# set random seed for replication
tf.set_random_seed(1000)

def load_data(dataset):
    assert dataset in DATA.get_supported_datasets()

    if (dataset == DATA.mnist):
        """
        Dataset of 60,000 28x28 grayscale images of the 10 digits,
        along with a test set of 10,000 images.
        """
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        nb_classes = 10
    elif (dataset == DATA.fation_mnist):
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
        nb_classes = 10
    elif (dataset == DATA.cifar_10):
        """
        Dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images.
        """
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        nb_classes = 10
    elif (dataset == DATA.cifar_100):
        (X_train, Y_train), (X_test, Y_test) = cifar100.load_data()
        nb_classes = 100

    # processing data
    _, img_rows, img_cols, nb_channels = X_test.shape

    X_train = X_train.reshape(-1, img_rows, img_cols, nb_channels)
    X_test = X_test.reshape(-1, img_rows, img_cols, nb_channels)

    # cast pixels to floats, normalize to [0, 1] range
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.

    # one-hot-encode the labels
    Y_train = keras.utils.to_categorical(Y_train, nb_classes)
    Y_test = keras.utils.to_categorical(Y_test, nb_classes)

    # summarize mnist data set
    print('Dataset({}) Summary:'.format(dataset.upper()))
    print('Train set: {}, {}'.format(X_train.shape, Y_train.shape))
    print('Test set: {}, {}'.format(X_test.shape, Y_test.shape))

    return (X_train, Y_train), (X_test, Y_test)

"""
for testing
"""
def main(args):
    load_data(args)

if __name__ == "__main__":
    main(DATA.cifar_10)