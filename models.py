"""
Define models and implement related operations.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.keras import layers, models

from config import *

FLAGS = flags.FLAGS

def create_model(dataset, input_shape, nb_classes):
    if (dataset == DATA.mnist):
        MODEL.set_dataset(DATA.mnist)
        MODEL.set_learning_rate(0.01)
        MODEL.set_batch_size(128)
        MODEL.set_epochs(50)
        return cnn_mnist(input_shape, nb_classes)
    elif (dataset == DATA.fation_mnist):
        MODEL.set_dataset(DATA.fation_mnist)
        MODEL.set_learning_rate(0.01)
        MODEL.set_batch_size(128)
        MODEL.set_epochs(50)
        return cnn_mnist(input_shape, nb_classes)
    elif (dataset == DATA.cifar_10):
        MODEL.set_dataset(DATA.cifar_10)
        MODEL.set_learning_rate(0.01)
        MODEL.set_batch_size(32)
        MODEL.set_epochs(350)
        return cnn_cifar(input_shape, nb_classes)

def cnn_cifar(input_shape, nb_classes):
    """
    a cnn for cifar
    :param input_shape:
    :param nb_classes:
    :return:
    """
    MODEL.ARCHITECTURE = 'cnn'

    struct = [
        layers.Conv2D(96, (3, 3), input_shape=input_shape),
        layers.Activation('relu'),
        layers.Conv2D(96, (3, 3)),
        layers.Activation('relu'),
        layers.Conv2D(96, (3, 3)),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(192, (3, 3)),
        layers.Activation('relu'),
        layers.Conv2D(192, (3, 3)),
        layers.Activation('relu'),
        layers.Conv2D(192, (3, 3)),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(192, (3, 3)),
        layers.Activation('relu'),
        layers.Conv2D(192, (1, 1)),
        layers.Activation('relu'),
        layers.Conv2D(10, (1, 1)),
        layers.Activation('relu'),
        layers.AveragePooling2D(pool_size=1),
        layers.Flatten(),
        layers.Dense(nb_classes),
        layers.Activation('softmax')
    ]

    model = models.Sequential()
    for layer in struct:
        model.add(layer)

    return model

def cnn_mnist(input_shape, nb_classes):
    """
    a simple cnn model.
    :param input_shape:
    :param nb_classes:
    :return: a simple cnn model
    """
    MODEL.ARCHITECTURE = 'cnn'

    struct = [
        layers.Conv2D(32, (3, 3), input_shape=input_shape),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, (3, 3)),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dense(64 * 64),
        layers.Dropout(rate=0.4),
        layers.Dense(nb_classes),
        layers.Activation('softmax')
    ]

    model = models.Sequential()
    for layer in struct:
        model.add(layer)

    return model

# --------------------------------------------
# OPERATIONS
# --------------------------------------------
def train(X, Y, model_name):
    """
    Train a model over given training set, then
    save the trained model as given name.
    :param X: training examples.
    :param Y: corresponding desired labels.
    :param model_type: the type of model to create and train.
    :param model_name: the name to save the model as
    :return: na
    """
    _, dataset, architect, trans_type = model_name.split('-')

    nb_validation = int(len(X) * DATA.valiation_rate)
    train_samples = X[: -nb_validation]
    train_labels = Y[: -nb_validation]
    val_sample = X[-nb_validation :]
    val_labels = Y[-nb_validation :]

    _, img_rows, img_cols, nb_channels = X.shape
    input_shape = (img_rows, img_cols, nb_channels)
    nb_classes = int(Y.shape[1])
    print('input_shape: {}; nb_classes: {}'.format(input_shape, nb_classes))

    model = create_model(dataset, input_shape=input_shape, nb_classes=nb_classes)

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # train the model
    print('Training {}...'.format(model_name))
    model.fit(train_samples, train_labels, epochs=MODEL.EPOCHS, batch_size=MODEL.BATCH_SIZE,
              shuffle=True, verbose=1, validation_data=(val_sample, val_labels))

    # save the trained model
    model.save('{}/{}'.format(PATH.MODEL, model_name))
    # delete the model after it's been saved.
    del model

    print('Trained model has been saved to data/{}'.format(model_name))

def evaluate_model(model, X, Y):
    """
    Evaluate the model on given test set.
    :param model: the name of the model to evaluate
    :param X: test set
    :param Y:
    :return: test accuracy, average confidences on correctly classified samples
            and misclassified samples respectively.
    """
    nb_corrections = 0
    nb_examples = Y.shape[0]

    conf_correct = 0.
    conf_misclassified = 0.

    pred_probs = model.predict(X, batch_size=128)

    # iterate over test set
    for pred_prob, true_prob in zip(pred_probs, Y):
        pred_label = np.argmax(pred_prob)
        true_label = np.argmax(true_prob)

        if (pred_label == true_label):
            nb_corrections += 1
            conf_correct += np.max(pred_prob)
        else:
            conf_misclassified += np.max(pred_prob)

    # test accuracy
    _, test_acc = model.evaluate(X, Y, verbose=0)
    # average confidences
    ave_conf_correct = conf_correct / nb_corrections
    ave_conf_miss = conf_misclassified / (nb_examples - nb_corrections)

    return test_acc, ave_conf_correct, ave_conf_miss


