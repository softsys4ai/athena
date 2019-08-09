"""
Define models and implement related operations.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

from config import *

FLAGS = flags.FLAGS
def cnn(input_shape, nb_classes):
    """
    a cnn model.
    :param input_shape:
    :param nb_classes:
    :return: a simple cnn model
    """
    MODEL.TYPE = 'cnn'

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
def train(X, Y, model_type, model_name):
    """
    Train a model over given training set, then
    save the trained model as given name.
    :param X: training examples.
    :param Y: corresponding desired labels.
    :param model_type: the type of model to create and train.
    :param model_name: the name to save the model as
    :return: na
    """
    nb_validation = int(len(X) * DATA.VALIDATION_RATE)
    train_samples = X[: -nb_validation]
    train_labels = Y[: -nb_validation]
    val_sample = X[-nb_validation :]
    val_labels = Y[-nb_validation :]

    img_rows = X.shape[1]
    img_cols = X.shape[2]
    nb_channels = X.shape[3]
    nb_classes = Y.shape[1]

    model = None

    if (model_type == 'cnn'):
        input_shape = (img_rows, img_cols, nb_channels)
        model = cnn(input_shape=input_shape, nb_classes=nb_classes)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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


