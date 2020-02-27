"""

@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import logging
import os
import time

from tensorflow.python import keras
from tensorflow.python.keras import layers, models

import data.data as data
import utils.data_utils as data_utils
from definitions import *
from models.transformation import transform
from utils import file
from utils.config import *
from utils.logger import get_logger
from utils.plot import plot_training_history

logger = get_logger('Athena')
logger.setLevel(logging.INFO)

ROOT_DIR = get_project_root()
LOG_DIR = os.path.join(str(ROOT_DIR), 'logs')

def create_model(input_shape=(28, 28, 1), nb_classes=10):
    MODEL.set_architecture('cnn')

    # define model architecture
    struct = [
        layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(filters=64, kernel_size=(3, 3)),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dense(64 * 64),
        layers.Dropout(rate=0.4),
        layers.Dense(nb_classes),
        layers.Activation('softmax'),
    ]

    # construct the model
    model = models.Sequential()
    for layer in struct:
        model.add(layer)

    logger.info(model.summary())
    return model

def train(dataset, model=None, trans_type=TRANSFORMATION.clean,
          save_path='cnn_mnist.h5', eval=True, **kwargs):
    """
    Train a cnn model on MNIST or Fashion-MNIST.
    :param dataset:
    :param model: a model to train.
    :param trans_type: transformation associated to the model.
    :param save_path: file name, including the path, to save the trained model.
    :param kwargs: customized loss function, optimizer, etc. for cleverhans to craft AEs.
    :return: the trained model
    """
    lr = 0.001
    validation_rate = 0.2
    optimizer = kwargs.get('optimizer', keras.optimizers.Adam(lr=lr))
    loss_fn = kwargs.get('loss', keras.losses.categorical_crossentropy)
    metrics = kwargs.get('metrics', 'default')

    logger.info('optimizer: [{}].'.format(optimizer))
    logger.info('loss function: [{}].'.format(loss_fn))
    logger.info('metrics: [{}].'.format(metrics))

    (X_train, Y_train), (X_test, Y_test) = data.load_data(dataset)
    X_train = data_utils.set_channels_last(X_train)
    X_test = data_utils.set_channels_last(X_test)

    # Apply transformation (associated to the weak defending model)
    X_train = transform(X_train, trans_type)
    X_test = transform(X_test, trans_type)

    nb_examples, img_rows, img_cols, nb_channels = X_train
    nb_train_samples = int(nb_examples * (1. - validation_rate))
    train_examples = X_train[:nb_train_samples]
    train_labels = Y_train[:nb_train_samples]
    val_examples = X_train[nb_train_samples:]
    val_labels = Y_train[nb_train_samples:]

    if model is None:
        model = create_model(input_shape=(img_rows, img_cols, nb_channels))

    # Compile model
    if ('default' == metrics):
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    else:
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy', metrics])

    # Train model
    batch_size = kwargs.get('batch_size', 128)
    epochs = kwargs.get('epochs', 20)

    start = time.monotonic()
    history = model.fit(train_examples, train_labels, batch_size=batch_size, epochs=epochs,
                        verbose=2, validation_data=(val_examples, val_labels))
    cost = time.monotonic() - start
    logger.info('Done training. It costs {} minutes.'.format(cost / 60.))

    if eval:
        scores_train = model.evaluate(train_examples, train_labels, batch_size=128, verbose=0)
        scores_val = model.evaluate(val_examples, val_labels, batch_size=128, verbose=0)
        scores_test = model.evaluate(X_test, Y_test, batch_size=128, verbose=0)

        logger.info('Evaluation on [{} set]: {}.'.format('training', scores_train))
        logger.info('Evaluation on [{} set]: {}.'.format('validation', scores_val))
        logger.info('Evaluation on [{} set]: {}.'.format('testing', scores_test))

    logger.info('Save the trained model to [{}].'.format(save_path))
    model.save(save_path)

    checkpoints_file = save_path.split('/')[-1].split('.')[0]
    checkpoints_file = 'checkpoints_train_' + checkpoints_file + '.csv'
    checkpoints_file = os.path.join(LOG_DIR, checkpoints_file)

    if not os.path.dirname(LOG_DIR):
        os.mkdir(LOG_DIR)

    logger.info('Training checkpoints have been saved to file [{}].'.format(checkpoints_file))
    file.dict2csv(history.history, checkpoints_file)
    save_path = save_path.split('/')[-1].split('.')[0]
    save_path = 'hist_train_' + save_path + '.pdf'
    plot_training_history(history, save_path)

    return model

