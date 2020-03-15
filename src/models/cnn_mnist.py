"""

@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import logging
import os
import time

import keras
from keras import layers, models
from sklearn import metrics

import utils.data_utils as data_utils
import utils.model_utils as model_utils
from data.data import load_data
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


train_conf = {
    'cnn_params': {
        'optimizer': keras.optimizers.Adam(lr=0.001),
        'loss': keras.losses.categorical_crossentropy,
    },
    'lr': 0.001,
    'val_rate': 0.2,
    'batch_size': 128,
    'epoch': 50,
}


def __get_cnn(input_shape=(28, 28, 1), nb_classes=10, activation='relu', pdrop=0.4):
    MODEL.set_architecture('cnn')

    # define model architecture
    struct = [
        layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape,
                      activation=activation),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(filters=64, kernel_size=(3, 3), activation=activation),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dense(64 * 64),
        layers.Dropout(rate=pdrop),
        layers.Dense(nb_classes),
        layers.Activation('softmax'),
    ]

    # construct the model
    model = models.Sequential()
    for layer in struct:
        model.add(layer)

    logger.info(model.summary())
    return model


def train(data, nb_classes=10, save_path='cnn_mnist.h5', eval=True, conf=train_conf):
    assert data is not None
    assert data['train'] is not None and len(data['train']) == 2

    # Get training data
    X_train, y_train = data['train']

    nb_examples, img_rows, img_cols, nb_channels = X_train.shape
    nb_trains = int(nb_examples * (1. - conf['val_rate']))
    train_examples = X_train[:nb_trains]
    train_labels = y_train[:nb_trains]
    val_examples = X_train[nb_trains:]
    val_labels = y_train[nb_trains:]

    cnn = __get_cnn(input_shape=(img_rows, img_cols, nb_channels), nb_classes=nb_classes)
    cnn.compile(optimizer=conf['cnn_params']['optimizer'], loss=conf['cnn_params']['loss'],
                metrics=['accuracy'])

    # train the model
    print('Training [{}] CNN, it may take a while...'.format(data['trans']))
    start = time.monotonic()
    history = cnn.fit(train_examples, train_labels, batch_size=conf['batch_size'], epochs=conf['epoch'],
                      verbose=2, validation_data=(val_examples, val_labels))
    train_cost = time.monotonic() - start
    logger.info('Done training. It costs {} minutes.'.format(train_cost / 60.))

    if eval:
        scores_train = cnn.evaluate(train_examples, train_labels, batch_size=128, verbose=0)
        scores_val = cnn.evaluate(val_examples, val_labels, batch_size=128, verbose=0)
        logger.info('Evaluation on [{} set]: {}.'.format('training', scores_train))
        logger.info('Evaluation on [{} set]: {}.'.format('validation', scores_val))

        if data['test'] is not None:
            X_test, y_test = data['test']
            scores_test = cnn.evaluate(X_test, y_test, batch_size=128, verbose=0)
            logger.info('Evaluation on [{} set]: {}'.format('testing', scores_test))

    # logger.info('Save the trained model to [{}].'.format(save_path))
    # cnn.save(save_path)

    checkpoints_file = save_path.split('/')[-1].split('.')[0]
    checkpoints_file = 'checkpoints_train-' + checkpoints_file + '.csv'
    checkpoints_file = os.path.join(LOG_DIR, checkpoints_file)

    if not os.path.dirname(LOG_DIR):
        os.mkdir(LOG_DIR)

    logger.info('Training checkpoints have been saved to file [{}].'.format(checkpoints_file))
    file.dict2csv(history.history, checkpoints_file)
    save_path = save_path.split('/')[-1].split('.')[0]
    save_path = 'hist_train-' + save_path + '.pdf'
    plot_training_history(history, save_path)

    return cnn


def get_logits(model, x):
    logits_layer = model.layers[-2].name
    logits_model = models.Model(
        inputs=model.input,
        outputs=model.get_layer(logits_layer).output)

    logits = logits_model.predict(x)
    return logits


def get_probabilities(model, x):
    return model.predict(x)


def get_predicted_labels(model, x):
    probabilities = get_probabilities(model, x)
    labels = [np.argmax(p) for p in probabilities]

    return labels

def evaluate(model, X, Y):

    if len(Y.shape) > 1:
        # convert probabilities to labels
        Y = [np.argmax(y) for y in Y]

    predictions = get_probabilities(model, X)
    predictions = [np.argmax(p) for p in predictions]

    return round(metrics.accuracy_score(y_true=Y, y_pred=predictions), 6)


if __name__ == '__main__':
    transformations = TRANSFORMATION.supported_types()

    data = {
        'dataset': DATA.mnist,
        'architecture': 'cnn',
    }

    (X_train, y_train), (X_test, y_test) = load_data(data['dataset'])
    nb_classes = y_train.shape[-1]

    X_train = data_utils.set_channels_last(X_train)
    X_test = data_utils.set_channels_last(X_test)
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    for trans in transformations:
        data['trans'] = trans

        data['train'] = (data_utils.rescale(transform(X_train, trans)), y_train)
        data['test'] = (data_utils.rescale(transform(X_test, trans)), y_test)

        model = train(data, nb_classes=nb_classes, eval=True, conf=train_conf)

        filename = 'model-{}-{}-{}.h5'.format(data['dataset'], data['architecture'], data['trans'])
        filename = os.path.join(PATH.MODEL, filename)
        model_utils.save(model, filename)


