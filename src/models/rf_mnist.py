"""
Random forest classifier on MNIST / Fashion-MNIST
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import logging
import os
import pickle
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import utils.data_utils as data_utils
from data.data import load_data
from definitions import *
from models.transformation import transform
from utils.config import *
from utils.logger import get_logger

logger = get_logger('Athena')
logger.setLevel(logging.INFO)

ROOT_DIR = get_project_root()
LOG_DIR = os.path.join(str(ROOT_DIR), 'logs')

validation_rate = 0.2

def train(train_set, test_set=None, save_path='mnist-rf-clean.rf', **training_params):
    X_train, Y_train = train_set

    if len(X_train.shape) > 2:
        X_train = __reshape(X_train)

    Y_train = data_utils.probs2labels(Y_train)

    nb_examples = X_train.shape[0]
    nb_training = int(nb_examples * (1. - validation_rate))
    train_exmaples = X_train[:nb_training]
    train_labels = Y_train[:nb_training]

    n_estimators = training_params.get('n_estimators', 100)
    criterion = training_params.get('criterion', 'gini')

    rf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
    print(rf)
    print('Training a random forest classifier, it may take a while...')
    start = time.monotonic()
    rf.fit(train_exmaples, train_labels)
    cost = time.monotonic() - start
    print('Trained! Cost:', cost)

    if save_path is not None:
        print('Save trained model to [{}].'.format(save_path))
        pickle.dump(rf, open(save_path, 'wb'))

    preds = rf.predict(train_exmaples)
    preds = data_utils.probs2labels(preds)
    train_acc = accuracy_score(train_labels, preds)
    preds = rf.predict(X_train[nb_training:])
    preds = data_utils.probs2labels(preds)
    val_acc = accuracy_score(Y_train[nb_training:], preds)

    print('Evaluation on training set:', train_acc)
    print('Evaluation on validation set:', val_acc)

    if test_set is not None:
        X_test, Y_test = test_set

        if len(X_test.shape) > 2:
            X_test = __reshape(X_test)

        Y_test = data_utils.probs2labels(Y_test)
        preds = rf.predict(X_test)
        preds = data_utils.probs2labels(preds)
        test_acc = accuracy_score(Y_test, preds)
        print('Evaluation on test set:', test_acc)


def get_logits(model, X):
    assert model is not None
    return model.predict_log_proba(X)


def get_probabilities(model, X):
    assert model is not None
    return model.predict_proba(X)


def __reshape(data):
    # assert len(data) == 4
    nb_samples, img_rows, img_cols, nb_channels = data.shape

    reshaped = np.reshape(data, (nb_samples, img_rows * img_cols * nb_channels))
    print('Reshaping...', data.shape, 'to', reshaped.shape)
    return reshaped

if __name__ == '__main__':

    training_params = {
        'model': 'rf',
        'dataset': DATA.mnist,
        'n_estimators': 100,
        'criterion': 'gini',
    }

    # transformations = TRANSFORMATION.supported_types()
    transformations = [TRANSFORMATION.clean]

    (X_train, Y_train), (X_test, Y_test) = load_data(DATA.mnist)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    MODEL_DIR = 'exp_results/models/rf'
    save_path = 'mnist-rf-'

    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    for trans in transformations:
        save_path = 'mnist-rf-' + trans + '.rf'
        save_path = os.path.join(MODEL_DIR, save_path)

        # apply transformation on data set
        X_train_trans = transform(X_train, trans)
        X_test_trans = transform(X_test, trans)
        train(train_set=(X_train_trans, Y_train), test_set=(X_test_trans, Y_test),
              save_path=save_path, **training_params)