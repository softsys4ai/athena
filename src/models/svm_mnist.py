"""
A Linear SVM classifier on MNIST/Fashion-MNIST.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import os

from sklearn import metrics
from sklearn.svm import SVC

import utils.model_utils as model_utils
from data.data import load_data
from models.transformation import transform
from utils.config import *

default_train_params = {
    'svm_params': {
        'C': 0.1,
        'probability': True,
    },
    'svm_kernel': 'linear',
}

def train(data, training_params=default_train_params):
    assert data is not None
    assert data['train'] is not None and len(data['train']) == 2, 'Require training samples and labels.'

    # Get training data
    X_train, Y_train = data['train']

    X_train = __reshape(X_train)

    print('Training [{}] svm, it may take a while...'.format(data['trans']))
    C, probability = training_params['svm_params'].values()

    svm = SVC(C=C, kernel=training_params['svm_kernel'], probability=probability)
    svm.fit(X_train, Y_train)
    print('Trained. Training accuracy:', evaluate(svm, X_train, Y_train))

    # Evaluate the model on test set
    if data['test'] is not None and len(data['test']) == 2:
        X_test, Y_test = data['test']
        X_test = __reshape(X_test)
        print('Test accuracy:', evaluate(svm, X_test, Y_test))

    return svm

def get_logits(model, X):
    # decision is a voting function
    votes = model.decision_function(X)
    votes_exp = np.exp(votes)

    return votes_exp

def get_probabilities(model, X):
    # get logtis
    votes_exp = get_logits(model, X)

    # apply softmax on logits
    probs = []
    for v_exp in votes_exp:
        v_sum = 0
        for v in v_exp:
            v_sum += v

        p = [v/v_sum for v in v_exp]
        probs.append(p)

    return np.asarray(probs)

def get_predicted_labels(model, X):
    assert model is not None
    assert  X is not X

    probs = get_probabilities(model, X)
    predictions = [np.argmax(p) for p in probs]

    return predictions

def __reshape(X):
    X = np.asarray(X)

    if len(X.shape) == 2:
        return X

    if len(X.shape) == 3:
        nb_samples, img_rows, img_cols = X.shape
        nb_channels = 1
    elif len(X.shape) == 4:
        nb_samples, img_rows, img_cols, nb_channels = X.shape
    else:
        raise ValueError('Incorrect dimension, expected 2, 3, or 4, however, it is {}.'.format(len(X.shape)))

    return X.reshape(nb_samples, (img_rows * img_cols * nb_channels))

def evaluate(model, X, Y):
    assert model is not None
    assert X is not None
    assert Y is not None

    X_test = __reshape(X)
    if len(Y.shape) > 1:
        # convert probabilities to labels
        Y = [np.argmax(y) for y in Y]

    predictions = get_predicted_labels(model, X_test)
    return round(metrics.accuracy_score(y_true=Y, y_pred=predictions), 6)

"""
def save(model, filename):
    assert model is not None
    assert filename is not None

    with open(filename, 'wb') as file:
        pickle.dump(model, file)
"""

"""
def load(filename):
    assert filename is not None

    model = None
    print('Loading model [{}]...'.format(filename))

    with open(filename, 'rb') as file:
        model = pickle.load(file)

    return model
"""

if __name__=='__main__':
    transformations = TRANSFORMATION.supported_types()

    data = {
        'dataset': DATA.mnist,
        'architecture': 'svm',
    }

    (X_train, Y_train), (X_test, Y_test) = load_data(data['dataset'])

    Y_train = np.argmax(Y_train, axis=1)
    Y_test = np.argmax(Y_test, axis=1)


    for trans in transformations:
        data['trans'] = trans

        data['train'] = (transform(X_train, trans), Y_train)
        data['test'] = (transform(X_test, trans), Y_test)

        model = train(data, training_params=default_train_params)

        filename = 'model-{}-{}-{}.pkl'.format(data['dataset'], data['architecture'], data['trans'])

        filename = os.path.join(PATH.MODEL, filename)
        model_utils.save(model, filename)

