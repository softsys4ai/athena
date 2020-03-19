"""

@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import os
import time

import models.svm_mnist as svm
import utils.model_utils as model_utils
from attacks import craft_w_art
from data.data import load_data
from utils.file import *
from utils.file import save_adv_examples

EXP_ROOT = os.path.join(PATH.PROJECT_DIR, 'exp_results')
MODLE_DIR = os.path.join(EXP_ROOT, 'svm')
RESULTS_DIR = os.path.join(EXP_ROOT, 'results')
AE_DIR = os.path.join(EXP_ROOT, 'ae_svm')

def generate_adversarial_exmaple(model_file, X, Y, nb_classes, attack=None):
    assert model_file is not None
    assert attack is not None

    print('Loading model [{}]'.format(model_file))
    model = model_utils.load(model_file)

    print('Wrapping the model...')
    mdoel = model_utils.wrap(model, target_type='sklearnclassifier', nb_classes=nb_classes)

    X_adv = craft_w_art.get_adversarial_examples(X, Y, model, nb_classes, attack)
    X_adv = np.clip(X_adv, 0., 1.)

    print('Test accuracy on clean: {}%'.format(100 * svm.evaluate(model, X, Y)))
    print('Test accuracy on AE: {}%'.format(100 * svm.evaluate(model, X_adv, Y)))
    # save adversarial examples
    save_adv_examples(X_adv, prefix='AE-test', dataset=DATA.mnist, transformation=TRANSFORMATION.clean,
                      attack_method=attack, attack_params='')


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = load_data(DATA.mnist)
    nb_tests, img_rows, img_cols, nb_channels = X_test.shape
    nb_samples = X_train.shape[0]
    nb_features = img_rows * img_cols * nb_channels
    nb_classes = Y_train.shape[1]

    # nb_samples = 200
    # nb_tests = 200
    # X_train = X_train[:nb_samples]
    # Y_train = Y_train[:nb_samples]
    # X_test = X_test[:nb_tests]
    # Y_test = Y_test[:nb_tests]

    X_train = X_train.reshape(nb_samples, nb_features)
    X_test = X_test.reshape(nb_tests, nb_features)

    Y_train = np.argmax(Y_train, axis=1)
    Y_test = np.argmax(Y_test, axis=1)

    model_file = 'test-mnist-svm-clean.pkl'
    model_file = os.path.join(PATH.MODEL, model_file)

    data = {
        'dataset': DATA.mnist,
        'architecture': 'svm',
    }

    data['trans'] = TRANSFORMATION.clean
    data['train'] = (X_train, Y_train)
    data['test'] = (X_test, Y_test)

    # start = time.monotonic()
    # model = svm.train(data, svm.default_train_params)
    # duration = time.monotonic() - start
    # print('Training cost:', duration)
    #
    # svm.save(model, model_file)

    start = time.monotonic()
    generate_adversarial_exmaple(model_file, X_test, Y_test,
                                 nb_classes, attack=ATTACK.CW_L2)
    duration = time.monotonic() - start
    print('Generation cost:', duration)