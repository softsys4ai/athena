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

MODLE_DIR = os.path.join(PATH.MODEL, 'svm_mnist')
AE_DIR = os.path.join(PATH.ADVERSARIAL_FILE, 'svm_mnist')

def generate_adversarial_exmaple(model_file, X, Y, nb_classes, **attack_settings):
    assert model_file is not None

    attack = attack_settings['attack']
    attack_settings.__delitem__('attack')

    print('Loading model [{}]'.format(model_file))
    model = model_utils.load(model_file)

    print('Wrapping the model...')
    wrap_settings = {
        'nb_classes': nb_classes,
    }
    model = model_utils.wrap(model, target_type='sklearnclassifier', **wrap_settings)

    X_adv = craft_w_art.craft(X, Y, model, attack=attack,
                              **attack_settings)
    X_adv = np.clip(X_adv, 0., 1.)

    print('Test accuracy on clean: {}'.format(svm.evaluate(model, X, Y)))
    print('Test accuracy on AE: {}'.format(svm.evaluate(model, X_adv, Y)))

    # save adversarial examples
    save_adv_examples(X_adv, prefix='AE-test', dataset=DATA.mnist, transformation=TRANSFORMATION.clean,
                      attack_method=attack, attack_params=attack_settings)
    return X_adv


def get_attack_conf(attack):
    # configs for paper
    attack_settings = {
        ATTACK.FGSM: {
            'attack': ATTACK.FGSM,
            'eps': [0.05, 0.15, 0.25],
        },
        ATTACK.BIM_L2: {
            'attack': ATTACK.BIM_L2,
            'eps': [0.75, 2.0, 9.5],
        },
        ATTACK.BIM_Li: {
            'attack': ATTACK.BIM_Li,
            'eps': [0.05, 0.1, 0.25],
        },
        ATTACK.JSMA: {
            'attack': ATTACK.JSMA,
            'theta': [2.0, 5.0, 35.0],
        },
        ATTACK.CW_L2: {
            'attack': ATTACK.CW_L2,
            'lr': [0.015, 0.015, 0.01],
            'bsearch_steps': [10, 12, 20],
        },
        ATTACK.PGD: {
            'attack': ATTACK.PGD,
            'eps': [0.05, 0.075, 0.25],
        },
    }

    return attack_settings[attack]


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = load_data(DATA.mnist)
    nb_tests, img_rows, img_cols, nb_channels = X_test.shape
    nb_samples = X_train.shape[0]
    nb_features = img_rows * img_cols * nb_channels
    nb_classes = Y_train.shape[1]

    # for test
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
    model_file = os.path.join(MODLE_DIR, model_file)

    data = {
        'dataset': DATA.mnist,
        'architecture': 'svm',
    }

    attack_settings = {

    }

    data['trans'] = TRANSFORMATION.clean
    data['train'] = (X_train, Y_train)
    data['test'] = (X_test, Y_test)

    # start = time.monotonic()
    # model = svm.train(data, svm.default_train_params)
    # duration = time.monotonic() - start
    # print('Training cost:', duration)
    # svm.save(model, model_file)

    start = time.monotonic()
    generate_adversarial_exmaple(model_file, X_test, Y_test,
                                 nb_classes, attack=ATTACK.CW_L2)
    duration = time.monotonic() - start
    print('Generation cost:', duration)