"""
Scripts to evaluate models.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
import numpy as np

import models

from utils.config import *
from data import normalize
from transformation import transform

def eval_single_model(model_name, testset_name, labels_name):
    """
    Evaluate model on test set
    :param model_name:
    :param testset_name:
    :return:
    """
    prefix, dataset, architect, trans_type = model_name.split('-')

    X_test = np.load('{}/{}.npy'.format(PATH.ADVERSARIAL_FILE, testset_name))
    labels = np.load('{}/{}.npy'.format(PATH.ADVERSARIAL_FILE, labels_name))

    if 'composition' in trans_type:
        trans_type = TRANSFORMATION.get_transformation_compositions()
        print(type(trans_type), trans_type)

    # apply transformation(s)
    X_test = transform(X_test, trans_type)

    # evaluate each of the composition
    if 'composition' in trans_type:
        for trans in trans_type:
            print(type(trans), trans)

            m_name = '{}-{}-{}-{}'.format(prefix, dataset, architect, trans)
            model = models.load_model(m_name)

            print('*** Evaluating ({}) on ({})...'.format(m_name, testset_name))
            scores = model.evaluate(X_test, labels, verbose=2)
            print(scores)
            del model

    # evaluate the model
    model = models.load_model(model_name)

    if (dataset == DATA.cifar_10):
        X_test = normalize(X_test)
    print('*** Evaluating ({}) on ({})...'.format(model_name, testset_name))
    scores = model.evaluate(X_test, labels, verbose=2)
    print(scores)
    return scores

def main():
    model_name = 'model-mnist-cnn-composition'
    testset_name = 'test_AE-mnist-cnn-clean-bim_ordinf_nbIter100_eps250'
    labels_name = 'test_Label-mnist-clean'
    eval_single_model(model_name, testset_name, labels_name)

if __name__ == '__main__':
    MODE.debug_off()
    main()

