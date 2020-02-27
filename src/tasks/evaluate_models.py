"""
Scripts to evaluate models.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
import numpy as np

import models

from utils.config import *
import os
from utils.csv_headers import IdealModelEvalHeaders as headers
from utils.file import *
from data.data import normalize
from models.transformation import transform
from evaluation.compute_accuracy import get_test_accuracies, PERFORM_MODE

PROJECT_DIR = PATH.PROJECT_DIR
ANALYSE_DIR = '{}/evaluation_results/2019-09-06_02-41-18'.format(PROJECT_DIR)
PREDICTION_DIR = '{}/prediction_result'.format(ANALYSE_DIR)
EVALUATION_DIR = '{}/evaluation'.format(ANALYSE_DIR)

attacks = [
        'bim_ord2_nbIter100_eps500',  # 0
        'bim_ord2_nbIter100_eps1000',  # 1
        'bim_ordinf_nbIter100_eps250',  # 2
        'bim_ordinf_nbIter100_eps500',  # 3
        'deepfool_maxIter100',  # 4
        'deepfool_maxIter10000',  # 5
        'fgsm_eps100',  # 6
        'fgsm_eps250',  # 7
        'fgsm_eps300',  # 8
        'jsma_theta30_gamma50',  # 9
        'jsma_theta50_gamma70',  # 10
        'pgd_eps250_nbIter100_epsIter10',  # 11
        'pgd_eps500_nbIter100_epsIter10',  # 12
        'pgd_eps750_nbIter100_epsIter10',  # 13
    ]

def eval_ideal_model(attack):
    """
    Test accuracy of ideal model is the test accuracy upper bound of an ensemble of k
    :return:
    """
    predictions_file = 'predProb.npy'
    label_file = 'labels.npy'

    ideal_accuracy = {}
    print('ATTACK ({})'.format(attack))

    # get the upper bound of ideal test accuracy
    print('Computing {}...'.format(headers.TOP_K.value))
    upper_bound = get_test_accuracies(attack, predictions_file,
                                      label_file, PERFORM_MODE.TOP)
    ideal_accuracy[headers.NUM_OF_WEAK_DEFENSES.value] = list(upper_bound.keys())
    ideal_accuracy[headers.TOP_K.value] = list(upper_bound.values())

    # get the lower bound of ideal test accuracy
    print('Computing {}...'.format(headers.BOTTOM_K.value))
    lower_bound = get_test_accuracies(attack, predictions_file,
                                      label_file, PERFORM_MODE.BOTTOM)
    ideal_accuracy[headers.BOTTOM_K.value] = list(lower_bound.values())

    # get test accuracies of randomly built ideal model
    # they will be used to estimate the certainty of test accuracy of an ideal model
    rand_acc = []
    rand_prefix = 'RandK_R'

    for i in range(30):
        key = '{}{}'.format(rand_prefix, i)
        print('Computing {}...'.format(key))

        acc = get_test_accuracies(attack, predictions_file,
                                  label_file, PERFORM_MODE.RANDOM)
        rand_acc.append(list(acc.values()))
        ideal_accuracy[key] = list(acc.values())

    # compute average, upper-bound, and lower-bound of test accuracy certainty
    nb_of_rounds, nb_of_samples = np.asarray(rand_acc).shape

    average = np.zeros(nb_of_samples)
    upperbounds = np.zeros(nb_of_samples)
    lowerbounds = np.zeros(nb_of_samples) + np.infty
    for i in range(nb_of_samples):
        for j in range(nb_of_rounds):
            average[i] += rand_acc[j][i]
            upperbounds[i] = max(upperbounds[i], rand_acc[j][i])
            lowerbounds[i] = min(lowerbounds[i], rand_acc[j][i])

    average = np.round(average / nb_of_rounds, 4)

    ideal_accuracy[headers.RANDK_AVG.value] = list(average)
    ideal_accuracy[headers.RANDK_UPPERBOUND.value] = list(upperbounds)
    ideal_accuracy[headers.RANDK_LOWERBOUND.value] = list(lowerbounds)
    ideal_accuracy[headers.GAP.value] = list(np.asarray(ideal_accuracy[headers.TOP_K.value])
                                             - np.asarray(ideal_accuracy[headers.RANDK_AVG.value]))

    if MODE.DEBUG:
        print(ideal_accuracy)

    file_name = 'acc-ideal_model-{}-{}.csv'.format('mnist', attack)
    file_path = os.path.join(EVALUATION_DIR, file_name)
    dict2csv(ideal_accuracy, file_path, list_as_value=True)


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
    """
    Evaluate single model
    """
    # model_name = 'model-mnist-cnn-composition'
    # testset_name = 'test_AE-mnist-cnn-clean-bim_ordinf_nbIter100_eps250'
    # labels_name = 'test_Label-mnist-clean'
    # eval_single_model(model_name, testset_name, labels_name)

    """
    Evaluate ideal model
    """
    attack = attacks[8]
    eval_ideal_model(attack)

if __name__ == '__main__':
    MODE.debug_off()
    # MODE.debug_on()
    main()

