"""

@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
import os
import operator
import random

from utils.config import *
from utils.file import *

PROJECT_DIR = PATH.PROJECT_DIR
# TODO: parameterize date-time folder
ANALYSE_DIR = '{}/evaluation_results/2019-09-06_02-41-18'.format(PROJECT_DIR)
PREDICTION_DIR = '{}/prediction_result'.format(ANALYSE_DIR)
EVALUATION_DIR = '{}/evaluation'.format(ANALYSE_DIR)

class PERFORM_MODE(object):
    TOP = 'top-k'
    BOTTOM = 'bottom-k'
    RANDOM = 'rand-k'

"""
Computing test accuracy upper bound for given k models
"""
def get_test_accuracies(attack, pred_file, label_file, mode=PERFORM_MODE.TOP):
    # load prediction matrix and true labels
    path = os.path.join(PREDICTION_DIR, attack)
    pred_matrix = np.load('{}/{}'.format(path, pred_file))
    labels = np.load('{}/{}'.format(PREDICTION_DIR, label_file))

    print('shape of data:', pred_matrix.shape)
    print(pred_matrix[0][0], np.argmax(pred_matrix[0][0]))

    print('shape of labels:', labels.shape)

    nb_of_models = pred_matrix.shape[0]
    accuracies = {}

    for m_id in range(nb_of_models):
        accuracies[m_id] = compute_model_accuracy(pred_matrix[m_id], labels)

    print(accuracies)

    if PERFORM_MODE.TOP == mode:
        acc_sorted = dict(sorted(accuracies.items(), key=operator.itemgetter(1), reverse=True))
        model_indices = list(acc_sorted.keys())
        upperbounds = compute_acc_upper_bounds(model_indices, pred_matrix, labels)

    elif PERFORM_MODE.BOTTOM == mode:
        acc_sorted = dict(sorted(accuracies.items(), key=operator.itemgetter(1), reverse=False))
        model_indices = list(acc_sorted.keys())
        upperbounds = compute_acc_upper_bounds(model_indices, pred_matrix, labels)

    elif PERFORM_MODE.RANDOM == mode:
        # for random-k weak defenses, compute the average of 10 rounds
        acc_sum = np.zeros(pred_matrix.shape[0])
        upperbounds = {}
        for i in range(10):
            model_indices = list(accuracies.items())
            random.shuffle(model_indices)
            acc_sorted = dict(model_indices)
            model_indices = list(acc_sorted.keys())
            acc = list(compute_acc_upper_bounds(model_indices, pred_matrix, labels).values())

            print('ROUND {}: {}'.format(i, acc))

            for s_id in range(pred_matrix.shape[0]):
                acc_sum[s_id] += acc[s_id]

        for k in range(pred_matrix.shape[0]):
            upperbounds[k+1] = acc_sum[k]/10.
    else:
        raise ValueError('Invalid performance mode.')

    # TODO: parameterize dataset
    file_name = 'acc_upperbounds-{}-{}-{}_2.csv'.format('mnist', mode, attack)
    file_path = os.path.join(EVALUATION_DIR, file_name)
    dict2csv(upperbounds, file_path)

    return upperbounds

"""
Computing test accuracy for a model
"""
def compute_model_accuracy(pred_matrix, true_labels):
    nb_of_hits = 0.
    nb_of_samples = true_labels.shape[0]
    for i in range(nb_of_samples):
        if (np.argmax(pred_matrix[i]) == true_labels[i]):
            nb_of_hits += 1.

    return round(nb_of_hits/nb_of_samples, 4)

"""
Computing the test accuracy upper bounds for k weak defenses
"""
def compute_acc_upper_bounds(model_indices, pred_matrix, true_labels):
    nb_of_models = pred_matrix.shape[0]
    nb_of_samples = true_labels.shape[0]

    upperbounds = {}
    print('model indices: ', model_indices)
    for k in range(nb_of_models):
        nb_of_hits = 0.0
        # print('model indices: ', model_indices[: k+1])
        for sample_id in range(nb_of_samples):
            for i in range(k + 1):
                model_id = model_indices[i]
                if (0 == model_id):
                    # skip the model trained on original data set.
                    # only consider the models trained on transformed data sets.
                    continue

                if (true_labels[sample_id] == np.argmax(pred_matrix[model_id][sample_id])):
                    nb_of_hits += 1.
                    break
        upperbounds[k+1] = round(nb_of_hits/nb_of_samples, 6)

    return upperbounds


def main():
    predictions_file = 'predProb.npy'
    label_file = 'labels.npy'

    attacks = [
        'bim_ord2_nbIter100_eps500', # 0
        'bim_ord2_nbIter100_eps1000', # 1
        'bim_ordinf_nbIter100_eps250', # 2
        'bim_ordinf_nbIter100_eps500',# 3
        'deepfool_maxIter100', # 4
        'deepfool_maxIter10000', # 5
        'fgsm_eps100', # 6
        'fgsm_eps250', # 7
        'fgsm_eps300', # 8
        'jsma_theta30_gamma50', # 9
        'jsma_theta50_gamma70', # 10
        'pgd_eps250_nbIter100_epsIter10', # 11
        'pgd_eps500_nbIter100_epsIter10', # 12
        'pgd_eps750_nbIter100_epsIter10', # 13
    ]

    mode = PERFORM_MODE.RANDOM
    attack = attacks[12]

    print('ATTACK ({}) / MODE ({})'.format(attack, mode))
    print(get_test_accuracies(attack, predictions_file, label_file, mode))

if __name__ == '__main__':
    main()