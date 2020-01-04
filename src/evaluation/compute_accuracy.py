"""

@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
import os
import operator
import random

from utils.file import *

PROJECT_DIR = PATH.PROJECT_DIR
# TODO: parameterize date-time folder
ANALYSE_DIR = '{}/evaluation_results/2019-09-06_02-41-18'.format(PROJECT_DIR)
PREDICTION_DIR = '{}/prediction_result'.format(ANALYSE_DIR)

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

    if MODE.DEBUG:
        print('shape of data:', pred_matrix.shape)
        print('shape of labels:', labels.shape)

    nb_of_models = pred_matrix.shape[0]
    accuracies = {}

    for m_id in range(nb_of_models):
        # clean model is not a weak defense
        accuracies[m_id] = compute_model_accuracy(pred_matrix[m_id], labels)

    if MODE.DEBUG:
        print(len(accuracies.keys()))
        print(accuracies)

    if PERFORM_MODE.TOP == mode:
        acc_sorted = dict(sorted(accuracies.items(), key=operator.itemgetter(1), reverse=True))

    elif PERFORM_MODE.BOTTOM == mode:
        acc_sorted = dict(sorted(accuracies.items(), key=operator.itemgetter(1), reverse=False))

    elif PERFORM_MODE.RANDOM == mode:
        model_indices = list(accuracies.items())
        random.shuffle(model_indices)
        acc_sorted = dict(model_indices)

    else:
        raise ValueError('Invalid performance mode.')

    model_indices = list(acc_sorted.keys())
    if MODE.DEBUG:
        # print('model indices:', model_indices)
        print('re-organized:', acc_sorted)

    upperbounds = get_ideal_accuracy(model_indices, pred_matrix, labels)

    return upperbounds

"""
Computing test accuracy for a model
"""
def compute_model_accuracy(pred_probabilities, true_labels):
    nb_of_hits = 0.
    nb_of_samples = true_labels.shape[0]
    for i in range(nb_of_samples):
        if (np.argmax(pred_probabilities[i]) == true_labels[i]):
            nb_of_hits += 1.

    return round(nb_of_hits/nb_of_samples, 4)

"""
Computing the test accuracy upper bounds for k weak defenses
"""
def get_ideal_accuracy(model_indices, pred_matrix, true_labels):
    nb_of_weak_defenses = len(model_indices)
    nb_of_samples = true_labels.shape[0]
    # print('number of weak defenses: ', nb_of_weak_defenses)

    upperbounds = {}
    for k in range(1, nb_of_weak_defenses + 1):
        nb_of_hits = 0.0
        # indices = model_indices[: k]
        for sample_id in range(nb_of_samples):
            # pred_set = set()
            for i in range(k):
                model_id = model_indices[i]

                # pred_set.add(np.argmax(pred_matrix[model_id][sample_id]))
                if (true_labels[sample_id] == np.argmax(pred_matrix[model_id][sample_id])):
                    nb_of_hits += 1.
                    break
            # print('sample({}): {} in {}?'.format(sample_id, true_labels[sample_id], pred_set))

            # if (true_labels[sample_id] in pred_set):
            #     nb_of_hits += 1

        upperbounds[k] = round(nb_of_hits/nb_of_samples, 4)

    return upperbounds
