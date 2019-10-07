"""

@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import models

from utils.config import *
import os
import utils.file as file
from utils.plot import *
from utils.csv_headers import IdealModelEvalHeaders as headers
from scripts.evaluate_models import eval_ideal_model
from data import normalize
from transformation import transform
from evaluation.compute_accuracy import get_test_accuracies, PERFORM_MODE

PROJECT_DIR = PATH.PROJECT_DIR
ANALYSE_DIR = '{}/evaluation_results/2019-09-06_02-41-18'.format(PROJECT_DIR)
RESULTS_DIR = '{}/result_of_post_analysis'.format(ANALYSE_DIR)
PREDICTION_DIR = '{}/prediction_result'.format(ANALYSE_DIR)
EVALUATION_DIR = '{}/evaluation'.format(ANALYSE_DIR)

FIGURES_DIR = '{}/figures'.format(ANALYSE_DIR)

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

def plot_ideal_accuracy(attack):
    file_name = 'acc-ideal_model-{}-{}.csv'.format('mnist', attack)
    file_path = os.path.join(EVALUATION_DIR, file_name)

    if not os.path.isfile(file_path):
        print('Cannot find {}, computing ideal test accuracy...'.format(file_path))
        eval_ideal_model(attack)

    # load data
    print('Loading data...')
    data = file.csv2dict(file_path)

    data_of_lines = {}
    data_of_lines[headers.NUM_OF_WEAK_DEFENSES.value] = data[headers.NUM_OF_WEAK_DEFENSES.value]
    data_of_lines[headers.TOP_K.value] = data[headers.TOP_K.value]
    data_of_lines[headers.BOTTOM_K.value] = data[headers.BOTTOM_K.value]
    data_of_lines[headers.RANDK_AVG.value] = data[headers.RANDK_AVG.value]
    # data_of_lines[headers.RANDK_UPPERBOUND.value] = data[headers.RANDK_UPPERBOUND.value]
    # data_of_lines[headers.RANDK_LOWERBOUND.value] = data[headers.RANDK_LOWERBOUND.value]
    data_of_lines[headers.GAP.value] = data[headers.GAP.value]

    certainty_borders = [(data[headers.NUM_OF_WEAK_DEFENSES.value],
                         data[headers.RANDK_UPPERBOUND.value],
                         data[headers.RANDK_LOWERBOUND.value])]

    print('data of lines:')
    print(data_of_lines)
    print('filled area:')
    print(certainty_borders)
    plot_scatter_with_certainty(data_of_lines, certainty_borders, title='PGD (eps: 0.75)',
                                ylabel='Test Accuracy', save=False, legend_loc=LEGEND_LOCATION.upper_center)

def main():
    """
    Evaluate ideal model
    """
    attack = attacks[13]
    plot_ideal_accuracy(attack)

if __name__ == '__main__':
    MODE.debug_off()
    main()
