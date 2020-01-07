"""
Implement the evaluation to white-box and gray-box threat models.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

from enum import Enum
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import copy
import random
import time
import tensorflow as tf
import tensorflow.keras as keras

import attacks.attacker
import attacks.whitebox
from attacks.attacker import attack_whitebox
from data.data import load_data
from models.models import load_model
from utils.plot import plot_image
from models.transformation import transform
from utils.config import *
from utils.file import save_adv_examples
import utils.measure as measure

class ATTACK_STRATEGY(Enum):
    MOST_EFFECTIVE = 'best'
    LEAST_EFFECTIVE = 'worst'
    RANDOM = 'random'

def get_waive_list(dataset=DATA.mnist):
    if dataset == DATA.mnist:
        DATA.set_current_dataset_name(DATA.mnist)
        waives = TRANSFORMATION.QUANTIZATIONS
        waives.extend([TRANSFORMATION.denoise_nl_means])
    else:
        # TODO: add the items later
        waives = []

    return waives

def get_attack_params(attack_method=ATTACK.FGSM):
    # TODO: support schedulable attack params
    attack_params = {
        'clip_min': 0.,
        'clip_max': 1.
    }

    if attack_method == ATTACK.FGSM:
        attack_params['eps'] = 0.10
        attack_params['ord'] = np.inf
    elif attack_method == ATTACK.BIM:
        attack_params['eps'] = 0.1
        attack_params['nb_iter'] = 100
        attack_params['ord'] = 2
    elif attack_method == ATTACK.DEEPFOOL:
        attack_params['max_iterations'] = 100
        attack_params['ord'] = 2
        attack_params['overshoot'] = 1.0
    elif attack_method == ATTACK.CW_L2:
        attack_params['initial_const'] = 10
        attack_params['learning_rate'] = 0.1
        attack_params['max_iterations'] = 100
    elif attack_method == ATTACK.JSMA:
        attack_params['theta'] = 0.3
        attack_params['gamma'] = 0.3
    elif attack_method == ATTACK.PGD:
        attack_params['eps'] = 0.1
    elif attack_method == ATTACK.ONE_PIXEL:
        attack_params['pixel_counts'] = 1
        attack_params['max_iter'] = 10
        attack_params['pop_size'] = 10
    elif attack_method == ATTACK.MIM:
        attack_params['eps'] = 0.1
        attack_params['nb_iter'] = 10

    return attack_params


def init_candidate_targets(weak_defenses_file):
    """
    :return: the list of candidate targets, which are
            sorted descendingly according to their effectiveness.
    """
    # load all models
    candidates = {}
    waives = get_waive_list()

    with open(os.path.join(PATH.MODEL, weak_defenses_file), 'r') as weak_defense_list:
        for weak_defense in weak_defense_list:
            weak_defense = weak_defense[:-1] # remove the \newline character
            print('...Loading weak defense {}'.format(weak_defense))
            name = weak_defense.split('.')[0].split('-')[-1]
            if name in waives:
                print('Ignore [{}]'.format(name))
                continue

            candidates[name] = load_model(weak_defense)

    return candidates


def sort_candidates(candidates, attacker):
    # TODO: sort candidates by their efficiency for certain attacker
    return candidates


def pick_target_model(candidate_names, strategy=ATTACK_STRATEGY.RANDOM.value):
    """
    :param candidate_names: the list of candidate targets, sorted by their effectiveness
    :param strategy: strategy to choose a target
    :return: a target model
    """
    if strategy == ATTACK_STRATEGY.MOST_EFFECTIVE.value:
        return candidate_names[0]
    elif strategy == ATTACK_STRATEGY.LEAST_EFFECTIVE.value:
        return candidate_names[-1]
    elif strategy == ATTACK_STRATEGY.RANDOM.value:
        name = random.choice(candidate_names)
        print('>>> Pick target [{}]'.format(name))
        return name


def reset(X, trans_type):
    if trans_type == TRANSFORMATION.rotate90:
        X = transform(X, TRANSFORMATION.rotate270)
    elif trans_type == TRANSFORMATION.rotate270:
        X = transform(X, TRANSFORMATION.rotate90)
    elif trans_type == TRANSFORMATION.rotate180:
        X = transform(X, TRANSFORMATION.rotate180)

    return X

def get_perturb_upperbound(attacker=ATTACK.FGSM):
    orig_file = 'test_BS-{}-clean.npy'.format(DATA.CUR_DATASET_NAME)
    orig_file = os.path.join(PATH.ADVERSARIAL_FILE, orig_file)

    adv_file = 'test_AE-{}-cnn-clean'.format(DATA.CUR_DATASET_NAME)
    adv_file = os.path.join(PATH.ADVERSARIAL_FILE, adv_file)

    if attacker == ATTACK.FGSM:
        # use the strongest as upperbound
        adv_file = '{}-{}.npy'.format(adv_file, ATTACK.get_fgsm_AETypes()[-1])
    elif attacker == ATTACK.JSMA:
        adv_file = '{}-{}.npy'.format(adv_file, ATTACK.get_jsma_AETypes()[-1])

    print('Loading original file [{}]...'.format(orig_file))
    X = np.load(orig_file)
    print('Loading perturbed file [{}]...'.format(adv_file))
    X_adv = np.load(adv_file)
    upperbound = np.round(measure.frobenius_norm(X1=X_adv, X2=X), 2)

    print('Upperbound of {} perturbation: {}'.format(attacker, upperbound))
    return upperbound


def attack_single(sess, target_model, attacker, x, y, **attack_params):
    if len(x.shape) < 4:
        x = np.expand_dims(x, axis=0)

    if len(y.shape) < 4:
        y = np.expand_dims(y, axis=0)

    x_adv, _ =  attack_whitebox(sess, target_model, attacker, x, y, **attack_params)

    return x_adv


def generate_single(sess, x, y, attacker=ATTACK.FGSM,
                    candidates=None,
                    attack_count=None,
                    max_perturb=get_perturb_upperbound(),
                    strategy=ATTACK_STRATEGY.RANDOM.value):
    # candidate_names = candidates.copy()
    candidate_names = copy.deepcopy(list(candidates.keys()))
    fooled = []

    attack_params = get_attack_params(attacker)
    x_adv = x
    perturbed_norm = measure.frobenius_norm(x_adv, x)

    max_iteration = len(candidate_names)
    iter = 0
    while ((len(fooled) < attack_count) and
           (iter < max_iteration)):
        # generate adversarial example for target model
        print('ITERATION {}: candidates/fooled ::: {}/{}'.format(iter, len(candidate_names), len(fooled)))
        iter += 1
        target_name = pick_target_model(candidate_names, strategy)
        transformation = target_name.split('.')[0].split('-')[-1]
        x_trans = transform(x_adv, transformation)
        if len(x_trans.shape) < 4:
            print('x_trans shape:', x_trans.shape)
            x_trans = np.expand_dims(x_trans, axis=0)

        x_tmp = attack_single(sess, candidates[target_name], attacker, x_trans, y, **attack_params)
        perturbed_norm = measure.frobenius_norm(x_tmp, transform(x, transformation))
        if perturbed_norm >= max_perturb:
            # keep the last x_adv if current one is out of the boundary
            print('out of perturbed boundary, stop.')
            break

        x_adv = reset(x_tmp, transformation)

        if MODE.DEBUG:
            plot_image(x_adv[0], transformation)

        del x_trans

        # filter out candidates that are fooled by x_adv
        true_label = np.argmax(y)
        for cand_name in candidate_names:
            transformation = cand_name.split('.')[0].split('-')[-1]

            # apply transformation
            x_trans = transform(x_adv, transformation)
            pred_label = np.argmax(candidates[cand_name].predict(x_trans))

            if MODE.DEBUG:
                print('prediction: [{}/{}/{}]'.format(transformation, true_label, pred_label))

            if (true_label != pred_label):
                # remove candidate being fooled by x_adv
                candidate_names.remove(cand_name)
                # record only the name of the weak defense
                print('+++ fooled [{}]'.format(cand_name))
                fooled.append(cand_name)
            # release
            del x_trans

        # use current adversarial example as the input of next iteration
        print('')
        del target_name

    return x_adv[0]

def gen_greedy(dataset, attacker=ATTACK.FGSM,
               attack_count=None, strategy=ATTACK_STRATEGY.RANDOM.value):

    config = tf.ConfigProto(intra_op_parallelism_threads=4,
                            inter_op_parallelism_threads=4)
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    candidates = init_candidate_targets('ensemble/mnist_weak_defenses_fsgm.list')

    print('...In total {} weak defenses.'.format(len(candidates)))

    prefix = 'wb' # white-box

    if attack_count == None or attack_count <= 0:
        prefix = 'gb' # gray-box
        attack_count = len(candidates.keys())

    X_adv = []

    _, (X, Y) = load_data(dataset=dataset)

    # generate 500 samples
    batch_size = 100
    nb_samples = Y.shape[0]
    nb_iter = int(nb_samples / batch_size)

    start = time.monotonic()
    for i in range(nb_iter):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, nb_samples)
        print(start_idx, end_idx)
        X_batch = X[start_idx:end_idx]
        Y_batch = Y[start_idx:end_idx]

        print('...In total {} inputs.'.format(Y.shape[0]))
        idx = 0
        for x, y in zip(X_batch, Y_batch):
            print('{}-th input...'.format(idx))

            x = np.expand_dims(x, axis=0)

            strategy = ATTACK_STRATEGY.RANDOM.value

            '''
            generate_single(sess, x, y, attacker=ATTACK.FGSM,
                        candidates=None,
                        attack_count=None,
                        max_perturb=get_perturb_upperbound(),
                        strategy=ATTACK_STRATEGY.RANDOM.value)
            '''
            start_sample = time.monotonic()
            X_adv.append(generate_single(sess, x, y, attacker, candidates, attack_count, strategy=strategy))
            end_sample = time.monotonic()
            print('({}, {})-th sample: {}\n\n'.format(i, idx, (end_sample - start_sample)))
            idx += 1

        save_adv_examples(np.asarray(X_adv), prefix=prefix, bs_samples=X_batch, dataset=dataset,
                          transformation=strategy, attack_method=attacker,
                          attack_params='eps100_batchsize{}_{}'.format(batch_size, i))

    duration = time.monotonic() - start
    print('----------------------------------')
    print('        Summary')
    print('----------------------------------')
    print('Number of inputs:', Y.shape[0])
    print('Adversary:', attacker)
    print('Strategy:', strategy)
    print('Time cost:', duration)

    sess.close()

def gen_separately(dataset, attacker=ATTACK.FGSM, attack_count=None,
                   strategy=ATTACK_STRATEGY.RANDOM.value):
    target_models = init_candidate_targets(attacker)


    pass

def main():
    DATA.set_current_dataset_name(DATA.mnist)
    gen_greedy(dataset=DATA.CUR_DATASET_NAME, attacker=ATTACK.FGSM)

if __name__ == '__main__':
    # MODE.debug_on()
    MODE.debug_off()
    main()
