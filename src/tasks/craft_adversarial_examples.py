"""
This is the script to craft adversarial examples.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import numpy as np

from attacks.attacker import get_adversarial_examples
from data.data import load_data
from models.transformation import transform
from utils.config import ATTACK, DATA, MODE, TRANSFORMATION
from utils.file import save_adv_examples

def reset(X, trans_type):
    if trans_type == TRANSFORMATION.rotate90:
        X = transform(X, TRANSFORMATION.rotate270)
    elif trans_type == TRANSFORMATION.rotate270:
        X = transform(X, TRANSFORMATION.rotate90)
    elif trans_type == TRANSFORMATION.rotate180:
        X = transform(X, TRANSFORMATION.rotate180)

    return X


def craft(dataset, gen_test=True, method=ATTACK.FGSM, trans_type=TRANSFORMATION.clean):
    print('loading original images...')

    if gen_test:
        # generate for test set
        _, (X, Y) = load_data(dataset)
        prefix = 'test'
    else:
        # generate for train set (the last 20% of the original train set)
        (X, Y), _ = load_data(dataset)
        nb_trainings = int(X.shape[0] * 0.8)
        X = X[nb_trainings:]
        Y = Y[nb_trainings:]
        prefix = 'val'

    """
    In debugging mode, crafting for 50 samples.
    """
    if MODE.DEBUG:
        X = X[:30]
        Y = Y[:30]

    X = transform(X, trans_type)
    model_name = 'model-{}-cnn-{}'.format(dataset, trans_type)

    if method == ATTACK.FGSM:
        for eps in ATTACK.get_fgsm_eps():
            print('{}: (eps={})'.format(method.upper(), eps))
            X_adv, _ = get_adversarial_examples(model_name, method, X, Y, eps=eps)

            attack_params = 'eps{}'.format(int(1000 * eps))

            reset(X, trans_type)
            reset(X_adv, trans_type)
            save_adv_examples(X_adv, prefix=prefix, dataset=dataset, transformation=trans_type,
                              attack_method=method, attack_params=attack_params)
    elif method == ATTACK.BIM:
        for ord in ATTACK.get_bim_norm():
            for nb_iter in ATTACK.get_bim_nbIter():
                for eps in ATTACK.get_bim_eps(ord):
                    print('{}: (ord={}, nb_iter={}, eps={})'.format(method.upper(), ord, nb_iter, eps))
                    X_adv, _ = get_adversarial_examples(model_name, method, X, Y,
                                                        ord=ord, nb_iter=nb_iter, eps=eps)

                    if ord == np.inf:
                        norm = 'inf'
                    else:
                        norm = ord
                    attack_params = 'ord{}_nbIter{}_eps{}'.format(norm, nb_iter, int(1000 * eps))
                    reset(X, trans_type)
                    reset(X_adv, trans_type)
                    save_adv_examples(X_adv, prefix=prefix, dataset=dataset, transformation=trans_type,
                                      attack_method=method, attack_params=attack_params)
    elif method == ATTACK.DEEPFOOL:
        for order in [2]:
            for overshoot in ATTACK.get_df_overshoots(order):
                print('attack {} -- order: {}; overshoot: {}'.format(method.upper(), order, overshoot))
                X_adv, _ = get_adversarial_examples(model_name, method, X, Y,
                                                    ord=order, overshoot=overshoot)

                attack_params = 'l{}_overshoot{}'.format(order, int(overshoot))
                reset(X, trans_type)
                reset(X_adv, trans_type)
                save_adv_examples(X_adv, prefix=prefix, bs_samples=X, dataset=dataset, transformation=trans_type,
                                  attack_method=method, attack_params=attack_params)

    elif method == ATTACK.CW_L2:
        binary_search_steps = 16 #9
        cw_batch_size = 2 #1
        initial_const = 1 #10

        for learning_rate in ATTACK.get_cwl2_lr():
            for max_iter in ATTACK.get_cwl2_maxIter():
                print('{}: (ord={}, max_iterations={})'.format(method.upper(), 2, max_iter))
                X_adv, _ = get_adversarial_examples(model_name, method, X, Y,
                                                    ord=2, max_iterations=max_iter,
                                                    binary_search_steps=binary_search_steps, cw_batch_size=cw_batch_size,
                                                    initial_const=initial_const, learning_rate=learning_rate)

                attack_params = 'lr{}_maxIter{}'.format(int(learning_rate * 1000), max_iter)
                reset(X, trans_type)
                reset(X_adv, trans_type)
                save_adv_examples(X_adv, prefix=prefix, bs_samples=X, dataset=dataset, transformation=trans_type,
                                  attack_method=method, attack_params=attack_params)

    elif method == ATTACK.CW_Linf:
        initial_const = 1e-5
        # X *= 255.

        for learning_rate in ATTACK.get_cwl2_lr():
            for max_iter in ATTACK.get_cwl2_maxIter():
                print('{}: (ord={}, max_iterations={})'.format(method.upper(), np.inf, max_iter))
                X_adv, _ = get_adversarial_examples(model_name, method, X, Y,
                                                    max_iterations=max_iter,
                                                    initial_const=initial_const,
                                                    learning_rate=learning_rate)

                attack_params = 'lr{}_maxIter{}'.format(int(learning_rate * 10), max_iter)
                reset(X, trans_type)
                reset(X_adv, trans_type)
                save_adv_examples(X_adv, prefix=prefix, bs_samples=X, dataset=dataset, transformation=trans_type,
                                  attack_method=method, attack_params=attack_params)

    elif method == ATTACK.CW_L0:
        initial_const = 1e-5

        for learning_rate in ATTACK.get_cwl2_lr():
            for max_iter in ATTACK.get_cwl2_maxIter():
                print('{}: (ord={}, max_iterations={})'.format(method.upper(), np.inf, max_iter))
                X_adv, _ = get_adversarial_examples(model_name, method, X, Y,
                                                    max_iterations=max_iter,
                                                    initial_const=initial_const,
                                                    learning_rate=learning_rate)

                attack_params = 'lr{}_maxIter{}'.format(int(learning_rate * 10), max_iter)
                reset(X, trans_type)
                reset(X_adv, trans_type)
                save_adv_examples(X_adv, prefix=prefix, bs_samples=X, dataset=dataset, transformation=trans_type,
                                  attack_method=method, attack_params=attack_params)

    elif method == ATTACK.JSMA:
        for theta in ATTACK.get_jsma_theta():
            for gamma in ATTACK.get_jsma_gamma():
                print('{}: (theta={}, gamma={})'.format(method.upper(), theta, gamma))
                X_adv, _ = get_adversarial_examples(model_name, method, X, Y,
                                                    theta=theta, gamma=gamma)

                attack_params = 'theta{}_gamma{}'.format(int(100 * theta), int(100 * gamma))
                reset(X, trans_type)
                reset(X_adv, trans_type)
                save_adv_examples(X_adv, prefix=prefix, bs_samples=X, dataset=dataset, transformation=trans_type,
                                  attack_method=method, attack_params=attack_params)

    elif method == ATTACK.PGD:
        nb_iter = 1000
        eps_iter = 0.05 #0.01

        for eps in ATTACK.get_pgd_eps():
            if eps < 0.05:
                eps_iter = 0.01
            elif eps <= 0.01:
                eps_iter = 0.005
            X_adv, _ = get_adversarial_examples(model_name, method, X, Y,
                                                eps=eps, nb_iter=nb_iter, eps_iter=eps_iter)
            attack_params = 'eps{}_nbIter{}_epsIter{}'.format(int(1000 * eps), nb_iter, int(1000 * eps_iter))
            reset(X, trans_type)
            reset(X_adv, trans_type)
            save_adv_examples(X_adv, prefix=prefix, bs_samples=X, dataset=dataset, transformation=trans_type,
                              attack_method=method, attack_params=attack_params)

    elif method == ATTACK.ONE_PIXEL:
        for pixel_counts in ATTACK.get_op_pxCnt():
            for max_iter in ATTACK.get_op_maxIter():
                for pop_size in ATTACK.get_op_popsize():
                    attack_params = {
                        'pixel_counts': pixel_counts,
                        'max_iter': max_iter,
                        'pop_size': pop_size
                    }
                    X_adv, _ = get_adversarial_examples(model_name, method, X, Y, **attack_params)
                    X_adv = np.asarray(X_adv)
                    attack_params = 'pxCount{}_maxIter{}_popsize{}'.format(pixel_counts, max_iter, pop_size)
                    reset(X, trans_type)
                    reset(X_adv, trans_type)
                    save_adv_examples(X_adv, prefix=prefix, bs_samples=X, dataset=dataset, transformation=trans_type,
                                      attack_method=method, attack_params=attack_params)
    elif method == ATTACK.MIM:
        for eps in ATTACK.get_mim_eps():
            for nb_iter in ATTACK.get_mim_nbIter():
                attack_params = {
                    'eps': eps,
                    'nb_iter': nb_iter
                }

                X_adv, _ = get_adversarial_examples(model_name, method, X, Y, **attack_params)
                attack_params = 'eps{}_nbIter{}'.format(int(eps * 100), nb_iter)
                reset(X, trans_type)
                reset(X_adv, trans_type)
                save_adv_examples(X_adv, prefix=prefix, bs_samples=X, dataset=dataset, transformation=trans_type,
                                  attack_method=method, attack_params=attack_params)

    del X
    del Y


def main(dataset, gen_test, attack_method):
    # trans_types = TRANSFORMATION.supported_types()
    trans_types = [TRANSFORMATION.clean]
    for trans in trans_types:
        try:
            craft(dataset, gen_test, attack_method, trans)
        except (FileNotFoundError, OSError) as e:
            print('Failed to load model [{}]: {}.'.format(trans, e))
            continue

if __name__ == '__main__':
    """
    switch on debugging mode
    """
    # MODE.debug_on()
    MODE.debug_off()
    gen_test = True
    main(DATA.mnist, gen_test, ATTACK.DEEPFOOL)
