"""
Entrance generating adversarial examples.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
import logging

from cleverhans.utils_keras import KerasModelWrapper
import time

from config import *
import attacks.whitebox as whitebox

logger = logging.getLogger('defence_transformers')
logger.setLevel(logging.INFO)

def get_adversarial_examples(model, attack_method, X, Y, **kwargs):
    # wrap keras model
    wrap_model = KerasModelWrapper(model)
    logger.info('Crafting adversarial examples using {} method...'.format(attack_method.upper()))

    if (attack_method == ATTACK.FGSM):
        X = (X - 0.5) * 0.5
        eps = kwargs.get('eps', 0.25)
        attack_params = {
            'eps': eps,
            'ord': np.inf
        }
        logger.info('{}: (eps={})'.format(attack_method.upper(), eps))

        start_time = time.time()
        X_adv, Y = whitebox.generate(wrap_model, attack_method, X, Y, attack_params)
        duration = time.time() - start_time
        logger.info('Time cost: {}'.format(duration))

        X_adv *= 2.

    elif (attack_method == ATTACK.BIM):
        X = (X - 0.5) * 0.5
        # iterative fast gradient method
        eps = kwargs.get('eps', 0.25)
        nb_iter = kwargs.get('nb_iter', 100)
        ord = kwargs.get('ord', np.inf)

        if eps < 0.005:
            raise ValueError('eps must be no less than 0.005.')
        elif eps < 0.05:
            # update eps_iter for small epsilons
            eps_iter = 0.003
        else:
            # otherwise, use the default setting
            eps_iter = 0.05
        attack_params = {
            'eps': eps,
            'eps_iter': eps_iter,
            'nb_iter': nb_iter,
            'ord': ord
        }

        logger.info('{}: (ord={}, nb_iter={}, eps={})'.format(attack_method.upper(), ord, nb_iter, eps))
        start_time = time.time()
        X_adv, Y = whitebox.generate(wrap_model, attack_method, X, Y, attack_params)

        duration = time.time() - start_time
        print('Time cost: {}'.format(duration))

        if eps < 0.01:
            X_adv *= 2.

    elif (attack_method == ATTACK.DEEPFOOL):
        # Images for inception classifier are normalized to be in [0, 255] interval.
        X *= 255.

        max_iterations = kwargs.get('max_iterations', 1)
        ord = kwargs.get('ord', 2)
        attack_params = {
            'ord': ord,
            'max_iterations': max_iterations,
            'nb_candidate': Y.shape[1]
        }

        logger.info('{}: (max_iterations={})'.format(attack_method.upper(), max_iterations))
        start_time = time.time()
        X_adv, Y = whitebox.generate(wrap_model, attack_method, X, Y, attack_params)
        duration = time.time() - start_time
        print('Time cost: {}'.format(duration))

        X_adv /= 255.
    elif (attack_method == ATTACK.CW):
        max_iterations = kwargs.get('max_iterations', 100)
        ord = kwargs.get('ord', 2)

        attack_params = {
            'ord': ord,
            'max_iterations': max_iterations
        }

        logger.info('{}: (ord={}, max_iterations={})'.format(attack_method.upper(), ord, max_iterations))

        start_time = time.time()
        X_adv, Y = whitebox.generate(wrap_model, attack_method, X, Y, attack_params)
        duration = time.time() - start_time
        logger.info('Time cost: {}'.format(duration))

    elif (attack_method == ATTACK.JSMA):
        X = (X - 0.5) * 0.5
        theta = kwargs.get('theta', 0.6)
        gamma = kwargs.get('gamma', 0.5)
        attack_params = {
            'theta': theta,
            'gamma': gamma
        }

        logger.info('{}: (theta={}, gamma={})'.format(attack_method.upper(), theta, gamma))
        start_time = time.time()
        X_adv, Y = whitebox.generate(wrap_model, attack_method, X, Y, attack_params)
        duration = time.time() - start_time
        logger.info('Time cost: {}'.format(duration))

        X_adv *= 2

    return X_adv, Y
