"""
Entrance of generating adversarial examples.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
import logging

import time

from utils.config import *
import attacks.whitebox as whitebox
import attacks.one_pixel as one_pixel

logger = logging.getLogger('attacks.attacker...')
logger.setLevel(logging.INFO)

def get_adversarial_examples(model_name, attack_method, X, Y, **kwargs):
    dataset = model_name.split('-')[1]

    logger.info('Crafting adversarial examples using {} method...'.format(attack_method.upper()))
    X_adv = None

    if (attack_method == ATTACK.FGSM):
        eps = kwargs.get('eps', 0.25)
        attack_params = {
            'eps': eps,
            'ord': np.inf,
            'clip_min': 0.,
            'clip_max': 1.
        }
        logger.info('{}: (eps={})'.format(attack_method.upper(), eps))

        start_time = time.time()
        X_adv, Y = whitebox.generate(model_name, X, Y, attack_method, attack_params)
        duration = time.time() - start_time
        logger.info('Time cost: {}'.format(duration))

    elif (attack_method == ATTACK.BIM):
        # iterative fast gradient method
        eps = kwargs.get('eps', 0.25)
        nb_iter = kwargs.get('nb_iter', 100)
        ord = kwargs.get('ord', np.inf)

        """
        Cleverhans requires an eps_iter that is smaller than the eps. 
        By default, eps_iter=0.05, so, update eps_iter for small epsilons.
        """
        if eps < 0.005:
            eps_iter = 0.001
        elif eps < 0.05:
            eps_iter = 0.005
        else:
            # for big enough eps, use the default setting
            eps_iter = 0.05
        attack_params = {
            'eps': eps,
            'eps_iter': eps_iter,
            'nb_iter': nb_iter,
            'ord': ord,
            'clip_min': 0.,
            'clip_max': 1.
        }

        logger.info('{}: (ord={}, nb_iter={}, eps={})'.format(attack_method.upper(), ord, nb_iter, eps))
        start_time = time.time()
        X_adv, Y = whitebox.generate(model_name, X, Y, attack_method, attack_params)

        duration = time.time() - start_time
        print('Time cost: {}'.format(duration))
    elif (attack_method == ATTACK.DEEPFOOL):
        # Images for inception classifier are normalized to be in [0, 255] interval.
        # max_iterations = kwargs.get('max_iterations', 100)
        max_iterations = 100
        ord = kwargs.get('ord', 2)
        overshoot = kwargs.get('overshoot', 1.0)

        attack_params = {
            'ord': ord,
            'max_iterations': max_iterations,
            'nb_candidate': Y.shape[1],
            'overshoot': overshoot,
            'clip_min': 0.,
            'clip_max': 1.
        }

        logger.info('{}: (max_iterations={})'.format(attack_method.upper(), max_iterations))
        start_time = time.time()
        X_adv, Y = whitebox.generate(model_name, X, Y, attack_method, attack_params)
        duration = time.time() - start_time
        print('Time cost: {}'.format(duration))

    elif (attack_method == ATTACK.CW_L2):
        binary_search_steps = kwargs.get('binary_search_steps', 10)
        batch_size = kwargs.get('cw_batch_size', 2)
        initial_const = kwargs.get('initial_const', 10)
        learning_rate = kwargs.get('learning_rate', 0.1)
        max_iterations = kwargs.get('max_iterations', 100)
        ord = kwargs.get('ord', 2)

        attack_params = {
            'ord': ord,
            'batch_size': batch_size,
            'binary_search_steps': binary_search_steps,
            'initial_const': initial_const,
            'learning_rate': learning_rate,
            'max_iterations': max_iterations,
            'clip_min': 0.,
            'clip_max': 1.
        }

        logger.info('{}: (ord={}, max_iterations={})'.format(attack_method.upper(), ord, max_iterations))

        start_time = time.time()
        X_adv, Y = whitebox.generate(model_name, X, Y, attack_method, attack_params)
        duration = time.time() - start_time
        logger.info('Time cost: {}'.format(duration))

    elif (attack_method == ATTACK.CW_Linf):
        decrease_factor = kwargs.get('decrease_factor', 0.9)
        initial_const = kwargs.get('initial_const', 1e-5)
        learning_rate = kwargs.get('learning_rate', 0.1)
        largest_const = kwargs.get('largest_const', 2e+1)
        max_iterations = kwargs.get('max_iterations', 1000)
        reduce_const = False
        const_factor = 3.0

        attack_params = {
            # 'descrease_factor': decrease_factor,
            'initial_const': initial_const,
            'learning_rate': learning_rate,
            'max_iterations': max_iterations,
            'largest_const': largest_const,
            'reduce_const': reduce_const,
            'const_factor': const_factor,
            'clip_min': 0.,
            'clip_max': 1.
        }

        logger.info('{}: (ord={}, max_iterations={})'.format(attack_method.upper(), np.inf, max_iterations))

        start_time = time.time()
        X_adv, Y = whitebox.generate(model_name, X, Y, attack_method, attack_params)
        duration = time.time() - start_time
        logger.info('Time cost: {}'.format(duration))

    elif (attack_method == ATTACK.JSMA):
        theta = kwargs.get('theta', 0.6)
        gamma = kwargs.get('gamma', 0.5)
        attack_params = {
            'theta': theta,
            'gamma': gamma,
            'clip_min': 0.,
            'clip_max': 1.
        }

        logger.info('{}: (theta={}, gamma={})'.format(attack_method.upper(), theta, gamma))
        start_time = time.time()
        X_adv, Y = whitebox.generate(model_name, X, Y, attack_method, attack_params)
        duration = time.time() - start_time
        logger.info('Time cost: {}'.format(duration))

    elif (attack_method == ATTACK.PGD):
        eps = kwargs.get('eps', 0.3)
        nb_iter = kwargs.get('nb_iter', 40)
        eps_iter = kwargs.get('eps_iter', 0.01)

        attack_params = {
            'eps': eps,
            'clip_min': 0.,
            'clip_max': 1.,
            'nb_iter': nb_iter,
            'eps_iter': eps_iter
        }

        start_time = time.time()
        X_adv, Y = whitebox.generate(model_name, X, Y, attack_method, attack_params)
        duration = time.time() - start_time
        logger.info('Time cost: {}'.format(duration))

    elif (attack_method == ATTACK.ONE_PIXEL):
        # one-pixel was implemented separately.
        targeted = kwargs.get('targeted', False)
        pixel_counts = kwargs.get('pixel_counts', 3)
        max_iter = kwargs.get('max_iter', 10)
        pop_size = kwargs.get('pop_size', 10)

        attack_params = {
            'targeted': targeted,
            'pixel_counts': pixel_counts,
            'max_iter': max_iter,
            'pop_size': pop_size,
            'clip_min': 0.,
            'clip_max': 1.,
        }

        start_time = time.monotonic()
        X_adv, Y = one_pixel.generate(model_name, X, Y, attack_params)
        duration = time.monotonic() - start_time
        logger.info('Time cost: {}'.format(duration))

    elif (attack_method == ATTACK.MIM):
        eps = kwargs.get('eps', 0.3)
        eps_iter = kwargs.get('eps_iter', 0.06)
        nb_iter = kwargs.get('nb_iter', 10)
        decay_factor = kwargs.get('decay_factor', 1.0)
        y_target = kwargs.get('y_target', None)

        attack_params = {
            'eps': eps,
            'eps_iter': eps_iter,
            'nb_iter': nb_iter,
            'ord': np.inf,
            'decay_factor': decay_factor,
            'y_target': y_target,
            'clip_min': 0.,
            'clip_max':1.
        }

        start_time = time.time()
        X_adv, Y = whitebox.generate(model_name, X, Y, attack_method, attack_params)
        duration = time.time() - start_time
        logger.info('Time cost: {}'.format(duration))

    print('*** SHAPE: {}'.format(np.asarray(X_adv).shape))
    return X_adv, Y
