"""
Entrance of generating adversarial examples.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
import logging

import time

from config import *
import attacks.whitebox as whitebox
import attacks.one_pixel as one_pixel

logger = logging.getLogger('defence_transformers')
logger.setLevel(logging.INFO)

def get_adversarial_examples(model_name, attack_method, X, Y, **kwargs):
    logger.info('Crafting adversarial examples using {} method...'.format(attack_method.upper()))
    X_adv = None

    if (attack_method == ATTACK.FGSM):
        eps = kwargs.get('eps', 0.25)
        attack_params = {
            'eps': eps,
            'ord': np.inf
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
            'ord': ord
        }

        logger.info('{}: (ord={}, nb_iter={}, eps={})'.format(attack_method.upper(), ord, nb_iter, eps))
        start_time = time.time()
        X_adv, Y = whitebox.generate(model_name, attack_method, X, Y, attack_params)

        duration = time.time() - start_time
        print('Time cost: {}'.format(duration))
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
        X_adv, Y = whitebox.generate(model_name, attack_method, X, Y, attack_params)
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
        X_adv, Y = whitebox.generate(model_name, attack_method, X, Y, attack_params)
        duration = time.time() - start_time
        logger.info('Time cost: {}'.format(duration))

    elif (attack_method == ATTACK.JSMA):
        theta = kwargs.get('theta', 0.6)
        gamma = kwargs.get('gamma', 0.5)
        attack_params = {
            'theta': theta,
            'gamma': gamma
        }

        logger.info('{}: (theta={}, gamma={})'.format(attack_method.upper(), theta, gamma))
        start_time = time.time()
        X_adv, Y = whitebox.generate(model_name, attack_method, X, Y, attack_params)
        duration = time.time() - start_time
        logger.info('Time cost: {}'.format(duration))

    elif (attack_method == ATTACK.ONE_PIXEL):
        # pixel format (x, y, r, g, b)
        samples = kwargs.get('samples', 500)
        pixel_counts = kwargs.get('pixel_counts', tuple([1]))
        max_iterations = kwargs.get('max_iterations', 100)
        targeted = kwargs.get('targeted', False)
        population = kwargs.get('population', 400)
        attack_params = {
            'samples': samples,
            'pixel_counts': pixel_counts,
            'max_iterations':max_iterations,
            'targeted': targeted,
            'population': population
        }

        logger.info('{}: (samples={}, pixel_counts={}, max_iterations={}, target={}, population={})'.format(attack_method.upper(),
                                                                                                            samples,
                                                                                                            pixel_counts,
                                                                                                            max_iterations,
                                                                                                            targeted,
                                                                                                            population
                                                                                        )
                    )
        start_time = time.time()
        X_adv, Y = one_pixel.generate(model_name, attack_method, X, Y, attack_params)
        duration = time.time() - start_time
        logger.info('Time cost: {}'.format(duration))


    return X_adv, Y
