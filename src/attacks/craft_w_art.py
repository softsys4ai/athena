"""
Craft adversarial examples for an SVM model.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

from art.attacks.evasion.carlini import CarliniL2Method, CarliniLInfMethod
from art.attacks.evasion.deepfool import DeepFool
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.attacks.evasion.iterative_method import BasicIterativeMethod
from art.attacks.evasion.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.evasion.saliency_map import SaliencyMapMethod

from utils.config import *


def craft(X, Y, art_classifier, attack=None,
          **attack_params):
    assert art_classifier is not None
    assert attack is not None

    attacker = None
    if attack == ATTACK.PGD:
        eps = attack_params.get('eps', 0.2)
        eps_step = attack_params.get('eps_step', eps/5.)
        max_iter = attack_params.get('max_iter', 3)
        targeted = attack_params.get('targeted', False)
        batch_size = attack_params.get('batch_size', 128)

        attacker = ProjectedGradientDescent(classifier=art_classifier, norm=np.inf, eps=eps, eps_step=eps_step,
                                            max_iter=max_iter, targeted=targeted, num_random_init=0, batch_size=batch_size)

    elif attack == ATTACK.DEEPFOOL:
        eps = attack_params.get('eps', 1e-6)
        max_iter = attack_params.get('max_iter', 5)
        nb_grads = attack_params.get('nb_grads', 3)
        batch_size = attack_params.get('batch_size', 1)

        attacker = DeepFool(classifier=art_classifier, max_iter=max_iter, epsilon=eps, nb_grads=nb_grads,
                            batch_size=batch_size)

    elif attack == ATTACK.FGSM:
        eps = attack_params.get('eps', 0.3)
        targeted = attack_params.get('targeted', False)
        batch_size = attack_params.get('batch_size', 128)

        attacker = FastGradientMethod(classifier=art_classifier, norm=np.inf, eps=eps, targeted=targeted,
                                      batch_size=batch_size)

    elif attack == ATTACK.BIM:
        eps = attack_params.get('eps', 0.3)
        eps_step = attack_params.get('eps_step', eps/5.)
        norm = attack_params.get('norm', np.inf)
        targeted = attack_params.get('targeted', False)
        batch_size = attack_params.get('batch_size', 128)

        attacker = BasicIterativeMethod(classifier=art_classifier, norm=norm,
                                        eps=eps, eps_step=eps_step,
                                        targeted=targeted, batch_size=batch_size)

    elif attack == ATTACK.JSMA:
        theta = attack_params.get('theta', 0.3)
        gamma = attack_params.get('gamma', 0.5)
        batch_size = attack_params.get('batch_size', 128)

        attacker = SaliencyMapMethod(classifier=art_classifier, theta=theta, gamma=gamma, batch_size=batch_size)

    elif attack == ATTACK.CW_L2:
        lr = attack_params.get('lr', 0.1)
        bsearch_steps = attack_params.get('bsearch_steps', 10)

        attacker = CarliniL2Method(classifier=art_classifier, learning_rate=lr,
                                   binary_search_steps=bsearch_steps)

    elif attack == ATTACK.CW_Linf:
        lr = attack_params.get('lr', 0.01)

        attacker = CarliniLInfMethod(classifier=art_classifier, learning_rate=lr)

    else:
        raise NotImplementedError(attack, 'is not implemented.')

    print('Generating [{}] adversarial examples, it will take a while...'.format(attack))
    X_adv = attacker.generate(X, y=Y)

    del attacker
    return X_adv
