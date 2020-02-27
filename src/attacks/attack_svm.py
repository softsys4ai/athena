"""

@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import os
import numpy as np

from art.classifiers import SklearnClassifier
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.attacks.evasion.deepfool import DeepFool
from art.attacks.evasion.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.evasion.saliency_map import SaliencyMapMethod
from art.attacks.evasion.carlini import CarliniL2Method, CarliniLInfMethod
from art.attacks.evasion.iterative_method import BasicIterativeMethod

from utils.config import *


def get_adversarial_examples(X, Y, model, nb_classes, attack=None):
    assert model is not None
    assert attack is not None

    art_classifier = SklearnClassifier(model=model, clip_values=(0, nb_classes))

    attacker = None
    if attack == ATTACK.PGD:
        attacker = ProjectedGradientDescent(classifier=art_classifier, norm=np.inf, eps=0.2, eps_step=0.1, max_iter=3,
                                            targeted=False, num_random_init=0, batch_size=128)
    elif attack == ATTACK.DEEPFOOL:
        attacker = DeepFool(classifier=art_classifier, max_iter=5, epsilon=1e-6, nb_grads=3, batch_size=1)
    elif attack == ATTACK.FGSM:
        attacker = FastGradientMethod(classifier=art_classifier, norm=np.inf, eps=0.3, targeted=False,
                                      batch_size=128)
    elif attack == ATTACK.BIM:
        attacker = BasicIterativeMethod(classifier=art_classifier, eps=0.3, eps_step=0.1, targeted=False, batch_size=128)
    elif attack == ATTACK.JSMA:
        attacker = SaliencyMapMethod(classifier=art_classifier, theta=0.3, gamma=0.5, batch_size=128)
    elif attack == ATTACK.CW_L2:
        attacker = CarliniL2Method(classifier=art_classifier, learning_rate=0.1)
    elif attack == ATTACK.CW_Linf:
        attacker = CarliniLInfMethod(classifier=art_classifier, learning_rate=0.01)
    else:
        raise NotImplementedError(attack, 'is not implemented.')

    print('Generating [{}] adversarial examples, it will take a while...'.format(attack))
    X_adv = attacker.generate(X, y=Y)

    del attacker
    return X_adv
