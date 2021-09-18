"""
Implement attacks on top of IBM Trusted-AI ART 1.2.0.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
import numpy as np
import torch

from art.attacks.evasion.carlini import CarliniL2Method, CarliniLInfMethod
from art.attacks.evasion.deepfool import DeepFool
from art.attacks.evasion.saliency_map import SaliencyMapMethod
from art.attacks.evasion.spatial_transformation import SpatialTransformation
# from art.attacks.evasion.hop_skip_jump import HopSkipJump
from art.attacks.evasion.zoo import ZooAttack

from attacks.evasion.fast_gradient import FastGradientMethod
from attacks.evasion.pgd import ProjectedGradientDescent
from attacks.evasion.bim import BasicIterativeMethod
from attacks.evasion.hsja import HopSkipJump
from attacks.utils import WHITEBOX_ATTACK as ATTACK


def generate(model, data_loader, attack_args, device=None):
    """
    Generate adversarial examples.
    :param model: an instances of art.classifiers.classifier. The targeted model.
    :param data_loader: a tuple of benign samples and corresponding true labels.
    :param attack_args: dictionary. adversarial configurations.
    :param device: string. cuda (for gpu) or cpu.
    :return:
    """
    attack = attack_args.get('attack').lower()
    eot = attack_args.get('eot')

    if eot and attack not in [ATTACK.FGSM.value, ATTACK.PGD.value]:
        raise NotImplementedError("`EOT` is not supported for {} attack yet.".format(attack))

    print(">>> Generating {}(EOT:{}) examples.".format(attack_args.get('description'),
                                                       "ON" if eot else "OFF"))

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images, labels = data_loader

    if attack == ATTACK.FGSM.value:
        return _fgsm(model, images, labels, attack_args)
    elif attack == ATTACK.CW.value:
        return _cw(model, images, labels, attack_args)
    elif attack == ATTACK.PGD.value:
        return _pgd(model, images, labels, attack_args)
    elif attack == ATTACK.BIM.value:
        return _bim(model, images, labels, attack_args)
    elif attack == ATTACK.JSMA.value:
        return _jsma(model, images, labels, attack_args)
    elif attack == ATTACK.DF.value:
        return _df(model, images, labels, attack_args)
    elif attack == ATTACK.MIM.value:
        return _mim(model, images, labels, attack_args)
    elif attack == ATTACK.OP.value:
        return _op(model, images, labels, attack_args)
    elif attack == ATTACK.HOP_SKIP_JUMP.value:
        return _hop_skip_jump(model, images, labels, attack_args)
    elif attack == ATTACK.SPATIAL_TRANS.value:
        return _spatial(model, images, labels, attack_args)
    elif attack == ATTACK.ZOO.value:
        return _zoo(model, images, labels, attack_args)
    else:
        raise ValueError('{} is not supported.'.format(attack))


def _fgsm(model, data, labels, attack_args):
    """
    Fast Gradient Sign Method
    Explaining and Harnessing Adversarial Examples
    by Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy
    ``https://arxiv.org/abs/1412.6572``
    :param model:
    :param data:
    :param labels:
    :param attack_args:
    :param distribution: dictionary. the configurations of distribution (for EOT)
    :return:
    """
    eps = attack_args.get('eps', 0.3)

    targeted = attack_args.get('targeted', False)
    num_random_init = attack_args.get('num_random_init', 0)
    minimal = attack_args.get('minimal', False)

    if attack_args.get("eot"):
        distribution = attack_args.get('distribution', None)
    else:
        distribution = None

    attacker = FastGradientMethod(model, eps=eps, eps_step=eps, targeted=targeted,
                                  num_random_init=num_random_init, minimal=minimal,
                                  distribution=distribution)

    return attacker.generate(data, labels)


def _cw(model, data, labels, attack_args):
    """
    Carlini & Wanger
    Towards Evaluating the Robustness of Neural Networks
    by Nicholas Carlini, David Wagner
    ``https://arxiv.org/abs/1608.04644``
    :param model:
    :param data:
    :param labels:
    :param attack_args:
    :return:
    """
    norm = attack_args.get('norm').lower()

    lr = attack_args.get('lr')
    max_iter = attack_args.get('max_iter', 100)

    # use default values for the following arguments
    confidence = attack_args.get('confidence', 0.0)
    targeted = attack_args.get('targeted', False)
    init_const = attack_args.get('init_const', 0.01)
    max_halving = attack_args.get('max_halving', 5)
    max_doubling = attack_args.get('max_doubling', 5)

    if norm == 'l2':
        print('>>> Generating CW_l2 examples.')
        binary_search_steps = attack_args.get('binary_search_steps', 10)

        attacker = CarliniL2Method(classifier=model, confidence=confidence, targeted=targeted, learning_rate=lr,
                                   binary_search_steps=binary_search_steps, max_iter=max_iter,
                                   initial_const=init_const, max_halving=max_halving,
                                   max_doubling=max_doubling)

    elif norm == 'linf':
        print('>>> Generating CW_linf examples.')
        eps = attack_args.get('eps', 0.3)
        attacker = CarliniLInfMethod(classifier=model, confidence=confidence, targeted=targeted, learning_rate=lr,
                                     max_iter=max_iter, max_halving=max_halving, max_doubling=max_doubling, eps=eps)
    else:
        raise ValueError('Support `l2` and `linf` norms. But found {}'.format(norm))

    return attacker.generate(data, labels)


def _pgd(model, data, labels, attack_args):
    """
    Projected Gradient Descent
    Towards deep learning models resistant to adversarial attacks
    by Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu.
    ``https://arxiv.org/abs/1706.06083``
    :param model:
    :param data:
    :param labels:
    :param attack_args:
    :return:
    """
    eps = attack_args.get('eps', 0.3)
    eps_step = attack_args.get('eps_step', eps/50.)
    max_iter = attack_args.get('max_iter', 200)

    norm = _get_norm_value(attack_args.get('norm', 'linf'))
    targeted = attack_args.get('targeted', False)
    num_random_init = attack_args.get('num_random_init', 0)
    random_eps = attack_args.get('random_eps', False)

    if attack_args.get("eot"):
        distribution = attack_args.get('distribution', None)
    else:
        distribution = None

    attacker = ProjectedGradientDescent(classifier=model, norm=norm, eps=eps, eps_step=eps_step,
                                        max_iter=max_iter, targeted=targeted,
                                        num_random_init=num_random_init, random_eps=random_eps,
                                        distribution=distribution)
    return attacker.generate(data, labels)


def _bim(model, data, labels, attack_args):
    """
    Basic Iteractive Method
    ADVERSARIAL EXAMPLES IN THE PHYSICAL WORLD
    Alexey Kurakin, Ian J. Goodfellow, Samy Bengio
    ``https://arxiv.org/pdf/1607.02533.pdf``
    :param model:
    :param data:
    :param labels:
    :param attack_args:
    :return:
    """
    eps = attack_args.get('eps', 0.3)
    eps_step = attack_args.get('eps_step', eps/50.)
    max_iter = attack_args.get('max_iter', 100)
    norm = _get_norm_value(attack_args.get('norm', 'linf'))

    targeted = attack_args.get('targeted', False)
    attacker = BasicIterativeMethod(classifier=model, norm=norm, eps=eps, eps_step=eps_step,
                                    max_iter=max_iter, targeted=targeted)
    return attacker.generate(data, labels)


def _jsma(model, data, labels, attack_args):
    theta = attack_args.get('theta', 0.15)
    gamma = attack_args.get('gamma', 0.5)

    batch_size = attack_args.get('batch_size', 1)

    attacker = SaliencyMapMethod(classifier=model, theta=theta, gamma=gamma, batch_size=batch_size)
    return attacker.generate(data, labels)


def _df(model, data, labels, attack_args):
    max_iter = attack_args.get('max_iter', 100)
    eps = attack_args.get('eps', 0.01)
    nb_grads = attack_args.get('nb_grads', 10)

    attacker = DeepFool(classifier=model, max_iter=max_iter, epsilon=eps, nb_grads=nb_grads)
    return attacker.generate(data, labels)


def _mim(model, data, labels, attack_args):
    raise NotImplementedError


def _op(model, data, labels, attack_args):
    raise NotImplementedError


def _spatial(model, data, labels, attack_args):
    max_translation = attack_args.get('max_translation', 0.2)
    num_translations = attack_args.get('num_translations', 10)
    max_rotation = attack_args.get('max_rotation', 15)
    num_rotations = attack_args.get('num_rotations', 10)

    if num_rotations <= 0:
        num_rotations = 1

    if num_translations <= 0:
        num_translations = 1

    attacker = SpatialTransformation(classifier=model,
                                     max_translation=max_translation, num_translations=num_translations,
                                     max_rotation=max_rotation, num_rotations=num_rotations)
    return attacker.generate(data, labels)


def _hop_skip_jump(model, data, labels, attack_args):
    norm = _get_norm_value(attack_args.get('norm', 'l2'))
    max_iter = attack_args.get('max_iter', 32)
    max_eval = attack_args.get('max_eval', 10000)
    max_queries = attack_args.get('max_queries', 1000)
    init_eval = attack_args.get('init_eval', 100)
    init_size = attack_args.get('init_size', 100)
    targeted = attack_args.get('targeted', False)

    attacker = HopSkipJump(classifier=model, targeted=targeted, norm=norm,
                           max_iter=max_iter, max_eval=max_eval, max_queries=max_queries,
                           init_eval=init_eval, init_size=init_size)

    return attacker.generate(data, labels)


def _zoo(model, data, labels, attack_args):
    lr = attack_args.get('learning_rate', 0.01)
    max_iter = attack_args.get('max_iter', 100)
    binary_search_steps = attack_args.get('binary_search_steps', 5)

    confidence = attack_args.get('confidence', 0.0)
    targeted = attack_args.get('targeted', False)
    init_const = attack_args.get('init_const', 1e-3)
    abort_early = attack_args.get('abort_early', True)
    use_resize = attack_args.get('use_resize', True)
    use_importance = attack_args.get('use_importance', True)
    nb_parallel = attack_args.get('nb_parallel', 128)
    variable_h = attack_args.get('variable_h', 1e-4)

    attacker = ZooAttack(classifier=model, confidence=confidence, targeted=targeted,
                         learning_rate=lr, max_iter=max_iter, binary_search_steps=binary_search_steps,
                         initial_const=init_const, abort_early=abort_early, use_resize=use_resize,
                         use_importance=use_importance, nb_parallel=nb_parallel, variable_h=variable_h)

    return attacker.generate(data, labels)


def _get_norm_value(norm):
    """
    Convert a string norm to a numeric value.
    :param norm: norm in string, defined in a format of `ln`,
            where `n` is `inf` or a number e.g., 0, 1, 2, etc.
    :return: the corresponding numeric value.
    """
    if norm[0] not in ['l', 'L']:
        raise ValueError('Norm should be defined in the form of `ln` (or `Ln`), where `n` is a number or `inf`. But found {}.'.format(norm))

    norm = norm.lower()[1:]
    if norm == 'inf':
        value = np.inf
    else:
        try:
            value = int(norm)
        except:
            raise ValueError('Norm should be defined in the form of `ln` (or `Ln`), where `n` is a number or `inf`. But found {}.'.format(norm))

    return value

