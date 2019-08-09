"""
Implement whitebox adversarial example generating approaches here,
mainly use cleverhans attack toolkits.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.python.platform import flags
import keras.backend as K

from config import *

from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import DeepFool
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.evaluation import batch_eval

FLAGS = flags.FLAGS

def generate(wrap_model, attack_method, X, Y, attack_params):
    """
    detect adversarial examples
    :param sess: target model session
    :param wrap_model: wrap model
    :param attack_method:  attack for generating adversarial examples
    :param X: examples to be attacked
    :param Y: correct label of the examples
    :return: x_adv: adversarial examples
    """
    img_rows, img_cols, nb_channels = X.shape[1:4]
    nb_classes = Y.shape[1]

    sess = tf.Session()
    K.set_session(sess)
    K.set_learning_phase(0)

    # input and output tensors
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nb_channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    batch_size = 128
    attacker = None

    # Define corresponding attack equations
    if (attack_method == ATTACK.FGSM):
        """
        The Fast Gradient Sign Method,
        by Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy 2014
        link: https://arxiv.org/abs/1412.6572
        """
        attacker = FastGradientMethod(wrap_model, sess=sess)
    elif (attack_method == ATTACK.JSMA):
        """
        The Jacobian-based Saliency Map Method
        by Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson, Z. Berkay Celik, Ananthram Swami 2016
        link: https://arxiv.org/abs/1511.07528
        """
        batch_size = 64
        attacker = SaliencyMapMethod(wrap_model, sess=sess)
    elif (attack_method == ATTACK.CW):
        """
        Untageted attack
        """
        ord = attack_params['ord']
        attack_params.pop('ord')
        attack_params['y'] = y

        if (ord == 2):
            # cleverhans supports only l2 norm so far.
            attacker = CarliniWagnerL2(wrap_model, sess=sess)
        elif (ord == 0):
            # TODO
            pass
        elif (ord == np.inf):
            # TODO
            pass
        else:
            raise ValueError('CW supports only l0, l2, and l-inf norms.')

    elif (attack_method == ATTACK.DEEPFOOL):
        """
        The DeepFool Method, is an untargeted & iterative attack
        which is based on an iterative linearization of the classifier.
        by Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard, 2016
        link: https://arxiv.org/abs/1511.04599
        """
        batch_size = 64
        ord = attack_params['ord']
        attack_params.pop('ord')

        if ord == 2:
            # cleverhans supports only l2 norm so far.
            attacker = DeepFool(wrap_model, sess=sess)
        elif ord == np.inf:
            # TODO
            pass
        else:
            raise ValueError('DeepFool supports only l2 and l-inf norms.')

    elif (attack_method == ATTACK.BIM):
        """
        The Basic Iterative Method (also, iterative FGSM)
        by Alexey Kurakin, Ian Goodfellow, Samy Bengio, 2016
        link: https://arxiv.org/abs/1607.02533
        """
        attacker = BasicIterativeMethod(wrap_model, back='tf', sess=sess)
    else:
        raise ValueError('{} attack is not supported.'.format(attack_method.upper()))

    # initialize the session
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    # define the symbolic
    adv_x = attacker.generate(x, **attack_params)
    # generating adversarial examples
    X_adv, = batch_eval(sess, [x, y], [adv_x], [X, Y], batch_size=batch_size)

    sess.close()

    return X_adv, Y

