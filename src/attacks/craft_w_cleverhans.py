"""
Implement whitebox adversarial example generating approaches here,
mainly use cleverhans attack toolkits.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from models import *
from utils.config import *

from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks import DeepFool
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.evaluation import batch_eval
from cleverhans.utils_keras import KerasModelWrapper

#from attacks.carlini_wagner_l0 import CarliniWagnerL0
from attacks.carlini_wagner_l2 import CarliniWagnerL2
#from attacks.carlini_wagner_li import CarliniWagnerLinf

# FLAGS = flags.FLAGS

validation_rate = 0.2


def generate(sess, model, X, Y, attack_method, dataset, attack_params):
    """
    detect adversarial examples
    :param model_name: the name of the target model. Models are named in the form of
                        model-<dataset>-<architecture>-<transform_type>.h5
    :param attack_method:  attack for generating adversarial examples
    :param X: examples to be attacked
    :param Y: correct label of the examples
    :return: adversarial examples
    """
    batch_size = 128

    img_rows, img_cols, nb_channels = X.shape[1:4]
    nb_classes = Y.shape[1]
    # label smoothing
    label_smoothing_rate = 0.1
    Y -= label_smoothing_rate * (Y - 1. / nb_classes)

    # to be able to call the model in the custom loss, we need to call it once before.
    # see https://github.com/tensorflow/tensorflow/issues/23769
    model(model.input)
    # wrap a keras model, making it fit the cleverhans framework
    wrap_model = KerasModelWrapper(model)

    # initialize the attack object
    attacker = None
    if attack_method == ATTACK.FGSM:
        """
        The Fast Gradient Sign Method,
        by Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy 2014
        link: https://arxiv.org/abs/1412.6572
        """
        attacker = FastGradientMethod(wrap_model, sess=sess)
    elif attack_method == ATTACK.JSMA:
        """
        The Jacobian-based Saliency Map Method
        by Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson, Z. Berkay Celik, Ananthram Swami 2016
        link: https://arxiv.org/abs/1511.07528
        """
        batch_size = 64
        attacker = SaliencyMapMethod(wrap_model, sess=sess)
    elif attack_method == ATTACK.CW_L2:
        """
        Untageted attack
        """
        attacker = CarliniWagnerL2(wrap_model, sess=sess)

    elif attack_method == ATTACK.CW_Linf:
        """
        Untageted attack
        """
        # TODO: bug fix --- cannot compute gradients correctly
        # attacker = CarliniWagnerLinf(wrap_model, sess=sess)

    elif attack_method == ATTACK.CW_L0:
        """
        Untargeted attack
        """
        # TODO: bug fix --- cannot compute gradients correctly
        # attacker = CarliniWagnerL0(wrap_model, sess=sess)

    elif attack_method == ATTACK.DEEPFOOL:
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

    elif attack_method == ATTACK.BIM:
        """
        The Basic Iterative Method (also, iterative FGSM)
        by Alexey Kurakin, Ian Goodfellow, Samy Bengio, 2016
        link: https://arxiv.org/abs/1607.02533
        """
        attacker = BasicIterativeMethod(wrap_model, back='tf', sess=sess)
    elif attack_method == ATTACK.PGD:
        """
        The Projected Gradient Descent approach.
        """
        attacker = ProjectedGradientDescent(wrap_model)
    elif attack_method == ATTACK.MIM:
        """
        The Momentum Iterative Method
        by Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su, Jun Zhu, Xiaolin Hu, Jianguo Li, 2018
        link: https://arxiv.org/abs/1710.06081
        """
        attacker = MomentumIterativeMethod(wrap_model, sess=sess)
    else:
        raise ValueError('{} attack is not supported.'.format(attack_method.upper()))

    # define custom loss function for adversary
    compile_params = get_compile_params(dataset,
                                        get_adversarial_metric(model, attacker, attack_params))

    print('#### Recompile the model')
    model.compile(optimizer=compile_params['optimizer'],
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy', compile_params['metrics']])

    # define the graph
    print('define the graph')
    adv_x = attacker.generate(model.input, **attack_params)
    # consider the attack to be constant
    adv_x = tf.stop_gradient(adv_x)

    # generating adversarial examples
    print('generating adversarial example...')
    adv_examples, = batch_eval(sess, [model.input, wrap_model(adv_x)], [adv_x],
                               [X, Y], batch_size=batch_size)

    if MODE.DEBUG:
        score = model.evaluate(adv_examples, Y, verbose=2)
        print('*** Evaluation on adversarial examples: {}'.format(score))

    return adv_examples, Y


"""
Prepare compile parameters 
"""
def get_compile_params(dataset=DATA.mnist, metrics=None):
    compile_params = {}

    if DATA.mnist in dataset:
        compile_params = {
            'optimizer': keras.optimizers.Adam(lr=0.001),
            'metrics': metrics
        }
    elif dataset == DATA.cifar_10:
        compile_params = {
            'optimizer': keras.optimizers.RMSprop(lr=0.001, decay=1e-6),
            'metrics': metrics
        }

    return compile_params

"""
Define custom loss functions
"""
def get_adversarial_metric(model, attacker, attack_params):
    print('INFO: create metrics for adversary generation.')

    def adversarial_accuracy(y, _):
        # get the adversarial examples
        x_adv = attacker.generate(model.input, **attack_params)

        # consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # get the prediction on the adversarial examples
        preds_adv = model(x_adv)
        return keras.metrics.categorical_accuracy(y, preds_adv)

    # return the loss function
    return adversarial_accuracy


def get_adversarial_loss(model, attacker, attack_params):
    def adversarial_loss(y, preds):
        # calculate the cross-entropy on the legitimate examples
        corss_entropy = keras.losses.categorical_crossentropy(y, preds)

        # get the adversarial examples
        x_adv = attacker.generate(model.input, **attack_params)

        # consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # calculate the cross-entropy on the adversarial examples
        preds_adv = model(x_adv)
        corss_entropy_adv = keras.losses.categorical_crossentropy(y, preds_adv)

        # return the average cross-entropy
        return 0.5 * corss_entropy + 0.5 * corss_entropy_adv

    # return the custom loss function
    return adversarial_loss
