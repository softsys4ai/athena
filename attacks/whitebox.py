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
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import DeepFool
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.evaluation import batch_eval
from cleverhans.utils_keras import KerasModelWrapper

# import attacks.cw_linf as cw_linf

# FLAGS = flags.FLAGS

validation_rate = 0.2


def generate(model_name, X, Y, attack_method, attack_params):
    """
    detect adversarial examples
    :param model_name: the name of the target model. Models are named in the form of
                        model-<dataset>-<architecture>-<transform_type>.h5
    :param attack_method:  attack for generating adversarial examples
    :param X: examples to be attacked
    :param Y: correct label of the examples
    :return: adversarial examples
    """
    label_smoothing_rate = 0.1

    prefix, dataset, architect, trans_type = model_name.split('-')

    # flag - whether to train a clean model
    train_new_model = True
    if (os.path.isfile('{}/{}/{}.h5'.format(PATH.MODEL, dataset, model_name)) or
            (os.path.isfile('{}/{}/{}.json'.format(PATH.MODEL, dataset, model_name)))):
        # found a trained model
        print('Found the trained model.')
        train_new_model = False

    img_rows, img_cols, nb_channels = X.shape[1:4]
    nb_classes = Y.shape[1]

    # Force TensorFlow to use single thread to improve reproducibility
    config = tf.ConfigProto(intra_op_parallelism_threads=4,
                            inter_op_parallelism_threads=4)
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    # input and output tensors
    # x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nb_channels))
    # y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    batch_size = 128

    # label smoothing
    Y -= label_smoothing_rate * (Y - 1. / nb_classes)

    model = None
    if train_new_model:
        print('INFO: train a new model then generate adversarial examples.')
        # create a new model
        input_shape = (img_rows, img_cols, nb_channels)
        model = create_model(dataset, input_shape=input_shape, nb_classes=nb_classes)
    else:
        # load model
        if dataset == DATA.mnist:
            model = keras.models.load_model('{}/{}/{}.h5'.format(PATH.MODEL, dataset, model_name))
        elif dataset == DATA.cifar_10:
            model = load_from_json(model_name)

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
    elif attack_method == ATTACK.CW:
        """
        Untageted attack
        """
        y = tf.placeholder(tf.float32, shape=(None, nb_classes))
        ord = attack_params['ord']
        attack_params.pop('ord')
        attack_params['y'] = y

        if ord == 2:
            # cleverhans supports only l2 norm so far.
            attacker = CarliniWagnerL2(wrap_model, sess=sess)
        elif ord == 0:
            # TODO
            pass
        elif ord == np.inf:
            # TODO
            pass
        else:
            raise ValueError('CW supports only l0, l2, and l-inf norms.')

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
        The Projected Gradient Descent approch.
        
        """
        attacker = ProjectedGradientDescent(wrap_model)
    else:
        raise ValueError('{} attack is not supported.'.format(attack_method.upper()))

    # initialize the session
    # init_op = tf.initialize_all_variables()
    # sess.run(init_op)
    # define custom loss
    adv_accuracy_metric = get_adversarial_metric(model, attacker, attack_params)

    augment = False
    compile_params = {
        'optimizer': keras.optimizers.Adam(lr=0.001),
        'metrics': adv_accuracy_metric
    }
    if DATA.cifar_10 == dataset:
        augment = True
        compile_params = {
            'optimizer': keras.optimizers.RMSprop(lr=0.001, decay=1e-6),
            'metrics': adv_accuracy_metric
        }

    print('#### Recompile the model')
    model.compile(optimizer=compile_params['optimizer'],
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy', adv_accuracy_metric])
    # train_model(model, dataset, model_name, need_augment=False, **kwargs)
    if train_new_model:
        model = train_model(model, dataset, model_name,
                            augment,
                            **compile_params)

    # define the graph
    adv_x = attacker.generate(model.input, **attack_params)
    # consider the attack to be constant
    adv_x = tf.stop_gradient(adv_x)

    # generating adversarial examples
    adv_examples, = batch_eval(sess, [model.input, model(adv_x)], [adv_x],
                               [X, Y], batch_size=batch_size)

    if MODE.DEBUG:
        score = model.evaluate(adv_examples, Y, verbose=2)
        print('*** Evaluation on adversarial examples: {}'.format(score))
        # title = '{}-{}'.format(dataset, attack_method)
        # draw_comparisons(X[10:20], adv_examples[10:20], title)

    if train_new_model:
        """
        recompile the trained model using default metrics,
        for the metrics related to adversarial approaches 
        are NOT required for model evaluation.
        """
        model.compile(
            optimizer=compile_params['optimizer'],
            loss=keras.losses.categorical_crossentropy,
            metrics=['accuracy']
        )

        # save to disk
        save_model(model, model_name)
        # evaluate the new model
        loaded_model = load_model(model_name)
        scores = loaded_model.evaluate(X, Y, verbose=2)
        print('*** Evaluating the new model: {}'.format(scores))
        del loaded_model

        # if DATA.cifar_10 == dataset:
        #     save_to_json(model, model_name)
        #
        #     # for test, evaluate the saved model
        #     loaded_model = load_from_json(model_name)
        #     scores = loaded_model.evaluate(X, Y, verbose=2)
        #     print('*** Evaluating the new model: {}'.format(scores))
        #     del loaded_model
        # elif DATA.mnist == dataset:
        #     model.save('{}/{}.h5'.format(PATH.MODEL, model_name),
        #                overwrite=True, include_optimizer=True)
        #     # for test
        #     # evaluate the saved model
        #     loaded_model = models.load_model('{}/{}.h5'.format(PATH.MODEL, model_name))
        #     scores = loaded_model.evaluate(X, Y, verbose=2)
        #     print('*** Evaluating the new model: {}'.format(scores))
        #     del loaded_model
    del model
    sess.close()

    return adv_examples, Y


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
