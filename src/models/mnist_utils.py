"""
Utilities of models for MNIST
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
from enum import Enum

import pickle
from keras.models import load_model, Model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import tensorflow as tf

from models.keraswrapper import WeakDefense as KerasWD
from models.sklearnwrapper import WeakDefense as SkLearnWD

tf.compat.v1.disable_eager_execution()

IMAGE_SHAPE = (28, 28, 1)


def load_pool(trans_configs, model_configs, active_list=False, use_logits=False, wrap=False):
    pool = {}
    trans_list = {}

    if active_list:
        wd_ids = trans_configs.get("active_wds")
    else:
        num_trans = trans_configs.get("num_transformations")
        wd_ids = [i for i in range(num_trans)]
    for i in wd_ids:
        key = "configs{}".format(i)
        trans = trans_configs.get(key).get("description")

        model_file = model_configs.get("wd_prefix") + trans + model_configs.get("wd_postfix")
        model_file = os.path.join(model_configs.get("dir"), model_file)

        if model_configs.get("architecture").lower() == MODEL_TYPE.LENET.value:
            wd = load_lenet(model_file, trans_configs=trans_configs.get(key), use_logits=use_logits, wrap=wrap)
        elif model_configs.get("architecture").lower() == MODEL_TYPE.SVM.value:
            wd = load_svm(model_file, trans_configs=trans_configs.get(key), wrap=wrap)
        else:
            raise ValueError("Expected `cnn` or `svm` model, but found {}.".format(model_configs.get("architecture")))

        pool[trans_configs.get(key).get("id")] = wd
        trans_list[trans_configs.get(key).get("id")] = trans_configs.get(key).get("description")

    print('>>> Loaded {} models.'.format(len(pool.keys())))
    return pool, trans_list


def load_lenet(file, trans_configs=None, use_logits=False, wrap=False):
    """
    Load a LeNet model (implemented in keras).
    :param file: str or path. The full-path file name to the trained model.
    :param trans_configs: dictionary. The corresponding transformation settings.
    :param use_logits: boolean. Use the logits or the probabilities.
    :param wrap: boolean. True, if want to wrap the model into a weak defense in Athena.
    :return:
    """
    print('>>> Loading model [{}]...'.format(file))
    with CustomObjectScope({"GlorotUniform": glorot_uniform()}):
        model = load_model(file)

    if use_logits:
        model = _convert2logits(model)

    if wrap:
        if trans_configs is None:
            # load the undefended model by default
            trans_configs = {
                "type": "clean",
                "subtype": "",
                "id": 0,
                "description": "clean"
            }

        model = KerasWD(model=model, trans_configs=trans_configs, use_logits=use_logits)

    return model


def _convert2logits(model):
    return Model(inputs=model.input,
                 outputs=model.get_layer(model.layers[-2].name).output)


def load_svm(file, trans_configs=None, wrap=False):
    """
    Load an (sklearn) SVM model (in pickle format)
    :param filename: str or path. The full-path file name to the model.
    :param trans_configs: dictionary. The corresponding transformation settings.
    :param wrap: boolean. True, if want to wrap to a weak defense in Athena.
    :return:
    """
    print(">>> Loading model [{}]...".format(file))
    with open(file, "rb") as f:
        model = pickle.load(f)

    if wrap:
        if trans_configs is None:
            # load the undefended model by default
            trans_configs = {
                "type": "clean",
                "subtype": "",
                "id": 0,
                "description": "clean"
            }

        model = SkLearnWD(model=model, trans_configs=trans_configs, image_shape=IMAGE_SHAPE)

    return model


class MODEL_TYPE(Enum):
    LENET = "lenet"
    SVM = "svm"

    @classmethod
    def supported_models(cls):
        return [
            cls.LENET.value,
            cls.SVM.value,
        ]