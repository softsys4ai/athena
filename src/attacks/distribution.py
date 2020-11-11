"""
Implement the distributions of transformations.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import numpy as np
from enum import Enum
import random

from models.image_processor import transform
from utils.data import set_channels_first, set_channels_last

def batch_sample_from_distribution(X, distribution_args):
    """
    Apply transformations from the specific distributions on given input x, in batch.
    :param X: the legitimate samples.
    :param distribution_args: dictionary. configuration of the distribution.
    :return:
    """
    raise NotImplemented("Sampling from distribution in batch is not implemented.")


def sample_from_distribution(x, distribution_args):
    """
    Apply transformations from the specific distributions on given input x.
    :param x: the legitimate sample.
    :param distribution_args: dictionary. configuration of the distribution.
    :return:
    """
    if len(x.shape) == 4 and x.shape[0] > 1:
        raise ValueError("This method does not support sampling for a batch. Function `batch_sample_from_distribution` is for batch sampling.")

    transformation = distribution_args.get("transformation", TRANSFORMATION_DISTRIBUTION.RANDOM.value)

    channel_index = distribution_args.get("channel_index", 3)
    if channel_index not in [1, 3]:
        raise ValueError("`channel_index` must be 1 or 3, but found {}.".format(channel_index))

    if channel_index == 1:
        x = set_channels_last(x)

    x = x.astype(np.float32)
    if transformation == TRANSFORMATION_DISTRIBUTION.RANDOM.value:
        # select a random transformation
        distribution = TRANSFORMATION_DISTRIBUTION.distributions()[1:]
        transformation = random.choice(distribution)

    if transformation == TRANSFORMATION_DISTRIBUTION.ROTATION.value:
        # randomly rotate the input x
        trans_args = {
            "type": "rotate",
            "subtype": "",
            "id": -1,
        }

        # the interval of rotation angle
        min_angle = distribution_args.get("min_angle", -15)
        max_angle = distribution_args.get("max_angle", 15)

        # rotate the x by a random angle
        angle = random.randint(min_angle, max_angle)
        trans_args["angle"] = angle
        trans_args["description"] = "rotate[{}]".format(angle)
        x_trans = transform(x, trans_args)[0]

    elif transformation == TRANSFORMATION_DISTRIBUTION.GAUSSIAN_NOISE.value:
        # add random gaussian noise to the input
        # magnitude of noise
        eta = distribution_args.get("eta", 0.03)

        clip_min = distribution_args.get("clip_min", 0.)
        clip_max = distribution_args.get("clip_max", 1.)

        # add noise
        noise = np.random.normal(loc=0, scale=1, size=x.shape)
        noisy_x = np.clip((x + noise * eta), clip_min, clip_max)
        x_trans = noisy_x.reshape(x.shape)

    elif transformation == TRANSFORMATION_DISTRIBUTION.TRANSLATION.value:
        # randomly translate the input
        trans_args = {
            "type": "shift",
            "subtype": "",
            "id": -1,
        }

        # interval of translation offsets
        min_offset = distribution_args.get("min_offset", -0.20)
        max_offset = distribution_args.get("max_offset", 0.20)
        # random offsets in x- and y-axis
        x_offset = random.uniform(min_offset, max_offset)
        y_offset = random.uniform(min_offset, max_offset)
        trans_args["x_offset"] = x_offset
        trans_args["y_offset"] = y_offset
        trans_args["description"] = "shift[{},{}]".format(x_offset, y_offset)
        x_trans = transform(x, trans_args)[0]

    elif transformation == TRANSFORMATION_DISTRIBUTION.AFFINE.value:
        # apply random affine transformation
        trans_args = {
            "type": "affine",
            "subtype": "",
            "id": -1,
        }

        # interval of transformation offsets
        min_offset = distribution_args.get("min_offset", 0.1)
        max_offset = distribution_args.get("max_offset", 0.5)

        if min_offset <= 0 or max_offset <= 0:
            raise ValueError("`min_offset` and `max_offset` must be positive, but found {} and {}.".format(min_offset, max_offset))

        if min_offset >= max_offset:
            raise ValueError("`min_offset` must be less than `max_offset`, but found {} and {}.".format(min_offset, max_offset))

        # apply random affine transformation
        op1 = random.uniform(min_offset, max_offset)
        op2 = random.uniform(min_offset, max_offset)
        origin_point1 = distribution_args.get("origin_point1", (op1, op1))
        origin_point2 = distribution_args.get("origin_point2", (op1, op2))
        origin_point3 = distribution_args.get("origin_point3", (op2, op1))
        np1 = random.uniform(min_offset, max_offset)
        np2 = random.uniform(min_offset, max_offset)
        np3 = random.uniform(min_offset, max_offset)
        np4 = random.uniform(min_offset, max_offset)
        new_point1 = distribution_args.get("new_point1", (np1, np2))
        new_point2 = distribution_args.get("new_point2", (np1, np3))
        new_point3 = distribution_args.get("new_point3", (np4, np2))

        trans_args["origin_point1"] = origin_point1
        trans_args["origin_point2"] = origin_point2
        trans_args["origin_point3"] = origin_point3
        trans_args["new_point1"] = new_point1
        trans_args["new_point2"] = new_point2
        trans_args["new_point3"] = new_point3
        trans_args["description"] = "affine[{},{},{};{},{},{}]".format(origin_point1, origin_point2, origin_point3,
                                                                       new_point1, new_point2, new_point3)

        x_trans = transform(x, trans_args)[0]

    else:
        raise ValueError("Distribution {} is not supported.".format(transformation))

    if channel_index == 1:
        x_trans = set_channels_first(x_trans)

    return x_trans.astype(np.float32)


class TRANSFORMATION_DISTRIBUTION(Enum):
    """
    Supported transformations
    """
    RANDOM = "random"
    ROTATION = "rotation"
    GAUSSIAN_NOISE = "gaussian_noise"
    TRANSLATION = "translation"
    AFFINE = "affine"

    @classmethod
    def distributions(cls):
        dist = [
            cls.RANDOM.value,
            cls.ROTATION.value,
            cls.GAUSSIAN_NOISE.value,
            cls.TRANSLATION.value,
            cls.AFFINE.value,
        ]
        return dist
