"""
Implement attack utilities.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

from enum import Enum
import numpy as np
# import cv2
# from models.image_processor import transform
# from utils.transformation import TRANSFORMATION


class WHITEBOX_ATTACK(Enum):
    FGSM = 'fgsm'
    CW = 'cw'
    PGD = 'pgd'
    JSMA = 'jsma'
    BIM = 'bim'
    MIM = 'mim'
    OP = 'one-pixel'
    DF = 'deepfool'
    SPATIAL_TRANS = 'spatial-transformation'
    HOP_SKIP_JUMP = 'hsja'
    ZOO = 'zoo'


def get_norm_value(norm):
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


# def random_samples(x, num_samples, args):
#     if args.get('transformation') == 'rotate':
#         return random_rotations(x, num_samples=num_samples,
#                                 minval=args.get('minval'), maxval=args.get('maxval'))
#     elif args.get('transformation') == 'shift':
#         return random_shifts(x, num_samples=num_samples,
#                                 minval=args.get('minval'), maxval=args.get('maxval'))
#     elif args.get('transformation') == 'flip':
#         return random_flips(x, num_samples=num_samples)
#     else:
#         return random_rotations(x, num_samples=num_samples,
#                                 minval=args.get('minval'), maxval=args.get('maxval'))
#
#
# def random_rotations(X, num_samples, minval, maxval):
#     samples = []
#     num_images = X.shape[0]
#     pool = np.asarray([i for i in range(minval, maxval+1)])
#     angles = np.random.choice(pool, size=num_samples, replace=False)
#
#     trans_params = {
#         'type': TRANSFORMATION.ROTATE.value,
#         'subtype': '',
#     }
#
#     print('Random angles:', angles)
#     rotated = []
#     for angle in angles:
#         trans_params['description'] = 'rotate{}'.format(angle)
#         trans_params['angle'] = angle
#         X_rot = transform(X, attack_args=trans_params)
#         rotated.append(X_rot)
#
#     rotated = np.asarray(rotated)
#     print('ROTATED SHAPE:', rotated.shape)
#     for i in range(num_images):
#         for j in range(num_samples):
#             samples.append(rotated[j][i])
#     samples = np.asarray(samples)
#
#     print('SAMPLE SHAPES:', samples.shape)
#     return samples
#
#
# def random_shifts(x, num_samples, minval, maxval):
#     img_rows, img_cols = x.shape[0], x.shape[1]
#
#     samples = []
#     x_offsets = np.random.uniform(low=minval, high=maxval, size=num_samples)
#     y_offsets = np.random.uniform(low=minval, high=maxval, size=num_samples)
#
#     for x_off in x_offsets:
#         for y_off in y_offsets:
#             tx = x_off * img_cols
#             ty = y_off * img_rows
#
#             trans_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
#             samples.append(cv2.warpAffine(x, trans_matrix, (img_cols, img_rows)))
#
#     return samples
#
#
# def random_flips(x, num_samples):
#     samples = []
#     directions = np.random.choice([-1, 0, 1], size=num_samples)
#
#     for direct in directions:
#         samples.append(cv2.flip(x, direct))
#
#     return samples
#

