"""
Define settings of transformations.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import cv2
from enum import Enum
from PIL import Image
from skimage import filters, morphology, transform
from scipy import ndimage

class TRANSFORMATION(Enum):
    CLEAN = 'clean'
    ROTATE = 'rotate'
    SHIFT = 'shift'
    FLIP = 'flip'
    AFFINE_TRANS = 'affine'
    MORPH_TRANS = 'morph'
    AUGMENT = 'augment'
    CARTOON = 'cartoon'
    QUANTIZATION = 'quant'
    DISTORTION = 'distort'
    NOISE = 'noise'
    FILTER = 'filter'
    COMPRESSION = 'compress'
    DENOISE = 'denoise'
    GEOMETRIC = 'geometric'
    SEGMENTATION = 'segment'


def get_flip_direction(flip_trans):
    return {
        0: 'AROUND_X_AXIS',
        1: 'AROUND_Y_AXIS',
        -1: 'AROUND_BOTH_AXIS',
    }[flip_trans]


class AUGMENT_TRANSFORMATIONS(Enum):
    SAMPLEWISE_AUGMENTATION = 'samplewise_std_norm'
    FEATURE_AUTMENTATION = 'feature_std_norm'
    ZCA_WHITENING = 'zca_whitening'
    PCA_WHITENING = 'pca_whitening'


class DISTORT_TRANSFORMATIONS(Enum):
    X = 'x'
    Y = 'y'
    PIXELATE = 'pixelate'
    CONTRAST = 'contrast'
    BRIGHTNESS = 'brightness'


class NOISE_TRANSFORMATIONS(Enum):
    GAUSSIAN = 'gaussian'
    LOCALVAR = 'localvar'
    POISSON = 'poisson'
    SALT = 'salt'
    PEPPER = 'pepper'
    SALTNPEPPER = 's&p'
    SPECKLE = 'speckle'


class DENOISE_TRANSFORMATIONS(Enum):
    WAVELET = 'wavelet'
    TV_CHAMBOLLE = 'tv_chambolle'
    TV_BREGMAN = 'tv_bregman'
    BILATERAL = 'bilateral' # TODO: bug fix
    NL_MEANS = 'nl_means'
    NL_MEANS_FAST = 'nl_means_fast'


class MORPH_TRANSFORMATIONS(Enum):
    EROSION = 'erosion'
    DILATION = 'dilation'
    OPENING = 'opening'
    CLOSING = 'closing'
    GRADIENT = 'gradient'


def get_morph_op(morph_trans):
    return {
        MORPH_TRANSFORMATIONS.EROSION.value: cv2.MORPH_ERODE,
        MORPH_TRANSFORMATIONS.DILATION.value: cv2.MORPH_DILATE,
        MORPH_TRANSFORMATIONS.OPENING.value: cv2.MORPH_OPEN,
        MORPH_TRANSFORMATIONS.CLOSING.value: cv2.MORPH_CLOSE,
        MORPH_TRANSFORMATIONS.GRADIENT.value: cv2.MORPH_GRADIENT,
    }[morph_trans]


class FLIP_DIRECTION(Enum):
    AROUND_X_AXIS = 0
    AROUND_Y_AXIS = 1
    AROUND_BOTH_AXIS = -1


class FILTER_TRANSFORMATION(Enum):
    SOBEL = 'sobel'
    GAUSSIAN = 'gaussian'
    RANK = 'rank'
    MEDIAN = 'median'
    MINIMUM = 'minimum'
    MAXIMUM = 'maximum'
    ENTROPY = 'entropy'
    ROBERTS = 'roberts'
    SCHARR = 'scharr'
    PREWITT = 'prewitt'
    MEIJERING = 'heijering' # TODO: bug fix
    SATO = 'sato' # TODO: bug fix
    FRANGI = 'frangi' # TODO: bug fix
    HESSIAN = 'hessian' # TODO: bug fix
    SKELETONIZE = 'skelentonize' # TODO: bug fix
    THIN = 'thin' # TODO: bug fix


def get_filter_op(filter):
    return {
        FILTER_TRANSFORMATION.SOBEL.value: filters.sobel,
        FILTER_TRANSFORMATION.GAUSSIAN.value: ndimage.gaussian_filter,
        FILTER_TRANSFORMATION.RANK.value: ndimage.rank_filter,
        FILTER_TRANSFORMATION.MEDIAN.value: ndimage.median_filter,
        FILTER_TRANSFORMATION.MINIMUM.value: ndimage.minimum_filter,
        FILTER_TRANSFORMATION.MAXIMUM.value: ndimage.maximum_filter,
        FILTER_TRANSFORMATION.ENTROPY.value: filters.rank.entropy,
        FILTER_TRANSFORMATION.ROBERTS.value: filters.roberts,
        FILTER_TRANSFORMATION.SCHARR.value: filters.scharr,
        FILTER_TRANSFORMATION.PREWITT.value: filters.prewitt,
        FILTER_TRANSFORMATION.MEIJERING.value: filters.meijering,
        FILTER_TRANSFORMATION.SATO.value: filters.sato,
        FILTER_TRANSFORMATION.FRANGI.value: filters.frangi,
        FILTER_TRANSFORMATION.HESSIAN.value: filters.hessian,
        FILTER_TRANSFORMATION.SKELETONIZE.value: morphology.skeletonize,
        FILTER_TRANSFORMATION.THIN.value: morphology.thin,
    }[filter]


class CARTOON_ADAPTIVE_METHODS(Enum):
    MEAN = 'mean'
    GAUSSIAN = 'gaussian'


class CARTOON_THRESH_METHODS(Enum):
    BINARY = 'thresh_binary'
    BINARY_INV = 'thresh_binary_inv'
    TRIANGLE = 'thresh_triangle'
    MASK = 'thresh_mask'
    TRUNC = 'thresh_trunc'
    OTSU = 'thresh_otsu'
    TOZERO = 'thresh_tozero'
    TOZERO_INV = 'thresh_tozero_inv'


def get_cartoon_adpative_method(adaptive_method):
    return {
        CARTOON_ADAPTIVE_METHODS.MEAN.value: cv2.ADAPTIVE_THRESH_MEAN_C,
        CARTOON_ADAPTIVE_METHODS.GAUSSIAN.value: cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    }[adaptive_method]


def get_cartoon_thresh_method(thresh_method):
    return {
        CARTOON_THRESH_METHODS.BINARY.value: cv2.THRESH_BINARY,
        CARTOON_THRESH_METHODS.BINARY_INV.value: cv2.THRESH_BINARY_INV,
        CARTOON_THRESH_METHODS.TRIANGLE.value: cv2.THRESH_TRIANGLE,
        CARTOON_THRESH_METHODS.MASK.value: cv2.THRESH_MASK,
        CARTOON_THRESH_METHODS.TRUNC.value: cv2.THRESH_TRUNC,
        CARTOON_THRESH_METHODS.OTSU.value: cv2.THRESH_OTSU,
        CARTOON_THRESH_METHODS.TOZERO.value: cv2.THRESH_TOZERO,
        CARTOON_THRESH_METHODS.TOZERO_INV.value: cv2.THRESH_TOZERO_INV,
    }[thresh_method]


class DISTORT_RESAMPLE_MEHTOD(Enum):
    NEAREST = 'nearest'
    LINEAR = 'linear'
    NORMAL = 'normal'
    BOX = 'box'

def get_distort_resample(resample):
    return {
        DISTORT_RESAMPLE_MEHTOD.NEAREST.value: Image.NEAREST,
        DISTORT_RESAMPLE_MEHTOD.LINEAR.value: Image.LINEAR,
        DISTORT_RESAMPLE_MEHTOD.NORMAL.value: Image.NORMAL,
        DISTORT_RESAMPLE_MEHTOD.BOX.value: Image.BOX,
    }[resample]


class COMPRESS_FORMAT(Enum):
    JPEG = '.jpeg'
    JPG = '.jpg'
    PNG = '.png'

def get_compress_encoder(format, rate):
    if format == COMPRESS_FORMAT.PNG.value:
        return [cv2.IMWRITE_PNG_COMPRESSION, rate]
    else:
        return [int(cv2.IMWRITE_JPEG_QUALITY), rate]


class GEOMETRIC_TRANSFORMATIONS(Enum):
    SWIRL = 'swirl'
    RADON = 'radon'
    IRADON = 'iradon'
    IRADON_SART = 'iradon_sart'


def get_geometric_op(geo_trans):
    return {
        GEOMETRIC_TRANSFORMATIONS.SWIRL.value: transform.swirl,
        GEOMETRIC_TRANSFORMATIONS.RADON.value: transform.radon,
        GEOMETRIC_TRANSFORMATIONS.IRADON.value: transform.iradon,
        GEOMETRIC_TRANSFORMATIONS.IRADON_SART.value: transform.iradon_sart,
    }[geo_trans]


class SEGMENT_TRANSFORMATIONS(Enum):
    GRADIENT = 'gradient'
    WATERSHED = 'watershed'
