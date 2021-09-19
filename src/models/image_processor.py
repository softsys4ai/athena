"""
Implement transformations
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from scipy import ndimage
from skimage import color, util
from skimage.filters import rank
from skimage.morphology import disk, watershed
from skimage.restoration import denoise_bilateral, denoise_nl_means, denoise_tv_bregman, denoise_tv_chambolle, \
    denoise_wavelet, estimate_sigma
from skimage.transform import radon
from sklearn.cluster import MiniBatchKMeans

import utils.transformation as trans_utils


# Entrance
def transform(X, trans_args):
    if len(X.shape) not in [3, 4]:
        raise ValueError("Expect an input with 3-4 dimensions, but received {}.".format(len(X.shape)))

    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=0)

    X = X.astype(np.float32)
    X = np.clip(X, 0., 1.,)
    if isinstance(trans_args, (list, np.ndarray)):
        raise NotImplementedError('Transformation combination is not implemented.')
    else:
        X_trans = _transform_images(X, trans_args).astype(np.float32)
        return np.clip(X_trans, 0., 1.)



def _transform_images(X, trans_args):
    # print('TRANSFORMATION [{}].'.format(trans_args.get('description')))

    if trans_args is None or type(trans_args)==str or trans_args.get('type') == trans_utils.TRANSFORMATION.CLEAN.value:
        return X
    elif trans_args.get('type') == trans_utils.TRANSFORMATION.ROTATE.value:
        return _rotate(X, trans_args)
    elif trans_args.get('type') == trans_utils.TRANSFORMATION.SHIFT.value:
        return _shift(X, trans_args)
    elif trans_args.get('type') == trans_utils.TRANSFORMATION.FLIP.value:
        return _flip(X, trans_args)
    elif trans_args.get('type') == trans_utils.TRANSFORMATION.AFFINE_TRANS.value:
        return _affine_trans(X, trans_args)
    elif trans_args.get('type') == trans_utils.TRANSFORMATION.MORPH_TRANS.value:
        return _morph_trans(X, trans_args)
    elif trans_args.get('type') == trans_utils.TRANSFORMATION.AUGMENT.value:
        return _augment_trans(X, trans_args)
    elif trans_args.get('type') == trans_utils.TRANSFORMATION.CARTOON.value:
        return _cartoon_trans(X, trans_args)
    elif trans_args.get('type') == trans_utils.TRANSFORMATION.QUANTIZATION.value:
        return _quant_trans(X, trans_args)
    elif trans_args.get('type') == trans_utils.TRANSFORMATION.DISTORTION.value:
        return _distort_trans(X, trans_args)
    elif trans_args.get('type') == trans_utils.TRANSFORMATION.NOISE.value:
        return _noise_trans(X, trans_args)
    elif trans_args.get('type') == trans_utils.TRANSFORMATION.FILTER.value:
        return _filter_trans(X, trans_args)
    elif trans_args.get('type') == trans_utils.TRANSFORMATION.COMPRESSION.value:
        return _compression_trans(X, trans_args)
    elif trans_args.get('type') == trans_utils.TRANSFORMATION.DENOISE.value:
        return _denoise_trans(X, trans_args)
    elif trans_args.get('type') == trans_utils.TRANSFORMATION.GEOMETRIC.value:
        return _geometric_trans(X, trans_args)
    elif trans_args.get('type') == trans_utils.TRANSFORMATION.SEGMENTATION.value:
        return _segment_trans(X, trans_args)
    else:
        raise ValueError('{} is not supported.'.format(trans_args.get('type')))


def _rotate(original_images, trans_args):
    """
    Rotate images.
    :param: original_images - the images to rotate.
    :param: process - an instance of Rotation class
    :return: the rotated images
    """
    angle = trans_args.get('angle', 90)
    scale = trans_args.get('scale', 1.0)
    transformed_images = []

    if len(original_images.shape) == 4:
        nb_images, img_rows, img_cols, nb_channels = original_images.shape
    else:
        nb_images, img_rows, img_cols = original_images.shape
        nb_channels = 1

    center = (img_rows/2, img_cols/2)
    trans_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)

    for img in original_images:
        transformed_images.append(cv2.warpAffine(img, trans_matrix, (img_cols, img_rows)))

    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))

    return transformed_images


def _shift(original_images, trans_args):
    """
    Shift/Translate images.
    :param: original_images - the images to shift.
    :param: process - an instance of Shift class.
    :return: the shifted images.
    """
    # -----------------------------------------
    # Shift images in (tx, ty) direction, by specific offsets in width and/or height.
    # Given shift direction (tx, ty), we can create the
    # transformation matrix M as follows:
    #
    # M = [[1, 0, tx],
    #      [0, 1, ty]]
    #
    # -----------------------------------------

    x_offset = trans_args.get('x_offset', 0.15)
    y_offset = trans_args.get('y_offset', 0.15)

    if len(original_images.shape) == 4:
        nb_images, img_rows, img_cols, nb_channels = original_images.shape
    else:
        nb_images, img_rows, img_cols = original_images.shape
        nb_channels = 1

    tx = x_offset * img_cols
    ty = y_offset * img_rows

    trans_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    transformed_images = []
    for img in original_images:
        transformed_images.append(cv2.warpAffine(img, trans_matrix, (img_cols, img_rows)))

    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        # reshape a 3d array to a 4d array
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))

    return transformed_images


def _flip(original_images, trans_args):
    """
    Flip images.
    :param: original_images - the images to applied transformations on.
    :param: process - the standard transformation to apply.
    :return: the flipped images.
    """
    direction = trans_args.get('direction', 0)

    if direction not in [-1, 0, 1]:
        raise ValueError('Invalid flipping direction. Available direction values are -1, 0, and 1.')

    if len(original_images.shape) == 4:
        nb_images, img_rows, img_cols, nb_channels = original_images.shape
    else:
        nb_images, img_rows, img_cols = original_images.shape
        nb_channels = 1

    # flip images
    transformed_images = []
    for img in original_images:
        transformed_images.append(cv2.flip(img, direction))

    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        # reshape a 3d array to a 4d array
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))

    return transformed_images


def _affine_trans(original_images, trans_args):
    """
    Apply affine transformation on images.
    :param: original_images - the images to applied transformations on.
    :param: transformation - the standard transformation to apply.
    :return: the transformed dataset.
    """
    """
    In affine transformation, all parallel lines in the original image will still be parallel in the transformed image.
    To find the transformation matrix, we need to specify 3 points from the original image 
    and their corresponding locations in transformed image. Then, the transformation matrix M (2x3) 
    can be generated by getAffineTransform()
    """

    origin_offset1 = trans_args.get('origin_point1', (0.25, 0.25))
    origin_offset2 = trans_args.get('origin_point2', (0.25, 0.5))
    origin_offset3 = trans_args.get('origin_point3', (0.5, 0.25))

    new_offset1 = trans_args.get('new_point1', (0.25, 0.32))
    new_offset2 = trans_args.get('new_point2', (0.25, 0.48))
    new_offset3 = trans_args.get('new_point3', (0.5, 0.32))

    if len(original_images.shape) == 4:
        nb_images, img_rows, img_cols, nb_channels = original_images.shape
    else:
        nb_images, img_rows, img_cols = original_images.shape
        nb_channels = 1

    origin_point1 = [origin_offset1[0] * img_cols, origin_offset1[1] * img_rows]
    origin_point2 = [origin_offset2[0] * img_cols, origin_offset2[1] * img_rows]
    origin_point3 = [origin_offset3[0] * img_cols, origin_offset3[1] * img_rows]
    # original locations
    pts_origin = np.float32([origin_point1, origin_point2, origin_point3])

    new_point1 = [new_offset1[0] * img_cols, new_offset1[1] * img_rows]
    new_point2 = [new_offset2[0] * img_cols, new_offset2[1] * img_rows]
    new_point3 = [new_offset3[0] * img_cols, new_offset3[1] * img_rows]
    # transformed locations
    pts_transformed = np.float32([new_point1, new_point2, new_point3])

    # transformation matrix
    trans_martix = cv2.getAffineTransform(pts_origin, pts_transformed)

    # applying an affine transformation over the dataset
    transformed_images = []
    for img in original_images:
        transformed_images.append(cv2.warpAffine(img, trans_martix, (img_cols, img_rows)))

    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        # reshape a 3d array to a 4d array
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))

    return transformed_images


def _morph_trans(original_images, trans_args):
    """
    Apply morphological transformations on images.
    :param: original_images - the images to applied transformations on.
    :param: transformation - the standard transformation to apply.
    :return: the transformed dataset.
    """
    if len(original_images.shape) == 4:
        nb_images, img_rows, img_cols, nb_channels = original_images.shape
    else:
        nb_images, img_rows, img_cols = original_images.shape
        nb_channels = 1

    morph_trans = trans_args.get('subtype', trans_utils.MORPH_TRANSFORMATIONS.OPENING.value)
    op = trans_utils.get_morph_op(morph_trans)
    kernel = trans_args.get('kernel', [2, 2])
    kernel = np.ones(tuple(kernel), np.uint8)

    transformed_images = []
    if morph_trans in [trans_utils.MORPH_TRANSFORMATIONS.EROSION.value,
                       trans_utils.MORPH_TRANSFORMATIONS.DILATION.value]:
        for img in original_images:
            iterations = trans_args.get('iterations', 1)
            transformed_images.append(cv2.morphologyEx(src=img, op=op, kernel=kernel, iterations=iterations))
    else:
        for img in original_images:
            transformed_images.append(cv2.morphologyEx(src=img, op=op, kernel=kernel))

    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))

    return transformed_images


def _augment_trans(original_images, trans_args):
    """
    Image augmentation.
    :param: original_images - the images to applied transformations on.
    :param: transformation - the standard transformation to apply.
    :return: the transformed dataset.
    """
    if len(original_images.shape) == 4:
        nb_images, img_rows, img_cols, nb_channels = original_images.shape
    else:
        nb_images, img_rows, img_cols = original_images.shape
        nb_channels = 1

    augment_trans = trans_args.get('subtype')

    data_generator = None
    if augment_trans == trans_utils.AUGMENT_TRANSFORMATIONS.SAMPLEWISE_AUGMENTATION.value:
        data_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
    elif augment_trans == trans_utils.AUGMENT_TRANSFORMATIONS.FEATURE_AUTMENTATION.value:
        data_generator = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    elif augment_trans == trans_utils.AUGMENT_TRANSFORMATIONS.ZCA_WHITENING.value:
        epsilon = augment_trans.get('epsilon', 1e-3)
        min_brightness = augment_trans.get('min_brightness', -100)
        max_brightness = augment_trans.get('max_brightness', 100)

        data_generator = ImageDataGenerator(zca_whitening=True, zca_epsilon=epsilon,
                                            brightness_range=(min_brightness, max_brightness))
    elif augment_trans == trans_utils.AUGMENT_TRANSFORMATIONS.PCA_WHITENING.value:
        raise NotImplementedError('{} is not implemented yet.'.format(augment_trans))
    else:
        raise ValueError('{} is not supported.'.format(augment_trans))

    # fig parameters from data
    data_generator.fit(original_images)
    batch_size = 128
    count = 0

    transformed_images = []
    for X_batch in data_generator.flow(original_images, shuffle=False, batch_size=batch_size):
        for img in X_batch:
            transformed_images.append(img)
            count += 1

        if (count >= nb_images):
            break

    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))

    return transformed_images


def _cartoon_trans(original_images, trans_args):
    """
    Configure for each type of cartoon effect.
    :param original_images:
    :param transformation:
    :return:
    """
    if len(original_images.shape) == 4:
        nb_images, img_rows, img_cols, nb_channels = original_images.shape
    else:
        nb_images, img_rows, img_cols = original_images.shape
        nb_channels = 1

    blur_ksize = trans_args.get('blur_ksize', 3)
    adaptive_method_name = trans_args.get('thresh_adaptive_method', trans_utils.CARTOON_ADAPTIVE_METHODS.MEAN.value)
    adaptive_method = trans_utils.get_cartoon_adpative_method(adaptive_method_name)
    thresh_method_name = trans_args.get('thresh_method', trans_utils.CARTOON_THRESH_METHODS.BINARY.value)
    thresh_method = trans_utils.get_cartoon_thresh_method(thresh_method_name)
    thresh_bsize = trans_args.get('thresh_bsize', 9)
    thresh_C = trans_args.get('thresh_C', 9)

    filter_d = trans_args.get('filter_d', 6)
    filter_sigma_color = trans_args.get('filter_sigma_color', 50)
    filter_sigma_space = trans_args.get('filter_sigma_space', 300)
    nb_downsampling = trans_args.get('nb_downsampling', 2)
    nb_bilateral = trans_args.get('nb_bilateral', 3)

    transformed_images = []
    original_images *= 255.
    for img in original_images:
        img = np.asarray(img, np.uint8)
        img_color = img

        """
        step 1. edge-aware smoothing using a bilateral filter
        """
        # downsample the image using Gaussian pyramid
        for _ in range(nb_downsampling):
            img_color = cv2.pyrDown(img_color)
        # repeatedly apply small bilateral filter instead of applying one large filter
        for _ in range(nb_bilateral):
            img_color = cv2.bilateralFilter(src=img_color, d=filter_d,
                                            sigmaColor=filter_sigma_color,
                                            sigmaSpace=filter_sigma_space)
        # upsample the image
        for _ in range(nb_downsampling):
            img_color = cv2.pyrUp(img_color)

        """
        step 2. reduce noise using a median filter
        """
        if (nb_channels == 3):
            # convert to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
        # apply median blur
        img_blur = cv2.medianBlur(src=img_gray, ksize=blur_ksize)

        """
        step 3. create an edge mask using adaptive thresholding
        """
        img_edges = cv2.adaptiveThreshold(src=img_blur, maxValue=255,
                                          adaptiveMethod=adaptive_method,
                                          thresholdType=thresh_method,
                                          blockSize=thresh_bsize, C=thresh_C)
        """
        step 4. combine color image with edge mask
        """
        if (nb_channels == 3):
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

        img_cartoon = cv2.bitwise_and(img_color, img_edges)
        transformed_images.append(img_cartoon/255.)

    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
    return transformed_images


def _quant_trans(original_images, trans_args):
    """
    Adapted from tutorial
    https://www.pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/
    :param original_images:
    :param transformation:
    :return:
    """
    if len(original_images.shape) == 4:
        nb_images, img_rows, img_cols, nb_channels = original_images.shape
    else:
        nb_images, img_rows, img_cols = original_images.shape
        nb_channels = 1

    nb_clusters = trans_args.get('nb_clusters', 4)

    transformed_images = []
    for img in original_images:
        img_type = img.dtype
        try:
            """
            Convert gray scale images to RGB color space such that
            we can further convert the image to LAB color space.
            This function will return a 3-channel gray image that
            each channel is a copy of the original gray image.
            """
            if (nb_channels == 1):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            """
            Convert the image from the RGB color space to the LAB color space,
            since we will be clustering using k-means which is based on
            the euclidean distance, we will use the LAB color space where
            the euclidean distance implies perceptual meaning.
            """
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            """
            reshape the image into a feature vector so that k-mean can be applied
            """
            img = img.reshape((img_rows * img_cols, 3))
            """
            apply k-means using the specified number of clusters and then
            create the quantized image based on the predictions.
            """
            cluster = MiniBatchKMeans(n_clusters=nb_clusters)
            labels = cluster.fit_predict(img)
            quant = cluster.cluster_centers_[labels]

            """
            reshape the feature vectors back to image
            """
            quant = quant.reshape((img_rows, img_cols, 3))
            """
            convert from LAB back to RGB
            """
            quant = cv2.cvtColor(quant, cv2.COLOR_Lab2RGB)
            """
            convert from RGB back to grayscale
            """
            if (nb_channels == 1):
                quant = cv2.cvtColor(quant, cv2.COLOR_RGB2GRAY)
        except:
            print('!!! Failed to apply cluster[{}].'.format(nb_clusters))
            quant = img
        transformed_images.append(quant.astype(img_type))

    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        # reshape a 3d array to a 4d array
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))

    return transformed_images


def _distort_trans(original_images, trans_args):
    if len(original_images.shape) == 4:
        nb_images, img_rows, img_cols, nb_channels = original_images.shape
    else:
        nb_images, img_rows, img_cols = original_images.shape
        nb_channels = 1

    distort_trans = trans_args.get('subtype')
    transformed_images = []
    if distort_trans in [trans_utils.DISTORT_TRANSFORMATIONS.X.value,
                         trans_utils.DISTORT_TRANSFORMATIONS.Y.value, ]:
        r1 = trans_args.get('r1', 5.)
        r2 = trans_args.get('r2', 2.)
        c = trans_args.get('c', 28.)

        a = c / r1
        w = r2 / c
        shift_func = lambda x: a * np.sin(np.pi * x * w)
        shift_func = trans_args.get('shift_func', shift_func)

        if distort_trans == trans_utils.DISTORT_TRANSFORMATIONS.X.value:
            for img in original_images:
                for i in range(img_rows):
                    img[:, i] = np.roll(img[:, i], int(shift_func(i)))
                transformed_images.append(img)
        else:
            for img in original_images:
                for i in range(img_cols):
                    img[i, :] = np.roll(img[i, :], int(shift_func(i)))
                transformed_images.append(img)

    elif distort_trans == trans_utils.DISTORT_TRANSFORMATIONS.PIXELATE.value:
        new_size = trans_args.get('new_size', (16, 16))
        resample = trans_args.get('resample')
        resample_method = trans_utils.get_distort_resample(resample)

        for img in original_images:
            img = Image.fromarray(img, 'RGB')
            # resize smoothly down
            img = img.resize(new_size, resample=resample_method)
            img = img.resize((img_rows, img_cols), resample=resample_method)
            img = np.array(img)
            transformed_images.append(img)
    elif distort_trans == trans_utils.DISTORT_TRANSFORMATIONS.CONTRAST.value:
        c = trans_args.get('c', 0.1)
        min_pixel_val = trans_args.get('min_pixel_val', 0.)
        max_pixel_val = trans_args.get('max_pixel_val', 1.)
        if nb_channels == 1:
            for img in original_images:
                means = np.mean(img, axis=0, keepdims=True)
                img = np.clip((img - means) * c + means, min_pixel_val, max_pixel_val)
                transformed_images.append(img)
        else:
            original_images *= 255.
            max_pixel_val *= 255.

            for img in original_images:
                means = np.mean(img, axis=(0, 1), keepdims=True)
                img = np.clip((img - means) * c + means, min_pixel_val, max_pixel_val)
                transformed_images.append(img/255.)
    elif distort_trans == trans_utils.DISTORT_TRANSFORMATIONS.BRIGHTNESS.value:
        c = trans_args.get('c', 0.99)
        min_pixel_val = trans_args.get('min_pixel_val', 0.)
        max_pixel_val = trans_args.get('max_pixel_val', 1.)

        if nb_channels == 1:
            for img in original_images:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = color.rgb2hsv(img)
                img[:, :, 2] = np.clip(img[:, :, 2] + c, min_pixel_val, max_pixel_val)
                img = color.hsv2rgb(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                transformed_images.append(img)
        else:
            original_images *= 255.
            max_pixel_val *= 255.
            for img in original_images:
                img = color.rgb2hsv(img)
                img[:, :, 2] = np.clip(img[:, :, 2] + c, min_pixel_val, max_pixel_val)
                img = color.hsv2rgb(img)
                transformed_images.append(img/255.)
    else:
        raise ValueError('{} is not supported.'.format(distort_trans))

    transformed_images = np.stack(transformed_images, axis=0)
    if nb_channels == 1:
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
    return transformed_images


def _noise_trans(original_images, trans_args):
    """
    Adding noise to given images.
    :param original_images:
    :param transformation:
    :return:
    """
    if len(original_images.shape) == 4:
        nb_images, img_rows, img_cols, nb_channels = original_images.shape
    else:
        nb_images, img_rows, img_cols = original_images.shape
        nb_channels = 1

    noise = trans_args.get('noise')
    transformed_images = []
    for img in original_images:
        try:
            img = util.random_noise(img, mode=noise)
        except:
            print('!!! Failed to add noise [{}].'.format(noise))
        # plt.imshow(img)
        transformed_images.append(img)
    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
    return transformed_images


def _filter_trans(original_images, trans_args):
    if len(original_images.shape) == 4:
        nb_images, img_rows, img_cols, nb_channels = original_images.shape
    else:
        nb_images, img_rows, img_cols = original_images.shape
        nb_channels = 1

    filter_trans = trans_args.get('subtype')
    op = trans_utils.get_filter_op(filter_trans)

    transformed_images = []
    if filter_trans in [trans_utils.FILTER_TRANSFORMATION.SOBEL.value,
                        trans_utils.FILTER_TRANSFORMATION.ROBERTS.value,
                        trans_utils.FILTER_TRANSFORMATION.SCHARR.value,
                        trans_utils.FILTER_TRANSFORMATION.PREWITT.value,
                        trans_utils.FILTER_TRANSFORMATION.SKELETONIZE.value]:
        for img in original_images:
            if (nb_channels == 3):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img.reshape(img_rows, img_cols)
            img = op(img)
            if (nb_channels == 3):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            transformed_images.append(img)
    elif filter_trans in [trans_utils.FILTER_TRANSFORMATION.MEDIAN.value,
                          trans_utils.FILTER_TRANSFORMATION.MINIMUM.value,
                          trans_utils.FILTER_TRANSFORMATION.MAXIMUM.value,
                          trans_utils.FILTER_TRANSFORMATION.SATO.value,
                          trans_utils.FILTER_TRANSFORMATION.FRANGI.value,
                          trans_utils.FILTER_TRANSFORMATION.HESSIAN.value]:
        size = trans_args.get('size', 3)

        for img in original_images:
            img = op(img, size=size)
            transformed_images.append(img)
    elif filter_trans == trans_utils.FILTER_TRANSFORMATION.RANK.value:
        size = trans_args.get('size', 3)
        rank = trans_args.get('rank', 15)

        for img in original_images:
            img = op(img, rank=rank, size=size)
            transformed_images.append(img)
    elif filter_trans == trans_utils.FILTER_TRANSFORMATION.GAUSSIAN.value:
        sigma = trans_args.get('sigma', 1)

        for img in original_images:
            img = op(img, sigma=sigma)
            transformed_images.append(img)
    elif filter_trans == trans_utils.FILTER_TRANSFORMATION.MEIJERING.value:
        sigmas = trans_args.get('sigmas', [0.01])

        for img in original_images:
            if nb_channels == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = op(img, sigmas=sigmas)
            if nb_channels == 1:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            transformed_images.append(img)
    elif filter_trans == trans_utils.FILTER_TRANSFORMATION.ENTROPY.value:
        radius = trans_args.get('radius', 2)

        for img in original_images:
            if (nb_channels == 3):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img.reshape((img_rows, img_cols))
            # requires values in range [-1., 1.]
            img = (img - 0.5) / 2.
            # skimage-entropy returns values in float64,
            # however, opencv supports only float32.
            img = np.float32(op(img, disk(radius=radius)))
            # rescale to [0., 1.]
            img = (img / 2.) + 0.5
            if (nb_channels == 3):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            transformed_images.append(img)
    elif filter_trans == trans_utils.FILTER_TRANSFORMATION.THIN.value:
        max_iter = trans_args.get('max_iter', 100)

        for img in original_images:
            if (nb_channels == 3):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img.reshape((img_rows, img_cols))
            img = op(img, max_iter=max_iter)
            if (nb_channels == 3):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            transformed_images.append(img)
    else:
        raise ValueError('{} is not supported.'.format(filter_trans))

    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))

    return transformed_images


def _compression_trans(original_images, trans_args):
    """
    :param original_images:
    :param transformation:
    :return:
    """
    if len(original_images.shape) == 4:
        nb_images, img_rows, img_cols, nb_channels = original_images.shape
    else:
        nb_images, img_rows, img_cols = original_images.shape
        nb_channels = 1

    format = trans_args.get('format', trans_utils.COMPRESS_FORMAT.PNG)
    rate = trans_args.get('rate', 80)
    encode_param = trans_utils.get_compress_encoder(format, rate)

    transformed_images = []
    for img in original_images:
        img *= 255.
        result, encoded_img = cv2.imencode(ext=format, img=img, params=encode_param)
        if False == result:
            print('Failed to encode image to {} format.'.format(format))
            quit()

        # decode the image from encoded image
        decoded_img = cv2.imdecode(buf=encoded_img, flags=1)
        if (nb_channels == 1):
            decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_RGB2GRAY)
        transformed_images.append(decoded_img/255.)

    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
    return transformed_images


def _denoise_trans(original_images, trans_args):
    """
    denoising transformation
    :param original_images:
    :param transformation:
    :return:
    """
    if len(original_images.shape) == 4:
        nb_images, img_rows, img_cols, nb_channels = original_images.shape
    else:
        nb_images, img_rows, img_cols = original_images.shape
        nb_channels = 1

    denoise_trans = trans_args.get('subtype')

    transformed_images = []
    if denoise_trans == trans_utils.DENOISE_TRANSFORMATIONS.WAVELET.value:
        method = trans_args.get('method', 'VisuShrink')  # any option in ['VisuShrink', 'BayesShrink']
        mode = trans_args.get('mode', 'soft')  # any option in ['soft', 'hard']
        wavelet = trans_args.get('wavelet', 'db1')  # any option in pywt.wavelist
        sigma = trans_args.get('sigma', None)  # float or list, optional

        for img in original_images:
            if sigma is None:
                sigma_est = estimate_sigma(img, multichannel=True, average_sigmas=True)
            else:
                sigma_est = sigma

            img = denoise_wavelet(img, wavelet=wavelet, multichannel=True,
                                  convert2ycbcr=False, method=method, mode=mode,
                                  sigma=sigma_est)
            transformed_images.append(img)
    elif denoise_trans == trans_utils.DENOISE_TRANSFORMATIONS.TV_CHAMBOLLE.value:
        # default 0.4 (grayscale); 0.07 (color image)
        weight = trans_args.get('weight', 0.4)
        epsilon = trans_args.get('epsilon', 2.e-4)
        max_iter = trans_args.get('max_iter', 200)

        for img in original_images:
            img = denoise_tv_chambolle(img, weight=weight, eps=epsilon,
                                       n_iter_max=max_iter, multichannel=True)
            transformed_images.append(img)
    elif denoise_trans == trans_utils.DENOISE_TRANSFORMATIONS.TV_BREGMAN.value:
        # default 2 (grayscale); 15 (color image)
        weight = trans_args.get('weight', 2)
        epsilon = trans_args.get('epsilon', 1e-6)
        max_iter = trans_args.get('max_iter', 50)

        for img in original_images:
            img_trans = denoise_tv_bregman(img, eps=epsilon, max_iter=max_iter, weight=weight)
            transformed_images.append(img_trans)
    elif denoise_trans == trans_utils.DENOISE_TRANSFORMATIONS.BILATERAL.value:
        sigma_color = np.double(trans_args.get('sigma_color', 0.05))
        sigma_spatial = np.double(trans_args.get('sigma_spatial', 15.0))

        for img in original_images:
            img_trans = denoise_bilateral(img, sigma_color=sigma_color,
                                          sigma_spatial=sigma_spatial, multichannel=True)
            transformed_images.append(img_trans)
    elif denoise_trans in [trans_utils.DENOISE_TRANSFORMATIONS.NL_MEANS.value,
                           trans_utils.DENOISE_TRANSFORMATIONS.NL_MEANS_FAST.value]:
        patch_kw = dict(patch_size=trans_args.get('patch_size', 5),  # 5x5 patch
                        patch_distance=trans_args.get('patch_distance', 6),  # 13x13 search area
                        multichannel=True)
        sigma = trans_args.get('sigma', None)
        hr = trans_args.get('hr', 0.8)
        # Athena default: 1 (mnist); 2.5 (cifar100)
        sr = trans_args.get('sr', 1)
        fast_mode = False if denoise_trans == trans_utils.DENOISE_TRANSFORMATIONS.NL_MEANS.value else True

        for img in original_images:
            # estimate the noise standard deviation from the noisy image
            if sigma is None:
                sigma_est = np.mean(estimate_sigma(img, multichannel=True))
            else:
                sigma_est = sigma
            img = denoise_nl_means(img, h=hr * sigma_est, sigma=sr * sigma_est,
                                   fast_mode=fast_mode, **patch_kw)
            transformed_images.append(img)
    else:
        raise ValueError('{} is not supported.'.format(denoise_trans))

    # stack images
    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
    return transformed_images


def _geometric_trans(original_images, trans_args):
    """
    geometric transformations
    :param original_images:
    :param transformation:
    :return:
    """
    if len(original_images.shape) == 4:
        nb_images, img_rows, img_cols, nb_channels = original_images.shape
    else:
        nb_images, img_rows, img_cols = original_images.shape
        nb_channels = 1

    geo_trans = trans_args.get('subtype')
    op = trans_utils.get_geometric_op(geo_trans)
    transformed_images = []
    if geo_trans == trans_utils.GEOMETRIC_TRANSFORMATIONS.SWIRL.value:
        # athena default: 3 (mnist); 1.5 (cifar100)
        strength = trans_args.get('strength', 3)
        # athena default: 65 (mnist); 45 (cifar100)
        radius = trans_args.get('radius', 65)
        center = trans_args.get('center', None)
        rotation = trans_args.get('rotation', 0)
        order = trans_args.get('order', 1)
        mode = trans_args.get('mode', 'reflect')

        for img in original_images:
            img = op(img, center=center, strength=strength,
                     radius=radius, rotation=rotation,
                     order=order, mode=mode)
            transformed_images.append(img)
    elif geo_trans in [trans_utils.GEOMETRIC_TRANSFORMATIONS.IRADON.value,
                       trans_utils.GEOMETRIC_TRANSFORMATIONS.IRADON_SART.value]:
        default_theta = np.linspace(start=trans_args.get('ls_start', -100),
                                    stop=trans_args.get('ls_stop', 150),
                                    num=trans_args.get('ls_num', 28),  # e.g., max(image.shape)
                                    endpoint=False)
        theta = trans_args.get('theta', default_theta)  # array_like, dtype=float, optional
        filter = trans_args.get('filter', 'ramp')  # ramp, shepp-logan, cosine, hamming, hann, or None
        interpolation = trans_args.get('interpolation', 'linear')  # 'linear', 'nearest', or 'cubic' (slow)
        circle = True

        for img in original_images:
            img = (img - 0.5) * 2.
            if (nb_channels == 3):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img.reshape((img_rows, img_cols))
            img = radon(img, theta=theta, circle=circle)
            if geo_trans == trans_utils.GEOMETRIC_TRANSFORMATIONS.IRADON.value:
                img = np.float32(op(img, theta=theta, filter=filter,
                                    interpolation=interpolation, circle=circle))
            else:
                img = np.float32(op(img, theta=theta))
            if (nb_channels == 3):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = (img / 2.) + 0.5
            transformed_images.append(img)
    elif geo_trans == trans_utils.GEOMETRIC_TRANSFORMATIONS.RADON.value:
        default_theta = np.linspace(start=trans_args.get('ls_start', -100),
                                    stop=trans_args.get('ls_stop', 150),
                                    num=trans_args.get('ls_num', 28),  # e.g., max(image.shape)
                                    endpoint=False)

        theta = trans_args.get('theta', default_theta)  # array_like, dtype=float, optional
        circle = True

        for img in original_images:
            img = (img - 0.5) * 2.
            if (nb_channels == 3):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img.reshape((img_rows, img_cols))
            img = op(img, theta=theta, circle=circle)
            if (nb_channels == 3):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = (img / 2.) + 0.5
            transformed_images.append(img)
    else:
        raise ValueError('{} is not supported.'.format(geo_trans))

    # stack images
    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
    return np.array(transformed_images)


def _segment_trans(original_images, trans_args):
    """
    Segmentation of objects
    :param original_images:
    :param transformation:
    :return:
    """
    if len(original_images.shape) == 4:
        nb_images, img_rows, img_cols, nb_channels = original_images.shape
    else:
        nb_images, img_rows, img_cols = original_images.shape
        nb_channels = 1

    segment_trans = trans_args.get('subtype').lower()
    transformed_images = []
    if segment_trans == trans_utils.SEGMENT_TRANSFORMATIONS.GRADIENT.value:
        median_radius = trans_args.get('median_radius', 2)
        gradient_radius = trans_args.get('gradient_radius', 1)
        for img in original_images:
            # denoise image
            if (nb_channels == 3):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img.reshape(img_rows, img_cols)
            denoised = rank.median(img, disk(median_radius))
            img_trans = rank.gradient(denoised, disk(gradient_radius))
            if (nb_channels == 3):
                img_trans = cv2.cvtColor(img_trans, cv2.COLOR_GRAY2RGB)
            transformed_images.append(img_trans)
    elif segment_trans == trans_utils.SEGMENT_TRANSFORMATIONS.WATERSHED.value:
        median_radius = trans_args.get('median_radius', 2)
        mark_radius = trans_args.get('mark_radius', 5)
        gradient_upper_bound = trans_args.get('gradient_upper_bound', 10)
        gradient_radius = trans_args.get('gradient_radius', 2)

        for img in original_images:
            if (nb_channels == 3):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img.reshape((img_rows, img_cols))
            # denoise image
            denoised = rank.median(img, disk(median_radius))
            # find continuous region (low gradient -
            # where less than 10 for this image) --> markers
            markers = rank.gradient(denoised, disk(mark_radius)) < gradient_upper_bound
            markers = ndimage.label(markers)[0]
            # local gradient (disk(2) is used to keep edges thin)
            gradient = rank.gradient(denoised, disk(gradient_radius))
            img = watershed(gradient, markers)
            if (nb_channels == 3):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            transformed_images.append(img)
    else:
        raise ValueError('{} is not supported.'.format(segment_trans))

    # stack images
    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
    return np.array(transformed_images)
