"""
Implement transformations.
@auther: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
import cv2
from scipy import ndimage

from keras.preprocessing.image import ImageDataGenerator
import skimage
from sklearn.cluster import MiniBatchKMeans
from skimage.restoration import (denoise_bilateral, denoise_nl_means, denoise_tv_bregman, denoise_tv_chambolle, denoise_wavelet, estimate_sigma)
from skimage.transform import (warp, swirl, radon, iradon, iradon_sart)
from skimage.morphology import disk, watershed, skeletonize, thin
from skimage.filters.rank import entropy
from skimage.filters import (rank, roberts, scharr, prewitt, meijering, sato, frangi, hessian)

from config import *
from data import load_data
from plot import draw_comparisons

def rotate(original_images, transformation):
    """
    Rotate images.
    :param: original_images - the images to applied transformations on.
    :param: transformation - the standard transformation to apply.
    :return: the transformed dataset.
    """
    if MODE.DEBUG:
        print('Rotating images({})...'.format(transformation))

    trans_matrix = None
    transformed_images = []
    nb_images, img_rows, img_cols, nb_channels = original_images.shape[:4]
    center = (img_rows/2, img_cols/2)
    
    # ---------------
    # rotate images
    # ---------------
    if (transformation == TRANSFORMATION.rotate90):
        # rotate 90-deg counterclockwise
        angle = 90
        scale = 1.0

        trans_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    elif (transformation == TRANSFORMATION.rotate180):
        # rotate 180-deg counterclockwise
        angle = 180
        scale = 1.0
        
        trans_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    elif (transformation == TRANSFORMATION.rotate270):
        # rotate 270-deg counterclockwise
        angle = 270
        scale = 1.0
        
        trans_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    else:
        raise ValueError('{} is not supported.'.format(transformation))
    
    # applying an affine transformation over the dataset
    transformed_images = []

    for img in original_images:
        transformed_images.append(cv2.warpAffine(img, trans_matrix, (img_cols, img_rows)))

    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        # reshape a 3d array to a 4d array
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))

    if MODE.DEBUG:
        print('shapes: original - {}; transformed - {}'.format(original_images.shape, transformed_images.shape))
        print('Applied transformation {}.'.format(transformation))

    return transformed_images

def shift(original_images, transformation):
    """
    Shift images.
    :param: original_images - the images to applied transformations on.
    :param: transformation - the standard transformation to apply.
    :return: the transformed dataset.
    """
    if MODE.DEBUG:
        print('Shifting images({})...'.format(transformation))

    # -----------------------------------------
    # Shift images in (tx, ty) direction, by 15% of width and/or height.
    # Given shift direction (tx, ty), we can create the
    # transformation matrix M as follows:
    #
    # M = [[1, 0, tx],
    #      [0, 1, ty]]
    #
    # -----------------------------------------
    nb_images, img_rows, img_cols, nb_channels = original_images.shape[:4]
    tx = int(0.15 * img_cols)
    ty = int(0.15 * img_rows)

    if (transformation == TRANSFORMATION.shift_left):
        tx = 0 - tx
        ty = 0
    elif (transformation == TRANSFORMATION.shift_right):
        tx = tx
        ty = 0
    elif (transformation == TRANSFORMATION.shift_up):
        tx = 0
        ty = 0 - ty
    elif (transformation == TRANSFORMATION.shift_down):
        tx = 0
        ty = ty
    elif (transformation == TRANSFORMATION.shift_top_right):
        tx = tx
        ty = 0 - ty
    elif (transformation == TRANSFORMATION.shift_top_left):
        tx = 0 - tx
        ty = 0 - ty
    elif (transformation == TRANSFORMATION.shift_bottom_left):
        tx = 0 - tx
        ty = ty
    elif (transformation == TRANSFORMATION.shift_bottom_right):
        tx = tx
        ty = ty
    else:
        raise ValueError('{} is not supported.'.format(transformation))

    transformed_images = []

    # define transformation matrix
    trans_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

    # applying an affine transformation over the dataset
    transformed_images = []
    for img in original_images:
        transformed_images.append(cv2.warpAffine(img, trans_matrix, (img_cols, img_rows)))
    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        # reshape a 3d array to a 4d array
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))

    return transformed_images

def flip(original_images, transformation):
    """
    Flip images.
    :param: original_images - the images to applied transformations on.
    :param: transformation - the standard transformation to apply.
    :return: the transformed dataset.
    """
    if MODE.DEBUG:
        print('Flipping images({})...'.format(transformation))
    nb_images, img_rows, img_cols, nb_channels = original_images.shape

    # set flipping direction
    flip_direction = 0
    if (transformation == TRANSFORMATION.vertical_flip):
        # flip around the x-axis
        flip_direction = 0
    elif (transformation == TRANSFORMATION.horizontal_flip):
        # flip around the y-axis
        flip_direction = 1
    elif (transformation == TRANSFORMATION.both_flip):
        # flip around both axes
        flip_direction = -1
    else:
        raise ValueError('{} is not supported.'.format(transformation))

    # flip images
    transformed_images = []
    for img in original_images:
        transformed_images.append(cv2.flip(img, flip_direction))
    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        # reshape a 3d array to a 4d array
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))

    if MODE.DEBUG:
        print('shapes: original - {}; transformed - {}'.format(original_images.shape, transformed_images.shape))

    return transformed_images

def affine_trans(original_images, transformation):
    """
    Apply affine transformation on images.
    :param: original_images - the images to applied transformations on.
    :param: transformation - the standard transformation to apply.
    :return: the transformed dataset.
    """
    if MODE.DEBUG:
        print('Applying affine transformation on images({})...'.format(transformation))

    """
    In affine transformation, all parallel lines in the original image will still be parallel in the transformed image.
    To find the transformation matrix, we need to specify 3 points from the original image 
    and their corresponding locations in transformed image. Then, the transformation matrix M (2x3) 
    can be generated by getAffineTransform()
    """
    nb_images, img_rows, img_cols, nb_channels = original_images.shape
    point1 = [0.25 * img_cols, 0.25 * img_rows]
    point2 = [0.25 * img_cols, 0.5 * img_rows]
    point3 = [0.5 * img_cols, 0.25 * img_rows]

    pts_original = np.float32([point1, point2, point3])

    if (transformation == TRANSFORMATION.affine_vertical_compress):
        point1 = [0.25 * img_cols, 0.32 * img_rows]
        point2 = [0.25 * img_cols, 0.48 * img_rows]
        point3 = [0.5 * img_cols, 0.32 * img_rows]
    elif (transformation == TRANSFORMATION.affine_vertical_stretch):
        point1 = [0.25 * img_cols, 0.2 * img_rows]
        point2 = [0.25 * img_cols, 0.55 * img_rows]
        point3 = [0.5 * img_cols, 0.2 * img_rows]
    elif (transformation == TRANSFORMATION.affine_horizontal_compress):
        point1 = [0.32 * img_cols, 0.25 * img_rows]
        point2 = [0.32 * img_cols, 0.5 * img_rows]
        point3 = [0.43 * img_cols, 0.25 * img_rows]
    elif (transformation == TRANSFORMATION.affine_horizontal_stretch):
        point1 = [0.2 * img_cols, 0.25 * img_rows]
        point2 = [0.2 * img_cols, 0.5 * img_rows]
        point3 = [0.55 * img_cols, 0.25 * img_rows]
    elif (transformation == TRANSFORMATION.affine_both_compress):
        point1 = [0.28 * img_cols, 0.28 * img_rows]
        point2 = [0.28 * img_cols, 0.47 * img_rows]
        point3 = [0.47 * img_cols, 0.28 * img_rows]
    elif (transformation == TRANSFORMATION.affine_both_stretch):
        point1 = [0.22 * img_cols, 0.22 * img_rows]
        point2 = [0.22 * img_cols, 0.55 * img_rows]
        point3 = [0.55 * img_cols, 0.22 * img_rows]
    else:
        raise ValueError('{} is not supported.'.format(transformation))

    # define transformation matrix
    pts_transformed = np.float32([point1, point2, point3])
    trans_matrix = cv2.getAffineTransform(pts_original, pts_transformed)

    # applying an affine transformation over the dataset
    transformed_images = []
    for img in original_images:
        transformed_images.append(cv2.warpAffine(img, trans_matrix, (img_cols, img_rows)))
    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        # reshape a 3d array to a 4d array
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))

    if MODE.DEBUG:
        print('shapes: original - {}; transformed - {}'.format(original_images.shape, transformed_images.shape))
        print('Applied transformation {}.'.format(transformation))

    return transformed_images

def morph_trans(original_images, transformation):
    """
    Apply morphological transformations on images.
    :param: original_images - the images to applied transformations on.
    :param: transformation - the standard transformation to apply.
    :return: the transformed dataset.
    """
    if MODE.DEBUG:    
        print('Applying morphological transformation ({})...'.format(transformation))

    nb_images, img_rows, img_cols, nb_channels = original_images.shape
    # set kernel as a matrix of size 2
    kernel = np.ones((2, 2),np.uint8)

    transformed_images = []
    if (transformation == TRANSFORMATION.dilation):
        # min filter (Graphics Mill)
        # It's opposite of erosion (max filter)
        # In dilation, a pixel element is '1' if at least one pixel
        # under the kernel is '1'. So it increases the white region
        # in the image or size of foreground object increases.
        for img in original_images:
            transformed_images.append(cv2.dilate(img, kernel, iterations=1))
    elif (transformation == TRANSFORMATION.erosion):
        # max filter (Graphic Mill)
        # The basic idea of erosion is like soil erosion.
        # It erodes away the boundaries of foreground object
        # (always try to keep foreground in white)
        # The kernel slides through the image as in 2D convolution.
        # A pixel in the original image will be considered 1 only if
        # all the pixels under the kernel is 1, otherwise, it's eroded.
        for img in original_images:
            transformed_images.append(cv2.erode(img, kernel, iterations=1))
    elif (transformation == TRANSFORMATION.opening):
        # erosion followed by dilation
        for img in original_images:
            transformed_images.append(cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel))
    elif (transformation == TRANSFORMATION.closing):
        # erosion followed by dilation
        for img in original_images:
            transformed_images.append(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel))
    elif (transformation == TRANSFORMATION.gradient):
        # keep the outline of the object
        for img in original_images:
            transformed_images.append(cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel))
    elif (transformation == TRANSFORMATION.skeletonize):
        for img in original_images:
            img_trans = skeletonize(img)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.thin):
        for img in original_images:
            img_trans = thin(img)
            transformed_images.append(img_trans, max_iter=25)
    else:
        raise ValueError('{} is not supported.'.format(transformation))

    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        # reshape a 3d array to a 4d array
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))

    if MODE.DEBUG:
        print('Applied transformation {}.'.format(transformation))

    return transformed_images

def augment(original_images, transformation):
    """
    Image augmentation.
    :param: original_images - the images to applied transformations on.
    :param: transformation - the standard transformation to apply.
    :return: the transformed dataset.
    """
    nb_images, img_rows, img_cols, nb_channels = original_images.shape
    data_generator = None
    transformed_images = np.zeros_like(original_images)

    if transformation == TRANSFORMATION.samplewise_std_norm:
        data_generator = ImageDataGenerator(samplewise_center=True,
                                            samplewise_std_normalization=True)
    elif transformation == TRANSFORMATION.feature_std_norm:
        data_generator = ImageDataGenerator(featurewise_center=True,
                                            featurewise_std_normalization=True)
    elif transformation == TRANSFORMATION.zca_whitening:
        data_generator = ImageDataGenerator(zca_whitening=True, brightness_range=(-200, 200))
    elif transformation == TRANSFORMATION.pca_whitening:
        raise NotImplementedError('{} is not ready yet.'.format(transformation))
    else:
        raise ValueError('{} is not supported.'.format(transformation))

    # fit parameters from data
    data_generator.fit(original_images)
    batch_size = 128
    cnt_trans = 0
    input_size = len(original_images)

    transformed_images = []
    for X_batch in data_generator.flow(original_images, shuffle=False,  batch_size=batch_size):
        for image in X_batch:
            # transformed_images[cnt_trans] = image
            transformed_images.append(image)
            cnt_trans += 1

        if (cnt_trans >= input_size):
            print('transformed {} inputs.'.format(cnt_trans))
            break;

    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        # reshape a 3d array to a 4d array
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))

    if MODE.DEBUG:
        print('Applied augmentations. ')

    return transformed_images

def cartoon_effect(original_images, **kwargs):
    """
    default type: cartoon_mean_type1
    """
    blur_ksize = kwargs.get('blur_ksize', 3)

    thresh_adaptive_method = kwargs.get('thresh_adaptive_method', cv2.ADAPTIVE_THRESH_MEAN_C)
    thresh_bsize = kwargs.get('thresh_bsize', 9)
    thresh_C = kwargs.get('thresh_C', 9)

    filter_d = kwargs.get('filter_d', 9)
    filter_sigma_color = kwargs.get('filter_sigma_color', 50)
    filter_sigma_space = kwargs.get('filter_sigma_space', 300)

    # number of downsampling steps
    nb_downsampling = kwargs.get('nb_downsampling', 2)
    # number of bilateral filtering steps
    nb_bilateral = kwargs.get('nb_bilateral', 3)

    nb_images, img_rows, img_cols, nb_channels = original_images.shape
    transformed_images = []

    for i in range(original_images.shape[0]):
        img = original_images[i] * 255
        img = np.asarray(img, np.uint8)

        img_color = img
        """
        step 1. edge-aware smoothing using a bilateral filter
        """
        # downsample image using Gaussian pyramid
        for _ in range(nb_downsampling):
            img_color = cv2.pyrDown(img_color)

        # repeatedly apply small bilateral filter instead of applying one large filter
        for _ in range(nb_bilateral):
            img_color = cv2.bilateralFilter(src=img_color,
                                        d=6,
                                        sigmaColor=filter_sigma_color,
                                        sigmaSpace=filter_sigma_space)

        # upsample image
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
        img_edges = cv2.adaptiveThreshold(src=img_blur,
                                          maxValue=255,
                                          adaptiveMethod=thresh_adaptive_method,
                                          thresholdType=cv2.THRESH_BINARY,
                                          blockSize=thresh_bsize,
                                          C=thresh_C)
        """
        step 4. combine color image with edge mask
        """
        if (nb_channels == 3):
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

        img_cartoon = cv2.bitwise_and(img_color, img_edges)

        transformed_images.append(img_cartoon/255.)
    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        # reshape a 3d array to a 4d array
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))

    if MODE.DEBUG:
        print(transformed_images.shape)
        print('Applied cartoon effects.')

    return transformed_images

def cartoonify(original_images, transformation):
    """
    Configure for each type of cartoon effect.
    :param original_images:
    :param transformation:
    :return:
    """
    if MODE.DEBUG:    
        print('Applying transformation {}...'.format(transformation))

    _, img_rows, img_cols, nb_channels = original_images.shape
    adaptive_method = transformation.split('_')[1]
    catoon_type = transformation.split('_')[2]

    """
    default type: cartoon_mean_type1
    """
    # number of downsampling steps
    if (nb_channels == 1):
        nb_downsampling = 0
    else:
        nb_downsampling = 2

    # number of bilateral filtering steps
    if (nb_channels == 1):
        nb_bilateral = 3
    else:
        nb_bilateral = 5

    blur_ksize = 3
    thresh_adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    if (nb_channels == 1):
        thresh_bsize = 9
        thresh_C = 9
    else:
        thresh_bsize = 3
        thresh_C = 3

    filter_d = 9
    if (nb_channels == 1):
        filter_sigma_color = 300
    else:
        filter_sigma_color = 2
    filter_sigma_space = 30

    if (adaptive_method == 'gaussian'):
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        nb_downsampling = 1
        nb_bilateral = 10
        filter_d = 250
    if (catoon_type == 'type2'):
        thresh_bsize = 3
        nb_downsampling = 1
        nb_bilateral = 100
    elif (catoon_type == 'type3'):
        thresh_C = 7
        nb_downsampling = 0
        nb_bilateral = 0
        if (nb_channels == 1):
            nb_downsampling = 2
            filter_sigma_color = 100

    elif (catoon_type == 'type4'):
        thresh_bsize = 5
        thresh_C = 5
        filter_d = 25

    return cartoon_effect(original_images, blur_ksize=blur_ksize,
                          thresh_adaptive_method=thresh_adaptive_method,
                          thresh_bsize=thresh_bsize, thresh_C=thresh_C,
                          filter_d=filter_d, filter_sigma_color=filter_sigma_color,
                          filter_sigma_space=filter_sigma_space,
                          nb_downsampling=nb_downsampling, nb_bilateral=nb_bilateral)

def quantize(original_images, transformation):
    """
    Adapted from tutorial
    https://www.pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/
    :param original_images:
    :param transformation:
    :return:
    """
    nb_clusters = int(transformation.split('_')[1])
    nb_images, img_rows, img_cols, nb_channels = original_images.shape
    transformed_images = []

    for i in range(nb_images):
        img = np.copy(original_images[i])
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

        transformed_images.append(quant)
        del img
    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        # reshape a 3d array to a 4d array
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))

    return transformed_images

def distort(original_images, transformation):
    transformed_images = []

    nb_images, img_rows, img_cols, nb_channels = original_images.shape

    r1 = 5.
    r2 = 2.
    # if (nb_channels == 3):
    #     r1 = 6.
    #     r2 = 1.5
    a = img_rows / r1
    w = r2 / img_cols
    shift = lambda x: a * np.sin(np.pi * x * w)

    if (transformation == TRANSFORMATION.distortion_y):
        for img in original_images:
            img_distorted = np.copy(img)
            for i in range(img_rows):
                img_distorted[i, :] = np.roll(img_distorted[i, :], int(shift(i)))
            transformed_images.append(img_distorted)
    elif (transformation == TRANSFORMATION.distortion_x):
        for img in original_images:
            img_distorted = np.copy(img)
            for i in range(img_rows):
                img_distorted[:, i] = np.roll(img_distorted[:, i], int(shift(i)))
            transformed_images.append(img_distorted)
    else:
        raise ValueError('{} is not supported.'.format(transformation))

    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        # reshape a 3d array to a 4d array
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
    return transformed_images

def filter(original_images, transformation):
    """
    :param original_images:
    :param transformation:
    :return:
    """
    nb_images, img_rows, img_cols, nb_channels = original_images.shape
    transformed_images = []

    if (transformation == TRANSFORMATION.sobel):
        if (nb_channels == 1):
            print('This transformation type ({}) does not support grayscale.'.format(transformation))
            return

        for img in original_images:
            img_trans = ndimage.sobel(img)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.median_filter):
        for img in original_images:
            img_trans = ndimage.median_filter(img, size=3)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.min_filter):
        for img in original_images:
            img_trans = ndimage.minimum_filter(img, size=3)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.max_filter):
        for img in original_images:
            img_trans = ndimage.maximum_filter(img, size=3)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.gaussian_filter):
        for img in original_images:
            img_trans = ndimage.gaussian_filter(img, sigma=1)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.rank_filter):
        for img in original_images:
            img_trans = ndimage.rank_filter(img, rank=15, size=3)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.entropy):
        for img in original_images:
            img_trans = entropy(img, disk(5))
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.roberts):
        for img in original_images:
            img_trans = roberts(img)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.scharr):
        for img in original_images:
            img_trans = scharr(img)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.prewitt):
        for img in original_images:
            img_trans = prewitt(img)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.meijering):
        for img in original_images:
            img_trans = meijering(img)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.sato):
        for img in original_images:
            img_trans = sato(img)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.frangi):
        for img in original_images:
            img_trans = frangi(img)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.hessian):
        for img in original_images:
            img_trans = hessian(img)
            transformed_images.append(img_trans)

    else:
        raise ValueError('{} is not supported.'.format(transformation))

    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        # reshape a 3d to a 4d
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
    return transformed_images

def add_noise(original_images, transformation):
    """
    Adding noise to given images.
    :param original_images:
    :param transformation:
    :return:
    """
    nb_images, img_rows, img_cols, nb_channels = original_images.shape
    transformed_images = []
    noise_mode = transformation.split('_')[1]

    for img in original_images:
        img_noised = skimage.util.random_noise(img, mode=noise_mode)
        transformed_images.append(img_noised)
    transformed_images = np.stack(transformed_images, axis=0)
    if (nb_channels == 1):
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
    return transformed_images

def compress(original_images, transformation):
    """

    :param original_images:
    :param transformation:
    :return:
    """
    original_images *= 255.
    compress_rate = int(transformation.split('_')[-1])
    format = '.{}'.format(transformation.split('_')[1])
    if MODE.DEBUG:
        print(compress_rate, format)
    nb_images, img_rows, img_cols, nb_channels = original_images.shape
    transformed_images = []
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_rate]

    if (format == '.png'):
        encode_param = [cv2.IMWRITE_PNG_COMPRESSION, compress_rate]

    for img in original_images:
        # encode an image
        result, encoded_img = cv2.imencode(format, img, encode_param)
        if False == result:
            print('Failed to encode image to jpeg format.')
            quit()

        # decode the image from encoded image
        decoded_img = cv2.imdecode(encoded_img, 1)
        if (nb_channels == 1):
            decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_RGB2GRAY)
        transformed_images.append(decoded_img/255.)

    transformed_images = np.stack(transformed_images, axis=0)

    if (nb_channels == 1):
        transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
    return transformed_images

def denoising(original_images, transformation):
    """
    denoising transformation
    :param original_images:
    :param transformation:
    :return:
    """
    nb_images, img_rows, img_cols, nb_channels = original_images.shape
    # TODO: checking number of channels and some customization for datasets
    transformed_images = []

    if (transformation == TRANSFORMATION.wavelet):
        for img in original_images:
            img_trans = denoise_wavelet(img, multichannel=True)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.tv_chambolle):
        for img in original_images:
            # TODO: better to consider different variations of weights
            img_trans = denoise_tv_chambolle(img, weight=0.1, multichannel=True)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.tv_bregman):
        for img in original_images:
            img_trans = denoise_tv_bregman(img, eps=1e-3, max_iter=100, weight=100)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.bilateral):
        for img in original_images:
            img_trans = denoise_bilateral(img, sigma_color=0.05, sigma_spatial=15,
                multichannel=True)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.nl):
        patch_kw = dict(patch_size=5,  # 5x5 patches
                        patch_distance=6,  # 13x13 search area
                        multichannel=True)
        for img in original_images:
            sigma_est = np.mean(estimate_sigma(img, multichannel=True))
            img_trans = denoise_nl_means(img, h=0.8 * sigma_est, sigma=sigma_est,
                            fast_mode=False, **patch_kw)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.nl_fast):
        patch_kw = dict(patch_size=5,  # 5x5 patches
                        patch_distance=6,  # 13x13 search area
                        multichannel=True)
        for img in original_images:
            sigma_est = np.mean(estimate_sigma(img, multichannel=True))
            img_trans = denoise_nl_means(img, h=0.6 * sigma_est, sigma=sigma_est,
                                 fast_mode=True, **patch_kw)
            transformed_images.append(img_trans)
    else:
        raise ValueError('{} is not supported.'.format(transformation))

    return transform_images

def geometric_transformations(original_images, transformation):
    """
    geometric transformations
    :param original_images:
    :param transformation:
    :return:
    """
    nb_images, img_rows, img_cols, nb_channels = original_images.shape
    # TODO: checking number of channels and some customization for datasets
    # TODO: more variations, after testing is done
    transformed_images = []

    if (transformation == TRANSFORMATION.randon):
        for img in original_images:
            theta = np.linspace(0., 180., max(img.shape), endpoint=False)
            img_trans = radon(img, theta=theta, circle=True)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.iradon):
        for img in original_images:
            theta = np.linspace(0., 180., max(img.shape), endpoint=False)
            img_trans = iradon(img, theta=theta, circle=True)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.iradon_sart):
        for img in original_images:
            theta = np.linspace(0., 180., max(img.shape), endpoint=False)
            img_trans = iradon_sart(img, theta=theta, circle=True)
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.swirl):
        for img in original_images:
            img_trans = swirl(img, rotation=0, strength=10, radius=120)
            transformed_images.append(img_trans)
    else:
        raise ValueError('{} is not supported.'.format(transformation))

    return transform_images

def segmentations(original_images, transformation):
    """
        Segmentation of objects¶
        :param original_images:
        :param transformation:
        :return:
        """
    nb_images, img_rows, img_cols, nb_channels = original_images.shape
    # TODO: checking number of channels and some customization for datasets
    # TODO: more variations, after testing is done
    transformed_images = []

    if (transformation == TRANSFORMATION.gradient):
        for img in original_images:
            # denoise image
            denoised = rank.median(img, disk(2))
            img_trans = rank.gradient(denoised, disk(2))
            transformed_images.append(img_trans)
    elif (transformation == TRANSFORMATION.watershed):
        for img in original_images:
            # denoise image
            denoised = rank.median(img, disk(2))
            # find continuous region (low gradient -
            # where less than 10 for this image) --> markers
            markers = rank.gradient(denoised, disk(5)) < 10
            markers = ndimage.label(markers)[0]
            # local gradient (disk(2) is used to keep edges thin)
            gradient = rank.gradient(denoised, disk(2))
            img_trans = watershed(gradient, markers)
            transformed_images.append(img_trans)
    else:
        raise ValueError('{} is not supported.'.format(transformation))

    return transform_images

def transform_images(X, transformation_type):
    """
    Main entrance applying transformations on images.
    :param X: the images to apply transformation.
    :param transformation_type:
    :return: the transformed images.
    """
    if (transformation_type == TRANSFORMATION.clean):
        """
        Do not apply any transformation for 'clean' images.
        """
        return X
    elif (transformation_type in TRANSFORMATION.ROTATE):
        return rotate(X, transformation_type)
    elif (transformation_type in TRANSFORMATION.FLIP):
        return flip(X, transformation_type)
    elif (transformation_type in TRANSFORMATION.SHIFT):
        return shift(X, transformation_type)
    elif (transformation_type in TRANSFORMATION.AFFINE_TRANS):
        return affine_trans(X, transformation_type)
    elif (transformation_type in TRANSFORMATION.MORPH_TRANS):
        return morph_trans(X, transformation_type)
    elif (transformation_type in TRANSFORMATION.AUGMENT):
        return augment(X, transformation_type)
    elif (transformation_type in TRANSFORMATION.CARTOONS):
        return cartoonify(X, transformation_type)
    elif (transformation_type in TRANSFORMATION.QUANTIZATIONS):
        return quantize(X, transformation_type)
    elif (transformation_type in TRANSFORMATION.DISTORTIONS):
        return distort(X, transformation_type)
    elif (transformation_type in TRANSFORMATION.FILTERS):
        return filter(X, transformation_type)
    elif (transformation_type in TRANSFORMATION.NOISES):
        return add_noise(X, transformation_type)
    elif (transformation_type in TRANSFORMATION.COMPRESSION):
        return compress(X, transformation_type)
    elif (transformation_type in TRANSFORMATION.DENOISING):
        return denoising(X, transformation_type)
    elif (transformation_type in TRANSFORMATION.GEOMETRIC):
        return geometric_transformations(X, transformation_type)
    elif (transformation_type in TRANSFORMATION.SEGMENTATION):
        return segmentations(X, transformation_type)
    else:
        raise ValueError('Transformation type {} is not supported.'.format(transformation_type.upper()))

"""
for testing
"""
def main(*args):
    print('Transform --- {}'.format(args))
    _, (X, _) = load_data(args[0])
    X_orig = np.copy(X[10:20])
    X_trans = transform_images(X_orig, args[1])
    draw_comparisons(X[10:20], X_trans, '{}-{}'.format(args[0], args[1]))

if __name__ == "__main__":
    MODE.debug_on()
    main(DATA.cifar_10, TRANSFORMATION.compress_png_compression_5)
