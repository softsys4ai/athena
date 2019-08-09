"""
Implement transformations.
@auther: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
import cv2

from keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import MiniBatchKMeans

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
    transformed_images = np.zeros_like(original_images)
    for i in range(nb_images):
        transformed_images[i] = np.expand_dims(cv2.warpAffine(original_images[i], trans_matrix, 
                                                              (img_cols, img_rows)), axis=2)
    if MODE.DEBUG:
        print('Applied transformation {}.'.format(transformation))
        draw_comparisons(original_images, transformed_images, transformation)
        
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
    transformed_images = np.zeros_like(original_images)
    for i in range(original_images.shape[0]):
        transformed_images[i] = np.expand_dims(cv2.warpAffine(original_images[i], trans_matrix,
                                                              (img_cols, img_rows)), axis=2)
    if MODE.DEBUG:
        print('Applied transformation {}.'.format(transformation))
        draw_comparisons(original_images, transformed_images, transformation)

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

    transformed_images = np.zeros_like(original_images)

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
    for i in range(original_images.shape[0]):
        transformed_images[i] = np.expand_dims(cv2.flip(original_images[i], flip_direction), axis=2)

    if MODE.DEBUG:
        draw_comparisons(original_images, transformed_images, transformation)

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

    # -----------------------------------------
    # In affine transformation, all parallel lines in the original image
    # will still be parallel in the transformed image.
    # To find the transformation matrix, we need to specify 3 points
    # from the original image and their corresponding locations in transformed image.
    # Then, the transformation matrix M (2x3) can be generated by getAffineTransform():
    # -----------------------------------------
    nb_images, img_rows, img_cols, nb_channels = original_images.shape[:4]
    point1 = [0.25 * img_cols, 0.25 * img_rows]
    point2 = [0.25 * img_cols, 0.5 * img_rows]
    point3 = [0.5 * img_cols, 0.25 * img_rows]

    pts_original = np.float32([point1, point2, point3])

    if (transformation == TRANSFORMATION.affine_vertical_compress):
        point1 = [0.25 * img_cols, 0.25 * img_rows]
        point2 = [0.25 * img_cols, 0.4 * img_rows]
        point3 = [0.5 * img_cols, 0.25 * img_rows]
    elif (transformation == TRANSFORMATION.affine_vertical_stretch):
        point1 = [0.25 * img_cols, 0.25 * img_rows]
        point2 = [0.25 * img_cols, 0.6 * img_rows]
        point3 = [0.5 * img_cols, 0.25 * img_rows]
    elif (transformation == TRANSFORMATION.affine_horizontal_compress):
        point1 = [0.25 * img_cols, 0.25 * img_rows]
        point2 = [0.25 * img_cols, 0.5 * img_rows]
        point3 = [0.4 * img_cols, 0.25 * img_rows]
    elif (transformation == TRANSFORMATION.affine_horizontal_stretch):
        point1 = [0.25 * img_cols, 0.25 * img_rows]
        point2 = [0.25 * img_cols, 0.5 * img_rows]
        point3 = [0.6 * img_cols, 0.25 * img_rows]
    elif (transformation == TRANSFORMATION.affine_both_compress):
        point1 = [0.25 * img_cols, 0.25 * img_rows]
        point2 = [0.25 * img_cols, 0.4 * img_rows]
        point3 = [0.4 * img_cols, 0.25 * img_rows]
    elif (transformation == TRANSFORMATION.affine_both_stretch):
        point1 = [0.25 * img_cols, 0.25 * img_rows]
        point2 = [0.25 * img_cols, 0.6 * img_rows]
        point3 = [0.6 * img_cols, 0.25 * img_rows]
    else:
        raise ValueError('{} is not supported.'.format(transformation))

    transformed_images = []
    # define transformation matrix
    pts_transformed = np.float32([point1, point2, point3])
    trans_matrix = cv2.getAffineTransform(pts_original, pts_transformed)

    # applying an affine transformation over the dataset
    transformed_images = np.zeros_like(original_images)
    for i in range(nb_images):
        transformed_images[i] = np.expand_dims(cv2.warpAffine(original_images[i], trans_matrix,
                                                              (img_cols, img_rows)), axis=2)

    if MODE.DEBUG:
        draw_comparisons(original_images, transformed_images, transformation)
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

    transformed_images = np.zeros_like(original_images)

    # set kernel as a matrix of size 2
    kernel = np.ones((2, 2),np.uint8)

    if (transformation == TRANSFORMATION.dilation):
        # min filter (Graphics Mill)
        # It's opposite of erosion (max filter)
        # In dilation, a pixel element is '1' if at least one pixel
        # under the kernel is '1'. So it increases the white region
        # in the image or size of foreground object increases.
        for i in range(original_images.shape[0]):
            transformed_images[i] = np.expand_dims(cv2.dilate(original_images[i],
                                                              kernel, iterations=1), axis=2)
    elif (transformation == TRANSFORMATION.erosion):
        # max filter (Graphic Mill)
        # The basic idea of erosion is like soil erosion.
        # It erodes away the boundaries of foreground object
        # (always try to keep foreground in white)
        # The kernel slides through the image as in 2D convolution.
        # A pixel in the original image will be considered 1 only if
        # all the pixels under the kernel is 1, otherwise, it's eroded.
        for i in range(original_images.shape[0]):
            transformed_images[i] = np.expand_dims(cv2.erode(original_images[i],
                                                              kernel, iterations=1), axis=2)
    elif (transformation == TRANSFORMATION.opening):
        # erosion followed by dilation
        for i in range(original_images.shape[0]):
            transformed_images[i] = np.expand_dims(cv2.morphologyEx(original_images[i],
                                                              cv2.MORPH_OPEN, kernel), axis=2)
    elif (transformation == TRANSFORMATION.closing):
        # erosion followed by dilation
        for i in range(original_images.shape[0]):
            transformed_images[i] = np.expand_dims(cv2.morphologyEx(original_images[i],
                                                              cv2.MORPH_CLOSE, kernel), axis=2)
    elif (transformation == TRANSFORMATION.gradient):
        # keep the outline of the object
        for i in range(original_images.shape[0]):
            transformed_images[i] = np.expand_dims(cv2.morphologyEx(original_images[i],
                                                              cv2.MORPH_GRADIENT, kernel), axis=2)
    else:
        raise ValueError('{} is not supported.'.format(transformation))

    if MODE.DEBUG:
        print('Applied transformation {}.'.format(transformation))
        draw_comparisons(original_images, transformed_images, transformation)

    return transformed_images

def augment(original_images, transformation):
    """
    Image augmentation.
    :param: original_images - the images to applied transformations on.
    :param: transformation - the standard transformation to apply.
    :return: the transformed dataset.
    """
    data_generator = None
    transformed_images = np.zeros_like(original_images)

    if transformation == TRANSFORMATION.samplewise_std_norm:
        data_generator = ImageDataGenerator(samplewise_center=True,
                                            samplewise_std_normalization=True)
    elif transformation == TRANSFORMATION.feature_std_norm:
        data_generator = ImageDataGenerator(featurewise_center=True,
                                            featurewise_std_normalization=True)
    elif transformation == TRANSFORMATION.zca_whitening:
        data_generator = ImageDataGenerator(zca_whitening=True)
    elif transformation == TRANSFORMATION.pca_whitening:
        raise NotImplementedError('{} is not ready yet.'.format(transformation))
    else:
        raise ValueError('{} is not supported.'.format(transformation))

    # fit parameters from data
    data_generator.fit(original_images)
    batch_size = 128
    cnt_trans = 0
    input_size = len(original_images)


    for X_batch in data_generator.flow(original_images, shuffle=False,  batch_size=batch_size):
        for image in X_batch:
            transformed_images[cnt_trans] = image
            cnt_trans += 1

        if (cnt_trans >= input_size):
            print('transformed {} inputs.'.format(cnt_trans))
            break;

    if MODE.DEBUG:
        draw_comparisons(original_images, transformed_images, transformation)

    return transformed_images

def cartoon_effect(original_images, **kwargs):
    transformed_images = np.zeros_like(original_images)

    # default type: cartoon_mean_type1
    blur_ksize = kwargs.get('blur_ksize', 3)

    thresh_adaptive_method = kwargs.get('thresh_adaptive_method', cv2.ADAPTIVE_THRESH_MEAN_C)
    thresh_bsize = kwargs.get('thresh_bsize', 9)
    thresh_C = kwargs.get('thresh_C', 9)

    filter_d = kwargs.get('filter_d', 9)
    filter_sigma_color = kwargs.get('filter_sigma_color', 300)
    filter_sigma_space = kwargs.get('filter_sigma_space', 300)

    for i in range(original_images.shape[0]):
        img = original_images[i] * 255
        img = np.asarray(img, np.uint8)

        # detecting edges
        gray = cv2.medianBlur(src=img, ksize=blur_ksize)
        edges = cv2.adaptiveThreshold(src=gray, maxValue=255, adaptiveMethod=thresh_adaptive_method,
                                      thresholdType=cv2.THRESH_BINARY, blockSize=thresh_bsize, C=thresh_C)

        # color
        color = cv2.bilateralFilter(src=img, d=filter_d, sigmaColor=filter_sigma_color, sigmaSpace=filter_sigma_space)
        # cartoon effect
        cartoon = cv2.bitwise_and(src1=color, src2=color, mask=edges)
        transformed_images[i] = np.expand_dims((1.0 * cartoon/255), axis=2)

    if MODE.DEBUG:
        print('Applied cartoon effects.')
        draw_comparisons(original_images, transformed_images, 'catoon type')

    return transformed_images

def cartoonify(original_images, transformation):
    if MODE.DEBUG:    
        print('Applying transformation {}...'.format(transformation))

    adaptive_method = transformation.split('_')[1]
    catoon_type = transformation.split('_')[2]

    # default type: cartoon_mean_type1
    blur_ksize = 3
    thresh_adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    thresh_bsize = 9
    thresh_C = 9

    filter_d = 9
    filter_sigma_color = 300
    filter_sigma_space = 300

    if (adaptive_method == 'gaussian'):
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    if (catoon_type == 'type2'):
        thresh_bsize = 3
    elif (catoon_type == 'type3'):
        thresh_C = 3
    elif (catoon_type == 'type4'):
        thresh_bsize = 5
        filter_d = 25

    return cartoon_effect(original_images, blur_ksize=blur_ksize,
                          thresh_adaptive_method=thresh_adaptive_method,
                          thresh_bsize=thresh_bsize, thresh_C=thresh_C,
                          filter_d=filter_d, filter_sigma_color=filter_sigma_color,
                          filter_sigma_space=filter_sigma_space)

def quantize(original_images, transformation):
    """
    Adapted from tutorial
    https://www.pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/
    :param original_images:
    :param transformation:
    :return:
    """
    transformed_images = np.zeros_like(original_images)
    nb_clusters = int(transformation.split('_')[1])

    original_images *= 255.
    nb_images, img_rows, img_cols, nb_channels = original_images.shape[:4]

    for i in range(nb_images):
        img = original_images[i]
        """
        Convert gray scale images to RGB color space such that
        we can further convert the image to LAB color space.
        This function will return a 3-channel gray image that
        each channel is a copy of the original gray image.
        """
        if (nb_channels == 1):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        """
        Convert the image from the RGB color space to the LAB color space,
        since we will be clustering using k-means which is based on
        the euclidean distance, we will use the LAB color space where
        the euclidean distance implies perceptual meaning.
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
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
        quant = cluster.cluster_centers_.astype('uint8')[labels]

        """
        reshape the feature vectors back to image
        """
        quant = quant.reshape((img_rows, img_cols, 3))
        img = img.reshape((img_rows, img_cols, 3))

        """
        convert from LAB back to RGB
        """
        quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        """
        convert from RGB back to grayscale
        """
        if (nb_channels == 1):
            quant = cv2.cvtColor(quant, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        transformed_images[i] = np.expand_dims(quant, axis=2)

    transformed_images /= 255.

    return transformed_images

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
    else:
        raise ValueError('Transformation type {} is not supported.'.format(transformation_type.upper()))

"""
for testing
"""
def main(*args):
    _, (X, _) = load_data(args[0])
    X = X[:10]

    X_trans = transform_images(X, args[1])
    draw_comparisons(X, X_trans, args[1])

if __name__ == "__main__":
    main(DATA.cifar_10, TRANSFORMATION.quant_12clusters)