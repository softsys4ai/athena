import cv2
import numpy as np

IMG_ROW = 28
IMG_COL = 28

# list of transformations
IMG_ROTATE = ['rotate90', 'rotate180', 'rotate270']
IMG_SHIFT = ['shift_left', 'shift_right', 'shift_up', 'shift_down',
          'shift_top_left', 'shift_top_right', 'shift_bottom_right', 'shift_bottom_left']
IMG_FLIP = ['horizontal_flip', 'vertical_flip', 'both_flip']
IMG_AFFINE_TRANS = ['affine_vertical_compress', 'affine_vertical_stretch',
                  'affine_horizontal_compress', 'affine_horizontal_stretch',
                  'affine_both_compress', 'affine_both_stretch']
IMG_MORPH_TRANS = ['erosion', 'dilation', 'opening', 'closing', 'gradient']
IMG_TRANSFORMATIONS = []

IMG_TRANSFORMATIONS.extend(IMG_ROTATE)
IMG_TRANSFORMATIONS.extend(IMG_SHIFT)
IMG_TRANSFORMATIONS.extend(IMG_FLIP)
IMG_TRANSFORMATIONS.extend(IMG_AFFINE_TRANS)
IMG_TRANSFORMATIONS.extend(IMG_MORPH_TRANS)


def rotate(original_images, transformation):
    """
    Rotate images.
    :param: original_images - the images to applied transformations on.
    :param: transformation - the standard transformation to apply.
    :return: the transformed dataset.
    """
    print('Rotating images({})...'.format(transformation))
    trans_matrix = None
    
    transformed_images = []
    center = (IMG_ROW/2, IMG_COL/2)
    
    # ---------------
    # rotate images
    # ---------------
    if transformation == 'rotate90':
        # rotate 90-deg counterclockwise
        angle = 90
        scale = 1.0

        trans_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    elif transformation == 'rotate180':
        # rotate 180-deg counterclockwise
        angle = 180
        scale = 1.0
        
        trans_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    elif transformation == 'rotate270':
        # rotate 270-deg counterclockwise
        angle = 270
        scale = 1.0
        
        trans_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    else:
        raise ValueError('{} is not supported.'.format(transformation))
    
    # applying an affine transformation over the dataset
    transformed_images = np.zeros_like(original_images)
    for i in range(original_images.shape[0]):
        transformed_images[i] = np.expand_dims(cv2.warpAffine(original_images[i], trans_matrix, 
                                                              (IMG_COL, IMG_ROW)), axis=2)

    print('Applied transformation {}.'.format(transformation))
        
    return transformed_images

def shift(original_images, transformation):
    """
    Shift images.
    :param: original_images - the images to applied transformations on.
    :param: transformation - the standard transformation to apply.
    :return: the transformed dataset.
    """
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
    tx = tf.cast(0.15 * IMG_COL, tf.int32)
    ty = tf.cast(0.15 * IMG_ROW, tf.int32)

    if transformation == 'shift_left':
        tx = 0 - tx
        ty = 0
    elif transformation == 'shift_right':
        tx = tx
        ty = 0
    elif transformation == 'shift_up':
        tx = 0
        ty = 0 - ty
    elif transformation == 'shift_down':
        tx = 0
        ty = ty
    elif transformation == 'shift_top_right':
        tx = tx
        ty = 0 - ty
    elif transformation == 'shift_top_left':
        tx = 0 - tx
        ty = 0 - ty
    elif transformation == 'shift_bottom_left':
        tx = 0 - tx
        ty = ty
    elif transformation == 'shift_bottom_right':
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
                                                              (IMG_COL, IMG_ROW)), axis=2)

    print('Applied transformation {}.'.format(transformation))

    return transformed_images

def flip(original_images, transformation):
    """
    Flip images.
    :param: original_images - the images to applied transformations on.
    :param: transformation - the standard transformation to apply.
    :return: the transformed dataset.
    """
    print('Flipping images({})...'.format(transformation))

    transformed_images = np.zeros_like(original_images)

    # set flipping direction
    flip_direction = 0
    if transformation == 'vertical_flip':
        # flip around the x-axis
        flip_direction = 0
    elif transformation == 'horizontal_flip':
        # flip around the y-axis
        flip_direction = 1
    elif transformation == 'both_flip':
        # flip around both axes
        flip_direction = -1
    else:
        raise ValueError('{} is not supported.'.format(transformation))

    # flip images
    for i in range(original_images.shape[0]):
        transformed_images[i] = np.expand_dims(cv2.flip(original_images[i], flip_direction), axis=2)


    return transformed_images

def affine_trans(original_images, transformation):
    """
    Apply affine transformation on images.
    :param: original_images - the images to applied transformations on.
    :param: transformation - the standard transformation to apply.
    :return: the transformed dataset.
    """
    print('Applying affine transformation on images({})...'.format(transformation))

    # -----------------------------------------
    # In affine transformation, all parallel lines in the original image
    # will still be parallel in the transformed image.
    # To find the transformation matrix, we need to specify 3 points
    # from the original image and their corresponding locations in transformed image.
    # Then, the transformation matrix M (2x3) can be generated by getAffineTransform():
    # -----------------------------------------
    point1 = [0.25 * IMG_COL, 0.25 * IMG_ROW]
    point2 = [0.25 * IMG_COL, 0.5 * IMG_ROW]
    point3 = [0.5 * IMG_COL, 0.25 * IMG_ROW]

    pts_original = np.float32([point1, point2, point3])

    if (transformation == 'affine_vertical_compress'):
        point1 = [0.25 * IMG_COL, 0.25 * IMG_ROW]
        point2 = [0.25 * IMG_COL, 0.4 * IMG_ROW]
        point3 = [0.5 * IMG_COL, 0.25 * IMG_ROW]
    elif (transformation == 'affine_vertical_stretch'):
        point1 = [0.25 * IMG_COL, 0.25 * IMG_ROW]
        point2 = [0.25 * IMG_COL, 0.6 * IMG_ROW]
        point3 = [0.5 * IMG_COL, 0.25 * IMG_ROW]
    elif (transformation == 'affine_horizontal_compress'):
        point1 = [0.25 * IMG_COL, 0.25 * IMG_ROW]
        point2 = [0.25 * IMG_COL, 0.5 * IMG_ROW]
        point3 = [0.4 * IMG_COL, 0.25 * IMG_ROW]
    elif (transformation == 'affine_horizontal_stretch'):
        point1 = [0.25 * IMG_COL, 0.25 * IMG_ROW]
        point2 = [0.25 * IMG_COL, 0.5 * IMG_ROW]
        point3 = [0.6 * IMG_COL, 0.25 * IMG_ROW]
    elif (transformation == 'affine_both_compress'):
        point1 = [0.25 * IMG_COL, 0.25 * IMG_ROW]
        point2 = [0.25 * IMG_COL, 0.4 * IMG_ROW]
        point3 = [0.4 * IMG_COL, 0.25 * IMG_ROW]
    elif (transformation == 'affine_both_stretch'):
        point1 = [0.25 * IMG_COL, 0.25 * IMG_ROW]
        point2 = [0.25 * IMG_COL, 0.6 * IMG_ROW]
        point3 = [0.6 * IMG_COL, 0.25 * IMG_ROW]
    else:
        raise ValueError('{} is not supported.'.format(transformation))

    transformed_images = []

    # define transformation matrix
    pts_transformed = np.float32([point1, point2, point3])
    trans_matrix = cv2.getAffineTransform(pts_original, pts_transformed)

    # applying an affine transformation over the dataset
    transformed_images = np.zeros_like(original_images)
    for i in range(original_images.shape[0]):
        transformed_images[i] = np.expand_dims(cv2.warpAffine(original_images[i], trans_matrix,
                                                              (IMG_COL, IMG_ROW)), axis=2)

    print('Applied transformation {}.'.format(transformation))

    return transformed_images

def morph_trans(original_images, transformation):
    """
    Apply morphological transformations on images.
    :param: original_images - the images to applied transformations on.
    :param: transformation - the standard transformation to apply.
    :return: the transformed dataset.
    """
    print('Applying morphological transformation ({})...'.format(transformation))

    transformed_images = np.zeros_like(original_images)

    # set kernel as a matrix of size 2
    kernel = np.ones((2,2),np.uint8)

    if transformation == 'dilation':
        # min filter (Graphics Mill)
        # It's opposite of erosion (max filter)
        # In dilation, a pixel element is '1' if at least one pixel
        # under the kernel is '1'. So it increases the white region
        # in the image or size of foreground object increases.
        for i in range(original_images.shape[0]):
            transformed_images[i] = np.expand_dims(cv2.dilate(original_images[i],
                                                              kernel, iterations=1), axis=2)
    elif transformation == 'erosion':
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
    elif transformation == 'opening':
        # erosion followed by dilation
        for i in range(original_images.shape[0]):
            transformed_images[i] = np.expand_dims(cv2.morphologyEx(original_images[i],
                                                              cv2.MORPH_OPEN, kernel), axis=2)
    elif transformation == 'closing':
        # erosion followed by dilation
        for i in range(original_images.shape[0]):
            transformed_images[i] = np.expand_dims(cv2.morphologyEx(original_images[i],
                                                              cv2.MORPH_CLOSE, kernel), axis=2)
    elif transformation == 'gradient':
        # keep the outline of the object
        for i in range(original_images.shape[0]):
            transformed_images[i] = np.expand_dims(cv2.morphologyEx(original_images[i],
                                                              cv2.MORPH_GRADIENT, kernel), axis=2)
    else:
        raise ValueError('{} is not supported.'.format(transformation))

    print('Applied transformation {}.'.format(transformation))


    return transformed_images


def transform_images(X, transformation_type):
    if (transformation_type in IMG_ROTATE):
        return rotate(X, transformation_type)
    elif (transformation_type in IMG_FLIP):
        return flip(X, transformation_type)
    elif (transformation_type in IMG_SHIFT):
        return shift(X, transformation_type)
    elif (transformation_type in IMG_AFFINE_TRANS):
        return affine_trans(X, transformation_type)
    elif (transformation_type in IMG_MORPH_TRANS):
        return morph_trans(X, transformation_type)


