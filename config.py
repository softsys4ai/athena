"""
Define global configurations.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com
"""

class TRANSFORMATION:
    """
    Define transformation types that are supported.
    """
    clean = 'clean' # clean image, no transformation is applied.

    rotate90 = 'rotate90'
    rotate180 = 'rotate180'
    rotate270 = 'rotate270'

    shift_left = 'shift_left'
    shift_right = 'shift_right'
    shift_up = 'shift_up'
    shift_down = 'shift_down'
    shift_top_left = 'shift_top_left'
    shift_top_right = 'shift_top_right'
    shift_bottom_left = 'shift_bottom_left'
    shift_bottom_right = 'shift_bottom_right'

    horizontal_flip = 'horizontal_flip'
    vertical_flip = 'vertical_flip'
    both_flip = 'both_flip'

    affine_vertical_compress = 'affine_vertical_compress'
    affine_vertical_stretch = 'affine_vertical_stretch'
    affine_horizontal_compress = 'affine_horizontal_compress'
    affine_horizontal_stretch = 'affine_horizontal_stretch'
    affine_both_compress = 'affine_both_compress'
    affine_both_stretch = 'affine_both_stretch'

    erosion = 'erosion'
    dilation = 'dilation'
    opening = 'opening'
    closing = 'closing'
    gradient = 'gradient'

    thresh_binary = 'thresh_binary'
    thresh_mean = 'thresh_mean'
    thresh_gaussian = 'thresh_gaussian'

    samplewise_std_norm = 'samplewise_std_norm'
    feature_std_norm = 'feature_std_norm'
    zca_whitening = 'zca_whitening'
    pca_whitening = 'pca_whitening'

    scaling = 'scaling'
    upsampling = 'upsampling'
    downsampling = 'downsampling'

    horizontal_shear = 'horizontal_shear'
    vertical_shear = 'vertical_shear'
    range_shear = 'range_shear'

    cartoon_mean_type1 = 'cartoon_mean_type1'
    cartoon_mean_type2 = 'cartoon_mean_type2'
    cartoon_mean_type3 = 'cartoon_mean_type3'
    cartoon_mean_type4 = 'cartoon_mean_type4'
    cartoon_gaussian_type1 = 'cartoon_gaussian_type1'
    cartoon_gaussian_type2 = 'cartoon_gaussian_type2'
    cartoon_gaussian_type3 = 'cartoon_gaussian_type3'
    cartoon_gaussian_type4 = 'cartoon_gaussian_type4'

    ROTATE = [rotate90, rotate180, rotate270]
    SHIFT = [shift_left, shift_right, shift_up, shift_down,
             shift_top_left, shift_top_right, shift_bottom_left, shift_bottom_right]
    FLIP = [horizontal_flip, vertical_flip, both_flip]
    AFFINE_TRANS = [affine_vertical_compress, affine_vertical_stretch,
                    affine_horizontal_compress, affine_horizontal_stretch,
                    affine_both_compress, affine_both_stretch]
    MORPH_TRANS = [erosion, dilation, opening, closing, gradient]
    THRESHING = [thresh_binary, thresh_mean, thresh_gaussian] # not ready yet
    AUGMENT = [samplewise_std_norm, feature_std_norm, zca_whitening]
    SCALING = [scaling, upsampling, downsampling] # not ready yet
    SHEAR = [horizontal_shear, vertical_shear, range_shear] # not ready yet
    CARTOONS = [cartoon_mean_type1, cartoon_mean_type2, cartoon_mean_type3, cartoon_mean_type4,
                cartoon_gaussian_type1, cartoon_gaussian_type2, cartoon_gaussian_type3, cartoon_gaussian_type4]
    GAUSSIAN_NOISES = []

    @classmethod
    def supported_types(cls):
        transformations = ['clean']
        transformations.extend(TRANSFORMATION.ROTATE)
        transformations.extend(TRANSFORMATION.SHIFT)
        transformations.extend(TRANSFORMATION.FLIP)
        transformations.extend(TRANSFORMATION.AFFINE_TRANS)
        transformations.extend(TRANSFORMATION.MORPH_TRANS)
        transformations.extend(TRANSFORMATION.AUGMENT)
        transformations.extend(TRANSFORMATION.CARTOONS)
        #transformations.extend(TRANSFORMATION.THRESHING) # not ready yet
        #transformations.extend(TRANSFORMATION.SCALING) # not ready yet
        #transformations.extend(TRANSFORMATION.SHEAR) # not ready yet
        #transformations.extend(TRANSFORMATION.GAUSSIAN_NOISES) # not ready yet

        return transformations

class ATTACK:
    """
    Define attack related configuration.
    """
    APPROACHES = ['fgsm', 'iter_fgsm', 'deepfool', 'cw', 'jsma']
    FGSM_EPS = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.5, 1]

class DATA:
    """
    Configuration for data.
    """
    DATASET = 'mnist'
    IMG_ROW = 28
    IMG_COL = 28
    NB_CLASSES = 10
    VALIDATION_RATE = 0.2

    @classmethod
    def info(cls):
        print('Dataset: {}'.format(DATA.DATASET))
        print('Image size: rows - {}; cols - {}'.format(DATA.IMG_ROW, DATA.IMG_COL))
        print('Validation ratio: {}'.format(DATA.VALIDATION_RATE))

    @classmethod
    def set_dataset(cls, dataset):
        DATA.DATASET = dataset

    @classmethod
    def set_img_size(cls, img_rows, img_cols):
        DATA.IMG_ROW = img_rows
        DATA.IMG_COL = img_cols

    @classmethod
    def set_number_classes(cls, nb_classes):
        DATA.NB_CLASSES = nb_classes

    @classmethod
    def set_validation_rate(cls, val_rate):
        DATA.VALIDATION_RATE = val_rate

class MODEL:
    """
    Configuration regarding model and training
    """
    TYPE = 'cnn'
    TRANS = TRANSFORMATION.clean
    NAME = '{}_{}_{}.model'.format(DATA.DATASET, TYPE, TRANS)
    LEARNING_RATE = 0.01
    BATCH_SIZE = 128
    EPOCHS = 5

    def set_model_type(self, type):
        self.TYPE = type

    def set_batch_size(self, batch_size):
        self.BATCH_SIZE = batch_size

    def set_learning_rate(self, lr):
        self.LEARNING_RATE = lr

    def set_model_name(self, model_name):
        self.NAME = model_name

    def set_epochs(self, epochs):
        self.EPOCHS = epochs

class MODE:
    DEBUG = False

    def debug_on(self):
        self.DEBUG = True

    def debug_off(self):
        self.DEBUG = False
