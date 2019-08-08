"""
Define global configurations.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com
"""
import numpy as np

# [For defenses]
numOfWCDefenses=3
numOfCVDefenses=2
cvDefenseNames=["Majority", "Max"] # strategies used to decide the label across clusters
# EM    :   expertise matrix
# 1s    :   every element in expertise matrix is 1
# SM    :   sum weighted confidence across models for one sample,
#           then return the label with largest sum of weighted confidence
# MMV   :   find the label with largest confidence for the input sample for each model
#           then run a majority vote across models to determine final label
wcDefenseNames=["1s_SM", "EM_SM", "EM_MMV"]
kmeansResultFoldName="KMeans_result"



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
    AUGMENT = [samplewise_std_norm, feature_std_norm, zca_whitening]
    CARTOONS = [cartoon_mean_type1, cartoon_mean_type2, cartoon_mean_type3, cartoon_mean_type4,
                cartoon_gaussian_type1, cartoon_gaussian_type2, cartoon_gaussian_type3, cartoon_gaussian_type4]

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

        return transformations

class ATTACK:
    """
    Define attack related configuration.
    """
    # ---------------------------
    # Supported methods
    # ---------------------------
    FGSM = 'fgsm'
    BIM = 'bim'
    DEEPFOOL = 'deepfool'
    CW = 'cw'
    JSMA = 'jsma'
    ONE_PIXEL = 'one-pixel'
    PGD = 'pgd'
    BLACKBOX = 'blackbox'

    @classmethod
    def get_supported_attacks(cls):
        return [ATTACK.FGSM, ATTACK.BIM, ATTACK.DEEPFOOL, ATTACK.JSMA, ATTACK.CW,
                ATTACK.ONE_PIXEL, ATTACK.PGD, ATTACK.BLACKBOX]

    # ---------------------------
    # FGSM Parameters
    # ---------------------------
    @classmethod
    def get_fgsm_eps(cls):
        # return [0.25, 0.3, 0.5, 0.1, 0.05, 0.01, 0.005] # full set
        return [0.01]

    # ---------------------------
    # i-FGSM/BIM Parameters
    # ---------------------------
    @classmethod
    def get_bim_nbIter(cls):
        # return [1000, 100, 10000, 10, 1, 100000] # full set
        return [1000]

    @classmethod
    def get_bim_norm(cls):
        # return [np.inf, 2] # full set
        return [np.inf]

    @classmethod
    def get_bim_eps(cls, order):
        if order == 2:
            # return [0.5, 1, 0.25, 0.1, 0.05]
            return [0.5]
        elif order == np.inf:
            # return [0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005]
            return [0.25]

    # ----------------------------
    # Deepfool parameters
    # ----------------------------
    @classmethod
    def get_df_maxIter(cls):
        # return [1000, 100, 10000, 10, 1, 100000] # full set
        return [100]

    # ----------------------------
    # JSMA parameters
    # ----------------------------
    @classmethod
    def get_jsma_theta(cls):
        # return [-1., -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0,7, 1.] # full set
        return [0.5]

    @classmethod
    def get_jsma_gamma(cls):
        # return [0.05, 0.1, 0.3, 0.5, 0.7] # full set. Need to double-check the meaning of gamma, values may change later.
        return [0.5]

    # ----------------------------
    # CW parameters
    # ----------------------------
    # TODO:
    CW_ORD = [np.inf, 2, 0]
    CW_BIN_SEARCH_STEPS = [1, 2, 3, 4, 5]
    CW_MAXITERATIONS = [10, 50, 100, 200, 300, 500]
    CW_LEARNING_RATES = [0.01, 0.05, 0.1, 0.25, 0.5]
    CW_INITIAL_CONST = [5, 10, 50, 75, 100]

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
    DEBUG = True
    @classmethod
    def debug_on(self):
        self.DEBUG = True
    @classmethod
    def debug_off(self):
        self.DEBUG = False

class PATH:
    MODEL = 'data/models'
    ADVERSARIAL_FILE = 'data/adversarial_examples'
    FIGURES = 'data/figures'
    RESULTS = 'data/results'
    ANALYSE = 'data/analyse'
