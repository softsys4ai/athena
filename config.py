"""
Define global configurations.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com,
        Jianhai Su
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
    # clean image, no transformation is applied.
    clean = 'clean'

    # rotation
    rotate90 = 'rotate90'
    rotate180 = 'rotate180'
    rotate270 = 'rotate270'

    # shift/translation
    shift_left = 'shift_left'
    shift_right = 'shift_right'
    shift_up = 'shift_up'
    shift_down = 'shift_down'
    shift_top_left = 'shift_top_left'
    shift_top_right = 'shift_top_right'
    shift_bottom_left = 'shift_bottom_left'
    shift_bottom_right = 'shift_bottom_right'

    # flipping
    horizontal_flip = 'horizontal_flip'
    vertical_flip = 'vertical_flip'
    both_flip = 'both_flip'

    # stretch/compress
    affine_vertical_compress = 'affine_vertical_compress'
    affine_vertical_stretch = 'affine_vertical_stretch'
    affine_horizontal_compress = 'affine_horizontal_compress'
    affine_horizontal_stretch = 'affine_horizontal_stretch'
    affine_both_compress = 'affine_both_compress'
    affine_both_stretch = 'affine_both_stretch'

    # morphology
    erosion = 'erosion'
    dilation = 'dilation'
    opening = 'opening'
    closing = 'closing'
    gradient = 'gradient'

    # augmentation
    samplewise_std_norm = 'samplewise_std_norm'
    feature_std_norm = 'feature_std_norm'
    zca_whitening = 'zca_whitening'
    pca_whitening = 'pca_whitening'

    # cartoonify
    cartoon_mean_type1 = 'cartoon_mean_type1'
    cartoon_mean_type2 = 'cartoon_mean_type2'
    cartoon_mean_type3 = 'cartoon_mean_type3'
    cartoon_mean_type4 = 'cartoon_mean_type4'
    cartoon_gaussian_type1 = 'cartoon_gaussian_type1'
    cartoon_gaussian_type2 = 'cartoon_gaussian_type2'
    cartoon_gaussian_type3 = 'cartoon_gaussian_type3'
    cartoon_gaussian_type4 = 'cartoon_gaussian_type4'

    # quantization
    quant_2clusters = 'quant_2_clusters'
    quant_4clusters = 'quant_4_clusters'
    quant_6clusters = 'quant_6_clusters'
    quant_8clusters = 'quant_8_clusters'
    quant_12clusters = 'quant_12_clusters'
    quant_16clusters = 'quant_16_clusters'

    thresh_binary = 'thresh_binary'
    thresh_mean = 'thresh_mean'
    thresh_gaussian = 'thresh_gaussian'

    scaling = 'scaling'
    upsampling = 'upsampling'
    downsampling = 'downsampling'

    horizontal_shear = 'horizontal_shear'
    vertical_shear = 'vertical_shear'
    range_shear = 'range_shear'

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
    QUANTIZATIONS = [quant_2clusters, quant_4clusters, quant_6clusters, quant_8clusters,
                     quant_12clusters, quant_16clusters]

    NOISES =[]
    FILTERS = []

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
        transformations.extend(TRANSFORMATION.QUANTIZATIONS)

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
        # return [0.25, 0.3, 0.1, 0.05, 0.01, 0.005] # full set
        return [0.25] # for test

    # ---------------------------
    # i-FGSM/BIM Parameters
    # ---------------------------
    @classmethod
    def get_bim_nbIter(cls):
        # return [1000, 100, 10000, 10, 1, 100000] # full set
        return [100] # for test

    @classmethod
    def get_bim_norm(cls):
        return [np.inf, 2] # full set
        # return [np.inf]

    @classmethod
    def get_bim_eps(cls, order):
        if order == 2:
            return [0.5, 1, 0.25, 0.1]
            # return [0.5, 0.25]
        elif order == np.inf:
            return [0.5, 0.25, 0.1, 0.05, 0.01, 0.005]
            # return [0.005, 0.01, 0.05]

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
        # theta: Perturbation introduced to modified components (can be positive or negative)
        # return [0.1, 0.3, 0.5, 0,7, 1.] # full set
        return [0.1]

    @classmethod
    def get_jsma_gamma(cls):
        # gamma: Maximum percentage of perturbed features
        # return [0.05, 0.1, 0.3, 0.5, 0.7, 1.] # full set.
        return [0.7, 0.3]

    # ----------------------------
    # CW parameters
    # ----------------------------
    @classmethod
    def get_cw_order(cls):
        # return [2, np.inf, 0] # full set
        return [2]

    @classmethod
    def get_cw_maxIter(cls):
        # return [1, 10, 100, 1000, 10000, 100000] # full set
        return [100]

class DATA:
    """
    Configuration for data.
    """
    mnist = 'mnist'
    fation_mnist = 'fmnist'
    cifar_10 = 'cifar10'
    cifar_100 = 'cifar100'

    @classmethod
    def get_supported_datasets(cls):
        datasets = [DATA.mnist, DATA.fation_mnist, DATA.cifar_10, DATA.cifar_100]
        return datasets

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
    @classmethod
    def debug_on(self):
        self.DEBUG = True
    @classmethod
    def debug_off(self):
        self.DEBUG = False

class PATH:
    if MODE.DEBUG:
        MODEL = 'data/debug/models'
        # ADVERSARIAL_FILE = 'data/debug/AEs'
    else:
        MODEL = 'data/models'
        # ADVERSARIAL_FILE = 'data/adversarial_examples'
    ADVERSARIAL_FILE = 'data/debug/AEs'
    FIGURES = 'data/figures'
    RESULTS = 'data/results'
    ANALYSE = 'data/analyse'
