"""
Define global configurations.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com,
        Jianhai Su
"""
import numpy as np

# [For defense]
numOfWCDefenses=3
numOfCVDefenses=2
cvDefenseNames=["CV_Maj", "CV_Max"] # strategies used to decide the label across clusters
# EM    :   expertise matrix
# 1s    :   every element in expertise matrix is 1
# SM    :   sum weighted confidence across models for one sample,
#           then return the label with largest sum of weighted confidence
# MMV   :   find the label with largest confidence for the input sample for each model
#           then run a majority vote across models to determine final label
wcDefenseNames=["1s_Mean", "EM_Mean", "EM_MXMV"]
kmeansResultFoldName="KMeans_result"
defensesList = ["CV_Maj", "CV_Max", "1s_Mean", "EM_Mean", "EM_MXMV", "1s_Mean_L", "EM_Mean_L", "EM_MXMV_L"]

dropout = 0.5

class TRANSFORMATION(object):
    """
    Define transformation types that are supported.
    """
    """
    clean image, no transformation is applied.
    """
    clean = 'clean'

    """
    rotation
    """
    rotate90 = 'rotate90'
    rotate180 = 'rotate180'
    rotate270 = 'rotate270'

    """
    shift/translation
    """
    shift_left = 'shift_left'
    shift_right = 'shift_right'
    shift_up = 'shift_up'
    shift_down = 'shift_down'
    shift_top_left = 'shift_top_left'
    shift_top_right = 'shift_top_right'
    shift_bottom_left = 'shift_bottom_left'
    shift_bottom_right = 'shift_bottom_right'

    """
    flipping
    """
    horizontal_flip = 'horizontal_flip'
    vertical_flip = 'vertical_flip'
    both_flip = 'both_flip'

    """
    stretch/compress
    """
    affine_vertical_compress = 'affine_vertical_compress'
    affine_vertical_stretch = 'affine_vertical_stretch'
    affine_horizontal_compress = 'affine_horizontal_compress'
    affine_horizontal_stretch = 'affine_horizontal_stretch'
    affine_both_compress = 'affine_both_compress'
    affine_both_stretch = 'affine_both_stretch'

    """
    morphology
    """
    erosion = 'erosion'
    dilation = 'dilation'
    opening = 'opening'
    closing = 'closing'
    gradient = 'gradient'

    """
    augmentation
    """
    samplewise_std_norm = 'samplewise_std_norm'
    feature_std_norm = 'feature_std_norm'
    zca_whitening = 'zca_whitening'
    pca_whitening = 'pca_whitening'

    """
    cartoonify
    """
    cartoon_mean_type1 = 'cartoon_mean_type1'
    cartoon_mean_type2 = 'cartoon_mean_type2'
    cartoon_mean_type3 = 'cartoon_mean_type3'
    cartoon_mean_type4 = 'cartoon_mean_type4'
    cartoon_gaussian_type1 = 'cartoon_gaussian_type1'
    cartoon_gaussian_type2 = 'cartoon_gaussian_type2'
    cartoon_gaussian_type3 = 'cartoon_gaussian_type3'
    cartoon_gaussian_type4 = 'cartoon_gaussian_type4'

    """
    quantization
    """
    quant_2clusters = 'quant_2_clusters'
    quant_4clusters = 'quant_4_clusters'
    quant_8clusters = 'quant_8_clusters'
    quant_16clusters = 'quant_16_clusters'
    quant_32clusters = 'quant_32_clusters'
    quant_64clusters = 'quant_64_clusters'

    """
    distortion
    """
    distortion_y = 'distortion_y'
    distortion_x = 'distortion_x'

    """
    noises
    """
    noise_gaussian = 'noise_gaussian'
    noise_localvar = 'noise_localvar'
    noise_poisson = 'noise_poisson'
    noise_salt = 'noise_salt'
    noise_pepper = 'noise_pepper'
    noise_saltNpepper = 'noise_s&p'
    noise_speckle = 'noise_speckle'

    """
    filter
    """
    sobel = 'sobel'
    gaussian_filter = 'gaussian_filter'
    rank_filter = 'rank_filter'
    median_filter = 'median_filter'
    min_filter = 'minimum_filter'
    max_filter = 'maximum_filter'

    """
    compression
    """
    compress_jpeg_quality_80 = 'compress_jpeg_quality_80'
    compress_jpeg_quality_50 = 'compress_jpeg_quality_50'
    compress_jpeg_quality_30 = 'compress_jpeg_quality_30'
    compress_jpeg_quality_10 = 'compress_jpeg_quality_10'
    compress_png_compression_1 = 'compress_png_compression_1'
    compress_png_compression_8 = 'compress_png_compression_8'
    compress_png_compression_5 = 'compress_png_compression_5'

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
    QUANTIZATIONS = [quant_2clusters, quant_4clusters, quant_8clusters,
                     quant_16clusters, quant_32clusters, quant_64clusters]
    DISTORTIONS = [distortion_x, distortion_y]
    NOISES =[noise_gaussian, noise_localvar, noise_poisson, noise_salt,
             noise_pepper, noise_saltNpepper, noise_speckle]
    FILTERS = [sobel, gaussian_filter, rank_filter, median_filter, min_filter, max_filter]
    COMPRESSION = [compress_jpeg_quality_80, compress_jpeg_quality_50,
                   compress_jpeg_quality_30, compress_jpeg_quality_10,
                   compress_png_compression_1, compress_png_compression_8, compress_png_compression_5]

    @classmethod
    def supported_types(cls):
        transformations = []
        transformations.extend(['clean'])
        transformations.extend(cls.ROTATE)
        transformations.extend(cls.SHIFT)
        transformations.extend(cls.FLIP)
        transformations.extend(cls.AFFINE_TRANS)
        transformations.extend(cls.MORPH_TRANS)
        transformations.extend(cls.AUGMENT)
        transformations.extend(cls.CARTOONS)
        transformations.extend(cls.QUANTIZATIONS)
        transformations.extend(cls.DISTORTIONS)
        transformations.extend(cls.NOISES)
        transformations.extend(cls.FILTERS)
        transformations.extend(cls.COMPRESSION)
        return transformations

class ATTACK(object):
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
        return [cls.FGSM, cls.BIM, cls.DEEPFOOL, cls.JSMA, cls.CW,
                cls.ONE_PIXEL, cls.PGD, cls.BLACKBOX]

    # ---------------------------
    # FGSM Parameters
    # ---------------------------
    @classmethod
    def get_fgsm_eps(cls):
        return [0.25, 0.3, 0.1, 0.05, 0.01, 0.005] # full set
        #return [0.25] # for test

    @classmethod
    def get_fgsm_AETypes(cls):
        attackApproach = cls.FGSM
        AETypes = []
        EPS = cls.get_fgsm_eps()
        EPS.sort()
        for eps in EPS:
            epsInt = int(1000*eps)
            AETypes.append(attackApproach+"_eps"+str(epsInt))
        return AETypes

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

class DATA(object):
    """
    Configuration for data.
    """
    mnist = 'mnist'
    fation_mnist = 'fmnist'
    cifar_10 = 'cifar10'
    cifar_100 = 'cifar100'

    valiation_rate = 0.2

    @classmethod
    def get_supported_datasets(cls):
        datasets = [cls.mnist, cls.fation_mnist, cls.cifar_10, cls.cifar_100]
        return datasets

    @classmethod
    def set_validation_rate(cls, rate):
        cls.valiation_rate = rate

class MODEL(object):
    """
    Configuration regarding model and training
    """
    ARCHITECTURE = 'cnn'
    DATASET = 'mnist'
    TRANS_TYPE = TRANSFORMATION.clean
    LEARNING_RATE = 0.01
    BATCH_SIZE = 128
    EPOCHS = 50

    @classmethod
    def set_architecture(cls, architecture):
        cls.ARCHITECTURE = architecture
    @classmethod
    def set_batch_size(cls, batch_size):
        cls.BATCH_SIZE = batch_size
    @classmethod
    def set_learning_rate(cls, lr):
        cls.LEARNING_RATE = lr
    @classmethod
    def set_dataset(cls, dataset):
        cls.NAME = dataset
    @classmethod
    def set_epochs(cls, epochs):
        cls.EPOCHS = epochs

class MODE(object):
    DEBUG = False
    @classmethod
    def debug_on(cls):
        cls.DEBUG = True
    @classmethod
    def debug_off(cls):
        cls.DEBUG = False

class PATH(object):
    # for debugging
    # MODEL = 'data/debug/models'
    # ADVERSARIAL_FILE = 'data/debug/AEs'

    # for experiment
    MODEL = 'data/models'
    ADVERSARIAL_FILE = 'data/adversarial_examples'

    FIGURES = 'data/figures'
    RESULTS = 'data/results'
    ANALYSE = 'data/analyse'
