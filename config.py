"""
Define global configurations.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com,
        Jianhai Su
"""
import numpy as np

# [For defense]
numOfWCDefenses = 3
numOfCVDefenses = 2
cvDefenseNames = ["CV_Maj", "CV_Max"]  # strategies used to decide the label across clusters
# EM    :   expertise matrix
# 1s    :   every element in expertise matrix is 1
# SM    :   sum weighted confidence across models for one sample,
#           then return the label with largest sum of weighted confidence
# MMV   :   find the label with largest confidence for the input sample for each model
#           then run a majority vote across models to determine final label
wcDefenseNames = ["1s_Mean", "EM_Mean", "EM_MXMV"]
kmeansResultFoldName = "KMeans_result"
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
    flip_horizontal = 'flip_horizontal'
    flip_vertical = 'flip_vertical'
    flip_both = 'flip_both'

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
    morph_erosion = 'morph_erosion'
    morph_dilation = 'morph_dilation'
    morph_opening = 'morph_opening'
    morph_closing = 'morph_closing'
    morph_gradient = 'morph_gradient'

    """
    augmentation
    """
    samplewise_std_norm = 'samplewise_std_norm'
    feature_std_norm = 'feature_std_norm'
    zca_whitening = 'zca_whitening'
    # pca_whitening = 'pca_whitening'

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
    filter_sobel = 'filter_sobel'
    filter_gaussian = 'filter_gaussian'
    filter_rank = 'filter_rank'
    filter_median = 'filter_median'
    filter_minimum = 'filter_minimum'
    filter_maximum = 'filter_maximum'
    filter_entropy = 'filter_entropy'
    filter_roberts = 'filter_roberts'
    filter_scharr = 'filter_scharr'
    filter_prewitt = 'filter_prewitt'
    filter_meijering = 'filter_meijering'
    filter_sato = 'filter_sato'
    filter_frangi = 'filter_frangi'
    filter_hessian = 'filter_hessian'
    filter_skeletonize = 'filter_skeletonize'
    filter_thin = 'filter_thin'

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

    """
    denoising
    """
    denoise_tv_chambolle = 'denoise_tv_chambolle'
    denoise_tv_bregman = 'denoise_tv_bregman'
    denoise_bilateral = 'denoise_bilateral'
    denoise_wavelet = 'denoise_wavelet'
    denoise_nl_means = 'denoise_nl_means'
    denoise_nl_fast = 'denoise_nl_means_fast'

    """
    geometric
    """
    geo_swirl = 'geo_swirl'
    geo_random = 'geo_random'
    geo_iradon = 'geo_iradon'
    geo_iradon_sart = 'geo_iradon_sart'

    """
    segmentation
    """
    seg_gradient = 'seg_gradient'
    seg_watershed = 'seg_watershed'

    ROTATE = [rotate90, rotate180, rotate270]
    SHIFT = [shift_left, shift_right, shift_up, shift_down,
             shift_top_left, shift_top_right, shift_bottom_left, shift_bottom_right]
    FLIP = [flip_horizontal, flip_vertical, flip_both]
    AFFINE_TRANS = [affine_vertical_compress, affine_vertical_stretch,
                    affine_horizontal_compress, affine_horizontal_stretch,
                    affine_both_compress, affine_both_stretch]
    MORPH_TRANS = [morph_erosion, morph_dilation, morph_opening, morph_closing, morph_gradient, filter_skeletonize,
                   filter_thin]
    AUGMENT = [samplewise_std_norm, feature_std_norm, zca_whitening]
    CARTOONS = [cartoon_mean_type1, cartoon_mean_type2, cartoon_mean_type3, cartoon_mean_type4,
                cartoon_gaussian_type1, cartoon_gaussian_type2, cartoon_gaussian_type3, cartoon_gaussian_type4]
    QUANTIZATIONS = [quant_2clusters, quant_4clusters, quant_8clusters,
                     quant_16clusters, quant_32clusters, quant_64clusters]
    DISTORTIONS = [distortion_x, distortion_y]
    NOISES = [noise_gaussian, noise_localvar, noise_poisson, noise_salt,
              noise_pepper, noise_saltNpepper, noise_speckle]
    FILTERS = [filter_sobel, filter_gaussian, filter_rank, filter_median, filter_minimum,
               filter_maximum, filter_entropy, filter_roberts, filter_scharr,
               filter_prewitt, filter_meijering, filter_sato, filter_frangi, filter_hessian,
               filter_skeletonize, filter_thin]
    COMPRESSION = [compress_jpeg_quality_80, compress_jpeg_quality_50,
                   compress_jpeg_quality_30, compress_jpeg_quality_10,
                   compress_png_compression_1, compress_png_compression_8, compress_png_compression_5]
    DENOISING = [denoise_tv_chambolle, denoise_tv_bregman, denoise_bilateral, denoise_wavelet, denoise_nl_means,
                 denoise_nl_fast]
    GEOMETRIC = [geo_swirl, geo_random, geo_iradon, geo_iradon_sart]
    SEGMENTATION = [seg_gradient, seg_watershed]

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
        transformations.extend(cls.DENOISING)
        transformations.extend(cls.GEOMETRIC)
        transformations.extend(cls.SEGMENTATION)

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
        return [0.25, 0.3, 0.1, 0.05, 0.01, 0.005]  # full set
        # return [0.25] # for test

    @classmethod
    def get_fgsm_AETypes(cls):
        attackApproach = cls.FGSM
        AETypes = []
        EPS = cls.get_fgsm_eps()
        EPS.sort()
        EPS = [0.25]
        for eps in EPS:
            epsInt = int(1000 * eps)
            AETypes.append(attackApproach + "_eps" + str(epsInt))
        return AETypes

    # ---------------------------
    # i-FGSM/BIM Parameters
    # ---------------------------
    @classmethod
    def get_bim_nbIter(cls):
        return [1, 10, 100, 1000, 10000, 100000]  # full set
        # return [100] # for test

    @classmethod
    def get_bim_norm(cls):
        return [np.inf, 2]  # full set
        # return [np.inf]

    @classmethod
    def get_bim_eps(cls, order):
        if order == 2:
            return [0.1, 0.25, 0.5, 1]
            # return [0.5, 0.25]
        elif order == np.inf:
            return [0.005, 0.01, 0.05, 0.1, 0.25, 0.5]
            # return [0.01, 0.005]

    @classmethod
    def get_bim_AETypes(cls):
        attackApproach = cls.BIM
        AETypes = []
        EPS = {}
        EPS["ord2"] = [0.25]
        EPS["ordinf"] = [0.01]
        for distType in ["ord2", "ordinf"]:
            curEPS = EPS[distType]
            for nbIter in [100]:
                for eps in curEPS:
                    epsInt = int(1000 * eps)
                    AETypes.append(attackApproach + "_" + distType + "_nbIter" + str(nbIter) + "_eps" + str(epsInt))
        return AETypes

    @classmethod
    def get_jsma_AETypes(cls):
        AETypes = [
            # "jsma_theta10_gamma30",
            # "jsma_theta10_gamma70",
            # "jsma_theta30_gamma50",
            "jsma_theta50_gamma50"]
        return AETypes

    @classmethod
    def get_AETypes(cls):
        AETypes = []
        AETypes.extend(cls.get_jsma_AETypes())
        AETypes.extend(cls.get_fgsm_AETypes())
        AETypes.extend(cls.get_bim_AETypes())

        return AETypes

    # ----------------------------
    # Deepfool parameters
    # ----------------------------
    @classmethod
    def get_df_maxIter(cls):
        return [1, 10, 100, 1000, 10000, 100000]  # full set
        # return [10]

    @classmethod
    def get_df_clip_max(cls):
        return [1.]

    # ----------------------------
    # JSMA parameters
    # ----------------------------
    @classmethod
    def get_jsma_theta(cls):
        # theta: Perturbation introduced to modified components (can be positive or negative)
        return [0.1, 0.3, 0.5, 0.7, 1.]  # full set
        # return [0.3, 0.5]

    @classmethod
    def get_jsma_gamma(cls):
        # gamma: Maximum percentage of perturbed features
        return [0.05, 0.1, 0.3, 0.5, 0.7, 1.]  # full set.
        # return [0.5]

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
