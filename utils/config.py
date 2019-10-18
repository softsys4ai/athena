"""
Define global configurations.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com,
        Jianhai Su
"""
import numpy as np
from pathlib import Path

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

class DATA(object):
    """
    Configuration for data.
    """
    mnist = 'mnist'
    fation_mnist = 'fmnist'
    cifar_10 = 'cifar10'
    cifar_100 = 'cifar100'

    valiation_rate = 0.2

    CUR_DATASET_NAME = mnist

    @classmethod
    def set_current_dataset_name(cls, dataset_name):
        supported_list = cls.get_supported_datasets()
        if not dataset_name in supported_list:
            raise ValueError(
                "{} is not supported. Currently only {} are supported.".format(dataset_name, supported_list))

        cls.CUR_DATASET_NAME = dataset_name

    @classmethod
    def get_supported_datasets(cls):
        datasets = [cls.mnist, cls.fation_mnist, cls.cifar_10, cls.cifar_100]
        return datasets

    @classmethod
    def set_validation_rate(cls, rate):
        cls.valiation_rate = rate


class TRANSFORMATION(object):
    """
    Define transformation types that are supported.
    Define transformation name in the form of
    <category>_<transformation>
    """
    """
    clean image, no transformation is applied.
    """
    clean = 'clean'

    """
    a global variable to store current transformation type
    """
    CUR_TRANS_TYPE = clean

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
    zca_whitening = 'zca_whitening' # TODO: bug fix
    #pca_whitening = 'pca_whitening' # TODO: add ?

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
    quant_2_clusters = 'quant_2_clusters'
    quant_4_clusters = 'quant_4_clusters'  # TODO: temporary removed from list
    quant_8_clusters = 'quant_8_clusters'
    quant_16_clusters = 'quant_16_clusters'  # TODO: temporary removed from list
    quant_32_clusters = 'quant_32_clusters'
    quant_64_clusters = 'quant_64_clusters'  # TODO: temporary removed from list

    """
    distortion
    """
    distort_y = 'distortion_y'
    distort_x = 'distortion_x'
    distort_pixelate = 'distortion_pixelate'
    distort_saturate = 'distortion_saturate'
    distort_brightness = 'distortion_brightness'
    distort_contrast = 'distortion_contrast'
    distort_motion_blur = 'distortion_motion_blur'
    distort_defocus_blur = 'distortion_defocus_blur'
    distort_glass_blur = 'distortion_glass_blur'
    distort_gaussian_blur = 'distortion_gaussian_blur'

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
    filter_meijering = 'filter_meijering'  # TODO: bug fix
    filter_sato = 'filter_sato'  # TODO: bug fix
    filter_frangi = 'filter_frangi'  # TODO: bug fix
    filter_hessian = 'filter_hessian'  # TODO: bug fix
    filter_skeletonize = 'filter_skeletonize'  # TODO: bug fix
    filter_thin = 'filter_thin'  # TODO: bug fix

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
    denoise_bilateral = 'denoise_bilateral'  # TODO: bug fix
    denoise_wavelet = 'denoise_wavelet'
    denoise_nl_means = 'denoise_nl_means'
    denoise_nl_fast = 'denoise_nl_means_fast'

    """
    geometric
    """
    geo_swirl = 'geo_swirl'
    geo_radon = 'geo_radon'  # TODO: remove from list
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
    MORPH_TRANS = [morph_erosion, morph_dilation, morph_opening, morph_closing, morph_gradient]
    AUGMENT = [samplewise_std_norm, feature_std_norm]
    CARTOONS = [cartoon_mean_type1, cartoon_mean_type2, cartoon_mean_type3, cartoon_mean_type4,
                cartoon_gaussian_type1, cartoon_gaussian_type2, cartoon_gaussian_type3, cartoon_gaussian_type4]

    # full set (for training)
    # QUANTIZATIONS = [quant_2_clusters, quant_4_clusters, quant_8_clusters,
    #                  quant_16_clusters, quant_32_clusters, quant_64_clusters] # full set

    """
    When evaluation, we choose 2 types of quants for each dataset, for the time-consuming issue.
    Extend corresponding array if necessary.
    """
    if (DATA.CUR_DATASET_NAME == DATA.cifar_10 or
        DATA.CUR_DATASET_NAME == DATA.cifar_100):
        # color images
        QUANTIZATIONS = [quant_16_clusters, quant_64_clusters]
    elif (DATA.CUR_DATASET_NAME == DATA.mnist or
          DATA.CUR_DATASET_NAME == DATA.fation_mnist):
        # greyscale
        QUANTIZATIONS = [quant_4_clusters, quant_8_clusters]

    DISTORTIONS = [distort_x, distort_y, distort_contrast, distort_brightness]
    NOISES = [noise_gaussian, noise_localvar, noise_poisson, noise_salt,
              noise_pepper, noise_saltNpepper, noise_speckle]

    # FILTERS = [filter_sobel, filter_gaussian, filter_rank, filter_median, filter_minimum,
    #            filter_maximum, filter_entropy, filter_roberts, filter_scharr,
    #            filter_prewitt, filter_meijering, filter_sato, filter_frangi, filter_hessian,
    #            filter_skeletonize, filter_thin] # TODO: full set

    FILTERS = [filter_sobel, filter_gaussian, filter_rank, filter_median, filter_minimum,
               filter_maximum, filter_roberts, filter_scharr, filter_entropy,
               filter_prewitt, filter_meijering]
    COMPRESSION = [compress_jpeg_quality_80, compress_jpeg_quality_50,
                   compress_jpeg_quality_30, compress_jpeg_quality_10,
                   compress_png_compression_1, compress_png_compression_8, compress_png_compression_5]
    DENOISING = [denoise_tv_chambolle, denoise_tv_bregman,  # denoise_bilateral,
                 denoise_wavelet, denoise_nl_means, denoise_nl_fast]
    GEOMETRIC = [geo_swirl, geo_iradon, geo_iradon_sart]
    SEGMENTATION = [seg_gradient]  # , seg_watershed]

    @classmethod
    def set_cur_transformation_type(cls, transformation):
        cls.CUR_TRANS_TYPE = transformation

    @classmethod
    def supported_types(cls):
        transformations = []
        transformations.extend([cls.clean])
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

        print('Support {} types of transformations.'.format(len(transformations)))
        return transformations

    """
    Transformation Compositions
    """
    composition1 = [noise_gaussian, affine_both_compress, filter_minimum]

    @classmethod
    def get_transformation_compositions(cls):
        compositions = []
        compositions.extend(cls.composition1)

        return compositions

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
    ONE_PIXEL = 'onepixel'
    MIM = 'mim'
    PGD = 'pgd'
    BLACKBOX = 'blackbox'

    @classmethod
    def get_supported_attacks(cls):
        return [cls.FGSM, cls.BIM, cls.DEEPFOOL, cls.JSMA, cls.CW,
                cls.ONE_PIXEL, cls.PGD, cls.BLACKBOX]

    @classmethod
    def get_AETypes(cls):
        AETypes = []
        AETypes.extend(cls.get_jsma_AETypes())
        AETypes.extend(cls.get_fgsm_AETypes())
        AETypes.extend(cls.get_bim_AETypes())
        AETypes.extend(cls.get_pgd_AETypes())
        AETypes.extend(cls.get_df_AETypes())

        return AETypes

    # ---------------------------
    # FGSM Parameters
    # ---------------------------
    @classmethod
    def get_fgsm_eps(cls):
        return [0.25, 0.3, 0.1, 0.05, 0.01, 0.005]  # full set
        # return [0.25] # for test

    @classmethod
    def get_fgsm_AETypes(cls):
        if DATA.CUR_DATASET_NAME == DATA.cifar_10:
            return ['fgsm_eps10', 'fgsm_eps50', 'fgsm_eps100']
        elif DATA.CUR_DATASET_NAME == DATA.mnist:
            return ['fgsm_eps100', 'fgsm_eps250', 'fgsm_eps300']

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
        if DATA.CUR_DATASET_NAME == DATA.cifar_10:
            return ['bim_ord2_nbIter100_eps500', 'bim_ord2_nbIter100_eps1000',
                    'bim_ordinf_nbIter100_eps50', 'bim_ordinf_nbIter100_eps100']
        elif DATA.CUR_DATASET_NAME == DATA.mnist:
            return ['bim_ord2_nbIter100_eps500', 'bim_ord2_nbIter100_eps1000',
                    'bim_ordinf_nbIter100_eps250', 'bim_ordinf_nbIter100_eps500']

    # ----------------------------
    # Deepfool parameters
    # ----------------------------
    @classmethod
    def get_df_maxIter(cls):
        return [1, 10, 100, 1000, 10000, 100000]  # full set
        # return [10]

    @classmethod
    def get_df_AETypes(cls):
        if DATA.CUR_DATASET_NAME == DATA.cifar_10:
            return ["deepfool_maxIter100", "deepfool_maxIter10000"]
        elif DATA.CUR_DATASET_NAME == DATA.mnist:
            return ['deepfool_maxIter100', 'deepfool_maxIter10000']

    # ----------------------------
    # JSMA parameters
    # ----------------------------
    @classmethod
    def get_jsma_theta(cls):
        # theta: Perturbation introduced to modified components (can be positive or negative)
        # for Grayscale, positive only.
        if DATA.CUR_DATASET_NAME == DATA.cifar_10:
            return [-1., -0.5, -0.3, 0.3, 0.5, 1.]
        else:
            return [0.3, 0.5]

    @classmethod
    def get_jsma_gamma(cls):
        # gamma: Maximum percentage of perturbed features
        return [0.05, 0.1, 0.3, 0.5, 0.7, 1.] # full set.
        # return [0.5]

    @classmethod
    def get_jsma_AETypes(cls):
        if DATA.CUR_DATASET_NAME == DATA.cifar_10:
            return ['jsma_theta30_gamma50', 'jsma_theta50_gamma70']
        elif DATA.CUR_DATASET_NAME == DATA.mnist:
            return ['jsma_theta30_gamma50', 'jsma_theta50_gamma70']

    # ----------------------------
    # CW parameters
    # ----------------------------
    @classmethod
    def get_cw_order(cls):
        return [0, 2, np.inf] # full set

    @classmethod
    def get_cw_maxIter(cls):
        # return [1, 10, 100, 1000, 10000, 100000] # full set
        return [100]

    @classmethod
    def get_cw_AETypes(cls):
        if DATA.CUR_DATASET_NAME == DATA.cifar_10:
            return []
        elif DATA.CUR_DATASET_NAME == DATA.mnist:
            return []

    # ----------------------------
    # PGD parameters
    # ----------------------------
    @classmethod
    def get_pgd_eps(cls):
        return [1., 0.75, 0.5, 0.3, 0.25, 0.1, 0.05]

    @classmethod
    def get_pgd_AETypes(cls):
        if DATA.CUR_DATASET_NAME == DATA.cifar_10:
            return ['pgd_eps50_nbIter100_epsIter10', 'pgd_eps100_nbIter100_epsIter10']
        elif DATA.CUR_DATASET_NAME == DATA.mnist:
            return ['pgd_eps250_nbIter100_epsIter10', 'pgd_eps500_nbIter100_epsIter10',
                    'pgd_eps750_nbIter100_epsIter10']

    # --------------------------
    # One-Pixel
    # --------------------------
    @classmethod
    def get_onepixel_pxCount(cls):
        # return [1, 3, 5, 10, 15]
        return [1]

    @classmethod
    def get_onepixel_maxIter(cls):
        # return [50, 100, 200]
        return [50]

    @classmethod
    def get_onepixel_popsize(cls):
        # return [30, 50, 100]
        return [30]

    @classmethod
    def get_onepixel_AETypes(cls):
        attack = ATTACK.ONE_PIXEL
        return ['{}_pxCount1_maxIter50_popsize30'.format(attack)]

class MODEL(object):
    """
    Configuration regarding model and training
    """
    ARCHITECTURE = 'cnn'
    DATASET = 'mnist'
    TRANS_TYPE = TRANSFORMATION.clean
    LEARNING_RATE = 0.01
    BATCH_SIZE = 128
    EPOCHS = 100

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
    # file.parent = utils folder
    # file.parent.parent = project base (utils' parent)
    PROJECT_DIR = Path(__file__).parent.parent.absolute()

    print('PROJECT DICTIONARY: {}'.format(PROJECT_DIR))
    MODEL = '{}/data/models'.format(PROJECT_DIR)
    ADVERSARIAL_FILE = '{}/data/adversarial_examples'.format(PROJECT_DIR)

    FIGURES = '{}/data/figures'.format(PROJECT_DIR)
    RESULTS = '{}/data/results'.format(PROJECT_DIR)
    ANALYSE = '{}/data/analyse'.format(PROJECT_DIR)

    @classmethod
    def set_path_of_models(cls, model_base):
        cls.MODEL = '{}/{}'.format(cls.PROJECT_DIR, model_base)

    @classmethod
    def set_path_of_ae(cls, ae_base):
        cls.ADVERSARIAL_FILE = '{}/{}'.format(cls.PROJECT_DIR, ae_base)

    @classmethod
    def set_path_of_figs(cls, figure_base):
        cls.FIGURES = '{}/{}'.format(cls.PROJECT_DIR, figure_base)

    @classmethod
    def set_path_of_results(cls, result_base):
        cls.RESULTS = '{}/{}'.format(cls.PROJECT_DIR, result_base)

    @classmethod
    def set_path_of_analyse(cls, analyse_base):
        cls.ANALYSE = '{}/{}'.format(cls.PROJECT_DIR, analyse_base)
