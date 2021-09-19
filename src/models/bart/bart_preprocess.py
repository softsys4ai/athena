"""
Randomize transformation parameters.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
import os
import random

def _get_str_transformation_type(transformation_type):
    if transformation_type == 0:
        type = 'clean'
    elif 1 <= transformation_type and transformation_type <= 6:
        type = 'affine'
    elif 7 <= transformation_type and transformation_type <= 14:
        type = 'cartoon'
    elif 15 <= transformation_type and transformation_type <= 21:
        type = 'compress'
    elif transformation_type == 22:
        type = 'denoise_nl_means_fast'
    elif transformation_type == 23:
        type = 'denoise_tv_bregman'
    elif transformation_type == 24:
        type = 'denoise_tv_chambolle'
    elif transformation_type == 25:
        type = 'denoise_wavelet'
    elif 26 <= transformation_type and transformation_type <= 27:
        type = 'distort'
    elif transformation_type in [28, 54]:
        type = 'augment'
    elif 29 <= transformation_type and transformation_type <= 34:
        type = 'filter'
    elif 35 <= transformation_type and transformation_type <= 37:
        type = 'flip'
    elif transformation_type == 38:
        type = 'goemetric_swirl'
    elif 39 <= transformation_type and transformation_type <= 43:
        type = 'morph'
    elif 44 <= transformation_type and transformation_type <= 50:
        type = 'noise'
    elif 51 <= transformation_type and transformation_type <= 53:
        type = 'rotate'
    elif 55 <= transformation_type and transformation_type <= 62:
        type = 'shift'
    else:
        raise ValueError(f'Cannot found transformation {transformation_type}.')

    return type


def build_transformation_configs(transformation_type):
    transformation_configs = {}

    if isinstance(transformation_type, int):
        transformation_type = _get_str_transformation_type(transformation_type)

    if transformation_type == 'clean':
        transformation_configs['type'] = 'clean'
        transformation_configs['description'] = 'clean'

    elif transformation_type == 'affine':
        transformation_configs['type'] = 'affine'
        transformation_configs['description'] = 'affine_random'
        val1 = random.uniform(0.1, 0.3)
        val2 = random.uniform(0.15, 0.55)
        transformation_configs['origin_point1'] = [val1, val1]
        transformation_configs['origin_point2'] = [val1, val2]
        transformation_configs['origin_point3'] = [val2, val1]
        val3 = random.uniform(val1-0.1, val1+0.1)
        val4 = random.uniform(val1-0.1, val1+0.1)
        val5 = random.uniform(val2-0.1, val2+0.1)
        val6 = random.uniform(val2-0.1, val2+0.1)
        transformation_configs['new_point1'] = [val3, val4]
        transformation_configs['new_point2'] = [val3, val5]
        transformation_configs['new_point3'] = [val6, val4]

    elif transformation_type == 'cartoon':
        transformation_configs['type'] = 'cartoon'
        subtype = random.choice(['gaussian', 'mean'])
        transformation_configs['subtype'] = f'{subtype}_random'
        transformation_configs['description'] = f'cartoon_{subtype}_random'

        transformation_configs['nb_downsampling'] = random.randint(0, 3)
        transformation_configs['nb_bilateral'] = random.randint(0, 120)
        transformation_configs['blur_ksize'] = random.choice(([3, 5]))
        transformation_configs['thresh_bsize'] = random.choice([3, 5])
        transformation_configs['thresh_C'] = random.choice([3, 5, 7])
        if subtype == 'gaussian':
            transformation_configs['filter_d'] = 250
        else:
            transformation_configs['filter_d'] = random.randint(5, 30)
        transformation_configs['filter_sigma_color'] = 2
        transformation_configs['filter_sigma_space'] = 30

    elif transformation_type == 'compress':
        transformation_configs['type'] = 'compress'
        format = random.choice(['jpeg', 'png'])
        transformation_configs['subtype'] = format
        transformation_configs['description'] = f'compress_{format}_random'
        transformation_configs['format'] = f'.{format}'
        if format == 'jpeg':
            transformation_configs['compress_rate'] = random.randint(10, 90)
        else:
            transformation_configs['compress_rate'] = random.randint(1, 9)

    elif transformation_type == 'denoise_nl_means_fast':
        transformation_configs['type'] = 'denoise'
        transformation_configs['subtype'] = 'nl_means_fast'
        transformation_configs['description'] = 'denoise_nl_means_fast_random'
        transformation_configs['patch_size'] = random.randint(2, 9)
        transformation_configs['patch_distance'] = random.randint(2, 9)
        transformation_configs['hr'] = random.uniform(0.05, 0.9)
        transformation_configs['sr'] = random.randint(1, 5)

    elif transformation_type == 'denoise_tv_bregman':
        transformation_configs['type'] = 'denoise'
        transformation_configs['subtype'] = 'tv_bregman'
        transformation_configs['description'] = 'denoise_tv_bregman_random'
        transformation_configs['weight'] = random.randint(10, 20)
        transformation_configs['epsilon'] = 1e-6
        transformation_configs['max_iter'] = random.randint(20, 100)

    elif transformation_type == 'denoise_tv_chambolle':
        transformation_configs['type'] = 'denoise'
        transformation_configs['subtype'] = 'tv_chambolle'
        transformation_configs['description'] = 'denoise_tv_chambolle_random'
        transformation_configs['weight'] = random.uniform(0.1, 0.5)
        transformation_configs['epsilon'] = 2e-4
        transformation_configs['max_iter'] = random.randint(300, 500)

    elif transformation_type == 'denoise_wavelet':
        transformation_configs['type'] = 'denoise'
        transformation_configs['subtype'] = 'wavelet'
        transformation_configs['description'] = 'denoise_wavelet_random'
        transformation_configs['method'] = 'BayesShrink'
        transformation_configs['mode'] = random.choice(['soft', 'hard'])
        transformation_configs['wavelet'] = 'db1'

    elif transformation_type == 'distort':
        transformation_configs['type'] = 'distort'
        direction = random.choice(['x', 'y'])
        transformation_configs['subtype'] = direction
        transformation_configs['description'] = f'distort_{direction}_random'
        transformation_configs['r1'] = random.uniform(5, 8)
        transformation_configs['r2'] = random.uniform(1, 4)
        transformation_configs['c'] = 32

    elif transformation_type == 'augment':
        transformation_configs['type'] = 'augment'
        subtype = random.choice(['feature_std_norm', 'samplewise_std_norm'])
        transformation_configs['subtype'] = subtype
        transformation_configs['description'] = f'augment_{subtype}_random'

    elif transformation_type == 'filter':
        transformation_configs['type'] = 'filter'
        subtype = random.choice(['minimum', 'median', 'maximum', 'rank', 'roberts'])
        transformation_configs['subtype'] = subtype
        transformation_configs['description'] = f'filter_{subtype}_random'
        if subtype != 'roberts':
            transformation_configs['size'] = random.randint(3, 7)
        if subtype == 'rank':
            transformation_configs['rank'] = random.randint(10, 20)

    elif transformation_type == 'flip':
        transformation_configs['type'] = 'flip'
        transformation_configs['description'] = 'flip_random'
        transformation_configs['direction'] = random.choice([-1, 0, 1])

    elif transformation_type == 'geometric_swirl':
        transformation_configs['type'] = 'geometric'
        transformation_configs['subtype'] = 'swirl'
        transformation_configs['description'] = 'geo_swirl_random'
        transformation_configs['strength'] = random.uniform(0.1, 5)
        transformation_configs['radius'] = random.randint(30, 60)
        transformation_configs['order'] = 1
        transformation_configs['mode'] = 'reflect'

    elif transformation_type == 'morph':
        transformation_configs['type'] = 'morph'
        subtype = random.choice(['closing', 'dilation', 'erosion', 'gradient', 'opening'])
        transformation_configs['subtype'] = subtype
        transformation_configs['description'] = f'morph_{subtype}_random'
        transformation_configs['kernel_w'] = random.randint(2, 6)
        transformation_configs['kernel_h'] = random.randint(2, 6)

    elif transformation_type == 'noise':
        transformation_configs['type'] = 'noise'
        noise = random.choice(['gaussian', 'localvar', 'pepper', 'poisson', 'salt', 's&p', 'speckle'])
        transformation_configs['subtype'] = noise
        transformation_configs['noise'] = noise
        if noise == 's&p':
            transformation_configs['description'] = 'noise_sNp_random'
        else:
            transformation_configs['description'] = f'noise_{noise}_random'

    elif transformation_type == 'rotate':
        transformation_configs['type'] = 'rotate'
        transformation_configs['description'] = 'rotate_random'
        transformation_configs['angle'] = random.randint(5, 30)

    elif transformation_type == 'shift':
        transformation_configs['type'] = 'shift'
        transformation_configs['description'] = 'shift_random'
        transformation_configs['x_offset'] = random.uniform(0.01, 0.2)
        transformation_configs['y_offset'] = random.uniform(0.01, 0.2)

    else:
        raise ValueError(f'Transformation [{transformation_type}] is not supported.')

    return transformation_configs


if __name__=='__main__':
    config_file = '../../configs/experiment/cifar100/full-pool.json'
    selected_candidates = 'revisionES4_ens1'
    save_file_name = f'{selected_candidates}-cifar100-train_data.npy'

    save_path = '../../../../data/cifar100/bart/'
    save_file = os.path.join(save_path, save_file_name)
