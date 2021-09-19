"""

@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import sys
sys.path.append('../')

import argparse
import numpy as np
import os
import time
import json

from models.cifar100_utils import load_model, load_pool
from utils.file import load_from_json
from models.athena import Ensemble, ENSEMBLE_STRATEGY


def collect_predictions(model, data_configs):
    folder = data_configs.get('dir')
    testsets = ['bs_full']

    for testset in testsets:
        data_file = os.path.join(folder, testset[0])

        print(f'>>> Evaluating model on test data {data_file}')
        data = np.load(data_file)
        raw_predictions = model.predict(data, raw=True)

        print(f'>>> Data shape: {data.shape}; Prediction shape: {raw_predictions.shape}')
        filename = f'Pred-{testset[0]}'
        savefile = os.path.join(folder, filename)
        print(f'>>> Save predictions to file [{savefile}].')
        np.save(savefile, raw_predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Cifar100')

    parser.add_argument('-r', '--experiment-root', required=False,
                        default='../../../')
    parser.add_argument('-p', '--pool-configs', required=False,
                        default='../configs/experiment/cifar100/full-pool.json')
    parser.add_argument('--selected-pool', required=False,
                        default=None)
    parser.add_argument('-m', '--model-configs', required=False,
                        default='../configs/experiment/cifar100/model-info.json')
    parser.add_argument('-d', '--data-configs', required=False,
                        default='../configs/experiment/cifar100/data-info.json')
    parser.add_argument('-b', '--benign-sample', required=False,
                        default='bs_full')
    parser.add_argument('-a', '--attack-configs', required=False,
                        default='../configs/experiment/cifar100/revision-attack-pgd-nonEot.json')
    parser.add_argument('--selected-attacks', required=False, default=None)
    parser.add_argument('-o', '--output-root', required=False,
                        default='../../experiment/cifar100/results/')
    parser.add_argument('--targeted-model', required=False, default='ensemble',
                        help='It has to be `single`, `adt_pgd`, or `ensemble`.')
    parser.add_argument('--eot', required=False, default='false',
                        help='`True` if use EOT, `False` otherwise.')
    parser.add_argument('--debug', required=False, default='false')

    args = parser.parse_args()

    eot = args.eot
    if isinstance(eot, str) and eot.lower() == 'true':
        print('set EOT to True')
        args.eot = True
    else:
        print('set EOT to False')
        args.eot = False

    debug = args.debug
    if isinstance(debug, str) and debug.lower() == 'true':
        print('Debug Mode')
        args.debug = True
    else:
        args.debug = False

    print('-------AUGMENT SUMMARY---------')
    print(f'EXPERIMENT ROOT: {args.experiment_root}')
    print(f'POOL CONFIGS: {args.pool_configs}')
    print(f'SELECTED POOL: {args.selected_pool}')
    print(f'MODEL CONFIGS: {args.model_configs}')
    print(f'DATA CONFIGS: {args.data_configs}')
    print(f'BENIGN SAMPLES: {args.benign_samples}')
    print(f'ATTACK CONFIGS: {args.attack_configs}')
    print(f'SELECTED ATTACK: {args.selected_attacks}')
    print(f'TARGETED MODEL: {args.targeted_model}')
    print(f'EOT or not: {args.eot}')
    print(f'OUTPUT ROOT: {args.output_root}')
    print(f'DEBUGGING MODE: {args.debug}')
    print('-------------------------------\n')

    # parse configurations from json file
    pool_configs = load_from_json(args.pool_configs)
    model_configs = load_from_json(args.model_configs)
    model_configs['wresnet']['dir'] = args.experiment_root + model_configs.get('wresnet').get('dir')
    model_configs['shake26']['dir'] = args.experiment_root + model_configs.get('wresnet').get('dir')
    data_configs = load_from_json(args.data_configs)
    data_configs['dir'] = args.experiment_root + data_configs.get('dir')
    attack_configs = load_from_json(args.attack_configs)

    # load the targeted model
    if args.targeted_model == 'single':
        prefix = f'AE-cifar100-wresnet-{args.targeted_model}'
        model_file = os.path.join(model_configs.get('wresnet').get('dir'),
                                  model_configs.get('wresnet').get('um_file'))
        model, _, _ = load_model(file=model_file,
                                  model_configs=model_configs.get('wresnet'),
                                  trans_configs=None)
    elif args.targeted_model == 'adt_pgd':
        prefix = f'AE-cifar100-wresnet-{args.targeted_model}'
        model_file = os.path.join(model_configs.get('wresnet').get('dir'),
                                  model_configs.get('wresnet').get('pgd_trained_cifar'))

        model, _, _ = load_model(file=model_file,
                                 model_configs=model_configs.get('wresnet'),
                                 trans_configs=None)
        print(model)
    elif args.targeted_model == 'ensemble':
        if args.selected_pool is None:
            selected_pool = 'demo_pool'
        else:
            selected_pool = args.selected_pool

        print(f'>>> [DEBUG][Target Ensemble]: {selected_pool}')
        print(model_configs.get('wresnet'))

        pool, _ = load_pool(trans_configs=pool_configs,
                            pool_name=selected_pool,
                            model_configs=model_configs.get('wresnet'),
                            active_list=True)

        wds = list(pool.values())
        model = Ensemble(classifiers=wds, strategy=ENSEMBLE_STRATEGY.AVEP.value, channel_index=1)

        prefix = f'AE-cifar100-wresnet-{selected_pool}'
    else:
        raise ValueError(f'Expect targeted model to be `single`, `adt_pgd`, or `ensemble`. But found {args.targeted_model}.')

    collect_predictions(model=model,
                        data_configs=data_configs)

    del model