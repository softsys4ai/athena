"""
Experiment:
    1. Objective: Generate adversarial examples.
    2. Context: zero-knowledge model and optimization-based white-box model.
    3. Dataset: CIFAR100.

@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
sys.path.append("../")

import argparse
import numpy as np
import os
import time
import json
from matplotlib import pyplot as plt

from models.cifar100_utils import load_model, load_pool
from utils.file import load_from_json
from models.utils.estimator import error_rate
from attacks.attacker_art import generate
from models.athena import Ensemble, ENSEMBLE_STRATEGY


def generate_ae(model, model_name, data, labels,
                attack_configs, selected_attacks,
                eot=False, save=True, prefix=None, output_dir=None):
    """
    Generate adversarial examples
    :param model: WeakDefense. The targeted model.
    :param data: array. The benign samples to generate adversarial for.
    :param labels: array or list. The true labels.
    :param attack_configs: dictionary. Attacks and corresponding settings.
    :param save: boolean. True, if save the adversarial examples.
    :param output_dir: str or path. Location to save the adversarial examples.
        It cannot be None when save is True.
    :return:
    """
    num_bs, img_rows, img_cols = data.shape[0], data.shape[1], data.shape[2]

    print(">>> Generating AEs for {} benign samples.".format(num_bs))
    adversaries = attack_configs.get(selected_attacks)
    if adversaries is None:
        # use all adversaries if not specified
        num_attacks = attack_configs.get("num_attacks")
        adversaries = [i for i in range(num_attacks)]

    data_loader = (data, labels)

    if len(labels.shape) > 1:
        labels = np.asarray([np.argmax(p) for p in labels])

    # generate attacks one by one
    cost = {}
    for id in adversaries:
        key = "configs{}".format(id)
        attack_args = attack_configs.get(key)
        attack_args["eot"] = eot
        print('[DEBUG][SAVE_FILE]:', "{}-{}.npy".format(prefix, attack_args.get("description")))

        start = time.monotonic()
        data_adv = generate(model=model,
                            data_loader=data_loader,
                            attack_args=attack_args
                            )
        end = time.monotonic()
        cost[attack_args.get("description")] = (end - start) / num_bs
        print('>>> Average Cost::: {}:{}'.format(attack_args.get("description"), cost[attack_args.get("description")]))

        # predict the adversarial examples
        predictions = model.predict(data_adv)
        predictions = np.asarray([np.argmax(p) for p in predictions])

        err = error_rate(y_pred=predictions, y_true=labels)
        print(">>> error rate:", err)

        # plotting some examples
        num_plotting = min(data.shape[0], 0)
        for i in range(num_plotting):
            img = data_adv[i].reshape((img_rows, img_cols, 3))
            plt.imshow(img)
            title = '{}(EOT:{}): {}->{}'.format(attack_args.get("description"),
                                                "ON" if eot else "OFF",
                                                labels[i],
                                                predictions[i]
                                                )
            plt.title(title)
            plt.show()
            plt.close()

        # save the adversarial example
        if save:
            if output_dir is None:
                raise ValueError("Cannot save images to a none path.")

            if prefix is None:
                file = "{}-{}-{}.npy".format(attack_args.get("description"), eot, time.monotonic())
            else:
                file = "{}-{}-{}.npy".format(prefix, attack_args.get("description"), eot)

            file = os.path.join(output_dir, file)
            print("Save the adversarial examples to file [{}].".format(file))
            np.save(file, data_adv)

        # delete generated data
        del data_adv

    if output_dir is not None:
        if prefix is None:
            cost_file = "Cost-GenAE-cifar100-cnn-{}-{}.json".format(selected_attacks, time.monotonic())
        else:
            cost_file = "Cost-Gen{}-{}.json".format(prefix, selected_attacks)

        print(">>> [COST] ({}): {}".format(cost_file, cost))

        cost_file = os.path.join(output_dir, cost_file)

        if os.path.exists(cost_file):
            append_write = 'a' # append to the file if already exists
        else:
            append_write = 'w' # otherwise, make a new file

        with open(cost_file, append_write) as file:
            file.write('\n------------------------\n')
            file.write('{}\n'.format(model_name))
            json.dump(cost, file)

def generate_ae_zk():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-r', '--experiment-root', required=False,
                        default='../../../')
    parser.add_argument('-p', '--pool-configs', required=False,
                        default='../configs/experiment/demo/pool.json')
    parser.add_argument('--selected-pool', required=False, default=None,
                        help='The key value of the selected pool.')
    parser.add_argument('-m', '--model-configs', required=False,
                        default='../configs/experiment/demo/model-info.json',
                        help='Folder where models stored in.')
    parser.add_argument('-d', '--data-configs', required=False,
                        default='../configs/experiment/demo/data.json',
                        help='Folder where test data stored in.')
    parser.add_argument('-b', '--benign-samples', required=False,
                        default='bs_ratio0002')
    parser.add_argument('-a', '--attack-configs', required=False,
                        default='../configs/experiment/demo/attack.json',
                        help='Folder where test data stored in.')
    parser.add_argument('--selected-attacks', required=False, default=None,
                        help='The key value of the selected attacks.')
    parser.add_argument('-o', '--output-root', required=False,
                        default='../../experiment/demo/results',
                        help='Folder for outputs.')
    parser.add_argument('--targeted-model', required=False, default='single',
                        help='The type of the adversarial target. It has to be `single`, `adt_pgd`, or `ensemble`.')
    parser.add_argument('--eot', required=False, default='false',
                        help='`True` if use EOT, `False` otherwise.')
    parser.add_argument('--debug', required=False, default='false')

    args = parser.parse_args()

    eot = args.eot
    if isinstance(eot, str) and eot.lower() == 'true':
        print("set EOT to True")
        args.eot = True
    else:
        print("set EOT to False")
        args.eot = False

    debug = args.debug
    if isinstance(debug, str) and debug.lower() == 'true':
        args.debug = True
    else:
        args.debug = False

    print("------AUGMENT SUMMARY-------")
    print("EXPERIMENT ROOT:", args.experiment_root)
    print("POOL CONFIGS:", args.pool_configs)
    print("SELECTED POOL:", args.selected_pool)
    print("MODEL CONFIGS:", args.model_configs)
    print("DATA CONFIGS:", args.data_configs)
    print("BENIGN SAMPLES:", args.benign_samples)
    print("ATTACK CONFIGS:", args.attack_configs)
    print("SELECTED ATTACKS:", args.selected_attacks)
    print("TARGETED MODEL:", args.targeted_model)
    print("EOT or not:", args.eot)
    print("OUTPUT ROOT:", args.output_root)
    print("DEBUGGING MODE:", args.debug)
    print('----------------------------\n')

    # ----------------------------
    # parse configurations (into a dictionary) from json file
    # ----------------------------
    pool_configs = load_from_json(args.pool_configs)
    model_configs = load_from_json(args.model_configs)
    model_configs["wresnet"]["dir"] = args.experiment_root + model_configs.get("wresnet").get("dir")
    model_configs["shake26"]["dir"] = args.experiment_root + model_configs.get("shake26").get("dir")
    data_configs = load_from_json(args.data_configs)
    data_configs["dir"] = args.experiment_root + data_configs.get("dir")
    attack_configs = load_from_json(args.attack_configs)

    # ---------------------------
    # load the targeted model
    # ---------------------------
    if args.targeted_model == 'single':
        # In the context of the zero-knowledge threat model,
        # we use the undefended model as adversary's target model.
        prefix = "AE-cifar100-wresnet-{}".format(args.targeted_model)

        model_file = os.path.join(model_configs.get("wresnet").get('dir'), model_configs.get("wresnet").get("um_file"))
        target, _, _ = load_model(file=model_file, model_configs=model_configs.get("wresnet"), trans_configs=None)
    elif args.targeted_model == 'adt_pgd':
        # In the context of the adversarially trained model,
        # we use the undefended model as adversary's target model.
        prefix = "AE-cifar100-wresnet-{}".format(args.targeted_model)

        model_file = os.path.join(model_configs.get("wresnet").get('dir'), model_configs.get("wresnet").get("pgd_trained_cifar"))

        if os.path.isfile(model_file):
            # model exist
            model, _, _ = load_model(file=model_file, model_configs=model_configs.get("wresnet"), trans_configs=None)
        else:
            model_file = os.path.join(model_configs.get("wresnet").get('dir'),
                                      model_configs.get("wresnet").get("um_file"))
            model, _, _ = load_model(file=model_file, model_configs=model_configs.get("wresnet"), trans_configs=None)

            # train a model first
            from data.data import load_data
            from adversarial_train import pgd_adv_train
            (x_train, y_train), _ = load_data('cifar100', channel_first=True)
            print('>>> Training the model...')

            target = pgd_adv_train(model=model,
                                   data=(x_train, y_train),
                                   outpath=model_configs.get("wresnet").get('dir'),
                                   model_name=model_configs.get("wresnet").get("pgd_trained_cifar"))

    elif args.targeted_model == 'ensemble':
        # In the context of the white-box threat model,
        # we use the ensemble as adversary's target model.
        # load weak defenses (in this example, load a tiny pool of 3 weak defenses)
        if args.selected_pool is None:
            selected_pool = "demo_pool"
        else:
            selected_pool = args.selected_pool

        print(">>> Target Ensemble:", selected_pool)
        print(model_configs.get("wresnet"))

        pool, _ = load_pool(trans_configs=pool_configs,
                            pool_name=selected_pool,
                            model_configs=model_configs.get("wresnet"),
                            active_list=True
                            )
        # create an AVEP ensemble as the target model
        wds = list(pool.values())
        target = Ensemble(classifiers=wds, strategy=ENSEMBLE_STRATEGY.AVEP.value, channel_index=1)

        # prefix = "AE-cifar100-wresnet-{}_{}".format(args.targeted_model, len(wds))
        # for revision
        prefix = "AE-cifar100-wresnet-{}".format(selected_pool)
    else:
        raise ValueError('Expect targeted model to be `single`, `adt_pgd`, or `ensemble`. But found {}.'.format(args.targeted_model))

    # -----------------------
    # Prepare benign samples and corresponding true labels for AE generation
    # -----------------------
    # load the benign samples & corresponding true labels
    bs = args.benign_samples if args.benign_samples else "bs_ratio0002"
    data_file = os.path.join(data_configs.get('dir'), data_configs.get(bs)[0])
    label_file = os.path.join(data_configs.get('dir'), data_configs.get(bs)[1])
    print(">>> Loading data from [{}].".format(data_file))
    data_bs = np.load(data_file)
    print(">>> Loading oracle labels from [{}].".format(label_file))
    labels = np.load(label_file)

    # ------------------------
    # Generate adversarial examples for a small subset
    # ------------------------
    # data_bs = data_bs[:1]
    # labels = labels[:1]

    # Normal approach (EOT=False)
    # Compute the loss w.r.t. a single input
    # For an ensemble target, averaging the losses of WDs'.
    # Adaptive approach (EOT=True)
    # Compute the loss expectation over a specific distribution.
    # For an ensemble target, averaging the EOT of WDs'.
    print('[DEBUG]prefix:', prefix)
    generate_ae(model=target,
                model_name=args.selected_pool,
                data=data_bs, labels=labels,
                eot=args.eot,
                save=True,
                prefix=prefix,
                attack_configs=attack_configs,
                selected_attacks=args.selected_attacks,
                output_dir=args.output_root,
                )

    # del useless objects
    del target, data_bs, labels
