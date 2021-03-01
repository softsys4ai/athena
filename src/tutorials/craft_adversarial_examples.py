"""
A sample to generate adversarial examples in the context of white-box threat model.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import sys
sys.path.append("../")

import argparse
import numpy as np
import os
import time
from matplotlib import pyplot as plt

from models.mnist_utils import load_pool
from utils.file import load_from_json
from models.utils.estimator import error_rate
from attacks.attacker_art import generate
from models.athena import Ensemble, ENSEMBLE_STRATEGY


def generate_ae(model, data, labels, attack_configs,
                eot=False,
                save=False, output_dir=None):
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
    img_rows, img_cols = data.shape[1], data.shape[2]

    adversaries = attack_configs.get("adversaries")
    if adversaries is None:
        # use all adversaries if not specified
        num_attacks = attack_configs.get("adversaries")
        adversaries = [i for i in range(num_attacks)]

    data_loader = (data, labels)

    if len(labels.shape) > 1:
        labels = np.asarray([np.argmax(p) for p in labels])

    # generate attacks one by one
    for id in adversaries:
        key = "configs{}".format(id)
        attack_args = attack_configs.get(key)

        attack_args["eot"] = eot
        data_adv = generate(model=model,
                            data_loader=data_loader,
                            attack_args=attack_args
                            )
        # predict the adversarial examples
        predictions = model.predict(data_adv)
        predictions = np.asarray([np.argmax(p) for p in predictions])

        err = error_rate(y_pred=predictions, y_true=labels)
        print(">>> error rate:", err)

        # plotting some examples
        num_plotting = min(data.shape[0], 1)
        for i in range(num_plotting):
            img = data_adv[i].reshape((img_rows, img_cols))
            plt.imshow(img, cmap='gray')
            title = '{}(EOT:{}): {}->{}'.format(attack_configs.get(key).get("description"),
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
            # save with a random name
            file = os.path.join(output_dir, "{}.npy".format(time.monotonic()))
            print("Save the adversarial examples to file [{}].".format(file))
            np.save(file, data_adv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-r', '--experiment-root', required=False,
                        default='../../../')
    parser.add_argument('-p', '--pool-configs', required=False,
                        default='../configs/experiment/mnist/vanilla-athena.json')
    parser.add_argument('-m', '--model-configs', required=False,
                        default='../configs/experiment/mnist/model-info.json',
                        help='Folder where models stored in.')
    parser.add_argument('-d', '--data-configs', required=False,
                        default='../configs/experiment/mnist/data-info.json',
                        help='Folder where test data stored in.')
    parser.add_argument('-a', '--attack-configs', required=False,
                        default='../configs/experiment/mnist/attack-zk.json',
                        help='Folder where test data stored in.')
    parser.add_argument('-o', '--output-root', required=False,
                        default='../../results',
                        help='Folder for outputs.')
    parser.add_argument('--debug', required=False, default=False)

    args = parser.parse_args()

    print("------AUGMENT SUMMARY-------")
    print("EXPERIMENT ROOT:", args.experiment_root)
    print("POOL CONFIGS:", args.pool_configs)
    print("MODEL CONFIGS:", args.model_configs)
    print("DATA CONFIGS:", args.data_configs)
    print("ATTACK CONFIGS:", args.attack_configs)
    print("OUTPUT ROOT:", args.output_root)
    print("DEBUGGING MODE:", args.debug)
    print('----------------------------\n')

    # ----------------------------
    # parse configurations (into a dictionary) from json file
    # ----------------------------
    pool_configs = load_from_json(args.pool_configs)
    model_configs = load_from_json(args.model_configs)
    model_configs["lenet"]["dir"] = args.experiment_root + model_configs.get("lenet").get("dir")
    model_configs["svm"]["dir"] = args.experiment_root + model_configs.get("svm").get("dir")
    data_configs = load_from_json(args.data_configs)
    data_configs["dir"] = args.experiment_root + data_configs.get("dir")
    attack_configs = load_from_json(args.attack_configs)

    # ---------------------------
    # load the targeted model
    # ---------------------------
    #
    # In the context of the zero-knowledge threat model,
    # we use the undefended model as adversary's target model.
    # model_file = os.path.join(model_configs.get("dir"), model_configs.get("um_file"))
    # target = load_lenet(file=model_file, wrap=True)

    # In the context of the white-box threat model,
    # we use the ensemble as adversary's target model.
    # load weak defenses (in this example, load a tiny pool of 3 weak defenses)
    pool_configs["active_wds"] = pool_configs.get("demo_pool")
    print(">>> POOL:", pool_configs.get("active_wds"))
    pool, _ = load_pool(trans_configs=pool_configs,
                        model_configs=model_configs.get("lenet"),
                        active_list=True,
                        wrap=True)
    # create an AVEP ensemble as the target model
    wds = list(pool.values())
    target = Ensemble(classifiers=wds, strategy=ENSEMBLE_STRATEGY.AVEP.value)

    # -----------------------
    # Prepare benign samples and corresponding true labels for AE generation
    # -----------------------
    # load the benign samples
    data_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
    data_bs = np.load(data_file)
    # load the corresponding true labels
    label_file = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
    labels = np.load(label_file)

    # ------------------------
    # Generate adversarial examples for a small subset
    # ------------------------
    data_bs = data_bs[:5]
    labels = labels[:5]

    # Normal approach
    # Compute the loss w.r.t. a single input
    # For an ensemble target, averaging the losses of WDs'.
    generate_ae(model=target,
                data=data_bs, labels=labels,
                eot=False,
                attack_configs=attack_configs
                )

    # Adaptive approach (with EOT)
    # Compute the loss expectation over specific distribution.
    # For an ensemble target, averaging the EOT of WDs'.
    generate_ae(model=target,
                data=data_bs, labels=labels,
                eot=True,
                attack_configs=attack_configs
                )
