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

from models.cifar100_utils import load_model
from utils.file import load_from_json
from data.data import load_data
from art.defences.trainer.adversarial_trainer import AdversarialTrainer
from art.attacks import ProjectedGradientDescent

def pgd_adv_train(model, data, outpath, model_name):
    attack = ProjectedGradientDescent(model,
                                      eps=0.015,
                                      eps_step=0.001,
                                      max_iter=2,
                                      targeted=False,
                                      num_random_init=0,
                                      )

    adv_trainer = AdversarialTrainer(model,
                                     attacks=attack,
                                     ratio=1.0)
    print('>>> Processing adversarial training, it will take a while...')
    x_train, y_train = data
    adv_trainer.fit(x_train, y_train, nb_epochs=30, batch_size=32)

    savefile = os.path.join(outpath, model_name)
    print('>>>Save the model to [{}]'.format(savefile))
    adv_trainer.classifier.save(savefile)

    return adv_trainer.classifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-r', '--experiment-root', required=False,
                        default='../../../')
    parser.add_argument('-m', '--model-configs', required=False,
                        default='../configs/experiment/demo/model-info.json',
                        help='Folder where models stored in.')
    parser.add_argument('-o', '--output-root', required=False,
                        default='../../experiment/demo/results',
                        help='Folder for outputs.')

    args = parser.parse_args()


    print("------AUGMENT SUMMARY-------")
    print("EXPERIMENT ROOT:", args.experiment_root)
    print("MODEL CONFIGS:", args.model_configs)
    print("OUTPUT ROOT:", args.output_root)
    print('----------------------------\n')

    # ----------------------------
    # parse configurations (into a dictionary) from json file
    # ----------------------------
    model_configs = load_from_json(args.model_configs)
    model_configs["wresnet"]["dir"] = args.experiment_root + model_configs.get("wresnet").get("dir")

    # ---------------------------
    # load the targeted model
    # ---------------------------
    # In the context of the adversarially trained model,
    # we use the undefended model as adversary's target model.
    savefile = "AdvTrained-cifar100.pth"
    model_file = os.path.join(model_configs.get("wresnet").get('dir'), model_configs.get("wresnet").get("pgd_trained_cifar"))
    model, _, _ = load_model(file=model_file, model_configs=model_configs.get("wresnet"), trans_configs=None)

    (x_train, y_train), _ = load_data('cifar100')

    pgd_adv_train(model=model,
                  data=(x_train, y_train),
                  outpath=args.output_root,
                  model_name=savefile
                  )
