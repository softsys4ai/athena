"""
Implement an adversarial detector on top of IBM Trusted-AI ART (1.2.0)
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
import sys
sys.path.append("../")

import argparse
import numpy as np
import os
import time
import json

from art.attacks import FastGradientMethod, ProjectedGradientDescent
from art.detection import BinaryInputDetector
from art.classifiers import KerasClassifier
import keras

from models.mnist_utils import load_lenet
import utils.data as data
import utils.file as file
from models.mnist_cnn import lenet


def train_detector(model_configs):
    # ------------------------------------
    # step 1. loading prereqs and data
    # ------------------------------------
    # load data
    (X_train, Y_train), (X_test, Y_test) = data.load_mnist()

    # size of dataset
    num_samples_train = 5000 #X_train.shape[0]
    num_samples_test = X_test.shape[0]
    X_train = X_train[:num_samples_train]
    Y_train = Y_train[:num_samples_train]
    X_test = X_test[:num_samples_test]
    Y_test = Y_test[:num_samples_test]

    class_descr = [i for i in range(10)]

    # load the underlying model
    lenet_configs = model_configs.get("lenet")
    file = os.path.join(lenet_configs.get("dir"), lenet_configs.get("um_file"))
    # classifier = load_lenet(file, trans_configs=None, use_logits=False, wrap=False)
    classifier = lenet(detector=True)

    # compile model
    classifier.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])

    classifier = KerasClassifier(model=classifier, use_logits=False)

    # ------------------------------------
    # step 2. training a detector
    # ------------------------------------
    # create a pre-trained detector on classifier
    detector = BinaryInputDetector(classifier)

    # To train the detector, we
    # 1. expand the training set with adversarial examples
    # 2. label the data with 0 (original) and 1 (adversarial)
    print(">>> Generating AEs, it will take a while...")
    # train the detector using FGSM
    X_train_adv = []
    for i, eps in enumerate([0.05, 0.075, 0.1, 0.13, 0.15]):
        attacker = ProjectedGradientDescent(classifier, eps=eps, eps_step=eps/5)
        X_train_adv.extend(attacker.generate(X_train[1000*i:1000*(i+1)]))
    X_train_adv = np.asarray(X_train_adv)
    print(">>> AE shape:", X_train_adv.shape)

    # attacker = ProjectedGradientDescent(classifier, eps=0.25)
    # X_train_adv = attacker.generate(X_train)
    num_train = X_train.shape[0]
    X_train_detector = np.concatenate((X_train, X_train_adv), axis=0)
    Y_train_detector = np.concatenate((np.array([[1, 0]] * num_train), np.array([[0, 1]] * num_train)), axis=0)

    # train the detector
    print(">>> Training a detector, it will take a while...")
    detector.fit(X_train_detector, Y_train_detector, nb_epochs=20, batch_size=128)
    # save the detector
    savefile = "{}detector-pgd_various{}".format(lenet_configs.get("wd_prefix"), lenet_configs.get("wd_postfix"))
    # savefile = os.path.join(lenet_configs.get("dir"), savefile)

    print(">>> Current working path:", os.getcwd())
    savefile = os.path.join(os.getcwd(), savefile)
    detector.save(savefile)
    print("Saved the trained detector to [{}]".format(savefile))

    print(">>> Evaluating test data...")
    detections = np.sum(np.argmax(detector.predict(X_test), axis=1) == 0)
    result = detections / X_test.shape[0]
    print(result)

    # load detector
    # print(">>> Loading detector from [{}]".format(savefile))
    # detector2 = keras.models.load_model(savefile)
    # print(detector2)


def eval_detector(classfile, data_configs):
    testsets = {
        "eot_dir": "eot_aes",
        "zk_dir": "zk_aes",
    }
    detector = keras.models.load_model(classfile)

    results = {}
    # evaluate clean data
    for folder, ae_list in testsets.items():
        folder = data_configs.get("dir") + "/" + data_configs.get(folder)
        data = np.load(os.path.join(data_configs.get("dir") + data_configs.get("zk_dir"), data_configs.get("bs_file")))
        detections = np.sum(np.argmax(detector.predict(data), axis=1) == 0)
        results["bs"] = detections / data.shape[0]

        # evaluate adversarial examples
        for aefile in data_configs.get(ae_list):
            print("Evaluating the detector on [{}]...".format(aefile))
            data = np.load(os.path.join(folder, aefile))
            name = aefile.replace(".npy", "")
            name = name.replace("AE-mnist-cnn-", "")
            # name = name.split("-")[-1]

            detections = np.sum(np.argmax(detector.predict(data), axis=1) == 1)
            results[name] = detections / data.shape[0]
        print("Evaluation results:\n", results)
        # save the results
    file.dump_to_json(results, "../../experiment/mnist/detectors/detections-mnist-fgsm_various.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-r", "--experiment-root", required=False,
                        default="../../../")
    parser.add_argument("-m", "--model-configs", required=False,
                        default="../configs/experiment/mnist/model-info.json")
    parser.add_argument("-d", "--data-configs", required=False,
                        default="../configs/experiment/mnist/data-info.json")
    parser.add_argument("-o", "--output-root", required=False,
                        default="../../experiment/mnist/results")

    args = parser.parse_args()
    model_configs = file.load_from_json(args.model_configs)
    model_configs["lenet"]["dir"] = args.experiment_root + model_configs.get("lenet").get("dir")
    data_configs = file.load_from_json(args.data_configs)
    data_configs["dir"] = args.experiment_root + data_configs.get("dir")

    # train_detector(model_configs=model_configs)
    print(data_configs.get("dir"))
    lenet_configs = model_configs.get("lenet")
    savefile = "{}detector-fgsm_various{}".format(lenet_configs.get("wd_prefix"), lenet_configs.get("wd_postfix"))
    savefile = os.path.join("../../../models/mnist/detectors", savefile)
    eval_detector(savefile, data_configs)












