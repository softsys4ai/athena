import sys
import os
import numpy as np
import time
from keras.utils import to_categorical

from attacks.attacker import get_adversarial_examples
from data.data import load_data
from models.transformation import transform
from utils.config import ATTACK, DATA, MODE, TRANSFORMATION
from utils.file import save_adv_examples


def craft(model_name, method, X, Y):
    '''
        input:
            X: nSamples X <dimension of a sample>
            Y: 2D numpy array - nSamples X nClasses
    '''
    model_prefix, dataset, architect, trans_type = model_name.split('-')
    prefix = "BB_{}".format(model_prefix)

    if method == ATTACK.FGSM:
        eps = 0.3
        print('{}: (eps={})'.format(method.upper(), eps))
        X_adv, _ = get_adversarial_examples(model_name, method, X, Y, eps=eps)

        attack_params = 'eps{}'.format(int(1000 * eps))

        save_adv_examples(X_adv, prefix=prefix, dataset=dataset, transformation=trans_type,
                          attack_method=method, attack_params=attack_params)

    elif method == "biml2": # BIM l2 norm
        order = 2
        nb_iter = 100
        norm = order
        eps = 2
        print('{}: (ord={}, nb_iter={}, eps={})'.format(ATTACK.BIM.upper(), order, nb_iter, eps))
        X_adv, _ = get_adversarial_examples(model_name, ATTACK.BIM, X, Y,
                                            ord=order, nb_iter=nb_iter, eps=eps)

        attack_params = 'ord{}_nbIter{}_eps{}'.format(norm, nb_iter, int(1000 * eps))
        save_adv_examples(X_adv, prefix=prefix, dataset=dataset, transformation=trans_type,
                          attack_method=ATTACK.BIM, attack_params=attack_params)

    elif method == "bimli": # BIM Inf norm
        order = np.inf
        nb_iter = 100
        norm = 'inf'
        eps = 0.3

        print('{}: (ord={}, nb_iter={}, eps={})'.format(ATTACK.BIM.upper(), order, nb_iter, eps))
        X_adv, _ = get_adversarial_examples(model_name, ATTACK.BIM, X, Y,
                                            ord=order, nb_iter=nb_iter, eps=eps)

        attack_params = 'ord{}_nbIter{}_eps{}'.format(norm, nb_iter, int(1000 * eps))
        save_adv_examples(X_adv, prefix=prefix, dataset=dataset, transformation=trans_type,
                          attack_method=ATTACK.BIM, attack_params=attack_params)


    elif method == ATTACK.DEEPFOOL:
        order=2
        overshoot = 0
        print('attack {} -- order: {}; overshoot: {}'.format(method.upper(), order, overshoot))
        X_adv, _ = get_adversarial_examples(model_name, method, X, Y,
                                            ord=order, overshoot=overshoot)

        attack_params = 'l{}_overshoot{}'.format(order, int(overshoot * 10))
        save_adv_examples(X_adv, prefix=prefix, bs_samples=X, dataset=dataset, transformation=trans_type,
                          attack_method=method, attack_params=attack_params)

    elif method == ATTACK.CW_L2:
        binary_search_steps = 9
        cw_batch_size = 1
        initial_const = 10

        learning_rate = 1
        max_iter = 100
        print('{}: (ord={}, max_iterations={})'.format(method.upper(), 2, max_iter))
        X_adv, _ = get_adversarial_examples(model_name, method, X, Y,
                                            ord=2, max_iterations=max_iter,
                                            binary_search_steps=binary_search_steps,
                                            cw_batch_size=cw_batch_size,
                                            initial_const=initial_const,
                                            learning_rate=learning_rate)

        attack_params = 'lr{}_maxIter{}'.format(int(learning_rate * 100), max_iter)
        save_adv_examples(
                X_adv, prefix=prefix, bs_samples=X, dataset=dataset,
                transformation=trans_type, attack_method=method, attack_params=attack_params)

    elif method == ATTACK.JSMA:
        theta = 0.5
        gamma = 0.7
        print('{}: (theta={}, gamma={})'.format(method.upper(), theta, gamma))
        X_adv, _ = get_adversarial_examples(model_name, method, X, Y,
                                            theta=theta, gamma=gamma)

        attack_params = 'theta{}_gamma{}'.format(int(100 * theta), int(100 * gamma))
        save_adv_examples(
                X_adv, prefix=prefix, bs_samples=X, dataset=dataset,
                transformation=trans_type, attack_method=method, attack_params=attack_params)

    elif method == ATTACK.PGD:
        nb_iter = 100
        eps_iter = 0.01
        eps = 0.30 
        X_adv, _ = get_adversarial_examples(model_name, method, X, Y,
                                            eps=eps, nb_iter=nb_iter, eps_iter=eps_iter)
        attack_params = 'eps{}'.format(int(1000 * eps))
       
        save_adv_examples(
                X_adv, prefix=prefix, bs_samples=X, dataset=dataset,
                transformation=trans_type, attack_method=method, attack_params=attack_params)

    elif method == ATTACK.ONE_PIXEL:
        pixel_counts = 30
        max_iter = 30
        pop_size = 100
        attack_params = {
            'pixel_counts': pixel_counts,
            'max_iter': max_iter,
            'pop_size': pop_size
        }
        X_adv, _ = get_adversarial_examples(model_name, method, X, Y, **attack_params)
        X_adv = np.asarray(X_adv)
        attack_params = 'pxCount{}_maxIter{}_popsize{}'.format(pixel_counts, max_iter, pop_size)
        save_adv_examples(
                X_adv, prefix=prefix, bs_samples=X, dataset=dataset,
                transformation=trans_type, attack_method=method, attack_params=attack_params)

    elif method == ATTACK.MIM:
        eps = 0.3
        nb_iter = 1000
        attack_params = {
            'eps': eps,
            'nb_iter': nb_iter
        }

        X_adv, _ = get_adversarial_examples(model_name, method, X, Y, **attack_params)
        attack_params = 'eps{}_nbIter{}'.format(int(eps * 1000), nb_iter)
        save_adv_examples(
                X_adv, prefix=prefix, bs_samples=X, dataset=dataset,
                transformation=trans_type, attack_method=method, attack_params=attack_params)

    del X
    del Y


def main(argv):

    BSDir=argv[0]
    ensembleTag = argv[1]

    nClasses = 10
    nSamplesList = [10, 50, 100, 500, 1000]

    attackMethods = [
            'fgsm',
            'biml2',
            'bimli',
            'deepfool',
            'cw_l2',
            'mim',
            'pgd',
            'jsma',
            'onepixel'
            ]

    timeCosts = np.zeros((len(attackMethods), len(nSamplesList)))
    row=0
    for attackMethod in attackMethods:
        for col, nSamples in zip(range(len(nSamplesList)), nSamplesList):
            modelNamePrefix = "model_"+str(nSamples)+"Samples_"+ensembleTag
            modelName = "{}-mnist-cnn-clean".format(modelNamePrefix)

            try:
                X = np.load(os.path.join(BSDir, "BS_1k_data_For_AE.npy"))
                Y = np.load(os.path.join(BSDir, "BS_1k_label_For_AE.npy"))
                Y = to_categorical(Y)

                print("Use {} to attack {}.h5".format(attackMethod, modelName))
                startTime = time.time()
                craft(modelName, attackMethod, X, Y)
                endTime = time.time()
                timeCost = round(endTime - startTime, 2)
                print("Time cost ({}): {}\n".format(attackMethod, timeCost))

                timeCosts[row, col] = timeCost

            except (OSError) as e:
                print('Failed to load model [{}]: {}.'.format(modelName, e))
                continue
        row += 1

    TC_FP = "AE_TimeCost_target-"+ensembleTag+".csv"
 
    with open(TC_FP, "w") as fp:
        fp.write("\t10\t50\t100\t500\t1000\n")
        for row in range(len(attackMethods)):
            fp.write(attackMethods[row]+"\t"
                    +str(timeCosts[row, 0])+"\t"
                    +str(timeCosts[row, 1])+"\t"
                    +str(timeCosts[row, 2])+"\t"
                    +str(timeCosts[row, 3])+"\t"
                    +str(timeCosts[row, 4])+"\n")

if __name__ == '__main__':

    main(sys.argv[1:])
