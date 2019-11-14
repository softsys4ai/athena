import sys
import os
import numpy as np
import time

from attacks.attacker import get_adversarial_examples
from data import load_data
from transformation import transform
from utils.config import ATTACK, DATA, MODE, TRANSFORMATION
from utils.file import save_adv_examples


def craft(model_name, method, X, Y):
    '''
        input:
            X: nSamples X <dimension of a sample>
            Y: 2D numpy array - nSamples X nClasses
    '''
    model_prefix, dataset, architect, trans_type = model_name.split('-')
    prefix = "BB"

    if method == ATTACK.FGSM:
        eps = 0.3
        print('{}: (eps={})'.format(method.upper(), eps))
        X_adv, _ = get_adversarial_examples(model_name, method, X, Y, eps=eps)

        attack_params = 'eps{}'.format(int(1000 * eps))

        save_adv_examples(X_adv, prefix=prefix, dataset=dataset, transformation=trans_type,
                          attack_method=method, attack_params=attack_params)

    elif method == ATTACK.BIM: # BIM l2 norm and BIM Inf norm
        for order in ATTACK.get_bim_norm():
            nb_iter = 100
            eps = ATTACK.get_bim_eps(order)[-1]
            print('{}: (ord={}, nb_iter={}, eps={})'.format(method.upper(), order, nb_iter, eps))
            X_adv, _ = get_adversarial_examples(model_name, method, X, Y,
                                                ord=order, nb_iter=nb_iter, eps=eps)

            if order == np.inf:
                norm = 'inf'
            else:
                norm = order
            attack_params = 'ord{}_nbIter{}_eps{}'.format(norm, nb_iter, int(1000 * eps))
            save_adv_examples(X_adv, prefix=prefix, dataset=dataset, transformation=trans_type,
                              attack_method=method, attack_params=attack_params)

    elif method == ATTACK.DEEPFOOL:
        order=2
        overshoot = 50
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

        learning_rate = 7
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
        eps = 0.75 
        X_adv, _ = get_adversarial_examples(model_name, method, X, Y,
                                            eps=eps, nb_iter=nb_iter, eps_iter=eps_iter)
        attack_params = 'eps{}_nbIter{}_epsIter{}'.format(
            int(1000 * eps), nb_iter, int(1000 * eps_iter)
        )
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
        eps = 0.5
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

    queriedDir=argv[0]
    attack_method = argv[1]
    #cleanModelsDir=argv[1]

    nAEs = 10
    
    nClasses = 10
    nSamplesList = [50, 100, 500, 1000, 5000]
    nSamplesList.reverse()

    attack_methods = [
            #'fgsm',
            #'bim',
            #'deepfool',
            #'cw_l2',
            #'jsma',
            #'mim',
            #'pgd',
            #'onepixel'
            ]
    #ensembleTags = ["prob0", "prob1", "prob2", "prob3", "logit2"]
    ensembleTag = "prob1"

    #timeCosts = np.zeros((len(attack_methods), len(nSamplesList)))

    for col, nSamples in zip(range(len(nSamplesList)), nSamplesList):
        datasetName = "mnist"+str(nSamples)+"Samples"+ensembleTag
        model_name = "model-{}-cnn-clean".format(datasetName)
        X = np.load(os.path.join(queriedDir, datasetName+"_data.npy"))[:nAEs]
        labels = np.load(os.path.join(queriedDir, datasetName+"_label.npy"))[:nAEs]
       
        print("Attacking {}.h5".format(model_name))
        #nSamples = nAEs
        #datasetName = "mnist"+str(nSamples)+"Samples"+ensembleTag
        #model_name = "model-{}-cnn-clean".format(datasetName)

        try:
            Y = np.zeros((nAEs, nClasses))
            for sIdx in range(nAEs):
                Y[sIdx, labels[sIdx]] = 1
            print("\n[{} - {}]".format(nAEs, attack_method))

            start_time = time.time()
            craft(model_name, attack_method, X, Y)
            end_time = time.time()
            timeCost = round(end_time - start_time, 2)
            print("{} - {}: {}".format(nAEs, attack_method, timeCost))

            #timeCosts[row, col] = timeCost
        except (FileNotFoundError, OSError) as e:
            print('Failed to load model [{}]: {}.'.format(model_name, e))
            continue

#    with open("AE_time_cost.txt", "w") as fp:
#        fp.write("\t50\t100\t500\t1000\t5000\n")
#        for row in range(len(attack_methods)):
#            fp.write(attack_methods[row]+"\t"
#                    +str(timeCosts[row, 0])+"\t"
#                    +str(timeCosts[row, 1])+"\t"
#                    +str(timeCosts[row, 2])+"\t"
#                    +str(timeCosts[row, 3])+"\t"
#                    +str(timeCosts[row, 4])+"\n")

if __name__ == '__main__':
    """
    switch on debugging mode
    """
    main(sys.argv[1:])
