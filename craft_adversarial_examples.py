"""
This is the script to craft adversarial examples.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

from config import *
from data import load_data
from file import save_adv_examples
from attacks.attacker import get_adversarial_examples

import numpy as np

def craft(dataset, method):
  print('loading original images...')
  # generate for test set
  _, (X, Y) = load_data(dataset)
  prefix = 'test'

  # ------ for black-box attack ------
  # generate for train set (the last 20% of the original train set)
  # (X, Y), _ = load_data(dataset)
  # nb_trainings = int(X.shape[0] * 0.8)
  # X = X[nb_trainings:]
  # Y = Y[nb_trainings:]
  # prefix = 'val'
  # ---------------------------------

  """
  In debugging mode, crafting for 50 samples.
  """
  if MODE.DEBUG:
    X = X[:50]
    Y = Y[:50]

  model_name = 'model-{}-cnn-clean'.format(dataset)

  X_adv = None
  if method == ATTACK.FGSM:
    for eps in ATTACK.get_fgsm_eps():
      print('{}: (eps={})'.format(method.upper(), eps))
      X_adv, _ = get_adversarial_examples(model_name, method, X, Y, eps=eps)

      attack_params = 'eps{}'.format(int(1000 * eps))
      save_adv_examples(X_adv, prefix=prefix, dataset=dataset, transformation='clean',
                        attack_method=method, attack_params=attack_params)
  elif method == ATTACK.BIM:
    for ord in ATTACK.get_bim_norm():
      for nb_iter in ATTACK.get_bim_nbIter():
        for eps in ATTACK.get_bim_eps(ord):
          print('{}: (ord={}, nb_iter={}, eps={})'.format(method.upper(), ord, nb_iter, eps))
          X_adv, _ = get_adversarial_examples(model_name, method, X, Y,
                                              ord=ord, nb_iter=nb_iter, eps=eps)

          if ord == np.inf:
            norm = 'inf'
          else:
            norm = ord
          attack_params = 'ord{}_nbIter{}_eps{}'.format(norm, nb_iter, int(1000 * eps))
          save_adv_examples(X_adv, prefix=prefix, dataset=dataset, transformation='clean',
                            attack_method=method, attack_params=attack_params)
  elif method == ATTACK.DEEPFOOL:
    for max_iter in ATTACK.get_df_maxIter():
      for clip_max in ATTACK.get_df_clip_max():
        print('{}: (max_iterations={})'.format(method.upper(), max_iter))
        X_adv, _ = get_adversarial_examples(model_name, method, X, Y, max_iterations=max_iter, clip_max=clip_max)

      attack_params = 'maxIter{}'.format(max_iter)
      save_adv_examples(X_adv, prefix=prefix, dataset=dataset, transformation='clean',
                        attack_method=method, attack_params=attack_params)
  elif method == ATTACK.CW:
    for ord in ATTACK.get_cw_order():
      for max_iter in ATTACK.get_cw_maxIter():
        print('{}: (ord={}, max_iterations={})'.format(method.upper(), ord, max_iter))
        if ord == 2:
          X_adv, _ = get_adversarial_examples(model_name, method, X, Y,
                                              ord=ord, max_iterations=max_iter)

          attack_params = 'L{}_maxIter{}'.format(ord, max_iter)
          save_adv_examples(X_adv, prefix=prefix, dataset=dataset, transformation='clean',
                            attack_method=method, attack_params=attack_params)
  elif method == ATTACK.JSMA:
    for theta in ATTACK.get_jsma_theta():
      for gamma in ATTACK.get_jsma_gamma():
        print('{}: (theta={}, gamma={})'.format(method.upper(), theta, gamma))
        X_adv, _ = get_adversarial_examples(model_name, method, X, Y,
                                            theta=theta, gamma=gamma)

        attack_params = 'theta{}_gamma{}'.format(int(100 * theta), int(100 * gamma))
        save_adv_examples(X_adv, prefix=prefix, dataset=dataset, transformation='clean',
                          attack_method=method, attack_params=attack_params)

  elif method == ATTACK.PGD:
    nb_iter = 100
    eps_iter = 0.01
    for eps in ATTACK.get_pgd_eps():
      X_adv, _ = get_adversarial_examples(model_name, method, X, Y,
                                          eps=eps, nb_iter=nb_iter, eps_iter=eps_iter)
      attack_params = 'eps{}_nbIter{}_epsIter{}'.format(
        int(1000 * eps), nb_iter, int(1000 * eps_iter)
      )
      save_adv_examples(X_adv, prefix=prefix, dataset=dataset, transformation='clean',
                        attack_method=method, attack_params=attack_params)

  del X
  del Y

def main(dataset, attack_method):
  craft(dataset, attack_method)

if __name__ == '__main__':
  """
  switch on debugging mode
  """
  MODE.debug_on()
  main(DATA.mnist, ATTACK.DEEPFOOL)

