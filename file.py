"""
Implement file operations such as read/write files here.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import csv
import os

from config import *
from data import load_data
from utils.plot import plot_comparisons

def dict2csv(dictionary, file_name):
  """
  Serialize values in given dictionary to a csv file.
  :param dictionary: the dictionary to save.
  :param file_name: the name of the csv file, including the path.
  """
  print(dictionary)
  with open(file_name, 'w') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    writer.writerow(dictionary.keys())
    writer.writerows(zip(*dictionary.values()))

def save_adv_examples(data, **kwargs):
  """
  Serialize generated adversarial examples to an npy file.
  :param data: the adversarial examples to store.
  :param kwargs: information needed to construct the file name
  :return: na
  """
  prefix = kwargs.get('prefix', 'test')
  dataset = kwargs.get('dataset', 'cifar10')
  architect = kwargs.get('architect', 'cnn')
  transformation = kwargs.get('transformation', 'clean')
  attack_method = kwargs.get('attack_method', 'fgsm')
  attack_params = kwargs.get('attack_params', None)

  prefix = '{}_AE'.format(prefix)
  attack_info = '{}_{}'.format(attack_method, attack_params)
  model_info = '{}-{}-{}'.format(dataset, architect, transformation)
  file_name = '{}-{}-{}.npy'.format(prefix, model_info, attack_info)
  np.save('{}/{}'.format(PATH.ADVERSARIAL_FILE, file_name), data)

  if MODE.DEBUG:
    title = '{}-{}'.format(model_info, attack_info)
    _, (bs_samples, _) = load_data(dataset)
    plot_comparisons(bs_samples[10:20], data[10:20], title)

  print('Adversarial examples saved to {}/{}.'.format(PATH.ADVERSARIAL_FILE, file_name))