"""
Implement file operations such as read/write files here.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import csv
from collections import namedtuple
from enum import Enum
import json

from utils.config import *
from utils.plot import plot_comparisons


# ------------------------------
# Manipulations on ``.txt`` files.
# ------------------------------
def read_list_from_txt(file):
    """
    Read a text file into a list, with each line as an element.
    :param file: the .txt file (with path).
    """
    with open(file) as txt_file:
        contents = txt_file.read().splitlines()

    print(">>> Loaded {} lines from [{}].".format(len(contents), file))
    return contents


# ------------------------------
# Manipulations on ``.json`` files.
# ------------------------------
def load_from_json(file):
    """
    Load a json file into a dictionary.
    :param file: the .json file (with path)
    """
    with open(file, "r") as json_file:
        contents = json.load(json_file)

    return contents


def dump_to_json(dict, file):
    """
    Dump a dictionary to specific json file.
    :param dict: the dictionary to save.
    :param file: the file to which the dictionary is dumped.
    """
    with open(file, "w") as json_file:
        json.dump(dict, json_file)

    print(">>> Dumped to [{}].".format(file))


# ------------------------------
# Manipulations on ``.csv`` files.
# ------------------------------
class CSV_ORIENT(Enum):
    COL = 'col'
    ROW = 'row'


def dump_to_csv(dictionary, file_name, list_as_value=False, append=False):
    """
    dump a dictionary to a csv file.
    :param dictionary: the dictionary to save.
    :param file_name: the name of the csv file, including the path.
    :param list_as_value: each key has a list of values.
    """
    # print(dictionary)

    if append:
        with open(file_name, 'a+', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            if list_as_value:
                writer.writerow(dictionary.keys())
                writer.writerows(zip(*dictionary.available_values()))
            else:
                for row in dictionary.items():
                    writer.writerow(row)
    else:
        with open(file_name, 'w') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            if list_as_value:
                """
                The dictionary is in form of
                {
                key : [values]
                }
                """
                writer.writerow(dictionary.keys())
                writer.writerows(zip(*dictionary.available_values()))
            else:
                """
                The dictionary is in form of
                {
                key : value
                }
                """
                for row in dictionary.items():
                    writer.writerow(row)


def load_from_csv(file_name, orient=CSV_ORIENT.COL, dtype='float'):
    """
    Load csv into a dictionary in the form of
    {
        key : [values]
    }
    :param file_name: csv file name includes the full path
    :param orient:
        orient.col: values of the keys are in a column
        orient.row: values of the keys are in a row
    :param dtype: data type of values
    :return: the dictionary
    """
    if CSV_ORIENT.COL == orient:
        with open(file_name) as file:
            reader = csv.reader(file)
            col_headers = next(reader, None)

            columns = {}
            for header in col_headers:
                columns[header] = []

            for row in reader:
                for header, value in zip(col_headers, row):
                    if 'float' == dtype:
                        value = float(value)
                    columns[header].append(value)

        return columns

    else: # ORIENT.ROW == orient
        # TODO: implement
        rows = {}
        return rows


# ------------------------------
# To remove
# ------------------------------
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
    bs_samples = kwargs.get('bs_samples', None)

    prefix = '{}_AE'.format(prefix)
    attack_info = '{}_{}'.format(attack_method, attack_params)
    model_info = '{}-{}-{}'.format(dataset, architect, transformation)
    file_name = '{}-{}-{}.npy'.format(prefix, model_info, attack_info)
    np.save('{}/{}'.format(PATH.ADVERSARIAL_FILE, file_name), data)

    if MODE.DEBUG:
        if bs_samples is not None:
            title = '{}-{}'.format(model_info, attack_info)
            plot_comparisons(bs_samples[:10], data[:10], title)
        else:
            print('Print AEs only')
            title = '{}-{}'.format(model_info, attack_info)
            plot_comparisons(data[:10], data[10:20], title)

    print('Adversarial examples saved to {}/{}.'.format(PATH.ADVERSARIAL_FILE, file_name))