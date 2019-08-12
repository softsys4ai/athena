"""
Implement file operations such as read/write files here.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import csv
import os

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

