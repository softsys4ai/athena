"""
Script for visualization representations.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import numpy as np
import os

from tasks.creat_models import load_model
from utils.config import PATH
import visualization.visualizact as vis

def keract_visual(model_path, x=None):
    if x is None:
        raise ValueError('x cannot be None.')

    model = load_model(model_path)

    visualizer = vis.Visualizer(model, x)
    visualizer.init_model_activations()
    visualizer.display_model_activations()

def main():
    model_path = 'model-mnist-cnn-clean'
    bs_file = 'test_BS-mnist-clean.npy'
    ae_file = 'test_AE-mnist-cnn-clean-fgsm_eps300.npy'

    bs_path = os.path.join(PATH.ADVERSARIAL_FILE, bs_file)
    ae_path = os.path.join(PATH.ADVERSARIAL_FILE, ae_file)
    bs_images = np.load(bs_path)
    adv_images = np.load(ae_path)

    x = bs_images[0] - adv_images[0]
    x = np.expand_dims(x, axis=0)

    keract_visual(model_path, x)

if __name__ == '__main__':
    main()


