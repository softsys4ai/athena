"""

@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""


from utils.config import PATH

from keract import get_activations, display_activations, display_heatmaps, display_gradients_of_trainable_weights

class Visualizer(object):
    def __init__(self, model, x):
        self.model = model
        self.x = x
        self.model_activations = None
        self.layer_name = None
        self.layer_activations = None

    def init_model_activations(self):
        self.model_activations = get_activations(self.model, self.x)

        return self.model_activations

    def init_layer_activations(self, layer_name):
        self.layer_name = layer_name
        self.layer_activations = get_activations(self.model, self.x, layer_name)

        return self.layer_activations

    def display_model_activations(self, cmap=None, save=False):
        if self.model_activations is None:
            raise ValueError('You must first initialize model activations, \n'
                             'by calling init_model_activations function.')

        output_dir = PATH.FIGURES

        if cmap is not None:
            display_activations(self.model_activations, cmap=cmap, save=save, directory=output_dir)
        else:
            display_activations(self.model_activations, save=save, directory=output_dir)

    def display_layer_activations(self, cmap=None, save=False):
        if self.layer_activations is None:
            raise ValueError('You must first initialize layer activations for specific layer, \n'
                             'by calling init_layer_activations function.')

        output_dir = PATH.FIGURES

        if cmap is not None:
            display_activations(self.layer_activations, cmap=cmap, save=save, directory=output_dir)
        else:
            display_activations(self.layer_activations, save=save, directory=output_dir)

    def display_model_heatmaps(self, save=False, fix=True):
        if self.model_activations is None:
            raise ValueError('You must first initialize model activations, \n'
                             'by calling init_model_activations function.')

        output_dir = PATH.FIGURES

        display_heatmaps(self.model_activations, self.x, directory=output_dir, save=save, fix=fix)