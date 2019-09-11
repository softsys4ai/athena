import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model

from config import *
import data

from cleverhans.compat import reduce_max
from cleverhans.model import Model
from cleverhans import utils
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import Attack
from scipy.optimize import differential_evolution

import matplotlib.pyplot as plt

class OnePixelAttack(Attack):
    def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
        self.model = model
        self.sess = sess
        self.dtypestr='float32'

    def generate(self, model_input, **kwargs):
        # Parse and save attack-specific parameters
        self.parse_params(**kwargs)
        return one_pixel(model_input,
                         model= self.model,
                         target=self.target,
                         pixel_count=self.pixel_count,
                         maxiter=self.maxiter,
                         population=self.population)

    def parse_params(self, target=None, pixel_count=(1,), maxiter=75, population=400):
        # Attack specific parameters
        self.target = target
        self.pixel_count = pixel_count
        self.maxiter = maxiter
        self.population = population

def perturb_image(xs, img):
    # Packing a flattened perturbation vector in list
    if xs.ndim < 2:
        xs = np.array([xs])

    tile = [len(xs)] + [1] * (xs.ndim + 1)
    imgs = np.tile(img, tile)

    # Flooring the members of xs as int types
    xs = xs.astype(int)

    for x, img in zip(xs, imgs):
        # Splitting x into an array of tuples (perturbation pixels)
        # i.e., [[x,y,r,g,b], ...]
        pixels = np.split(x, len(x) // len(img.shape))

        for pixel in pixels:
            # At each pixel's x,y position, assigning rgb value
            x_pos, y_pos, *rgb = pixel
            img[x_pos][y_pos] = rgb

    return imgs

def predict_classes(xs, img, target_class, model):
    imgs_perturbed = perturb_image(xs, img)
    predictions = model.predict(imgs_perturbed)[:, target_class]
    print(f"Predictions: {predictions}")
    return predictions

def attack_success(x, img, target_class, model, targeted_attack=False):
    # Perturbing image with the given pixel(s) and getting prediction of model
    attack_image = perturb_image(x, img)
    confidence = model.predict(attack_image)[0]
    predicted_class = np.argmax(confidence)

    # If misclassification or
    # targeted classification), return True
    if ((targeted_attack and predicted_class == target_class) or
            (not targeted_attack and predicted_class != target_class)):
        return True

def one_pixel(model_input, model, target=None, pixel_count=1, maxiter=75, population=400):
    targeted_attack = target is not None
    target_class = target if targeted_attack else None
    nb_channels = int(model_input.shape[-1])
    print(nb_channels)
    upper_bound = 1.0

    # Defining the bounds of a flat vector of (x, y, nb_channels)
    dim_x, dim_y = model_input.shape[1:3].as_list()
    bounds = [(0, dim_x), (0, dim_y)]
    for i in range(nb_channels):
        bounds.append((0, upper_bound))
    print(f"Count: {pixel_count}")
    bounds = bounds * pixel_count

    # Population multiplier, in terms of the size of the perturbation vector x
    popmul = max(1, population // len(bounds))

    # Formatting predict and callback functions for differential evolution
    predict_fn = lambda xs: predict_classes(xs, img, target_class, model)
    callback_fn = lambda x, convergence: attack_success(
            x, img, target_class, model, targeted_attack)

    # Calling Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=callback_fn, polish=False)

    attack_image = attack_result.x
    plt.imshow(attack_image)
    plt.show()
    if attack_result.success:
        adv_x = tf.identity(attack_image)
        return adv_x
    return None