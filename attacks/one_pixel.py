"""
Implement one-pixel attack,
adapted from https://github.com/Hyperparticle/one-pixel-attack-keras/blob/master/1_one-pixel-attack-cifar10.ipynb

@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
import copy
import logging
import numpy as np
from scipy.optimize import differential_evolution
import tensorflow as tf

from models import *
from attacks import attacker

np_dtype = np.dtype('float32')

class OnePixel(object):
    def __init__(self, model, X, Y, kwargs):

        self.model = model
        self.X = X
        self.Y = Y

        self.targeted = kwargs.get('targeted', False)
        self.pixel_count = kwargs.get('pixel_count', 1)
        self.max_iter = kwargs.get('max_iter', 30)
        self.pop_size = kwargs.get('pop_size', 30)
        self.clip_min = kwargs.get('clip_min', 0.)
        self.clip_max = kwargs.get('clip_max', 1.)

        # Rescale value of pixels to [0, 255]
        self.X *= (255.0 / self.X.max())
        self.nb_samples, self.img_rows, self.img_cols, self.nb_channels = self.X.shape

    def perturb_image(self, noises, x):

        # copy for perturbation
        x_adv = copy.copy(x)

        new_pixels = np.split(noises.astype(int), len(noises) // 5)
        for pixel in new_pixels:
            # At each x_adv's (x_pos, y_pos), update its rgb value
            x_pos, y_pos, *rgb = pixel
            x_adv[x_pos, y_pos] = rgb

        print('perturb_image, x_adv.shape:', x_adv.shape)
        return x_adv

    def predict(self, noises, x, target_class, minimize=True):
        x_adv = self.perturb_image(noises, x)
        pred_probs = self.model.predict(x_adv)

        return pred_probs if minimize else 1 - pred_probs

    def attack_success(self, noises, x, target_label, targeted_attack=False):
        x_adv = self.perturb_image(noises, x)
        pred_probs = self.model.predict(x_adv)
        pred_label = np.argmax(pred_probs)

        if ((targeted_attack and pred_label == target_label) or
            (not targeted_attack and pred_label != target_label)):
            return True
        else:
            return False

    def attack(self, x, target_label=None):
        if not self.targeted and target_label is None:
            target_label = np.argmax(self.model.predict(x))

        bounds = [(0, self.img_rows), (0, self.img_cols)]
        for i in range(self.nb_channels):
            bounds.append((0, 255))
        bounds *= self.pixel_count

        popmul = max(1, self.pop_size // len(bounds))

        prediction_func = lambda noises: self.predict(noises, x, target_label, (not self.targeted))
        callback_func = lambda noises, convergence: self.attack_success(noises, x, target_label, self.targeted)

        perturbations = differential_evolution(
            prediction_func, bounds, maxiter=self.max_iter, popsize=popmul,
            recombination=1, atol=1, callback=callback_func, polish=False
        )

        x_adv = self.perturb_image(perturbations.noises, x)

        prior_probs = self.model.predict(x)
        prior_label = np.argmax(prior_probs)
        pred_probs = self.model.predict(x_adv)
        pred_label = np.argmax(pred_probs)
        success = pred_label != prior_label
        confidence_diff = prior_probs[prior_label] - pred_probs[pred_label]

        print('{}: ({}, {}), ({}, {}), {}'.format(success, prior_probs, prior_label,
                                                  pred_probs, pred_label, confidence_diff))

        return [x, x_adv, perturbations.noises, prior_probs, prior_label,
                pred_probs, pred_label, confidence_diff, success]

    def attack_all(self):
        X_adv = []

        log_batch = 10
        log_iter = self.nb_samples / log_batch

        for i, x in enumerate(self.X):
            if i % log_iter == 0:
                print('Perturbing {}-th input...'.format(i))

            if not self.targeted:
                # untargeted attack
                target_labels = [None]
            else:
                # targeted attack
                raise NotImplementedError('targeted attack is not supported yet')

            for target_label in target_labels:
                x_adv = self.attack(x, target_label)[1]
                X_adv.append(x_adv)

        return X_adv

def generate(model, X, Y, attack_params):
    clip_min = attack_params.get('clip_min', 0.)
    clip_max = attack_params.get('clip_max', 1.)

    attacker = OnePixel(model, X, Y, attack_params)
    X_adv = attacker.attack_all()

    # clipping as required
    if clip_min is not None and clip_max is not None:
        X_adv = np.clip(X_adv, clip_min, clip_max)

    return X_adv


