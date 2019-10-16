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
from utils.plot import plot_image

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
        self.True_Labels = np.array(
            [np.where (y == 1)[0][0] for y in self.Y]
        )

    def perturb_image(self, noises, x):
        if noises.ndim < 2:
            noises = np.array([noises])

        tile = [len(noises)] + [1] * (noises.ndim + 1)
        X_adv = np.tile(x, tile)

        noises = noises.astype(int)

        for pixel, x_adv in zip(noises, X_adv):
            new_pixels = np.split(pixel, len(pixel) // len(x_adv.shape))
            for pixel in new_pixels:
                # At each x_adv's (x_pos, y_pos), update its rgb value
                x_pos, y_pos, *rgb = pixel
                x_adv[x_pos, y_pos] = rgb

        # print('perturb_image, x_adv.shape:', X_adv.shape)
        return X_adv

    def predict_class(self, noises, x, target_class, minimize=True):
        x_adv = self.perturb_image(noises, x)
        pred_probs = self.model.predict(x_adv)[:, target_class]

        return pred_probs if minimize else 1 - pred_probs

    def attack_success(self, noises, x, target_label, targeted_attack=False):
        x_adv = self.perturb_image(noises, x)
        pred_probs = self.model.predict(x_adv)[0]
        pred_label = np.argmax(pred_probs)

        if ((targeted_attack and pred_label == target_label) or
            (not targeted_attack and pred_label != target_label)):
            return True
        else:
            return False

    def attack(self, img, target_label=None):

        if not self.targeted and target_label is None:
            target_label = np.argmax(self.model.predict(img)[0])

        bounds = [(0, self.img_rows), (0, self.img_cols)]
        for i in range(self.nb_channels):
            bounds.append((0, 256))
        bounds *= self.pixel_count

        popmul = max(1, self.pop_size // len(bounds))

        prediction_func = lambda X: self.predict_class(X, img, target_label, (not self.targeted))
        callback_func = lambda x, convergence: self.attack_success(x, img, target_label, self.targeted)

        print('Differential Evolution')
        perturbations = differential_evolution(
            prediction_func, bounds, maxiter=self.max_iter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_func, polish=False
        )

        x_adv = self.perturb_image(perturbations.x, img)

        # img = img.reshape(self.img_rows, self.img_cols, self.nb_channels)
        # x_adv = x_adv.reshape(self.img_rows, self.img_cols, self.nb_channels)

        x_adv /= 255.
        if self.clip_min is not None and self.clip_max is not None:
            x_adv = np.clip(x_adv, self.clip_min, self.clip_max)

        prior_probs = self.model.predict(img)[0]
        prior_label = np.argmax(prior_probs)
        pred_probs = self.model.predict(x_adv)[0]
        pred_label = np.argmax(pred_probs)
        success = pred_label != prior_label
        confidence_diff = prior_probs[prior_label] - pred_probs[pred_label]

        print('x_adv shape:', x_adv.shape)
        print('{}: ({}, {}), ({}, {}), {}'.format(success, prior_probs, prior_label,
                                                  pred_probs, pred_label, confidence_diff))

        return [img[0], x_adv[0], perturbations.x, prior_probs, prior_label,
                pred_probs, pred_label, confidence_diff, success]

    def attack_all(self):
        X_adv = []

        log_batch = 10
        log_iter = self.nb_samples / log_batch

        for i in range(self.nb_samples):
            x = self.X[i:i+1]

            if i % log_iter == 0:
                print('Perturbing {}-th input...'.format(i))

            if not self.targeted:
                # untargeted attack
                target_labels = [None]
            else:
                # targeted attack
                raise NotImplementedError('targeted attack is not supported yet')

            for target_label in target_labels:
                x_orig, x_adv, perturbation, _, prior_label, _, pred_label, _, _ = self.attack(x, target_label)

                if i % 2 == 0:
                    plot_image(x_adv, title='{} -> {}'.format(prior_label, pred_label))
                    plot_image((x_orig - x_adv), title='perturbation {}'.format(i))

                X_adv.append(x_adv)

        # X_adv = self.attack(self.X, None)

        return X_adv, self.Y

def generate(model_name, X, Y, attack_params):
    model = load_model(model_name)
    attacker = OnePixel(model, X, Y, attack_params)
    X_adv = attacker.attack_all()

    return X_adv


