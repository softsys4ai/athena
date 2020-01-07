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

from models.models import *
from attacks import attacker
from utils.plot import plot_image

np_dtype = np.dtype('float32')

class OnePixel(object):
    def __init__(self, model, X, Y, kwargs):

        self.model = model
        self.X = X
        self.Y = Y

        self.targeted = kwargs.get('targeted', False)
        self.pixel_counts = kwargs.get('pixel_counts', 1)
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

        if MODE.DEBUG:
            self.summary()


    def summary(self):
        print('--------------------------------')
        print('        Summary')
        print('--------------------------------')
        print('target model name:', self.model.name)
        print('targeted attack:', self.targeted)
        print('pixel counts:', self.pixel_counts)
        print('max iteration:', self.max_iter)
        print('pop size:', self.pop_size)


    def perturb_image(self, xs, img):
        if xs.ndim < 2:
            xs = np.array([xs])

        tile = [len(xs)] + [1] * (xs.ndim + 1)
        imgs = np.tile(img, tile)

        xs = xs.astype(int)

        for x, img in zip(xs, imgs):
            pixels = np.split(x, len(x) // len(img.shape))
            # pixels = np.split(x, len(x) // 5)

            for pixel in pixels:
                # At each x_adv's (x_pos, y_pos), update its rgb value
                x_pos, y_pos, *rgb = pixel
                img[x_pos, y_pos] = rgb

        return imgs

    def predict_class(self, xs, img, target_class, minimize=True):
        imgs_perturbed = self.perturb_image(xs, img)
        x_perturbed = imgs_perturbed / 255.
        pred_probs = self.model.predict(x_perturbed)[:, target_class]

        return pred_probs if minimize else 1 - pred_probs

    def attack_success(self, x, img, target_label, targeted_attack=False):
        img_perturbed = self.perturb_image(x, img)
        x_perturbed = img_perturbed / 255.
        pred_probs = self.model.predict(x_perturbed)[0]
        pred_label = np.argmax(pred_probs)

        if targeted_attack:
            if MODE.DEBUG:
                print('for targeted attack, we expect pred_label == target_label ({})'.format(pred_label == target_label))
            return (pred_label == target_label)
        else: # untargeted attack
            if MODE.DEBUG:
                print('for untargeted attack, we expect pred_label({}) != target_label({}) ({})'.format(pred_label,
                                                                                             target_label,
                                                                                             pred_label != target_label))
            return (pred_label != target_label)

    def attack(self, img, true_label, target_label=None):

        if not self.targeted and target_label is None:
            target_label = np.argmax(self.model.predict(img)[0])

        bounds = [(0, self.img_rows), (0, self.img_cols)]
        for i in range(self.nb_channels):
            bounds.append((0, 256))
        bounds *= self.pixel_counts

        if MODE.DEBUG:
            print('bounds: len/shape -- {}/{}\n{}'.format(len(bounds), np.asarray(bounds).shape, bounds))

        popmul = max(1, self.pop_size // len(bounds))

        prediction_func = lambda xs: self.predict_class(xs, img, target_label, (not self.targeted))
        callback_func = lambda x, convergence: self.attack_success(x, img, target_label, self.targeted)

        if MODE.DEBUG:
            print('Differential Evolution')
        perturbations = differential_evolution(
            prediction_func, bounds, maxiter=self.max_iter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_func, polish=False
        )

        if MODE.DEBUG:
            print('perturbations:', perturbations)
        x_adv = self.perturb_image(perturbations.x, img)

        # img = img.reshape(self.img_rows, self.img_cols, self.nb_channels)
        # x_adv = x_adv.reshape(self.img_rows, self.img_cols, self.nb_channels)
        # re-scale as demanded
        x_adv /= 255.
        if self.clip_min is not None and self.clip_max is not None:
            x_adv = np.clip(x_adv, self.clip_min, self.clip_max)

        prior_probs = self.model.predict(img)[0]
        prior_label = np.argmax(prior_probs)
        pred_probs = self.model.predict(x_adv)[0]
        pred_label = np.argmax(pred_probs)
        success = (pred_label != true_label)
        confidence_diff = prior_probs[true_label] - pred_probs[true_label]

        # return [img[0], x_adv[0], perturbations.x, prior_probs, prior_label,
        #         pred_probs, pred_label, confidence_diff, success]
        return [img[0], x_adv[0], perturbations, prior_label, pred_label, success]

    def attack_all(self):
        X_adv = []

        log_batch = 10
        log_iter = self.nb_samples / log_batch

        for i, img in enumerate(self.X):
        # for i in range(self.nb_samples):
        #     x = self.X[i:i+1]
            img = np.expand_dims(img, axis=0)
            y_true = self.True_Labels[i]

            if i % log_iter == 0:
                print('Perturbing {}-th input...'.format(i))

            if not self.targeted:
                # untargeted attack
                target_labels = [None]
            else:
                # targeted attack
                raise NotImplementedError('targeted attack is not supported yet')

            for target_label in target_labels:
                # print('y_true/target: {}/{}'.format(y_true, target_label))
                # x_orig, x_adv, perturbation, _, prior_label, _, pred_label, _, _ = self.attack(img, true_label=y_true, target_label=target_label)
                x_orig, x_adv, perturbations, prior_label, pred_label, _ = self.attack(img, true_label=y_true, target_label=target_label)

                print('{}-th >>> true/legitimate/adv: {}/{}/{}'.format(i, y_true, prior_label, pred_label))

                X_adv.append(x_adv)

        return X_adv, self.Y

def generate(model, X, Y, attack_params):
    # model = load_model(model_name)
    attacker = OnePixel(model, X, Y, attack_params)
    X_adv = attacker.attack_all()

    return X_adv


