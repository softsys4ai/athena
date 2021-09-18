"""
Adapted from the hop_skip_jump in ART.

This module implements the HopSkipJump attack `HopSkipJump`. This is a black-box attack that only requires class
predictions. It is an advanced version of the Boundary attack.

| Paper link: https://arxiv.org/abs/1904.02144
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.utils import compute_success, to_categorical, check_and_transform_label_format

from utils.data import set_channels_first, set_channels_last

logger = logging.getLogger(__name__)


class HopSkipJump(EvasionAttack):
    """
    Implementation of the HopSkipJump attack from Jianbo et al. (2019). This is a powerful black-box attack that
    only requires final class prediction, and is an advanced version of the boundary attack.

    | Paper link: https://arxiv.org/abs/1904.02144
    """

    attack_params = EvasionAttack.attack_params + [
        "targeted",
        "norm",
        "max_iter",
        "max_eval",
        "max_queries",
        "init_eval",
        "init_size",
        "curr_iter",
        "batch_size",
    ]

    def __init__(self, classifier, targeted=False, norm=2,
                 max_iter=50, max_eval=10000, max_queries=1000,
                 init_eval=100, init_size=100):
        """
        Create a HopSkipJump attack instance.

        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        :param targeted: Should the attack target one specific class.
        :type targeted: `bool`
        :param norm: Order of the norm. Possible values: np.inf or 2.
        :type norm: `int`
        :param max_iter: Maximum number of iterations.
        :type max_iter: `int`
        :param max_eval: Maximum number of evaluations for estimating gradient.
        :type max_eval: `int`
        :param max_queries: Maximum number of model queries.
        :type max_queries: `int`
        :param init_eval: Initial number of evaluations for estimating gradient.
        :type init_eval: `int`
        :param init_size: Maximum number of trials for initial generation of adversarial examples.
        :type init_size: `int`
        """
        super(HopSkipJump, self).__init__(classifier=classifier)
        params = {
            "targeted": targeted,
            "norm": norm,
            "max_iter": max_iter,
            "max_eval": max_eval,
            "max_queries": max_queries,
            "init_eval": init_eval,
            "init_size": init_size,
            "curr_iter": 0,
            "batch_size": 1,
        }
        self.set_params(**params)

        self.curr_iter = 0

        # Set binary search threshold
        if norm == 2:
            self.theta = 0.01 / np.sqrt(np.prod(self.classifier.input_shape))
        else:
            self.theta = 0.01 / np.prod(self.classifier.input_shape)

    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :param x_adv_init: Initial array to act as initial adversarial examples. Same shape as `x`.
        :type x_adv_init: `np.ndarray`
        :param resume: Allow users to continue their previous attack.
        :type resume: `bool`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        y = check_and_transform_label_format(y, self.classifier.nb_classes())

        # Check whether users need a stateful attack
        resume = kwargs.get("resume")

        if resume is not None and resume:
            start = self.curr_iter
        else:
            start = 0

        # Get clip_min and clip_max from the classifier or infer them from data
        if hasattr(self.classifier, "clip_values") and self.classifier.clip_values is not None:
            clip_min, clip_max = self.classifier.clip_values
        else:
            clip_min, clip_max = np.min(x), np.max(x)

        # Prediction from the original images
        preds = np.argmax(self.classifier.predict(x, batch_size=self.batch_size), axis=1)

        # Prediction from the initial adversarial examples if not None
        x_adv_init = kwargs.get("x_adv_init")

        if x_adv_init is not None:
            init_preds = np.argmax(self.classifier.predict(x_adv_init, batch_size=self.batch_size), axis=1)
        else:
            init_preds = [None] * len(x)
            x_adv_init = [None] * len(x)

        # Assert that, if attack is targeted, y is provided
        if self.targeted and y is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # Some initial setups
        x_adv = x.astype(ART_NUMPY_DTYPE)
        if y is not None:
            y = np.argmax(y, axis=1)

        # Generate the adversarial samples
        for ind, val in enumerate(x_adv):
            self.curr_iter = start

            # print('[DEBUG][generate()] {}: {}'.format(ind, val.shape))

            if self.targeted:
                x_adv[ind] = self._perturb(
                    x=val,
                    y=y[ind],
                    y_p=preds[ind],
                    init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
            else:
                x_adv[ind] = self._perturb(
                    x=val,
                    y=-1,
                    y_p=preds[ind],
                    init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

        if y is not None:
            y = to_categorical(y, self.classifier.nb_classes())

        logger.info(
            "Success rate of HopSkipJump attack: %.2f%%",
            100 * compute_success(self.classifier, x, y, x_adv, self.targeted, batch_size=self.batch_size),
        )

        return x_adv

    def _perturb(self, x, y, y_p, init_pred, adv_init, clip_min, clip_max):
        """
        Internal attack function for one example.

        :param x: An array with one original input to be attacked.
        :type x: `np.ndarray`
        :param y: If `self.targeted` is true, then `y` represents the target label.
        :type y: `int`
        :param y_p: The predicted label of x.
        :type y_p: `int`
        :param init_pred: The predicted label of the initial image.
        :type init_pred: `int`
        :param adv_init: Initial array to act as an initial adversarial example.
        :type adv_init: `np.ndarray`
        :param clip_min: Minimum value of an example.
        :type clip_min: `float`
        :param clip_max: Maximum value of an example.
        :type clip_max: `float`
        :return: an adversarial example.
        :rtype: `np.ndarray`
        """
        # reshape if necessary
        if self.classifier.channel_index == 1:
            x = set_channels_first(x)
            # print("[DEBUG][hsja._perturb]: set channels first")
        # First, create an initial adversarial sample
        initial_sample = self._init_sample(x, y, y_p, init_pred, adv_init, clip_min, clip_max)

        # If an initial adversarial example is not found, then return the original image
        if initial_sample is None:
            return x

        # If an initial adversarial example found, then go with hopskipjump attack
        x_adv = self._attack(initial_sample[0], x, initial_sample[1], clip_min, clip_max)

        return set_channels_last(x_adv)

    def _init_sample(self, x, y, y_p, init_pred, adv_init, clip_min, clip_max):
        """
        Find initial adversarial example for the attack.

        :param x: An array with 1 original input to be attacked.
        :type x: `np.ndarray`
        :param y: If `self.targeted` is true, then `y` represents the target label.
        :type y: `int`
        :param y_p: The predicted label of x.
        :type y_p: `int`
        :param init_pred: The predicted label of the initial image.
        :type init_pred: `int`
        :param adv_init: Initial array to act as an initial adversarial example.
        :type adv_init: `np.ndarray`
        :param clip_min: Minimum value of an example.
        :type clip_min: `float`
        :param clip_max: Maximum value of an example.
        :type clip_max: `float`
        :return: an adversarial example.
        :rtype: `np.ndarray`
        """
        nprd = np.random.RandomState()
        initial_sample = None

        if self.targeted:
            # Attack satisfied
            if y == y_p:
                return None

            # Attack unsatisfied yet and the initial image satisfied
            if adv_init is not None and init_pred == y:
                return adv_init.astype(ART_NUMPY_DTYPE), init_pred

            # Attack unsatisfied yet and the initial image unsatisfied
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                random_class = np.argmax(
                    self.classifier.predict(np.array([random_img]), batch_size=self.batch_size), axis=1
                )[0]

                if random_class == y:
                    # Binary search to reduce the l2 distance to the original image
                    random_img = self._binary_search(
                        current_sample=random_img,
                        original_sample=x,
                        target=y,
                        norm=2,
                        clip_min=clip_min,
                        clip_max=clip_max,
                        threshold=0.001,
                    )
                    initial_sample = random_img, random_class

                    logger.info("Found initial adversarial image for targeted attack.")
                    break
            else:
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        else:
            # The initial image satisfied
            if adv_init is not None and init_pred != y_p:
                return adv_init.astype(ART_NUMPY_DTYPE), y_p

            # The initial image unsatisfied
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                random_class = np.argmax(
                    self.classifier.predict(np.array([random_img]), batch_size=self.batch_size), axis=1
                )[0]

                if random_class != y_p:
                    # Binary search to reduce the l2 distance to the original image
                    random_img = self._binary_search(
                        current_sample=random_img,
                        original_sample=x,
                        target=y_p,
                        norm=2,
                        clip_min=clip_min,
                        clip_max=clip_max,
                        threshold=0.001,
                    )
                    initial_sample = random_img, y_p

                    logger.info("Found initial adversarial image for untargeted attack.")
                    break
            else:
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        return initial_sample

    def _attack(self, initial_sample, original_sample, target, clip_min, clip_max):
        """
        Main function for the boundary attack.

        :param initial_sample: An initial adversarial example.
        :type initial_sample: `np.ndarray`
        :param original_sample: The original input.
        :type original_sample: `np.ndarray`
        :param target: The target label.
        :type target: `int`
        :param clip_min: Minimum value of an example.
        :type clip_min: `float`
        :param clip_max: Maximum value of an example.
        :type clip_max: `float`
        :return: an adversarial example.
        :rtype: `np.ndarray`
        """
        # Set current perturbed image to the initial image
        current_sample = initial_sample

        # Main loop to wander around the boundary
        iter = 0
        queries = 0
        self.classifier.reset_model_queries()
        while iter < self.max_iter and queries < self.max_queries:
        # for _ in range(self.max_iter):
            # First compute delta
            delta = self._compute_delta(
                current_sample=current_sample, original_sample=original_sample, clip_min=clip_min, clip_max=clip_max
            )

            # Then run binary search
            current_sample = self._binary_search(
                current_sample=current_sample,
                original_sample=original_sample,
                norm=self.norm,
                target=target,
                clip_min=clip_min,
                clip_max=clip_max,
            )

            # Next compute the number of evaluations and compute the update
            num_eval = min(int(self.init_eval * np.sqrt(self.curr_iter + 1)), self.max_eval)
            update = self._compute_update(
                current_sample=current_sample,
                num_eval=num_eval,
                delta=delta,
                target=target,
                clip_min=clip_min,
                clip_max=clip_max,
            )

            # Finally run step size search by first computing epsilon
            if self.norm == 2:
                dist = np.linalg.norm(original_sample - current_sample)
            else:
                dist = np.max(abs(original_sample - current_sample))

            epsilon = 2.0 * dist / np.sqrt(self.curr_iter + 1)
            success = False

            while not success:
                epsilon /= 2.0
                potential_sample = current_sample + epsilon * update
                success = self._adversarial_satisfactory(
                    samples=potential_sample[None], target=target, clip_min=clip_min, clip_max=clip_max
                )

            # Update current sample
            current_sample = np.clip(potential_sample, clip_min, clip_max)

            # Update current iteration
            self.curr_iter += 1

            # update the number of iterations
            iter += 1
            # update the number of model queries
            queries += self.classifier.num_queries

        return current_sample

    def _binary_search(self, current_sample, original_sample, target, norm, clip_min, clip_max, threshold=None):
        """
        Binary search to approach the boundary.

        :param current_sample: Current adversarial example.
        :type current_sample: `np.ndarray`
        :param original_sample: The original input.
        :type original_sample: `np.ndarray`
        :param target: The target label.
        :type target: `int`
        :param norm: Order of the norm. Possible values: np.inf or 2.
        :type norm: `int`
        :param clip_min: Minimum value of an example.
        :type clip_min: `float`
        :param clip_max: Maximum value of an example.
        :type clip_max: `float`
        :param threshold: The upper threshold in binary search.
        :type threshold: `float`
        :return: an adversarial example.
        :rtype: `np.ndarray`
        """
        # First set upper and lower bounds as well as the threshold for the binary search
        if norm == 2:
            (upper_bound, lower_bound) = (1, 0)

            if threshold is None:
                threshold = self.theta

        else:
            (upper_bound, lower_bound) = (np.max(abs(original_sample - current_sample)), 0)

            if threshold is None:
                threshold = np.minimum(upper_bound * self.theta, self.theta)

        # Then start the binary search
        while (upper_bound - lower_bound) > threshold:
            # Interpolation point
            alpha = (upper_bound + lower_bound) / 2.0
            interpolated_sample = self._interpolate(
                current_sample=current_sample, original_sample=original_sample, alpha=alpha, norm=norm
            )

            # Update upper_bound and lower_bound
            satisfied = self._adversarial_satisfactory(
                samples=interpolated_sample[None], target=target, clip_min=clip_min, clip_max=clip_max
            )[0]
            lower_bound = np.where(satisfied == 0, alpha, lower_bound)
            upper_bound = np.where(satisfied == 1, alpha, upper_bound)

        result = self._interpolate(
            current_sample=current_sample, original_sample=original_sample, alpha=upper_bound, norm=norm
        )

        return result

    def _compute_delta(self, current_sample, original_sample, clip_min, clip_max):
        """
        Compute the delta parameter.

        :param current_sample: Current adversarial example.
        :type current_sample: `np.ndarray`
        :param original_sample: The original input.
        :type original_sample: `np.ndarray`
        :param clip_min: Minimum value of an example.
        :type clip_min: `float`
        :param clip_max: Maximum value of an example.
        :type clip_max: `float`
        :return: Delta value.
        :rtype: `float`
        """
        # Note: This is a bit different from the original paper, instead we keep those that are
        # implemented in the original source code of the authors
        if self.curr_iter == 0:
            return 0.1 * (clip_max - clip_min)

        if self.norm == 2:
            dist = np.linalg.norm(original_sample - current_sample)
            delta = np.sqrt(np.prod(self.classifier.input_shape)) * self.theta * dist
        else:
            dist = np.max(abs(original_sample - current_sample))
            delta = np.prod(self.classifier.input_shape) * self.theta * dist

        return delta

    def _compute_update(self, current_sample, num_eval, delta, target, clip_min, clip_max):
        """
        Compute the update in Eq.(14).

        :param current_sample: Current adversarial example.
        :type current_sample: `np.ndarray`
        :param num_eval: The number of evaluations for estimating gradient.
        :type num_eval: `int`
        :param delta: The size of random perturbation.
        :type delta: `float`
        :param target: The target label.
        :type target: `int`
        :param clip_min: Minimum value of an example.
        :type clip_min: `float`
        :param clip_max: Maximum value of an example.
        :type clip_max: `float`
        :return: an updated perturbation.
        :rtype: `np.ndarray`
        """
        # Generate random noise
        rnd_noise_shape = [num_eval] + list(self.classifier.input_shape)
        if self.norm == 2:
            rnd_noise = np.random.randn(*rnd_noise_shape).astype(ART_NUMPY_DTYPE)
        else:
            rnd_noise = np.random.uniform(low=-1, high=1, size=rnd_noise_shape).astype(ART_NUMPY_DTYPE)

        # Normalize random noise to fit into the range of input data
        rnd_noise = rnd_noise / np.sqrt(
            np.sum(rnd_noise ** 2, axis=tuple(range(len(rnd_noise_shape)))[1:], keepdims=True)
        )
        eval_samples = np.clip(current_sample + delta * rnd_noise, clip_min, clip_max)
        rnd_noise = (eval_samples - current_sample) / delta

        # Compute gradient: This is a bit different from the original paper, instead we keep those that are
        # implemented in the original source code of the authors
        satisfied = self._adversarial_satisfactory(
            samples=eval_samples, target=target, clip_min=clip_min, clip_max=clip_max
        )
        f_val = 2 * satisfied.reshape([num_eval] + [1] * len(self.classifier.input_shape)) - 1.0
        f_val = f_val.astype(ART_NUMPY_DTYPE)

        if np.mean(f_val) == 1.0:
            grad = np.mean(rnd_noise, axis=0)
        elif np.mean(f_val) == -1.0:
            grad = -np.mean(rnd_noise, axis=0)
        else:
            f_val -= np.mean(f_val)
            grad = np.mean(f_val * rnd_noise, axis=0)

        # Compute update
        if self.norm == 2:
            result = grad / np.linalg.norm(grad)
        else:
            result = np.sign(grad)

        return result

    def _adversarial_satisfactory(self, samples, target, clip_min, clip_max):
        """
        Check whether an image is adversarial.

        :param samples: A batch of examples.
        :type samples: `np.ndarray`
        :param target: The target label.
        :type target: `int`
        :param clip_min: Minimum value of an example.
        :type clip_min: `float`
        :param clip_max: Maximum value of an example.
        :type clip_max: `float`
        :return: An array of 0/1.
        :rtype: `np.ndarray`
        """
        samples = np.clip(samples, clip_min, clip_max)
        preds = np.argmax(self.classifier.predict(samples, batch_size=self.batch_size), axis=1)

        if self.targeted:
            result = preds == target
        else:
            result = preds != target

        return result

    @staticmethod
    def _interpolate(current_sample, original_sample, alpha, norm):
        """
        Interpolate a new sample based on the original and the current samples.

        :param current_sample: Current adversarial example.
        :type current_sample: `np.ndarray`
        :param original_sample: The original input.
        :type original_sample: `np.ndarray`
        :param alpha: The coefficient of interpolation.
        :type alpha: `float`
        :param norm: Order of the norm. Possible values: np.inf or 2.
        :type norm: `int`
        :return: an adversarial example.
        :rtype: `np.ndarray`
        """
        if norm == 2:
            result = (1 - alpha) * original_sample + alpha * current_sample
        else:
            result = np.clip(current_sample, original_sample - alpha, original_sample + alpha)

        return result

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param targeted: Should the attack target one specific class.
        :type targeted: `bool`
        :param norm: Order of the norm. Possible values: np.inf or 2.
        :type norm: `int`
        :param max_iter: Maximum number of iterations.
        :type max_iter: `int`
        :param max_eval: Maximum number of evaluations for estimating gradient.
        :type max_eval: `int`
        :param init_eval: Initial number of evaluations for estimating gradient.
        :type init_eval: `int`
        :param init_size: Maximum number of trials for initial generation of adversarial examples.
        :type init_size: `int`
        """
        # Save attack-specific parameters
        super(HopSkipJump, self).set_params(**kwargs)

        # Check if order of the norm is acceptable given current implementation
        if self.norm not in [np.inf, int(2)]:
            raise ValueError("Norm order must be either `np.inf` or 2.")

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")

        if not isinstance(self.max_eval, (int, np.int)) or self.max_eval <= 0:
            raise ValueError("The maximum number of evaluations must be a positive integer.")

        if not isinstance(self.init_eval, (int, np.int)) or self.init_eval <= 0:
            raise ValueError("The initial number of evaluations must be a positive integer.")

        if self.init_eval > self.max_eval:
            raise ValueError("The maximum number of evaluations must be larger than the initial number of evaluations.")

        if not isinstance(self.init_size, (int, np.int)) or self.init_size <= 0:
            raise ValueError("The number of initial trials must be a positive integer.")

        return True
