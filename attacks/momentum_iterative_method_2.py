import numpy as np
import tensorflow as tf

from cleverhans.model import Model
from cleverhans.compat import reduce_sum, reduce_mean, softmax_cross_entropy_with_logits

class MomentumIterativeMethod(object):
    def __init__(self, model, X, Y, **kwargs):

        self.model = model
        self.X = X
        self.Y = Y

        self.eps = kwargs.get('eps', 0.3)
        self.eps_iter = kwargs.get('eps_iter', 0.06)
        self.nb_iter = kwargs.get('nb_iter', 10)
        self.decay_factor = kwargs.get('decay_factor', 1.0)
        self.y_target = kwargs.get('y_target', None)
        self.clip_min = kwargs.get('clip_min', 0.)
        self.clip_max = kwargs.get('clip_max', 1.)

        self.alpha = self.eps/self.nb_iter
        self.nb_samples, self.img_rows, self.img_cols, self.nb_channels = self.X.shape
        self.True_Labels = np.array(
            [np.where (y == 1)[0][0] for y in self.Y]
        )

    def body(self, i , ax, m):
        logits = self.model.layers[-2].output
        loss = softmax_cross_entropy_with_logits(labels=y, logits=logits)
        if targeted:
            loss = -loss

        # Define gradient of loss w.r.t input
        grad, _ = tf.gradients(loss, ax)

        # Normalize current gradient and add to accumulated gradient
        red_ind = list(range(1, len(grad.get_shape())))
        avoid_zero_div = tf.cast(1e-12, grad.dtype)

