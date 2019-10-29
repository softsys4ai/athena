import tensorflow.python as tf
from models import *
from utils.config import *

class MomentumIterativeMethod(object):
    def __init__(self, model, X, Y, **kwargs):

        self.model = model
        self.X = X
        self.Y = Y

        self.eps = kwargs.get('eps', 0.3)
        self.nb_iter = kwargs.get('nb_iter', 10)
        self.decay_factor = kwargs.get('decay_factor', 1.0)
        self.y_target = kwargs.get('y_target', None)
        self.clip_min = kwargs.get('clip_min', 0.)
        self.clip_max = kwargs.get('clip_max', 1.)

        self.alpha = self.eps/self.nb_iter

    def create_adversarial_pattern(self, input_image, input_label, grad):
        loss_object = tf.keras.losses.CategoricalCrossentropy()

        # Finding gradient of loss w.r.t input
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = self.model(input_image)
            loss = loss_object(input_label, prediction, sample_weight=1.0)
            loss += loss_object(self.model.output, prediction, sample_weight=0.4)
            if (self.y_target is not None):
                loss = -loss

        noise, = tf.gradients(loss, input_image)
        # Avoiding zero div
        noise = tf.cast(1e-12, noise.dtype)
        # Normalizing gradient
        noise = noise / (keras.backend.mean(keras.backend.abs(noise), keepdims=True))
        noise = (self.decay_factor * grad) + noise
        noise = self.alpha * keras.backend.sign(noise)

        return noise

    def attack(self, X, Y, **kwargs):
        X = np.expand_dims(X, axis=0)
        X = tf.cast(X, tf.float32)
        image_probs = self.model.predict(X, steps=1)

        sess = keras.backend.get_session()
        x_adv = X
        grad = tf.zeros(shape=X.shape)
        for i in range(self.nb_iter):
            grad = self.create_adversarial_pattern(x_adv, image_probs, grad)
            x_adv += grad
            x_adv = tf.clip_by_value(x_adv, self.clip_min, self.clip_max)
        return x_adv.eval(session=sess)

    def attack_all(self, X, Y):
        adv_examples = []
        for index, img in enumerate(X):
            print(f"Attacking img {index}")
            x_adv = self.attack(img, Y[index])
            adv_examples.append(x_adv)

        return adv_examples

def generate(model_name, X, Y, **attack_params):
    model = load_model(model_name)
    attacker = MomentumIterativeMethod(model, X, Y, **attack_params)
    adv_examples = attacker.attack_all(X, Y)
    for index, img in enumerate(adv_examples):
        np.save(f'results/example{index}', img)
    return adv_examples, Y

if __name__ == '__main__':
    (X, Y) = data.load_data('mnist')[0]
    X = X[:50]
    Y = Y[:50]
    #model = models.load_model('../data/models/model-mnist-cnn-clean.h5')
    model_name = 'model-mnist-cnn-clean'
    attack_params = {'eps': 0.3, 'nb_iter': 10, 'decay_factor': 1.0, 'y_target': None, 'clip_min': 0.0, 'clip_max': 1.0}
    print("generating")
    generate(model_name, X, Y, **attack_params)




