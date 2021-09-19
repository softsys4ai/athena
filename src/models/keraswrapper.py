"""
Implement weak defense model for Athena on top of IBM Trusted-AI ART 1.2.0.
It wraps a keras model to a weak defense in Athena ensemble.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import six
from art.classifiers.classifier import Classifier, ClassifierNeuralNetwork, ClassifierGradients

from models.image_processor import transform

logger = logging.getLogger(__name__)


class WeakDefense(ClassifierNeuralNetwork, ClassifierGradients, Classifier):
    def __init__(self, model, trans_configs, use_logits=False, channel_index=3,
                 clip_values=(0., 1.), input_layer=0, output_layer=0, ):
        super(WeakDefense, self).__init__(clip_values=clip_values,
                                          preprocessing_defences=None,
                                          postprocessing_defences=None,
                                          preprocessing=(0, 1),
                                          channel_index=channel_index, )

        self.classifier = model
        self._trans_configs = trans_configs
        self._channel_index = channel_index
        self._input_layer = input_layer
        self._output_layer = output_layer
        self._num_queries = 0

        if "<class 'tensorflow" in str(type(model)):
            self.is_tensorflow = True
        elif "<class 'keras" in str(type(model)):
            self.is_tensorflow = False
        else:
            raise TypeError("Type of model not recognized:" + type(model))

        self._initialize_params(model, use_logits, input_layer, output_layer)


    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        # Check shape of `x` because of custom function for `_loss_gradients`
        if self._input_shape != x.shape[1:]:
            raise ValueError(
                "Error when checking x: expected x to have shape {} but got array with shape {}".format(
                    self._input_shape, x.shape[1:]
                )
            )

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        # Adjust the shape of y for loss functions that do not take labels in one-hot encoding
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        # Compute gradients
        gradients = self._loss_gradients([x_preprocessed, y_preprocessed])[0]
        gradients = self._apply_preprocessing_gradient(x, gradients)
        assert gradients.shape == x_preprocessed.shape

        return gradients

    def class_gradient(self, x, label=None, **kwargs):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :type label: `int` or `list`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        # Check value of label for computing gradients
        if not (
            label is None
            or (isinstance(label, (int, np.integer)) and label in range(self.nb_classes()))
            or (
                isinstance(label, np.ndarray)
                and len(label.shape) == 1
                and (label < self.nb_classes()).all()
                and label.shape[0] == x.shape[0]
            )
        ):
            raise ValueError("Label %s is out of range." % str(label))

        # Check shape of `x` because of custom function for `_loss_gradients`
        if self._input_shape != x.shape[1:]:
            raise ValueError(
                "Error when checking x: expected x to have shape {} but got array with shape {}".format(
                    self._input_shape, x.shape[1:]
                )
            )

        self._init_class_gradients(label=label)

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        if label is None:
            # Compute the gradients w.r.t. all classes
            gradients = np.swapaxes(np.array(self._class_gradients([x_preprocessed])), 0, 1)

        elif isinstance(label, (int, np.integer)):
            # Compute the gradients only w.r.t. the provided label
            gradients = np.swapaxes(np.array(self._class_gradients_idx[label]([x_preprocessed])), 0, 1)
            assert gradients.shape == (x_preprocessed.shape[0], 1) + self.input_shape

        else:
            # For each sample, compute the gradients w.r.t. the indicated target class (possibly distinct)
            unique_label = list(np.unique(label))
            gradients = np.array([self._class_gradients_idx[l]([x_preprocessed]) for l in unique_label])
            gradients = np.swapaxes(np.squeeze(gradients, axis=1), 0, 1)
            lst = [unique_label.index(i) for i in label]
            gradients = np.expand_dims(gradients[np.arange(len(gradients)), lst], axis=1)

        gradients = self._apply_preprocessing_gradient(x, gradients)

        return gradients

    def predict(self, x, batch_size=128, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        from art.config import ART_NUMPY_DTYPE

        # Apply transformation
        x_preprocessed = transform(x, self._trans_configs)

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x_preprocessed, y=None, fit=False)

        # Run predictions with batching
        predictions = np.zeros((x_preprocessed.shape[0], self.nb_classes()), dtype=ART_NUMPY_DTYPE)
        for batch_index in range(int(np.ceil(x_preprocessed.shape[0] / float(batch_size)))):
            begin, end = batch_index * batch_size, min((batch_index + 1) * batch_size, x_preprocessed.shape[0])
            predictions[begin:end] = self.classifier.predict([x_preprocessed[begin:end]])

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=predictions, fit=False)

        # increase number of queries
        self._num_queries += x.shape[0]

        return predictions

    def fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit_generator` function in Keras and will be passed to this function as such. Including the number of
               epochs or the number of steps per epoch as part of this argument will result in as error.
        :type kwargs: `dict`
        :return: `None`
        """
        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        # Adjust the shape of y for loss functions that do not take labels in one-hot encoding
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        gen = generator_fit(x_preprocessed, y_preprocessed, batch_size)
        steps_per_epoch = max(int(x_preprocessed.shape[0] / batch_size), 1)
        self.classifier.fit_generator(gen, steps_per_epoch=steps_per_epoch, epochs=nb_epochs, **kwargs)

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        """
        Fit the classifier using the generator that yields batches as specified.

        :param generator: Batch generator providing `(x, y)` for each epoch. If the generator can be used for native
                          training in Keras, it will.
        :type generator: :class:`.DataGenerator`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit_generator` function in Keras and will be passed to this function as such. Including the number of
               epochs as part of this argument will result in as error.
        :type kwargs: `dict`
        :return: `None`
        """
        from art.data_generators import KerasDataGenerator

        # Try to use the generator as a Keras native generator, otherwise use it through the `DataGenerator` interface
        if isinstance(generator, KerasDataGenerator) and \
                (self.preprocessing_defences is None or self.preprocessing_defences == []) and \
                self.preprocessing == (0, 1):
            try:
                self.classifier.fit_generator(generator.iterator, epochs=nb_epochs, **kwargs)
            except ValueError:
                logger.info('Unable to use data generator as Keras generator. Now treating as framework-independent.')
                super(WeakDefense, self).fit_generator(generator, nb_epochs=nb_epochs, **kwargs)
        else:
            super(WeakDefense, self).fit_generator(generator, nb_epochs=nb_epochs, **kwargs)

    def reset_model_queries(self):
        """
        Set the number of model queries to 0.
        """
        self._num_queries = 0

    @property
    def num_queries(self):
        """
        Return the number of model queries.
        """
        return self._num_queries

    @property
    def layer_names(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`

        .. warning:: `layer_names` tries to infer the internal structure of the model.
                     This feature comes with no guarantees on the correctness of the result.
                     The intended order of the layers tries to match their order in the model, but this is not
                     guaranteed either.
        """
        return self._layer_names

    def get_activations(self, x, layer, batch_size):
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :type x: `np.ndarray`
        :param layer: Layer for computing the activations
        :type layer: `int` or `str`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        :rtype: `np.ndarray`
        """
        # pylint: disable=E0401
        if self.is_tensorflow:
            import tensorflow.keras.backend as k
        else:
            import keras.backend as k
        from art.config import ART_NUMPY_DTYPE

        if isinstance(layer, six.string_types):
            if layer not in self._layer_names:
                raise ValueError("Layer name %s is not part of the graph." % layer)
            layer_name = layer
        elif isinstance(layer, int):
            if layer < 0 or layer >= len(self._layer_names):
                raise ValueError(
                    "Layer index %d is outside of range (0 to %d included)." % (layer, len(self._layer_names) - 1)
                )
            layer_name = self._layer_names[layer]
        else:
            raise TypeError("Layer must be of type `str` or `int`.")

        layer_output = self.classifier.get_layer(layer_name).output
        output_func = k.function([self._input], [layer_output])

        if x.shape == self.input_shape:
            x_expanded = np.expand_dims(x, 0)
        else:
            x_expanded = x

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x=x_expanded, y=None, fit=False)

        # Determine shape of expected output and prepare array
        output_shape = output_func([x_preprocessed[0][None, ...]])[0].shape
        activations = np.zeros((x_preprocessed.shape[0],) + output_shape[1:], dtype=ART_NUMPY_DTYPE)

        # Get activations with batching
        for batch_index in range(int(np.ceil(x_preprocessed.shape[0] / float(batch_size)))):
            begin, end = batch_index * batch_size, min((batch_index + 1) * batch_size, x_preprocessed.shape[0])
            activations[begin:end] = output_func([x_preprocessed[begin:end]])[0]

        return activations

    def set_learning_phase(self, train):
        """
        Set the learning phase for the backend framework.

        :param train: True to set the learning phase to training, False to set it to prediction.
        :type train: `bool`
        """
        # pylint: disable=E0401
        if self.is_tensorflow:
            import tensorflow.keras.backend as k
        else:
            import keras.backend as k

        if isinstance(train, bool):
            self._learning_phase = train
            k.set_learning_phase(int(train))

    def nb_classes(self):
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        :rtype: `int`
        """
        return self._nb_classes

    def save(self, filename, path=None):
        """
        Save a model to file in the format specific to the backend framework. For Keras, .h5 format is used.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `ART_DATA_PATH`.
        :type path: `str`
        :return: None
        """
        import os

        if path is None:
            from art.config import ART_DATA_PATH

            full_path = os.path.join(ART_DATA_PATH, filename)
        else:
            full_path = os.path.join(path, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.classifier.save(str(full_path))
        logger.info("Model saved in path: %s.", full_path)

    def _init_class_gradients(self, label=None):
        # pylint: disable=E0401
        if self.is_tensorflow:
            import tensorflow.keras.backend as k
        else:
            import keras.backend as k

        if len(self._output.shape) == 2:
            nb_outputs = self._output.shape[1]
        else:
            raise ValueError("Unexpected output shape for classification in Keras model.")

        if label is None:
            logger.debug("Computing class gradients for all %i classes.", self.nb_classes())
            if not hasattr(self, "_class_gradients"):
                class_gradients = [k.gradients(self._predictions_op[:, i], self._input)[0] for i in range(nb_outputs)]
                self._class_gradients = k.function([self._input], class_gradients)

        else:
            if isinstance(label, int):
                unique_labels = [label]
            else:
                unique_labels = np.unique(label)
            logger.debug("Computing class gradients for classes %s.", str(unique_labels))

            if not hasattr(self, "_class_gradients_idx"):
                self._class_gradients_idx = [None for _ in range(nb_outputs)]

            for current_label in unique_labels:
                if self._class_gradients_idx[current_label] is None:
                    class_gradients = [k.gradients(self._predictions_op[:, current_label], self._input)[0]]
                    self._class_gradients_idx[current_label] = k.function([self._input], class_gradients)

    def _initialize_params(self, model, use_logits, input_layer, output_layer, synthesis=True, num_synthesis=10):
        """
        Initialize most parameters of the classifier. This is a convenience function called by `__init__` and
        `__setstate__` to avoid code duplication.

        :param model: Keras model
        :type model: `keras.models.Model`
        :param use_logits: True if the output of the model are logits.
        :type use_logits: `bool`
        :param input_layer: Which layer to consider as the Input when the model has multiple input layers.
        :type input_layer: `int`
        :param output_layer: Which layer to consider as the Output when the model has multiple output layers.
        :type output_layer: `int`
        """
        # pylint: disable=E0401
        if self.is_tensorflow:
            import tensorflow as tf

            if tf.executing_eagerly():
                raise ValueError("TensorFlow is executing eagerly. Please disable eager execution.")
            import tensorflow.keras as keras
            import tensorflow.keras.backend as k
        else:
            import keras
            import keras.backend as k

        if hasattr(model, "inputs"):
            self._input_layer = input_layer
            self._input = model.inputs[input_layer]
        else:
            self._input = model.input
            self._input_layer = 0

        if hasattr(model, "outputs"):
            self._output = model.outputs[output_layer]
            self._output_layer = output_layer
        else:
            self._output = model.output
            self._output_layer = 0

        _, self._nb_classes = k.int_shape(self._output)
        self._input_shape = k.int_shape(self._input)[1:]
        logger.debug(
            "Inferred %i classes and %s as input shape for Keras classifier.", self.nb_classes(), str(self.input_shape)
        )

        self._use_logits = use_logits

        # Get loss function
        if not hasattr(self.classifier, "loss"):
            logger.warning("Keras model has no loss set. Classifier tries to use `k.sparse_categorical_crossentropy`.")
            loss_function = k.sparse_categorical_crossentropy
        else:

            if isinstance(self.classifier.loss, six.string_types):
                loss_function = getattr(k, self.classifier.loss)

            elif "__name__" in dir(self.classifier.loss) and self.classifier.loss.__name__ in [
                "categorical_hinge",
                "categorical_crossentropy",
                "sparse_categorical_crossentropy",
                "binary_crossentropy",
                "kullback_leibler_divergence",
            ]:
                if self.classifier.loss.__name__ in ["categorical_hinge", "kullback_leibler_divergence"]:
                    loss_function = getattr(keras.losses, self.classifier.loss.__name__)
                else:
                    loss_function = getattr(keras.backend, self.classifier.loss.__name__)

            elif isinstance(
                self.classifier.loss,
                (
                    keras.losses.CategoricalHinge,
                    keras.losses.CategoricalCrossentropy,
                    keras.losses.SparseCategoricalCrossentropy,
                    keras.losses.BinaryCrossentropy,
                    keras.losses.KLDivergence,
                ),
            ):
                loss_function = self.classifier.loss
            else:
                loss_function = getattr(k, self.classifier.loss.__name__)

        # Check if loss function is an instance of loss function generator, the try is required because some of the
        # modules are not available in older Keras versions
        try:
            flag_is_instance = isinstance(
                loss_function,
                (
                    keras.losses.CategoricalHinge,
                    keras.losses.CategoricalCrossentropy,
                    keras.losses.BinaryCrossentropy,
                    keras.losses.KLDivergence,
                ),
            )
        except AttributeError:
            flag_is_instance = False

        # Check if the labels have to be reduced to index labels and create placeholder for labels
        if (
            "__name__" in dir(loss_function)
            and loss_function.__name__
            in ["categorical_hinge", "categorical_crossentropy", "binary_crossentropy", "kullback_leibler_divergence"]
        ) or (self.is_tensorflow and flag_is_instance):
            self._reduce_labels = False
            label_ph = k.placeholder(shape=self._output.shape)
        elif (
            "__name__" in dir(loss_function) and loss_function.__name__ in ["sparse_categorical_crossentropy"]
        ) or isinstance(loss_function, keras.losses.SparseCategoricalCrossentropy):
            self._reduce_labels = True
            label_ph = k.placeholder(shape=[None,])
        else:
            raise ValueError("Loss function not recognised.")

        # Define the loss using the loss function
        if "__name__" in dir(loss_function,) and loss_function.__name__ in [
            "categorical_crossentropy",
            "sparse_categorical_crossentropy",
            "binary_crossentropy",
        ]:
            loss_ = loss_function(label_ph, self._output, from_logits=self._use_logits)

        elif "__name__" in dir(loss_function) and loss_function.__name__ in [
            "categorical_hinge",
            "kullback_leibler_divergence",
        ]:
            loss_ = loss_function(label_ph, self._output)

        elif isinstance(
            loss_function,
            (
                keras.losses.CategoricalHinge,
                keras.losses.CategoricalCrossentropy,
                keras.losses.SparseCategoricalCrossentropy,
                keras.losses.KLDivergence,
                keras.losses.BinaryCrossentropy,
            ),
        ):
            loss_ = loss_function(label_ph, self._output)

        loss_gradients = k.gradients(loss_, self._input)

        if k.backend() == "tensorflow":
            loss_gradients = loss_gradients[0]
        elif k.backend() == "cntk":
            raise NotImplementedError("Only TensorFlow and Theano support is provided for Keras.")

        # Set loss, gradients and prediction functions
        self._predictions_op = self._output
        self._loss = loss_
        self._loss_gradients = k.function([self._input, label_ph], [loss_gradients])

        # Get the internal layer
        self._layer_names = self._get_layers()

    def _get_layers(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`
        """
        # pylint: disable=E0401
        if self.is_tensorflow:
            from tensorflow.keras.layers import InputLayer
        else:
            from keras.engine.topology import InputLayer

        layer_names = [layer.name for layer in self.classifier.layers[:-1] if not isinstance(layer, InputLayer)]
        logger.info("Inferred %i hidden layers on Keras classifier.", len(layer_names))

        return layer_names

    def __getstate__(self):
        """
        Use to ensure `KerasClassifier` can be pickled.

        :return: State dictionary with instance parameters.
        :rtype: `dict`
        """
        import time

        state = self.__dict__.copy()

        # Remove the unpicklable entries
        del state["_model"]
        del state["_input"]
        del state["_output"]
        del state["_predictions_op"]
        del state["_loss"]
        del state["_loss_gradients"]
        del state["_layer_names"]

        model_name = str(time.time()) + ".h5"
        state["model_name"] = model_name
        self.save(model_name)
        return state

    def __setstate__(self, state):
        """
        Use to ensure `KerasClassifier` can be unpickled.

        :param state: State dictionary with instance parameters to restore.
        :type state: `dict`
        """
        self.__dict__.update(state)

        # Load and update all functionality related to Keras
        # pylint: disable=E0401
        import os
        from art.config import ART_DATA_PATH

        if self.is_tensorflow:
            from tensorflow.keras.models import load_model
        else:
            from keras.models import load_model

        full_path = os.path.join(ART_DATA_PATH, state["model_name"])
        model = load_model(str(full_path))

        self.classifier = model
        self._initialize_params(model, state["_use_logits"], state["_input_layer"], state["_output_layer"])

    def __repr__(self):
        repr_ = (
            "%s(model=%r, use_logits=%r, channel_index=%r, clip_values=%r, preprocessing_defences=%r, "
            "postprocessing_defences=%r, preprocessing=%r, input_layer=%r, output_layer=%r)"
            % (
                self.__module__ + "." + self.__class__.__name__,
                self.classifier,
                self._use_logits,
                self.channel_index,
                self.clip_values,
                self.preprocessing_defences,
                self.postprocessing_defences,
                self.preprocessing,
                self._input_layer,
                self._output_layer,
            )
        )

        return repr_


def generator_fit(x, y, batch_size=128):
    """
    Minimal data generator for randomly batching large datasets.

    :param x: The data sample to batch.
    :type x: `np.ndarray`
    :param y: The labels for `x`. The first dimension has to match the first dimension of `x`.
    :type y: `np.ndarray`
    :param batch_size: The size of the batches to produce.
    :type batch_size: `int`
    :return: A batch of size `batch_size` of random samples from `(x, y)`
    :rtype: `tuple(np.ndarray, np.ndarray)`
    """
    while True:
        indices = np.random.randint(x.shape[0], size=batch_size)
        yield x[indices], y[indices]

