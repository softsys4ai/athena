"""
Implement ATHENA ensemble on top of IBM Trusted-AI ART 1.2.0.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
from enum import Enum
import random

from art.classifiers.classifier import Classifier, ClassifierNeuralNetwork, ClassifierGradients
from utils.data import set_channels_first, set_channels_last

logger = logging.getLogger(__name__)


class Ensemble(ClassifierNeuralNetwork, ClassifierGradients, Classifier):
    def __init__(self, classifiers, strategy, classifier_weights=None,
                 channel_index=3, clip_values=(0., 1.), preprocessing_defences=None,
                 postprocessing_defences=None, preprocessing=(0, 1)):
        if preprocessing_defences is not None:
            raise NotImplementedError('Preprocessing is not applicable in this classifier.')

        super(Ensemble, self).__init__(
            clip_values=clip_values,
            channel_index=channel_index,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        self._num_queries = 0

        if classifiers is None or not classifiers:
            raise ValueError('No classifiers provided for the whitebox.')

        for classifier in classifiers:
            # if not isinstance(classifier, ClassifierNeuralNetwork):
            #     raise TypeError('Expected type `Classifier`, found {} instead.'.format(type(classifier)))

            if clip_values != classifier.clip_values:
                raise ValueError('Incompatible `clip_values` between classifiers in the whitebox. '
                                 'Found {} and {}.'.format(str(clip_values), str(classifier.clip_values)))

            if classifier.nb_classes() != classifiers[0].nb_classes():
                raise ValueError('Incompatible output shape between classifiers in the whitebox. '
                                 'Found {} and {}'.format(classifier.nb_classes(), classifiers[0].nb_classes()))

            # if classifier.input_shape != classifiers[0].input_shape:
            #     raise ValueError('Incompatible input shape between classifiers in the whitebox. '
            #                      'Found {} and {}'.format(classifier.input_shape, classifiers[0].input_shape))

        self._input_shape = classifiers[0].input_shape
        self._nb_classes = classifiers[0].nb_classes()

        # check for consistent channel_index in whitebox members
        # for i_cls, cls in enumerate(classifiers):
        #     if cls.channel_index != self.channel_index:
        #         raise ValueError(
        #             "The channel_index value of classifier {} is {} while this whitebox expects a "
        #             "channel_index value of {}. The channel_index values of all classifiers and the "
        #             "whitebox need ot be identical.".format(i_cls, cls.channel_index, self.channel_index)
        #         )

        self._classifiers = classifiers
        self._nb_classifiers = len(classifiers)

        # Set weights for classifiers
        if classifier_weights is None:
            classifier_weights = np.ones(self._nb_classifiers) / self._nb_classifiers
        self._classifier_weights = classifier_weights

        if strategy not in ENSEMBLE_STRATEGY.available_names() or strategy not in ENSEMBLE_STRATEGY.available_values():
            strategy = ENSEMBLE_STRATEGY.AVEP.name
        self._strategy = strategy

        self._learning_phase = None

    def predict(self, x, batch_size=128, raw=False, **kwargs):
        raw_predictions = np.array(
            [self._classifier_weights[i] * self._classifiers[i].predict(x) for i in range(self._nb_classifiers)]
        )

        self._num_queries += x.shape[0]

        if raw:
            return raw_predictions
        else:
            return self.predict_by_predictions(raw_predictions=raw_predictions)

    def predict_by_predictions(self, raw_predictions):
        """
        Produce the final prediction given the collection of predictions from the WDs.
        :param raw_predictions: numpy array. the collection of predictions from the WDs.
        :return:
        """
        ensemble_preds = None
        if self._strategy == ENSEMBLE_STRATEGY.RD.name or self._strategy == ENSEMBLE_STRATEGY.RD.value:
            id = random.choice(self._nb_classifiers)
            ensemble_preds =  raw_predictions[id]

        elif self._strategy == ENSEMBLE_STRATEGY.MV.name or self._strategy == ENSEMBLE_STRATEGY.MV.value:
            num_samples = raw_predictions.shape[1]
            predicted_labels = []
            for probs in raw_predictions:
                labels = [np.argmax(p) for p in probs]
                predicted_labels.append(labels)
            predicted_labels = np.asarray(predicted_labels)

            # count frequency of each class
            votes = []
            ensemble_preds = []
            for s_id in range(num_samples):
                labels = [predicted_labels[wd_id][s_id] for wd_id in range(self._nb_classifiers)]
                values, freqs = np.unique(labels, return_counts=True)
                votes.append((values, freqs))
                rates = np.zeros((self._nb_classes, ), dtype=np.float32)
                amount = 0.
                for v, f in zip(values, freqs):
                    rates[v] = f
                    amount += f
                rates = rates / amount
                ensemble_preds.append(rates)

            ensemble_preds = np.asarray(ensemble_preds)

        elif self._strategy in [ENSEMBLE_STRATEGY.AVEP.name, ENSEMBLE_STRATEGY.AVEL.name, ENSEMBLE_STRATEGY.AVEO.name] or \
            self._strategy in [ENSEMBLE_STRATEGY.AVEP.value, ENSEMBLE_STRATEGY.AVEL.value, ENSEMBLE_STRATEGY.AVEO.value]:
            # averaging predictions
            ensemble_preds = np.average(raw_predictions, axis=0)

        # Apply postprocessing
        ensemble_preds = self._apply_postprocessing(preds=ensemble_preds, fit=False)
        return ensemble_preds

    def class_gradient(self, x, label=None, raw=False, **kwargs):
        grads = np.array(
            [
                self._classifier_weights[i] * self._classifiers[i].class_gradient(x, label)
                for i in range(self._nb_classifiers)
            ]
        )

        if raw:
            return grads

        return np.sum(grads, axis=0)

    def loss_gradient(self, x, y, raw=False, **kwargs):
        grads = np.array(
            [
                self._classifier_weights[i] * self._classifiers[i].loss_gradient(x, y)
                for i in range(self._nb_classifiers)
            ]
        )

        if raw:
            return grads

        return np.sum(grads, axis=0)

    def set_learning_phase(self, train):
        """
        Set the learning phase for the backend framework.

        :param train: True to set the learning phase to training, False to set it to prediction.
        :type train: `bool`
        """
        if self._learning is not None and isinstance(train, bool):
            for classifier in self._classifiers:
                classifier.set_learning_phase(train)
            self._learning_phase = train

    def nb_classes(self):
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        :rtype: `int`
        """
        return self._nb_classes

    def fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs):
        raise NotImplementedError

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        raise NotImplementedError

    def reset_model_queries(self):
        """
        Set the number of model queries to 0.
        """
        self._num_queries = 0

    @property
    def num_queries(self):
        """
        Return the number of model queries
        """
        return self._num_queries

    @property
    def layer_names(self):
        raise NotImplementedError

    def get_activations(self, x, layer, batch_size):
        raise NotImplementedError

    def __repr__(self):
        repr_ = (
            "%s(classifiers=%r, classifier_weights=%r, channel_index=%r, clip_values=%r, "
            "preprocessing_defences=%r, postprocessing_defences=%r, preprocessing=%r)"
            % (
                self.__module__ + "." + self.__class__.__name__,
                self._classifiers,
                self._classifier_weights,
                self.channel_index,
                self.clip_values,
                self.preprocessing_defences,
                self.postprocessing_defences,
                self.preprocessing,
            )
        )

        return repr_

    def save(self, filename, path=None):
        """
        Save a model to file in the format specific to the backend framework. This function is not supported for
        ensembles.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `ART_DATA_PATH`.
        :type path: `str`
        :raises: `NotImplementedException`
        :return: None
        """
        import json
        import os

        pool = [
            self._classifiers[i].trans_configs.get('description')
            for i in range(self._nb_classifiers)
        ]

        ensemble = {
            'pool': pool,
            'strategy': self._strategy,
            'channel_index': self._channel_index,
            'nb_classes': self._nb_classes,
            'clip_values': self._clip_values,
        }

        filename = os.path.join(path, filename)
        with open(filename, 'w') as f:
            json.dump(ensemble, f)


class ENSEMBLE_STRATEGY(Enum):
    RD = 0
    MV = 1
    T2MV = 2
    AVEP = 3
    AVEL = 4
    AVEO = 5

    @classmethod
    def available_strategies(cls):
        return {
            cls.RD.name: cls.RD.value,
            cls.MV.name: cls.MV.value,
            cls.T2MV.name: cls.T2MV.value,
            cls.AVEP.name: cls.AVEP.value,
            cls.AVEL.name: cls.AVEL.value,
            cls.AVEO.name: cls.AVEO.value,
        }

    @classmethod
    def available_names(cls):
        return [
            cls.RD.name, cls.MV.name, cls.T2MV.name,
            cls.AVEP.name, cls.AVEL.name, cls.AVEO.name,
        ]

    @classmethod
    def available_values(cls):
        return [
            cls.RD.value, cls.MV.value, cls.T2MV.value,
            cls.AVEP.value, cls.AVEL.value, cls.AVEO.value,
        ]