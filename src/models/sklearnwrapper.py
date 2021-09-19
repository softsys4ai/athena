"""
Implement weak defense model for Athena on top of IBM Trusted-AI ART 1.2.0.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

# MIT License
#
# Copyright (C) IBM Corporation 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the classifiers for scikit-learn models.
"""

import logging
import importlib

import numpy as np

from art.classifiers.classifier import Classifier, ClassifierGradients, ClassifierDecisionTree
from art.utils import to_categorical

from models.image_processor import transform

logger = logging.getLogger(__name__)


# pylint: disable=C0103
def WeakDefense(
    model, trans_configs, image_shape, clip_values=(0., 1.),
    preprocessing_defences=None, postprocessing_defences=None,
    preprocessing=(0, 1)
):
    """
    Create a `Classifier` instance from a scikit-learn Classifier model. This is a convenience function that
    instantiates the correct wrapper class for the given scikit-learn model.

    :param model: scikit-learn Classifier model.
    :type model: `sklearn.base.BaseEstimator`
    :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
            for features.
    :type clip_values: `tuple`
    :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
    :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
    :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
    :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
    :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
            used for data preprocessing. The first value will be subtracted from the input. The input will then
            be divided by the second one.
    :type preprocessing: `tuple`
    """
    if model.__class__.__module__.split(".")[0] != "sklearn":
        raise TypeError("Model is not an sklearn model. Received '%s'" % model.__class__)

    sklearn_name = model.__class__.__name__
    module = importlib.import_module("models.sklearnwrapper")
    if hasattr(module, "Scikitlearn%s" % sklearn_name):
        return getattr(module, "Scikitlearn%s" % sklearn_name)(
            model=model,
            trans_configs=trans_configs,
            image_shape=image_shape,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

    # This basic class at least generically handles `fit`, `predict` and `save`
    return ScikitlearnClassifier(model, trans_configs, image_shape, clip_values,
                                 preprocessing_defences, postprocessing_defences,
                                 preprocessing)


class ScikitlearnClassifier(Classifier):
    """
    Wrapper class for scikit-learn classifier models.
    """

    def __init__(
        self,
        model,
        trans_configs,
        image_shape,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0, 1),
    ):
        """
        Create a `Classifier` instance from a scikit-learn classifier model.

        :param model: scikit-learn classifier model.
        :type model: `sklearn.base.BaseEstimator`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        super(ScikitlearnClassifier, self).__init__(
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self.classifier = model
        self._trans_configs = trans_configs
        self._image_shape = image_shape
        self._input_shape = self._get_input_shape(model)
        self._nb_classes = self._get_nb_classes()
        self._num_queries = 0

    def fit(self, x, y, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit` function in `sklearn` classifier and will be passed to this function as such.
        :type kwargs: `dict`
        :return: `None`
        """
        # Apply transformation
        _input_shape = self._get_input_shape(self.classifier)
        if x.shape == _input_shape:
            x = x.reshape((-1, self._image_shape[0], self._image_shape[1], self._image_shape[2]))
        x_preprocessed = transform(x, self._trans_configs)
        x_preprocessed = x_preprocessed.reshape(self._get_input_shape(self.classifier))

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x_preprocessed, y, fit=True)
        y_preprocessed = np.argmax(y_preprocessed, axis=1)

        self.classifier.fit(x_preprocessed, y_preprocessed, **kwargs)
        self._input_shape = self._get_input_shape(self.classifier)
        self._nb_classes = self._get_nb_classes()

    def predict(self, x, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """

        # Apply transformation
        _num_examples = x.shape[0]
        if len(x.shape) <= 2:
            x = x.reshape((-1, self._image_shape[0], self._image_shape[1], self._image_shape[2]))
        x_preprocessed = transform(x, self._trans_configs)
        _input_shape = [i for i in self._input_shape]
        _input_shape = tuple([_num_examples] + _input_shape)
        x_preprocessed = x_preprocessed.reshape(_input_shape)

        # Apply defences
        x_preprocessed, _ = self._apply_preprocessing(x_preprocessed, y=None, fit=False)

        if hasattr(self.classifier, "predict_proba") and callable(getattr(self.classifier, "predict_proba")):
            y_pred = self.classifier.predict_proba(x_preprocessed)
        elif hasattr(self.classifier, "predict") and callable(getattr(self.classifier, "predict")):
            y_pred = to_categorical(self.classifier.predict(x_preprocessed), nb_classes=self.classifier.classes_.shape[0])
        else:
            raise ValueError("The provided model does not have methods `predict_proba` or `predict`.")

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=y_pred, fit=False)

        # Increase the number of model queries
        self._num_queries += x.shape[0]

        return predictions

    def nb_classes(self):
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        :rtype: `int` or `None`
        """
        if hasattr(self.classifier, "n_classes_"):
            _nb_classes = self.classifier.n_classes_
        else:
            _nb_classes = None
        return _nb_classes

    def save(self, filename, path=None):
        import pickle

        with open(filename + ".pickle", "wb") as file_pickle:
            pickle.dump(self.classifier, file=file_pickle)

    def reset_model_queries(self):
        """
        Set the number of model queries to 0
        """
        self._num_queries = 0

    @property
    def num_queries(self):
        """
        Return the number of model queries
        """
        return self._num_queries

    def _get_input_shape(self, model):
        if hasattr(model, "n_features_"):
            _input_shape = (model.n_features_,)
        elif hasattr(model, "feature_importances_"):
            _input_shape = (len(model.feature_importances_),)
        elif hasattr(model, "coef_"):
            if len(model.coef_.shape) == 1:
                _input_shape = (model.coef_.shape[0],)
            else:
                _input_shape = (model.coef_.shape[1],)
        elif hasattr(model, "support_vectors_"):
            _input_shape = (model.support_vectors_.shape[1],)
        elif hasattr(model, "steps"):
            _input_shape = self._get_input_shape(model.steps[0][1])
        else:
            logger.warning("Input shape not recognised. The model might not have been fitted.")
            _input_shape = None
        return _input_shape

    def _get_nb_classes(self):
        if hasattr(self.classifier, "n_classes_"):
            _nb_classes = self.classifier.n_classes_
        elif hasattr(self.classifier, "classes_"):
            _nb_classes = self.classifier.classes_.shape[0]
        else:
            logger.warning("Number of classes not recognised. The model might not have been fitted.")
            _nb_classes = None
        return _nb_classes

#
# class ScikitlearnDecisionTreeClassifier(ScikitlearnClassifier):
#     """
#     Wrapper class for scikit-learn Decision Tree Classifier models.
#     """
#
#     def __init__(
#         self, model, clip_values=None, preprocessing_defences=None, postprocessing_defences=None, preprocessing=(0, 1)
#     ):
#         """
#         Create a `Classifier` instance from a scikit-learn Decision Tree Classifier model.
#
#         :param model: scikit-learn Decision Tree Classifier model.
#         :type model: `sklearn.tree.DecisionTreeClassifier`
#         :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
#                for features.
#         :type clip_values: `tuple`
#         :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
#         :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
#         :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
#         :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
#         :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
#                used for data preprocessing. The first value will be subtracted from the input. The input will then
#                be divided by the second one.
#         :type preprocessing: `tuple`
#         """
#         # pylint: disable=E0001
#         from sklearn.tree import DecisionTreeClassifier
#
#         if not isinstance(model, DecisionTreeClassifier) and model is not None:
#             raise TypeError("Model must be of type sklearn.tree.DecisionTreeClassifier.")
#
#         super(ScikitlearnDecisionTreeClassifier, self).__init__(
#             model=model,
#             clip_values=clip_values,
#             preprocessing_defences=preprocessing_defences,
#             postprocessing_defences=postprocessing_defences,
#             preprocessing=preprocessing,
#         )
#         self.classifier = model
#
#     def get_classes_at_node(self, node_id):
#         """
#         Returns the classification for a given node.
#
#         :return: major class in node.
#         :rtype: `float`
#         """
#         return np.argmax(self.classifier.tree_.value[node_id])
#
#     def get_threshold_at_node(self, node_id):
#         """
#         Returns the threshold of given id for a node.
#
#         :return: threshold value of feature split in this node.
#         :rtype: `float`
#         """
#         return self.classifier.tree_.threshold[node_id]
#
#     def get_feature_at_node(self, node_id):
#         """
#         Returns the feature of given id for a node.
#
#         :return: feature index of feature split in this node.
#         :rtype: `int`
#         """
#         return self.classifier.tree_.feature[node_id]
#
#     def get_left_child(self, node_id):
#         """
#         Returns the id of the left child node of node_id.
#
#         :return: the indices of the left child in the tree.
#         :rtype: `int`
#         """
#         return self.classifier.tree_.children_left[node_id]
#
#     def get_right_child(self, node_id):
#         """
#         Returns the id of the right child node of node_id.
#
#         :return: the indices of the right child in the tree.
#         :rtype: `int`
#         """
#         return self.classifier.tree_.children_right[node_id]
#
#     def get_decision_path(self, x):
#         """
#         Returns the path through nodes in the tree when classifying x. Last one is leaf, first one root node.
#
#         :return: the indices of the nodes in the array structure of the tree.
#         :rtype: `np.ndarray`
#         """
#         if len(np.shape(x)) == 1:
#             return self.classifier.decision_path(x.reshape(1, -1)).indices
#
#         return self.classifier.decision_path(x).indices
#
#     def get_values_at_node(self, node_id):
#         """
#         Returns the feature of given id for a node.
#
#         :return: Normalized values at node node_id.
#         :rtype: `nd.array`
#         """
#         return self.classifier.tree_.value[node_id] / np.linalg.norm(self.classifier.tree_.value[node_id])
#
#     def _get_leaf_nodes(self, node_id, i_tree, class_label, box):
#         from copy import deepcopy
#         from art.metrics.verification_decisions_trees import LeafNode, Box, Interval
#
#         leaf_nodes = list()
#
#         if self.get_left_child(node_id) != self.get_right_child(node_id):
#
#             node_left = self.get_left_child(node_id)
#             node_right = self.get_right_child(node_id)
#
#             box_left = deepcopy(box)
#             box_right = deepcopy(box)
#
#             feature = self.get_feature_at_node(node_id)
#             box_split_left = Box(intervals={feature: Interval(-np.inf, self.get_threshold_at_node(node_id))})
#             box_split_right = Box(intervals={feature: Interval(self.get_threshold_at_node(node_id), np.inf)})
#
#             if box.intervals:
#                 box_left.intersect_with_box(box_split_left)
#                 box_right.intersect_with_box(box_split_right)
#             else:
#                 box_left = box_split_left
#                 box_right = box_split_right
#
#             leaf_nodes += self._get_leaf_nodes(node_left, i_tree, class_label, box_left)
#             leaf_nodes += self._get_leaf_nodes(node_right, i_tree, class_label, box_right)
#
#         else:
#             leaf_nodes.append(
#                 LeafNode(
#                     tree_id=i_tree,
#                     class_label=class_label,
#                     node_id=node_id,
#                     box=box,
#                     value=self.get_values_at_node(node_id)[0, class_label],
#                 )
#             )
#
#         return leaf_nodes
#
#
# class ScikitlearnDecisionTreeRegressor(ScikitlearnDecisionTreeClassifier):
#     """
#     Wrapper class for scikit-learn Decision Tree Regressor models.
#     """
#
#     def __init__(
#         self, model, clip_values=None, preprocessing_defences=None, postprocessing_defences=None, preprocessing=(0, 1)
#     ):
#         """
#         Create a `Regressor` instance from a scikit-learn Decision Tree Regressor model.
#
#         :param model: scikit-learn Decision Tree Regressor model.
#         :type model: `sklearn.tree.DecisionTreeRegressor`
#         :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
#                for features.
#         :type clip_values: `tuple`
#         :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
#         :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
#         :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
#         :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
#         :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
#                used for data preprocessing. The first value will be subtracted from the input. The input will then
#                be divided by the second one.
#         :type preprocessing: `tuple`
#         """
#         # pylint: disable=E0001
#         from sklearn.tree import DecisionTreeRegressor
#
#         if not isinstance(model, DecisionTreeRegressor):
#             raise TypeError("Model must be of type sklearn.tree.DecisionTreeRegressor.")
#
#         ScikitlearnDecisionTreeClassifier.__init__(
#             self,
#             model=None,
#             clip_values=clip_values,
#             preprocessing_defences=preprocessing_defences,
#             postprocessing_defences=postprocessing_defences,
#             preprocessing=preprocessing,
#         )
#         self.classifier = model
#
#     def get_values_at_node(self, node_id):
#         """
#         Returns the feature of given id for a node.
#
#         :return: Normalized values at node node_id.
#         :rtype: `nd.array`
#         """
#         return self.classifier.tree_.value[node_id]
#
#     def _get_leaf_nodes(self, node_id, i_tree, class_label, box):
#         from copy import deepcopy
#         from art.metrics.verification_decisions_trees import LeafNode, Box, Interval
#
#         leaf_nodes = list()
#
#         if self.get_left_child(node_id) != self.get_right_child(node_id):
#
#             node_left = self.get_left_child(node_id)
#             node_right = self.get_right_child(node_id)
#
#             box_left = deepcopy(box)
#             box_right = deepcopy(box)
#
#             feature = self.get_feature_at_node(node_id)
#             box_split_left = Box(intervals={feature: Interval(-np.inf, self.get_threshold_at_node(node_id))})
#             box_split_right = Box(intervals={feature: Interval(self.get_threshold_at_node(node_id), np.inf)})
#
#             if box.intervals:
#                 box_left.intersect_with_box(box_split_left)
#                 box_right.intersect_with_box(box_split_right)
#             else:
#                 box_left = box_split_left
#                 box_right = box_split_right
#
#             leaf_nodes += self._get_leaf_nodes(node_left, i_tree, class_label, box_left)
#             leaf_nodes += self._get_leaf_nodes(node_right, i_tree, class_label, box_right)
#
#         else:
#             leaf_nodes.append(
#                 LeafNode(
#                     tree_id=i_tree,
#                     class_label=class_label,
#                     node_id=node_id,
#                     box=box,
#                     value=self.get_values_at_node(node_id)[0, 0],
#                 )
#             )
#
#         return leaf_nodes
#
#
# class ScikitlearnExtraTreeClassifier(ScikitlearnDecisionTreeClassifier):
#     """
#     Wrapper class for scikit-learn Extra TreeClassifier Classifier models.
#     """
#
#     def __init__(
#         self, model, clip_values=None, preprocessing_defences=None, postprocessing_defences=None, preprocessing=(0, 1)
#     ):
#         """
#         Create a `Classifier` instance from a scikit-learn Extra TreeClassifier Classifier model.
#
#         :param model: scikit-learn Extra TreeClassifier Classifier model.
#         :type model: `sklearn.tree.ExtraTreeClassifier`
#         :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
#                for features.
#         :type clip_values: `tuple`
#         :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
#         :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
#         :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
#         :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
#         :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
#                used for data preprocessing. The first value will be subtracted from the input. The input will then
#                be divided by the second one.
#         :type preprocessing: `tuple`
#         """
#         # pylint: disable=E0001
#         from sklearn.tree import ExtraTreeClassifier
#
#         if not isinstance(model, ExtraTreeClassifier):
#             raise TypeError("Model must be of type sklearn.tree.ExtraTreeClassifier.")
#
#         super(ScikitlearnExtraTreeClassifier, self).__init__(
#             model=model,
#             clip_values=clip_values,
#             preprocessing_defences=preprocessing_defences,
#             postprocessing_defences=postprocessing_defences,
#             preprocessing=preprocessing,
#         )
#         self.classifier = model
#
#
# class ScikitlearnAdaBoostClassifier(ScikitlearnClassifier):
#     """
#     Wrapper class for scikit-learn AdaBoost Classifier models.
#     """
#
#     def __init__(
#         self, model, clip_values=None, preprocessing_defences=None, postprocessing_defences=None, preprocessing=(0, 1)
#     ):
#         """
#         Create a `Classifier` instance from a scikit-learn AdaBoost Classifier model.
#
#         :param model: scikit-learn AdaBoost Classifier model.
#         :type model: `sklearn.ensemble.AdaBoostClassifier`
#         :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
#                for features.
#         :type clip_values: `tuple`
#         :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
#         :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
#         :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
#         :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
#         :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
#                used for data preprocessing. The first value will be subtracted from the input. The input will then
#                be divided by the second one.
#         :type preprocessing: `tuple`
#         """
#         # pylint: disable=E0001
#         from sklearn.ensemble import AdaBoostClassifier
#
#         if not isinstance(model, AdaBoostClassifier):
#             raise TypeError("Model must be of type sklearn.ensemble.AdaBoostClassifier.")
#
#         super(ScikitlearnAdaBoostClassifier, self).__init__(
#             model=model,
#             clip_values=clip_values,
#             preprocessing_defences=preprocessing_defences,
#             postprocessing_defences=postprocessing_defences,
#             preprocessing=preprocessing,
#         )
#         self.classifier = model
#
#
# class ScikitlearnBaggingClassifier(ScikitlearnClassifier):
#     """
#     Wrapper class for scikit-learn Bagging Classifier models.
#     """
#
#     def __init__(
#         self, model, clip_values=None, preprocessing_defences=None, postprocessing_defences=None, preprocessing=(0, 1)
#     ):
#         """
#         Create a `Classifier` instance from a scikit-learn Bagging Classifier model.
#
#         :param model: scikit-learn Bagging Classifier model.
#         :type model: `sklearn.ensemble.BaggingClassifier`
#         :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
#                for features.
#         :type clip_values: `tuple`
#         :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
#         :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
#         :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
#         :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
#         :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
#                used for data preprocessing. The first value will be subtracted from the input. The input will then
#                be divided by the second one.
#         :type preprocessing: `tuple`
#         """
#         # pylint: disable=E0001
#         from sklearn.ensemble import BaggingClassifier
#
#         if not isinstance(model, BaggingClassifier):
#             raise TypeError("Model must be of type sklearn.ensemble.BaggingClassifier.")
#
#         super(ScikitlearnBaggingClassifier, self).__init__(
#             model=model,
#             clip_values=clip_values,
#             preprocessing_defences=preprocessing_defences,
#             postprocessing_defences=postprocessing_defences,
#             preprocessing=preprocessing,
#         )
#         self.classifier = model
#
#
# class ScikitlearnExtraTreesClassifier(ScikitlearnClassifier, ClassifierDecisionTree):
#     """
#     Wrapper class for scikit-learn Extra Trees Classifier models.
#     """
#
#     def __init__(
#         self, model, clip_values=None, preprocessing_defences=None, postprocessing_defences=None, preprocessing=(0, 1)
#     ):
#         """
#         Create a `Classifier` instance from a scikit-learn Extra Trees Classifier model.
#
#         :param model: scikit-learn Extra Trees Classifier model.
#         :type model: `sklearn.ensemble.ExtraTreesClassifier`
#         :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
#                for features.
#         :type clip_values: `tuple`
#         :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
#         :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
#         :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
#         :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
#         :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
#                used for data preprocessing. The first value will be subtracted from the input. The input will then
#                be divided by the second one.
#         :type preprocessing: `tuple`
#         """
#         # pylint: disable=E0001
#         from sklearn.ensemble import ExtraTreesClassifier
#
#         if not isinstance(model, ExtraTreesClassifier):
#             raise TypeError("Model must be of type sklearn.ensemble.ExtraTreesClassifier.")
#
#         super(ScikitlearnExtraTreesClassifier, self).__init__(
#             model=model,
#             clip_values=clip_values,
#             preprocessing_defences=preprocessing_defences,
#             postprocessing_defences=postprocessing_defences,
#             preprocessing=preprocessing,
#         )
#         self.classifier = model
#
#     def get_trees(self):  # lgtm [py/similar-function]
#         """
#         Get the decision trees.
#
#         :return: A list of decision trees.
#         :rtype: `[Tree]`
#         """
#         from art.metrics.verification_decisions_trees import Box, Tree
#
#         trees = list()
#
#         for i_tree, decision_tree_model in enumerate(self.classifier.estimators_):
#             box = Box()
#
#             #     if num_classes == 2:
#             #         class_label = -1
#             #     else:
#             #         class_label = i_tree % num_classes
#
#             extra_tree_classifier = ScikitlearnExtraTreeClassifier(model=decision_tree_model)
#
#             for i_class in range(self.classifier.n_classes_):
#                 class_label = i_class
#
#                 # pylint: disable=W0212
#                 trees.append(
#                     Tree(
#                         class_id=class_label,
#                         leaf_nodes=extra_tree_classifier._get_leaf_nodes(0, i_tree, class_label, box),
#                     )
#                 )
#
#         return trees
#
#
# class ScikitlearnGradientBoostingClassifier(ScikitlearnClassifier, ClassifierDecisionTree):
#     """
#     Wrapper class for scikit-learn Gradient Boosting Classifier models.
#     """
#
#     def __init__(
#         self, model, clip_values=None, preprocessing_defences=None, postprocessing_defences=None, preprocessing=(0, 1)
#     ):
#         """
#         Create a `Classifier` instance from a scikit-learn Gradient Boosting Classifier model.
#
#         :param model: scikit-learn Gradient Boosting Classifier model.
#         :type model: `sklearn.ensemble.GradientBoostingClassifier`
#         :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
#                for features.
#         :type clip_values: `tuple`
#         :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
#         :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
#         :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
#         :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
#         :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
#                used for data preprocessing. The first value will be subtracted from the input. The input will then
#                be divided by the second one.
#         :type preprocessing: `tuple`
#         """
#         # pylint: disable=E0001
#         from sklearn.ensemble import GradientBoostingClassifier
#
#         if not isinstance(model, GradientBoostingClassifier):
#             raise TypeError("Model must be of type sklearn.ensemble.GradientBoostingClassifier.")
#
#         super(ScikitlearnGradientBoostingClassifier, self).__init__(
#             model=model,
#             clip_values=clip_values,
#             preprocessing_defences=preprocessing_defences,
#             postprocessing_defences=postprocessing_defences,
#             preprocessing=preprocessing,
#         )
#         self.classifier = model
#
#     def get_trees(self):
#         """
#         Get the decision trees.
#
#         :return: A list of decision trees.
#         :rtype: `[Tree]`
#         """
#         from art.metrics.verification_decisions_trees import Box, Tree
#
#         trees = list()
#         num_trees, num_classes = self.classifier.estimators_.shape
#
#         for i_tree in range(num_trees):
#             box = Box()
#
#             for i_class in range(num_classes):
#                 decision_tree_classifier = ScikitlearnDecisionTreeRegressor(
#                     model=self.classifier.estimators_[i_tree, i_class]
#                 )
#
#                 if num_classes == 2:
#                     class_label = None
#                 else:
#                     class_label = i_class
#
#                 # pylint: disable=W0212
#                 trees.append(
#                     Tree(
#                         class_id=class_label,
#                         leaf_nodes=decision_tree_classifier._get_leaf_nodes(0, i_tree, class_label, box),
#                     )
#                 )
#
#         return trees
#
#
# class ScikitlearnRandomForestClassifier(ScikitlearnClassifier):
#     """
#     Wrapper class for scikit-learn Random Forest Classifier models.
#     """
#
#     def __init__(
#         self, model, clip_values=None, preprocessing_defences=None, postprocessing_defences=None, preprocessing=(0, 1)
#     ):
#         """
#         Create a `Classifier` instance from a scikit-learn Random Forest Classifier model.
#
#         :param model: scikit-learn Random Forest Classifier model.
#         :type model: `sklearn.ensemble.RandomForestClassifier`
#         :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
#                for features.
#         :type clip_values: `tuple`
#         :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
#         :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
#         :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
#         :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
#         :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
#                used for data preprocessing. The first value will be subtracted from the input. The input will then
#                be divided by the second one.
#         :type preprocessing: `tuple`
#         """
#         # pylint: disable=E0001
#         from sklearn.ensemble import RandomForestClassifier
#
#         if not isinstance(model, RandomForestClassifier):
#             raise TypeError("Model must be of type sklearn.ensemble.RandomForestClassifier.")
#
#         super(ScikitlearnRandomForestClassifier, self).__init__(
#             model=model,
#             clip_values=clip_values,
#             preprocessing_defences=preprocessing_defences,
#             postprocessing_defences=postprocessing_defences,
#             preprocessing=preprocessing,
#         )
#         self.classifier = model
#
#     def get_trees(self):  # lgtm [py/similar-function]
#         """
#         Get the decision trees.
#
#         :return: A list of decision trees.
#         :rtype: `[Tree]`
#         """
#         from art.metrics.verification_decisions_trees import Box, Tree
#
#         trees = list()
#
#         for i_tree, decision_tree_model in enumerate(self.classifier.estimators_):
#             box = Box()
#
#             #     if num_classes == 2:
#             #         class_label = -1
#             #     else:
#             #         class_label = i_tree % num_classes
#
#             decision_tree_classifier = ScikitlearnDecisionTreeClassifier(model=decision_tree_model)
#
#             for i_class in range(self.classifier.n_classes_):
#                 class_label = i_class
#
#                 # pylint: disable=W0212
#                 trees.append(
#                     Tree(
#                         class_id=class_label,
#                         leaf_nodes=decision_tree_classifier._get_leaf_nodes(0, i_tree, class_label, box),
#                     )
#                 )
#
#         return trees
#
#
# class ScikitlearnLogisticRegression(ScikitlearnClassifier, ClassifierGradients):
#     """
#     Wrapper class for scikit-learn Logistic Regression models.
#     """
#
#     def __init__(
#         self, model, clip_values=None, preprocessing_defences=None, postprocessing_defences=None, preprocessing=(0, 1)
#     ):
#         """
#         Create a `Classifier` instance from a scikit-learn Logistic Regression model.
#
#         :param model: scikit-learn LogisticRegression model
#         :type model: `sklearn.linear_model.LogisticRegression`
#         :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
#                for features.
#         :type clip_values: `tuple`
#         :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
#         :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
#         :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
#         :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
#         :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
#                used for data preprocessing. The first value will be subtracted from the input. The input will then
#                be divided by the second one.
#         :type preprocessing: `tuple`
#         """
#         super(ScikitlearnLogisticRegression, self).__init__(
#             model=model,
#             clip_values=clip_values,
#             preprocessing_defences=preprocessing_defences,
#             postprocessing_defences=postprocessing_defences,
#             preprocessing=preprocessing,
#         )
#         self.classifier = model
#
#     def nb_classes(self):
#         """
#         Return the number of output classes.
#
#         :return: Number of classes in the data.
#         :rtype: `int` or `None`
#         """
#         if hasattr(self.classifier, "coef_"):
#             _nb_classes = self.classifier.classes_.shape[0]
#         else:
#             _nb_classes = None
#         return _nb_classes
#
#     def class_gradient(self, x, label=None, **kwargs):
#         """
#         Compute per-class derivatives w.r.t. `x`.
#
#         | Paper link: http://cs229.stanford.edu/proj2016/report/ItkinaWu-AdversarialAttacksonImageRecognition-report.pdf
#         | Typo in https://arxiv.org/abs/1605.07277 (equation 6)
#
#         :param x: Sample input with shape as expected by the model.
#         :type x: `np.ndarray`
#         :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
#                       output is computed for all samples. If multiple values as provided, the first dimension should
#                       match the batch size of `x`, and each value will be used as target for its corresponding sample in
#                       `x`. If `None`, then gradients for all classes will be computed for each sample.
#         :type label: `int` or `list`
#         :return: Array of gradients of input features w.r.t. each class in the form
#                  `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
#                  `(batch_size, 1, input_shape)` when `label` parameter is specified.
#         :rtype: `np.ndarray`
#         """
#         if not hasattr(self.classifier, "coef_"):
#             raise ValueError(
#                 """Model has not been fitted. Run function `fit(x, y)` of classifier first or provide a
#             fitted model."""
#             )
#
#         nb_samples = x.shape[0]
#
#         # Apply preprocessing
#         x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)
#
#         y_pred = self.classifier.predict_proba(X=x_preprocessed)
#         weights = self.classifier.coef_
#
#         if self.nb_classes() > 2:
#             w_weighted = np.matmul(y_pred, weights)
#
#         def _f_class_gradient(i_class, i_sample):
#             if self.nb_classes() == 2:
#                 return (-1.0) ** (i_class + 1.0) * y_pred[i_sample, 0] * y_pred[i_sample, 1] * weights[0, :]
#
#             return weights[i_class, :] - w_weighted[i_sample, :]
#
#         if label is None:
#             # Compute the gradients w.r.t. all classes
#             class_gradients = list()
#
#             for i_class in range(self.nb_classes()):
#                 class_gradient = np.zeros(x.shape)
#                 for i_sample in range(nb_samples):
#                     class_gradient[i_sample, :] += _f_class_gradient(i_class, i_sample)
#                 class_gradients.append(class_gradient)
#
#             gradients = np.swapaxes(np.array(class_gradients), 0, 1)
#
#         elif isinstance(label, (int, np.integer)):
#             # Compute the gradients only w.r.t. the provided label
#             class_gradient = np.zeros(x.shape)
#             for i_sample in range(nb_samples):
#                 class_gradient[i_sample, :] += _f_class_gradient(label, i_sample)
#
#             gradients = np.swapaxes(np.array([class_gradient]), 0, 1)
#
#         elif (
#             (isinstance(label, list) and len(label) == nb_samples)
#             or isinstance(label, np.ndarray)
#             and label.shape == (nb_samples,)
#         ):
#             # For each sample, compute the gradients w.r.t. the indicated target class (possibly distinct)
#             class_gradients = list()
#             unique_labels = list(np.unique(label))
#
#             for unique_label in unique_labels:
#                 class_gradient = np.zeros(x.shape)
#                 for i_sample in range(nb_samples):
#                     # class_gradient[i_sample, :] += label[i_sample, unique_label] * (weights[unique_label, :]
#                     # - w_weighted[i_sample, :])
#                     class_gradient[i_sample, :] += _f_class_gradient(unique_label, i_sample)
#
#                 class_gradients.append(class_gradient)
#
#             gradients = np.swapaxes(np.array(class_gradients), 0, 1)
#             lst = [unique_labels.index(i) for i in label]
#             gradients = np.expand_dims(gradients[np.arange(len(gradients)), lst], axis=1)
#
#         else:
#             raise TypeError("Unrecognized type for argument `label` with type " + str(type(label)))
#
#         gradients = self._apply_preprocessing_gradient(x, gradients)
#
#         return gradients
#
#     def loss_gradient(self, x, y, **kwargs):
#         """
#         Compute the gradient of the loss function w.r.t. `x`.
#
#         | Paper link: http://cs229.stanford.edu/proj2016/report/ItkinaWu-AdversarialAttacksonImageRecognition-report.pdf
#         | Typo in https://arxiv.org/abs/1605.07277 (equation 6)
#
#         :param x: Sample input with shape as expected by the model.
#         :type x: `np.ndarray`
#         :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
#                   (nb_samples,).
#         :type y: `np.ndarray`
#         :return: Array of gradients of the same shape as `x`.
#         :rtype: `np.ndarray`
#         """
#         # pylint: disable=E0001
#         from sklearn.utils.class_weight import compute_class_weight
#
#         if not hasattr(self.classifier, "coef_"):
#             raise ValueError(
#                 """Model has not been fitted. Run function `fit(x, y)` of classifier first or provide a
#             fitted model."""
#             )
#
#         # Apply preprocessing
#         x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)
#
#         num_samples, _ = x_preprocessed.shape
#         gradients = np.zeros(x_preprocessed.shape)
#
#         y_index = np.argmax(y_preprocessed, axis=1)
#         if self.classifier.class_weight is None or self.classifier.class_weight == "balanced":
#             class_weight = np.ones(self.nb_classes())
#         else:
#             class_weight = compute_class_weight(
#                 class_weight=self.classifier.class_weight, classes=self.classifier.classes_, y=y_index
#             )
#
#         y_pred = self.classifier.predict_proba(X=x_preprocessed)
#         weights = self.classifier.coef_
#
#         # Consider the special case of a binary logistic regression model:
#         if self.nb_classes() == 2:
#             for i_sample in range(num_samples):
#                 gradients[i_sample, :] += (
#                     class_weight[1] * (1.0 - y_preprocessed[i_sample, 1])
#                     - class_weight[0] * (1.0 - y_preprocessed[i_sample, 0])
#                 ) * (y_pred[i_sample, 0] * y_pred[i_sample, 1] * weights[0, :])
#         else:
#             w_weighted = np.matmul(y_pred, weights)
#
#             for i_sample in range(num_samples):
#                 for i_class in range(self.nb_classes()):
#                     gradients[i_sample, :] += (
#                         class_weight[i_class]
#                         * (1.0 - y_preprocessed[i_sample, i_class])
#                         * (weights[i_class, :] - w_weighted[i_sample, :])
#                     )
#
#         gradients = self._apply_preprocessing_gradient(x, gradients)
#
#         return gradients


class ScikitlearnSVC(ScikitlearnClassifier, ClassifierGradients):
    """
    Wrapper class for scikit-learn C-Support Vector Classification models.
    """

    def __init__(
        self,
        model,
        trans_configs,
        image_shape,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0, 1)
    ):
        """
        Create a `Classifier` instance from a scikit-learn C-Support Vector Classification model.

        :param model: scikit-learn C-Support Vector Classification model.
        :type model: `sklearn.svm.SVC` or `sklearn.svm.LinearSVC`
        :param trans_configs: the corresponding transformation(s)
        :type trans_configs: dictionary.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        # pylint: disable=E0001
        from sklearn.svm import SVC, LinearSVC

        if not isinstance(model, SVC) and not isinstance(model, LinearSVC):
            raise TypeError(
                "Model must be of type sklearn.svm.SVC or sklearn.svm.LinearSVC. Found type {}".format(type(model))
            )

        super(ScikitlearnSVC, self).__init__(
            model=model,
            trans_configs=trans_configs,
            image_shape=image_shape,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self.classifier = model
        self._trans_configs = trans_configs
        self._image_shape = image_shape
        self._kernel = self._kernel_func()
        self._num_queries = 0

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
        # pylint: disable=E0001
        from sklearn.svm import SVC, LinearSVC

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        num_samples, _ = x_preprocessed.shape

        if isinstance(self.classifier, SVC):
            if self.classifier.fit_status_:
                raise AssertionError("Model has not been fitted correctly.")

            support_indices = [0] + list(np.cumsum(self.classifier.n_support_))

            if self.nb_classes() == 2:
                sign_multiplier = -1
            else:
                sign_multiplier = 1

            if label is None:
                gradients = np.zeros((x_preprocessed.shape[0], self.nb_classes(), x_preprocessed.shape[1]))

                for i_label in range(self.nb_classes()):
                    for i_sample in range(num_samples):
                        for not_label in range(self.nb_classes()):
                            if i_label != not_label:
                                if not_label < i_label:
                                    label_multiplier = -1
                                else:
                                    label_multiplier = 1

                                for label_sv in range(support_indices[i_label], support_indices[i_label + 1]):
                                    alpha_i_k_y_i = self.classifier.dual_coef_[
                                        not_label if not_label < i_label else not_label - 1, label_sv
                                    ]
                                    grad_kernel = self._get_kernel_gradient_sv(label_sv, x_preprocessed[i_sample])
                                    gradients[i_sample, i_label] += label_multiplier * alpha_i_k_y_i * grad_kernel

                                for not_label_sv in range(support_indices[not_label], support_indices[not_label + 1]):
                                    alpha_i_k_y_i = self.classifier.dual_coef_[
                                        i_label if i_label < not_label else i_label - 1, not_label_sv
                                    ]
                                    grad_kernel = self._get_kernel_gradient_sv(not_label_sv, x_preprocessed[i_sample])
                                    gradients[i_sample, i_label] += label_multiplier * alpha_i_k_y_i * grad_kernel

            elif isinstance(label, (int, np.integer)):
                gradients = np.zeros((x_preprocessed.shape[0], 1, x_preprocessed.shape[1]))

                for i_sample in range(num_samples):
                    for not_label in range(self.nb_classes()):
                        if label != not_label:
                            if not_label < label:
                                label_multiplier = -1
                            else:
                                label_multiplier = 1

                            for label_sv in range(support_indices[label], support_indices[label + 1]):
                                alpha_i_k_y_i = self.classifier.dual_coef_[
                                    not_label if not_label < label else not_label - 1, label_sv
                                ]
                                grad_kernel = self._get_kernel_gradient_sv(label_sv, x_preprocessed[i_sample])
                                gradients[i_sample, 0] += label_multiplier * alpha_i_k_y_i * grad_kernel

                            for not_label_sv in range(support_indices[not_label], support_indices[not_label + 1]):
                                alpha_i_k_y_i = self.classifier.dual_coef_[
                                    label if label < not_label else label - 1, not_label_sv
                                ]
                                grad_kernel = self._get_kernel_gradient_sv(not_label_sv, x_preprocessed[i_sample])
                                gradients[i_sample, 0] += label_multiplier * alpha_i_k_y_i * grad_kernel

            elif (
                (isinstance(label, list) and len(label) == num_samples)
                or isinstance(label, np.ndarray)
                and label.shape == (num_samples,)
            ):
                gradients = np.zeros((x_preprocessed.shape[0], 1, x_preprocessed.shape[1]))

                for i_sample in range(num_samples):
                    for not_label in range(self.nb_classes()):
                        if label[i_sample] != not_label:
                            if not_label < label[i_sample]:
                                label_multiplier = -1
                            else:
                                label_multiplier = 1

                            for label_sv in range(
                                support_indices[label[i_sample]], support_indices[label[i_sample] + 1]
                            ):
                                alpha_i_k_y_i = self.classifier.dual_coef_[
                                    not_label if not_label < label[i_sample] else not_label - 1, label_sv
                                ]
                                grad_kernel = self._get_kernel_gradient_sv(label_sv, x_preprocessed[i_sample])
                                gradients[i_sample, 0] += label_multiplier * alpha_i_k_y_i * grad_kernel

                            for not_label_sv in range(support_indices[not_label], support_indices[not_label + 1]):
                                alpha_i_k_y_i = self.classifier.dual_coef_[
                                    label[i_sample] if label[i_sample] < not_label else label[i_sample] - 1,
                                    not_label_sv,
                                ]
                                grad_kernel = self._get_kernel_gradient_sv(not_label_sv, x_preprocessed[i_sample])
                                gradients[i_sample, 0] += label_multiplier * alpha_i_k_y_i * grad_kernel

            else:
                raise TypeError("Unrecognized type for argument `label` with type " + str(type(label)))

            gradients = self._apply_preprocessing_gradient(x, gradients * sign_multiplier)

        elif isinstance(self.classifier, LinearSVC):
            if label is None:
                gradients = np.zeros((x_preprocessed.shape[0], self.nb_classes(), x_preprocessed.shape[1]))

                for i in range(self.nb_classes()):
                    for i_sample in range(num_samples):
                        if self.nb_classes() == 2:
                            gradients[i_sample, i] = self.classifier.coef_[0] * (2 * i - 1)
                        else:
                            gradients[i_sample, i] = self.classifier.coef_[i]

            elif isinstance(label, (int, np.integer)):
                gradients = np.zeros((x_preprocessed.shape[0], 1, x_preprocessed.shape[1]))

                for i_sample in range(num_samples):
                    if self.nb_classes() == 2:
                        gradients[i_sample, 0] = self.classifier.coef_[0] * (2 * label - 1)
                    else:
                        gradients[i_sample, 0] = self.classifier.coef_[label]

            elif (
                (isinstance(label, list) and len(label) == num_samples)
                or isinstance(label, np.ndarray)
                and label.shape == (num_samples,)
            ):
                gradients = np.zeros((x_preprocessed.shape[0], 1, x_preprocessed.shape[1]))

                for i_sample in range(num_samples):
                    if self.nb_classes() == 2:
                        gradients[i_sample, 0] = self.classifier.coef_[0] * (2 * label[i_sample] - 1)
                    else:
                        gradients[i_sample, 0] = self.classifier.coef_[label[i_sample]]

            else:
                raise TypeError("Unrecognized type for argument `label` with type " + str(type(label)))

            gradients = self._apply_preprocessing_gradient(x, gradients)

        return gradients

    def _kernel_grad(self, sv, x_sample):
        """
        Applies the kernel gradient to a support vector.

        :param sv: A support vector.
        :type sv: `np.ndarray`
        :param x_sample: The sample the gradient is taken with respect to.
        :type x_sample: `np.ndarray`
        :return: the kernel gradient.
        :rtype: `np.ndarray`
        """
        # pylint: disable=W0212
        if self.classifier.kernel == "linear":
            grad = sv
        elif self.classifier.kernel == "poly":
            grad = (
                self.classifier.degree
                * (self.classifier._gamma * np.sum(x_sample * sv) + self.classifier.coef0) ** (self.classifier.degree - 1)
                * sv
            )
        elif self.classifier.kernel == "rbf":
            grad = (
                2
                * self.classifier._gamma
                * (-1)
                * np.exp(-self.classifier._gamma * np.linalg.norm(x_sample - sv, ord=2))
                * (x_sample - sv)
            )
        elif self.classifier.kernel == "sigmoid":
            raise NotImplementedError
        else:
            raise NotImplementedError("Loss gradients for kernel '{}' are not implemented.".format(self.classifier.kernel))
        return grad

    def _get_kernel_gradient_sv(self, i_sv, x_sample):
        """
        Applies the kernel gradient to all of a model's support vectors.

        :param i_sv: A support vector index.
        :type i_sv: `int`
        :param x_sample: A sample vector.
        :type x_sample: `np.ndarray`
        :return: The kernelized product of the vectors.
        :rtype: `np.ndarray`
        """
        x_i = self.classifier.support_vectors_[i_sv, :]
        return self._kernel_grad(x_i, x_sample)

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.
        Following equation (1) with lambda=0.

        | Paper link: https://pralab.diee.unica.it/sites/default/files/biggio14-svm-chapter.pdf

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        # pylint: disable=E0001
        from sklearn.svm import SVC, LinearSVC

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        num_samples, _ = x_preprocessed.shape
        gradients = np.zeros_like(x_preprocessed)

        y_index = np.argmax(y_preprocessed, axis=1)

        if isinstance(self.classifier, SVC):

            if self.classifier.fit_status_:
                raise AssertionError("Model has not been fitted correctly.")

            if y_preprocessed.shape[1] == 2:
                sign_multiplier = 1
            else:
                sign_multiplier = -1

            i_not_label_i = None
            label_multiplier = None

            support_indices = [0] + list(np.cumsum(self.classifier.n_support_))

            for i_sample in range(num_samples):

                i_label = y_index[i_sample]

                for i_not_label in range(self.nb_classes()):

                    if i_label != i_not_label:

                        if i_not_label < i_label:
                            i_not_label_i = i_not_label
                            label_multiplier = -1
                        elif i_not_label > i_label:
                            i_not_label_i = i_not_label - 1
                            label_multiplier = 1

                        for i_label_sv in range(support_indices[i_label], support_indices[i_label + 1]):
                            alpha_i_k_y_i = self.classifier.dual_coef_[i_not_label_i, i_label_sv] * label_multiplier
                            grad_kernel = self._get_kernel_gradient_sv(i_label_sv, x_preprocessed[i_sample])
                            gradients[i_sample, :] += sign_multiplier * alpha_i_k_y_i * grad_kernel

                        for i_not_label_sv in range(support_indices[i_not_label], support_indices[i_not_label + 1]):
                            alpha_i_k_y_i = self.classifier.dual_coef_[i_not_label_i, i_not_label_sv] * label_multiplier
                            grad_kernel = self._get_kernel_gradient_sv(i_not_label_sv, x_preprocessed[i_sample])
                            gradients[i_sample, :] += sign_multiplier * alpha_i_k_y_i * grad_kernel

        elif isinstance(self.classifier, LinearSVC):

            for i_sample in range(num_samples):

                i_label = y_index[i_sample]
                if self.nb_classes() == 2:
                    i_label_i = 0
                    if i_label == 0:
                        label_multiplier = 1
                    elif i_label == 1:
                        label_multiplier = -1
                    else:
                        raise ValueError("Label index not recognized because it is not 0 or 1.")
                else:
                    i_label_i = i_label
                    label_multiplier = -1

                gradients[i_sample] = label_multiplier * self.classifier.coef_[i_label_i]
        else:
            raise TypeError("Model not recognized.")

        gradients = self._apply_preprocessing_gradient(x, gradients)

        return gradients

    def _kernel_func(self):
        """
        Return the function for the kernel of this SVM.

        :return: A callable kernel function.
        :rtype: `str` or callable
        """
        # pylint: disable=E0001
        from sklearn.svm import SVC, LinearSVC
        from sklearn.metrics.pairwise import polynomial_kernel, linear_kernel, rbf_kernel

        if isinstance(self.classifier, LinearSVC):
            kernel = "linear"
        elif isinstance(self.classifier, SVC):
            kernel = self.classifier.kernel
        else:
            raise NotImplementedError("SVM model not yet supported.")

        if kernel == "linear":
            kernel_func = linear_kernel
        elif kernel == "poly":
            kernel_func = polynomial_kernel
        elif kernel == "rbf":
            kernel_func = rbf_kernel
        elif callable(kernel):
            kernel_func = kernel
        else:
            raise NotImplementedError("Kernel '{}' not yet supported.".format(kernel))

        return kernel_func

    def q_submatrix(self, rows, cols):
        """
        Returns the q submatrix of this SVM indexed by the arrays at rows and columns.

        :param rows: the row vectors.
        :type rows: `np.ndarray`
        :param cols: the column vectors.
        :type cols: `np.ndarray`
        :return: a submatrix of Q.
        :rtype: `np.ndarray`
        """
        submatrix_shape = (rows.shape[0], cols.shape[0])
        y_row = self.classifier.predict(rows)
        y_col = self.classifier.predict(cols)
        y_row[y_row == 0] = -1
        y_col[y_col == 0] = -1
        q_rc = np.zeros(submatrix_shape)
        for row in range(q_rc.shape[0]):
            for col in range(q_rc.shape[1]):
                q_rc[row][col] = self._kernel([rows[row]], [cols[col]])[0][0] * y_row[row] * y_col[col]

        return q_rc

    def predict(self, x, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        # pylint: disable=E0001
        from sklearn.svm import SVC

        # Apply transformation
        _num_examples = x.shape[0]
        if len(x.shape) <= 2:
            x = x.reshape((-1, self._image_shape[0], self._image_shape[1], self._image_shape[2]))
        x_preprocessed = transform(x, self._trans_configs)
        _input_shape = [i for i in self._input_shape]
        _input_shape = tuple([_num_examples] + _input_shape)
        x_preprocessed = x_preprocessed.reshape(_input_shape)

        # Apply defences
        x_preprocessed, _ = self._apply_preprocessing(x_preprocessed, y=None, fit=False)

        if isinstance(self.classifier, SVC) and self.classifier.probability:
            y_pred = self.classifier.predict_proba(X=x_preprocessed)
        else:
            y_pred_label = self.classifier.predict(X=x_preprocessed)
            targets = np.array(y_pred_label).reshape(-1)
            one_hot_targets = np.eye(self.nb_classes())[targets]
            y_pred = one_hot_targets

        # increase the number of model queries
        self._num_queries += x.shape[0]

        return y_pred

    def reset_model_queries(self):
        """
        Set the number of model queries to 0
        """
        self._num_queries = 0

    @property
    def queries(self):
        """
        Return the number of model queries
        """
        return self._num_queries

    def nb_classes(self):
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        :rtype: `int` or `None`
        """
        if hasattr(self.classifier, "classes_"):
            _nb_classes = len(self.classifier.classes_)
        else:
            _nb_classes = None
        return _nb_classes


ScikitlearnLinearSVC = ScikitlearnSVC
