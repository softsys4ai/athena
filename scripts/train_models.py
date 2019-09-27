"""
Script to train models.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
import models
from utils.config import *
from data import load_data
from transformation import transform, composite_transforms


def train_model_batch(dataset):
    """
    Train models on specific dataset.
    :param dataset:
    """
    for trans in TRANSFORMATION.supported_types():
        # for trans in TRANSFORMATION.AFFINE_TRANS:
        # train a model per type of transformation
        train_model(dataset, trans)

def train_model(dataset, transform_type):
    """
    Train specific model on given dataset.
    :param dataset:
    :param transform_type:
    """
    print('Training model ({}) on {}...'.format(transform_type, dataset))
    (X_train, Y_train), (X_test, Y_test) = load_data(dataset)
    if transform_type != TRANSFORMATION.clean:
        # transform images on demand.
        X_train = transform(X_train, transform_type)

    model_name = 'model-{}-cnn-{}'.format(dataset, transform_type)
    require_preprocess = False
    if (dataset == DATA.cifar_10):
        require_preprocess = True

    # train
    model = models.train(model_name, X_train, Y_train, require_preprocess)
    # save to disk
    models.save_model(model, model_name)
    # evaluate the new model
    loaded_model = models.load_model(model_name)
    scores = loaded_model.evaluate(X_test, Y_test, verbose=2)
    print('*** Evaluating the new model: {}'.format(scores))
    del loaded_model

def train_composition(dataset, transformation_list):
    """
    Train a model on dataset on which a sequence of transformations applied
    :param dataset: the original dataset
    :param transformation_list: the sequence of transformations
    :return:
    """
    # Apply a sequence of transformations
    (X_train, Y_train), (X_test, Y_test) = load_data(dataset)
    X_trans = composite_transforms(X_train, transformation_list)
    # Train a model and save
    model_name = 'model-{}-cnn-{}'.format(dataset, 'composition')
    require_preprocess = (dataset == DATA.cifar_10)

    # train(model, X, Y, model_name, need_augment=False, **kwargs)
    input_shape = X_trans.shape[1:4]
    nb_classes = Y_train.shape[1]
    model = models.create_model(dataset, input_shape, nb_classes)
    models.train(model, X_trans, Y_train, model_name, require_preprocess)
    # save to disk
    models.save_model(model, model_name)
    # evaluate the new model
    loaded_model = models.load_model(model_name)
    X_trans = composite_transforms(X_test, transformation_list)
    scores = loaded_model.evaluate(X_trans, Y_test, verbose=2)
    print('*** Evaluating the new model: {}'.format(scores))
    del loaded_model


def train_models_with_newLabels(
        dataset_name,
        AE_type_tag,
        defense_tag,
        transform_type,
        num_of_samples,
        X,
        Y,
        validation_rate=0.2,
        need_argument=False):
    print('Training model ({}) on {} {} new labels collected from ensemble ({}) built upon {}...'.format(transform_type,
                                                                                                         num_of_samples,
                                                                                                         dataset_name,
                                                                                                         defense_tag,
                                                                                                         AE_type_tag))

    if transform_type != TRANSFORMATION.clean:
        # transform images on demand.
        X = transform(X, transform_type)

    model_name = 'model-{}-cnn-{}-{}-{}-{}'.format(
        dataset_name,
        transform_type,
        AE_type_tag,
        defense_tag,
        num_of_samples)

    models.train_and_save(
        model_name,
        X,
        Y,
        validation_rate,
        need_argument)


"""
For testing
"""
def main(dataset, trans_type=TRANSFORMATION.clean, batch=False):
    if len(trans_type) > 1:
        train_composition(dataset, trans_type)
    elif batch:
        train_model_batch(dataset)
    else:
        train_model(dataset, trans_type)


if __name__ == "__main__":
    MODE.debug_on()
    # MODE.debug_off()
    compositions = TRANSFORMATION.get_transformation_compositions()

    main(DATA.mnist, compositions[0], batch=True)
