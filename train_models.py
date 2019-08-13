"""
Script to train models.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
import models
from config import *
from data import load_data
from transformation import transform_images

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
    (X_train, Y_train), _ = load_data(dataset)
    if transform_type != TRANSFORMATION.clean:
        # transform images on demand.
        X_train = transform_images(X_train, transform_type)

    model_name = 'model-{}-cnn-{}'.format(dataset, transform_type)
    require_preprocess = False
    if (dataset == DATA.cifar_10):
        require_preprocess = True

    models.train(model_name, X_train, Y_train, require_preprocess)

"""
For testing
"""
def main(dataset, trans_type=TRANSFORMATION.clean, batch=False):
    if batch:
        train_model_batch(dataset)
    else:
        train_model(dataset, trans_type)

if __name__ == "__main__":
    MODE.debug_on()
    # MODE.debug_off()
    main(DATA.mnist, TRANSFORMATION.affine_both_compress, batch=True)
