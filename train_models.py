"""
Train models
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
import models
from config import *
from data import load_data
from transformation import transform_images

def train_model_batch(dataset):
    for trans in TRANSFORMATION.supported_types():
        train_model(dataset, trans)

def train_model(dataset, transform_type):
    print('Training model ({}) on {}...'.format(transform_type, dataset))
    (X_train, Y_train), _ = load_data(dataset)
    if transform_type != TRANSFORMATION.clean:
        X_train = transform_images(X_train, transform_type)

    models.train(X_train, Y_train, 'model-{}-cnn-{}.h5'.format(dataset, transform_type))

def main(dataset, trans_type=TRANSFORMATION.clean, batch=False):
    if batch:
        train_model_batch(dataset)
    else:
        train_model(dataset, trans_type)

if __name__ == "__main__":
    main(DATA.mnist, TRANSFORMATION.clean, batch=False)
