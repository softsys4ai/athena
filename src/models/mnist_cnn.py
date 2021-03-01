"""
Model structure for MNIST
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import argparse
import os

import keras

from utils.data import load_mnist
import utils.file as file
from models.image_processor import transform

# -------------------------
# Network Architecture
# -------------------------
def lenet(input_shape=(28, 28, 1), num_classes=10, detector=False):
    if detector:
        num_classes = 2

    struct = [
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dense(64 * 64),
        keras.layers.Dropout(rate=0.4),
        keras.layers.Dense(num_classes, activation='softmax'),
    ]

    model = keras.models.Sequential()
    for layer in struct:
        model.add(layer)

    print(model.summary())
    return model


# -------------------------
# Train a cnn classifier
# -------------------------
def train(trainset, testset, detector=False,
          trans_configs=None, training_configs=None, model_configs=None):
    if trainset is None:
        raise ValueError("Training data cannot be None.")

    # default transformation setting
    if trans_configs is None:
        # by default, train on the clean data
        trans_configs = {
            "type": "clean",
            "subtype": "",
            "id": 0,
            "description": "clean",
        }

    # default training config
    if training_configs is None:
        training_configs = {
            "learning_rate": 0.001,
            "validation_rate": 0.2,
            "optimizer": keras.optimizers.Adam(lr=0.001),
            "loss": keras.losses.categorical_crossentropy,
            "metrics": ["accuracy"],
        }

    # default model configuration
    if model_configs is None:
        model_configs = {
            "model_dir": "model",
            "model_name": "model-cnn-{}.h5".format(trans_configs.get("description")),
        }

    # get network
    model = lenet(detector=detector)

    # prepare dataset
    X_train, Y_train = trainset

    # apply transformation
    X_train = transform(X=X_train, trans_args=trans_configs)

    # prepare to train
    if training_configs.get("batch_size") is None:
        training_configs["batch_size"] = 64
    if training_configs.get("epochs") is None:
        training_configs["epochs"] = 20

    # summary
    print("\n------------------------------")
    print("--- Training Configuration ---")
    print("Transformation: {}".format(trans_configs.get("description")))
    print("Optimizer: {}".format(training_configs.get("optimizer")))
    print("Loss function: {}".format(training_configs.get("loss")))
    print("Metrics: {}".format(training_configs.get("metrics")))
    print("Bach size: {}".format(training_configs.get("batch_size")))
    print("Epochs: {}".format(training_configs.get("epochs")))
    print("------------------------------\n")

    num_examples, img_rows, img_cols, num_channels = X_train.shape

    num_training = int(num_examples * (1 - training_configs.get("validation_rate")))
    train_examples = X_train[:num_training]
    train_labels = Y_train[:num_training]
    val_examples = X_train[num_training:]
    val_labels = Y_train[num_training:]

    # compile model
    model.compile(optimizer=training_configs.get("optimizer"),
                  loss=training_configs.get("loss"),
                  metrics=training_configs.get("metrics"))

    # train
    history = model.fit(train_examples, train_labels,
                        batch_size=training_configs.get("batch_size"),
                        epochs=training_configs.get("epochs"),
                        verbose=1, validation_data=(val_examples, val_labels))

    # save the trained model
    savefile = os.path.join(model_configs.get("model_dir"),
                            model_configs.get("model_name"))
    print("Save the trained model to [{}].".format(savefile))
    model.save(savefile)

    # evaluate model
    scores_train = model.evaluate(train_examples, train_labels, batch_size=128, verbose=0)
    scores_val = model.evaluate(val_examples, val_labels, batch_size=128, verbose=0)

    # evaluate on test data
    if testset:
        X_test, Y_test = testset
        X_test = transform(X=X_test, trans_args=trans_configs)
        scores_test = model.evaluate(X_test, Y_test, batch_size=128, verbose=0)
    else:
        scores_test = "na"

    # report
    print("\n------------------------------")
    print("--- Model Evaluation ---")
    print("dataset \t\t\t loss  \t acc (BS/AE)")
    print("training set: {}".format(scores_train))
    print("validation set: {}".format(scores_val))
    print("test set: {}".format(scores_test))
    print("------------------------------\n")

    # save training checkpoints
    log_dir = training_configs.get("checkpoint_folder", "checkpoints")
    savefile = savefile.replace(".h5", "")
    file.dump_to_csv(dictionary=history.history,
                     file_name="{}/checkpoints-{}.csv".format(log_dir, savefile))
