"""
Define models and implement related operations.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.platform import flags
from tensorflow.python.keras import layers, models, optimizers, regularizers
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from config import *
import data
import file
from plot import plotTrainingResult

FLAGS = flags.FLAGS

def create_model(dataset, input_shape, nb_classes):
  if (dataset == DATA.mnist):
    MODEL.set_dataset(DATA.mnist)
    MODEL.set_learning_rate(0.01)
    MODEL.set_batch_size(128)
    MODEL.set_epochs(10)
    return cnn_mnist(input_shape=input_shape, nb_classes=nb_classes)
  elif (dataset == DATA.fation_mnist):
    MODEL.set_dataset(DATA.fation_mnist)
    MODEL.set_learning_rate(0.01)
    MODEL.set_batch_size(128)
    MODEL.set_epochs(50)
    return cnn_mnist(input_shape, nb_classes)
  elif (dataset == DATA.cifar_10):
    MODEL.set_dataset(DATA.cifar_10)
    MODEL.set_learning_rate(0.01)
    MODEL.set_batch_size(64)
    MODEL.set_epochs(100)
    return cnn_cifar(input_shape, nb_classes)

def cnn_cifar(input_shape, nb_classes):
  """
  a cnn for cifar
  :param input_shape:
  :param nb_classes:
  :return:
  """
  MODEL.ARCHITECTURE = 'cnn'
  weight_decay = 1e-4

  struct = [
    layers.Conv2D(32, (3, 3), padding='same',
                  kernel_regularizer=regularizers.l2(weight_decay),
                  input_shape=input_shape),
    layers.Activation('elu'),
    layers.BatchNormalization(),

    layers.Conv2D(32, (3, 3), padding='same',
                  kernel_regularizer=regularizers.l2(weight_decay)),
    layers.Activation('elu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3, 3), padding='same',
                  kernel_regularizer=regularizers.l2(weight_decay)),
    layers.Activation('elu'),
    layers.BatchNormalization(),

    layers.Conv2D(64, (3, 3), padding='same',
                  kernel_regularizer=regularizers.l2(weight_decay)),
    layers.Activation('elu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3, 3), padding='same',
                  kernel_regularizer=regularizers.l2(weight_decay)),
    layers.Activation('elu'),
    layers.BatchNormalization(),

    layers.Conv2D(128, (3, 3), padding='same',
                  kernel_regularizer=regularizers.l2(weight_decay)),
    layers.Activation('elu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(nb_classes, activation='softmax')
  ]

  model = models.Sequential()
  for layer in struct:
    model.add(layer)

  if MODE.DEBUG:
    print(model.summary())
  return model

def cnn_cifar10(input_shape, nb_classes):
  """
  a cnn for cifar
  :param input_shape:
  :param nb_classes:
  :return:
  """
  MODEL.ARCHITECTURE = 'cnn'

  struct = [
    layers.Conv2D(32, (3, 3), activation='relu',
                  kernel_initializer='he_uniform',
                  padding='same',
                  input_shape=input_shape),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu',
                  kernel_initializer='he_uniform',
                  padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(rate=0.2),

    layers.Conv2D(64, (3, 3), activation='relu',
                  kernel_initializer='he_uniform',
                  padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu',
                  kernel_initializer='he_uniform',
                  padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(rate=0.3),

    layers.Conv2D(128, (3, 3), activation='relu',
                  kernel_initializer='he_uniform',
                  padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu',
                  kernel_initializer='he_uniform',
                  padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(rate=0.4),

    layers.Flatten(),
    layers.Dense(128, activation='relu',
                 kernel_initializer='he_uniform'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.5),
    layers.Dense(nb_classes, activation='softmax')
  ]

  model = models.Sequential()
  for layer in struct:
    model.add(layer)

  return model

def cnn_mnist(input_shape=(28, 28, 1), nb_classes=10, logits=False, input_placeholder=None):
  """
  Defines a CNN model using Keras sequential model
  :param logits: If set to False, returns a Keras model, otherwise will also
                  return logits tensor
  :param input_placeholder: The TensorFlow tensor for the input
                  (needed if returning logits)
                  ("ph" stands for placeholder but it need not actually be a
                  placeholder)
  :param img_rows: number of row in the image
  :param img_cols: number of columns in the image
  :param channels: number of color channels (e.g., 1 for MNIST)
  :param nb_filters: number of convolutional filters per layer
  :param nb_classes: the number of output classes
  :return:
  """

  MODEL.ARCHITECTURE = 'cnn'
  img_rows, img_cols, nb_channels = input_shape

  # Define the layers successively (convolution layers are version dependent)
  if tf.keras.backend.image_data_format() == 'channels_first':
    input_shape = (nb_channels, img_rows, img_cols)

  struct = [
    layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3)),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(64 * 64),
    layers.Dropout(rate=0.4),
    layers.Dense(nb_classes),
    layers.Activation('softmax')
  ]
  # construct the cnn model
  model = models.Sequential()
  for layer in struct:
    model.add(layer)

  # logits_tensor = None
  # if logits:
  #   logits_tensor = model(input_placeholder)
  # model.add(layers.Activation('softmax'))
  #
  # if logits:
  #   return model, logits_tensor
  # else:
  #   return model
  return model

# --------------------------------------------
# OPERATIONS
# --------------------------------------------
def train_model(model, dataset, model_name, need_augment=False, **kwargs):
  (X_train, Y_train), _ = data.load_data(dataset)
  return train(model, X_train, Y_train, model_name, need_augment, **kwargs)

def train(model, X, Y, model_name, need_augment=False, **kwargs):
  """
  Train a model on given dataset.
  :param model: the model to train
  :param dataset: the name of the dataset
  :param need_augment: a flag - whether we need to augment the data before training the model
  :param kwargs: for optimizer, loss function, and metrics
  :return: the trained model
  """
  print('INFO: model name: {}'.format(model_name))
  learning_rate = 0.001
  validation_rate = 0.2

  prefix, dataset, architect, trans_type = model_name.split('-')

  optimizer = kwargs.get('optimizer', keras.optimizers.Adam(lr=learning_rate))
  loss_func = kwargs.get('loss', keras.losses.categorical_crossentropy)
  metrics = kwargs.get('metrics', 'default')

  print('INFO: compiler')
  print('>>> optimizer: {}'.format(optimizer))
  print('>>> loss function: {}'.format(loss_func))
  print('>>> metrics: {}'.format(metrics))

  nb_examples, img_rows, img_cols, nb_channels = X.shape

  if (DATA.cifar_10 == dataset):
    """
    mean-std normalization
    """
    X = data.normalize(X)

  nb_training = int(nb_examples * (1. - validation_rate))
  train_examples = X[:nb_training]
  train_labels = Y[:nb_training]
  val_examples = X[nb_training:]
  val_labels = Y[nb_training:]

  """
  augment data
  """
  datagen = None
  if (DATA.cifar_10 == dataset):
    # normalize data (has been handled when loading the data)
    # data augmentation
    datagen = ImageDataGenerator(
      rotation_range=15,
      width_shift_range=0.1,
      height_shift_range=0.1,
      horizontal_flip=True,
    )
    datagen.fit(train_examples)

  """
  compile data
  """

  if ('default' == metrics):
    model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])
  else:
    model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy', metrics])

  """
  train the model
  """
  if (DATA.cifar_10 == dataset):
    history = model.fit_generator(datagen.flow(train_examples, train_labels, batch_size=MODEL.BATCH_SIZE),
                                  steps_per_epoch=nb_training // MODEL.BATCH_SIZE, epochs=MODEL.EPOCHS,
                                  verbose=2, validation_data=(val_examples, val_labels),
                                  callbacks=[LearningRateScheduler(lr_schedule)])

  else: # do not need to augment data
    history = model.fit(train_examples, train_labels,
                        batch_size=MODEL.BATCH_SIZE,
                        epochs=MODEL.EPOCHS, verbose=2,
                        validation_data=(val_examples, val_labels))

  """
  evaluate the model
  """
  scores_train = model.evaluate(train_examples, train_labels, batch_size=128, verbose=0)
  scores_val = model.evaluate(val_examples, val_labels, batch_size=128, verbose=0)

  """
  report
  """
  print('\t\t\t loss, \tacc, \tadv_acc')
  print('Evaluation score on training set: {}'.format(scores_train))
  print('Evaluation score on validation set: {}'.format(scores_val))
  file_name = 'checkpoints-{}-{}-{}.csv'.format(dataset, architect, trans_type)
  file.dict2csv(history.history, '{}/{}'.format(PATH.RESULTS, file_name))
  plotTrainingResult(history, model_name)

  return model

def train_and_save(model_name, X, Y, validation_rate=0.2, need_augment=False):
  """
  Train a model over given training set, then
  save the trained model as given name.
  :param model_name: the name to save the model as
  :param X: training examples.
  :param Y: corresponding desired labels.
  :param need_augment: a flag whether to perform data augmentation before training.
  """

  prefix, dataset, architect, trans_type = model_name.split('-')
  nb_examples, img_rows, img_cols, nb_channels = X.shape

  nb_validation = int(nb_examples * validation_rate)
  nb_training = nb_examples - nb_validation
  train_samples = X[:nb_training]
  train_labels = Y[:nb_training]
  val_sample = X[nb_training:]
  val_labels = Y[nb_training:]
  input_shape = (img_rows, img_cols, nb_channels)
  nb_classes = int(Y.shape[1])

  print('input_shape: {}; nb_classes: {}'.format(input_shape, nb_classes))
  print('{} training sample; {} validation samples.'.format(nb_training, nb_validation))

  # get corresponding model
  model = create_model(dataset, input_shape=input_shape, nb_classes=nb_classes)
  history = []
  scores = []
  if (need_augment):
    # normalize samples
    train_samples = data.normalize(train_samples)
    val_sample = data.normalize(val_sample)
    # data augmentation
    datagen = ImageDataGenerator(
      rotation_range=15,
      width_shift_range=0.1,
      height_shift_range=0.1,
      horizontal_flip=True,
    )
    datagen.fit(train_samples)
    # define a optimizer
    opt_rms = optimizers.RMSprop(lr=0.001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
    # perform training
    with tf.device('/device:GPU:0'): # to run in google colab
      print("Found GPU:0")
      # train
      history = model.fit_generator(datagen.flow(train_samples, train_labels, batch_size=MODEL.BATCH_SIZE),
                                    steps_per_epoch= nb_training//MODEL.BATCH_SIZE, epochs=MODEL.EPOCHS,
                                    verbose=2, validation_data=(val_sample, val_labels),
                                    callbacks=[LearningRateScheduler(lr_schedule)])
      # test, this will be run with GPU
      # verbose: integer. 0 = silent; 1 = progress bar; 2 = one line per epoch
      # train the model silently
      scores = model.evaluate(val_sample, val_labels, batch_size=128, verbose=0)
  else:
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    with tf.device('/device:GPU:0'):
      # train
      history = model.fit(train_samples, train_labels,
                          epochs=MODEL.EPOCHS, batch_size=MODEL.BATCH_SIZE,
                          shuffle=True, verbose=1, validation_data=(val_sample, val_labels))
      # test
      # verbose: integer. 0 = silent; 1 = progress bar; 2 = one line per epoch
      # train the model silently
      scores = model.evaluate(val_sample, val_labels, batch_size=128, verbose=0)

  # save the trained model
  model.save('{}/{}.h5'.format(PATH.MODEL, model_name))
  keras.models.save_model(model, '{}/{}_2.h5'.format(PATH.MODEL, model_name))
  # report
  print('Trained model has been saved to data/{}'.format(model_name))
  print('Test accuracy: {:.4f}; loss: {:.4f}'.format(scores[1], scores[0]))
  file_name = 'CheckPoint-{}-{}-{}.csv'.format(dataset, architect, trans_type)
  file.dict2csv(history.history, '{}/{}'.format(PATH.RESULTS, file_name))
  plotTrainingResult(history, model_name)
  # delete the model after it's been saved.
  del model

def lr_schedule(epoch):
  """
  schedule a dynamic learning rate
  :param epoch:
  :return:
  """
  lr = 0.001
  if (epoch > 75):
    lr = 0.0005
  elif (epoch > 100):
    lr = 0.0003
  return lr

def evaluate_model(model, X, Y):
  """
  Evaluate the model on given test set.
  :param model: the name of the model to evaluate
  :param X: test set
  :param Y:
  :return: test accuracy, average confidences on correctly classified samples
          and misclassified samples respectively.
  """
  nb_corrections = 0
  nb_examples = Y.shape[0]

  conf_correct = 0.
  conf_misclassified = 0.

  pred_probs = model.predict(X, batch_size=128)

  # iterate over test set
  for pred_prob, true_prob in zip(pred_probs, Y):
    pred_label = np.argmax(pred_prob)
    true_label = np.argmax(true_prob)

    if (pred_label == true_label):
      nb_corrections += 1
      conf_correct += np.max(pred_prob)
    else:
      conf_misclassified += np.max(pred_prob)

  # test accuracy
  _, test_acc = model.evaluate(X, Y, verbose=0)
  # average confidences
  ave_conf_correct = conf_correct / nb_corrections
  ave_conf_miss = conf_misclassified / (nb_examples - nb_corrections)

  return test_acc, ave_conf_correct, ave_conf_miss

def save_to_json(model, model_name):
  file_name = '{}/{}.json'.format(PATH.MODEL, model_name)
  model_json = model.to_json()
  with open(file_name, 'w') as json_file:
    json_file.write(model_json)

  model.save_weights('{}/weights_{}.h5'.format(PATH.MODEL, model_name))

def load_from_json(model_name,
                   optimizer=keras.optimizers.RMSprop(lr=0.001, decay=1e-6)):
  dataset = model_name.split('-')[1]
  json_file = open('{}/{}/{}.json'.format(PATH.MODEL, dataset, model_name), 'r')
  loaded_model_json = json_file.read()
  json_file.close()

  model = keras.models.model_from_json(loaded_model_json)
  model.load_weights('{}/{}/weights_{}.h5'.format(PATH.MODEL, dataset, model_name))

  model.compile(
    optimizer=optimizer,
    loss=keras.losses.categorical_crossentropy,
    metrics=['accuracy']
  )

  return model


def main():
  from data import load_data

  model = models.load_model('data/models/mnist/model-mnist-cnn-clean.h5')
  _, (X, Y) = load_data(DATA.mnist)
  print(model.evaluate(X, Y, verbose=1))

if __name__ == "__main__":
  main()
