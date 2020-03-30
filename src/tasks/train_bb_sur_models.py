import os
import sys
from keras.utils import to_categorical

from tasks import creat_models
from utils.config import *

def train_model(data, model_name):
    X, Y = data
    transformation_type=TRANSFORMATION.clean

    model = creat_models.create_model(DATA.CUR_DATASET_NAME)
    print('Training model [{}]...'.format(model_name))
    model = creat_models.train(model, X, Y, model_name, need_augment=False, is_BB=True)
    print('Saving model...')
    creat_models.save_model(model, model_name)
    print('Done.')

    return model

def main(argv):

    trainDir = argv[0]
    valDir  = argv[1]

    valDataPath = os.path.join(valDir, "validation_set_9k_data.npy")
    valLabelPath = os.path.join(valDir, "validation_set_9k_label.npy")
    trainDataDir = os.path.join(trainDir, "data")

    nClasses=10
    DATA.set_current_dataset_name(DATA.mnist)
    TRANSFORMATION.set_cur_transformation_type(TRANSFORMATION.clean)

    targetNames = ["RD", "MV", "AVEP", "T2MV", "AVEL", "UM"]
    budgets = [10, 50, 100, 500, 1000]

    for targetName in targetNames:
        trainLabelDir = os.path.join(
                os.path.join(trainDir, "label"),
                targetName)

        for budget in budgets:
            print("Targeting {} with a budget {}".format(targetName, budget))

            X = np.load(valDataPath)
            Y = np.load(valLabelPath)
            trainX = np.load(os.path.join(trainDataDir, "budget{}_train_set.npy".format(budget)))
            trainY = np.load(os.path.join(trainLabelDir, "{}.npy".format(budget)))

            X = np.vstack((X, trainX)) # four dimentions. concatenate along the first dimension, which represents the sample ID.
            Y = np.hstack((Y, trainY)) # one dimension

            Y = to_categorical(Y)

            modelNamePrefix = "model_"+str(budget)+"Samples_"+targetName
            try:
                modelName = '{}-mnist-cnn-clean'.format(modelNamePrefix)

                model = train_model((X, Y), modelName)
                print('Created {}'.format(modelName))
                print('')
            except (FileNotFoundError, OSError) as e:
                print(e)
                print('')
                continue

            del model

if __name__ == '__main__':
    main(sys.argv[1:])

