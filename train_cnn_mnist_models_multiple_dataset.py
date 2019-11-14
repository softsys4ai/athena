import os
import sys

import models
from utils.config import *

def train_model(data, model_name):
    X, Y = data
    transformation_type=TRANSFORMATION.clean

    model = models.create_model(DATA.CUR_DATASET_NAME)
    print('Training model [{}]...'.format(model_name))
    model = models.train(model, X, Y, model_name)
    print('Saving model...')
    models.save_model(model, model_name)
    print('Done.')

    return model

def main(argv):
    BSDataFP=argv[0]
    newLabelDir=argv[1]
    queriedDataDir=argv[2]

    newLabelFNs = [  "label_EnsembleID0_prob.npy",
                    "label_EnsembleID1_prob.npy",
                    "label_EnsembleID2_prob.npy",
                    "label_EnsembleID3_prob.npy",
                    "label_EnsembleID2_logit.npy"]
    ensembleTags = ["prob0", "prob1", "prob2", "prob3", "logit2"]
    nSamplesList = [50, 100, 500, 1000, 5000]

    nClasses=10
    DATA.set_current_dataset_name(DATA.mnist)
    TRANSFORMATION.set_cur_transformation_type(TRANSFORMATION.clean)

    X = np.load(BSDataFP)
    for newLabelFN, ensembleTag in zip(newLabelFNs, ensembleTags):
        labels = np.load(os.path.join(newLabelDir, newLabelFN))
        inds = np.array(range(len(labels)))
        for nSamples in nSamplesList:
            datasetName = "mnist"+str(nSamples)+"Samples"+ensembleTag
            try:
                model_name = 'model-{}-cnn-clean'.format(datasetName)

                np.random.shuffle(inds)
                selectedLabels = labels[inds[:nSamples]]
                selectedX = X[inds[:nSamples]]
                # saved for generating AEs later
                np.save(os.path.join(queriedDataDir, datasetName+"_data.npy"), selectedX)
                np.save(os.path.join(queriedDataDir, datasetName+"_label.npy"), selectedLabels)

                Y = np.zeros((nSamples, nClasses))
                for sIdx in range(nSamples):
                    Y[sIdx, selectedLabels[sIdx]] = 1

                model = train_model((selectedX, Y), model_name)
                print('')
            except (FileNotFoundError, OSError) as e:
                print(e)
                print('')
                continue

            del model

if __name__ == '__main__':
    main(sys.argv[1:])

