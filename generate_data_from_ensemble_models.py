import os
import sys
import time

import numpy as np

from config import *
from util import *

def usage():
    print("=======================================================================================")
    print("python <this script> inputImagesFP modelsDir experimentRootDir datasetName numOfClasses")
    print("=======================================================================================")

if len(sys.argv) != 6:
    usage()
    exit(1)


inputImagesFP = sys.argv[1]
modelsDir = sys.argv[2]
experimentRootDir = sys.argv[3]
datasetName = sys.argv[4]
numOfClasses = int(sys.argv[5])

# Basic parameters for k-fold experiment setup
architecture = MODEL.ARCHITECTURE
AETypes = ATTACK.get_AETypes()
numOfAETypes = len(AETypes)

targetModelName = "clean"
transformConfig = TRANSFORMATION()
transformationList = transformConfig.supported_types() 
transformationList = transformationList[1:] # exclude the 'clean' transformation - no transformation


# Load models and create models to output logits
modelFilenamePrefix = datasetName+"-"+architecture
models, logitsModels = loadModels(modelsDir, modelFilenamePrefix, transformationList)


# Prediction
inputSamples = np.load(inputImagesFP)
numOfSamples = inputSamples.shape[0]
numOfTrans = len(transformationList)
predShape = (numOfTrans, numOfSamples, numOfClasses)
predProb   = np.zeros(predShape)
predLogits = np.zeros(predShape)

for modelID in range(numOfTrans):
    transformType = transformationList[modelID]
    print("\t\t\t [{}] prediction on {} model".format(modelID, transformType))
    # Transformation cost
    tranSamples = transform_images(inputSamples, transformType)

    # model prediction cost - using probability-based defense
    predProb[modelID, :, :],  _  = prediction(
            tranSamples,
            models[modelID])
    # model prediction cost - using logits-based defense
    predLogits[modelID, :, :], _ = prediction(
            tranSamples,
            logitsModels[modelID])


predProbLC = np.zeros((predProb.shape[0], predProb.shape[1], 2))
predProbLC[:, :, 0] = np.argmax(predProb, axis=2)
predProbLC[:, :, 1] = np.max(predProb, axis=2)

trainModelDir = os.path.join(experimentRootDir, "train_models")
newLabelsDir = os.path.join(experimentRootDir, "newLabels")
createDirSafely(newLabelsDir)
numOfDefenses = numOfCVDefenses+2*numOfWCDefenses
# Test each ensemble model trained by each type of AEs
for AETypeIdx in range(numOfAETypes):
    AEType = AETypes[AETypeIdx]
    curTrainModelDir = os.path.join(trainModelDir, AEType)
    curNewLabelsDir = os.path.join(newLabelsDir, AEType)
    createDirSafely(curNewLabelsDir)

    print("Collecting new labels from the ensemble models built upon "+AEType)

    # accuracy of clustering-and-voting based defenses
    for defenseIdx in range(numOfCVDefenses):
        defenseName = cvDefenseNames[defenseIdx] 
        newLabelsFilename = defenseName+"_newLabels.npy"
        clusters = loadCAVModel(os.path.join(curTrainModelDir, defenseName+".txt"))

        # getting new labels
        votedResults, _ = votingAsDefense(
                predProbLC,
                clusters,
                vsac=cvDefenseNames[defenseIdx],
                measureTC=False)
        newLabels = votedResults[:, 0]
        np.save(os.path.join(curNewLabelsDir, newLabelsFilename), newLabels)


    # accuracy of weithed-confidence based defenses
    for defenseIdx in range(numOfWCDefenses):
        defenseName = wcDefenseNames[defenseIdx]
        for plIdx in range(2):
            wcMatFilename = defenseName+"_EM.npy"
            mIDsFilename  = defenseName+"_modelIDs.npy"
            newLabelsFilename = defenseName+"_newLabels.npy"
            pred = predProb
            if plIdx == 1: # predict logit instead of probability
                wcMatFilename = "LG_" + wcMatFilename
                mIDsFilename  = "LG_" +  mIDsFilename
                newLabelsFilename = "LG_" + newLabelsFilename
                pred = predLogits

            wcMat = np.load(os.path.join(curTrainModelDir, wcMatFilename))
            # ID of transform models: starts from 0.
            mIDs  = np.load(os.path.join(curTrainModelDir, mIDsFilename))
           
            curPred = pred[mIDs] 

            # getting new labels
            predLabels,  _ = wcdefenses(
                    curPred, wcMat, defenseName, measureTC=False)

            newLabels = predLabels
            np.save(os.path.join(curNewLabelsDir, newLabelsFilename), newLabels)

