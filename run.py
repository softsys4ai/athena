import os
import sys
import time

from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

import numpy as np
import cv2

from config import *
from util import *
from transformation import transform_images

def usage():
    print("===================================================================")
    print("python <this script> samplesDir rootDir modelsDir numOfSamples kFold")
    print("Note:")
    print("\trootDir: directory for hold each run of evaluation result.")
    print("\t         For each run of evaluation, a directory named by a time stamp")
    print("\t         will be created inside 'rootDir'")
    print("===================================================================")

if len(sys.argv) != 6:
    usage()
    exit(1)


samplesDir = sys.argv[1]
rootDir = sys.argv[2]
modelsDir = sys.argv[3]
numOfSamples  = int(sys.argv[4])
kFold = int(sys.argv[5])

# Basic parameters for k-fold experiment setup
#rootDir = "experiment"
timeStamp=time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
experimentRootDir=os.path.join(rootDir,timeStamp)
createDirSafely(experimentRootDir)

#kFold = 5
isKFolderUponTestSet=True
datasetName = DATA.DATASET
architecture = MODEL.TYPE
#modelsDir = "models"
#numOfSamples  = 10000
#samplesDir     = "samples_0808"
numOfClasses = DATA.NB_CLASSES

#EPS = ATTACK.FGSM_EPS
attackApproach = ATTACK.FGSM
AETypes = []
EPS = [0.25, 0.3, 0.5, 0.1, 0.05, 0.01, 0.005]
for eps in EPS:
    epsInt = int(1000*eps)
    AETypes.append(attackApproach+"_eps"+str(epsInt))
    if len(AETypes) >= 1:
        break

numOfAETypes = len(AETypes)
sampleTypes =["BS"]
sampleTypes.extend(AETypes)

targetModelName = "clean"
transformConfig = TRANSFORMATION()
transformationList = transformConfig.supported_types() 

# Create fold directories for evaluation
foldDirs = createKFoldDirs(experimentRootDir, kFold)
predictionResultDir = os.path.join(experimentRootDir, "prediction_result")


# Prediction
kFoldPredictionSetup(
        experimentRootDir,
        kFold,
        predictionResultDir,
        datasetName,
        architecture,
        numOfClasses,
        targetModelName,
        modelsDir,
        samplesDir,
        numOfSamples,
        AETypes,
        transformationList,
        isKFolderUponTestSet)


# Evaluation: training and testing
oneFoldAmount = numOfSamples//kFold
print(oneFoldAmount)
predProbBS  = np.load(os.path.join(predictionResultDir, "BS/predProb.npy"))
predProbBS  = predProbBS[1:]
predLogitBS = np.load(os.path.join(predictionResultDir, "BS/predLogit.npy"))
predLogitBS = predLogitBS[1:]
labels      = np.load(os.path.join(predictionResultDir, "labels.npy"))
predLCBS    = np.zeros((predProbBS.shape[0], predProbBS.shape[1], 2))
predLCBS[:, :, 0] = np.argmax(predProbBS, axis=2)
predLCBS[:, :, 1] = np.max(predProbBS, axis=2)
labelsBS    = labels

for foldIdx in range(1, 1+kFold):
    # foldIndices: testing samples' indices for this fold
    print("fold " + str(foldIdx))
    if foldIdx != kFold:
        testingIndices  = np.array(range(
            (foldIdx-1)*oneFoldAmount,
            foldIdx*oneFoldAmount))
        trainingIndices = np.hstack((
            np.array(range(0, (foldIdx-1)*oneFoldAmount)),
            np.array(range(foldIdx*oneFoldAmount, numOfSamples))))
        trainingIndices = trainingIndices.astype(int)
    else:
        testingIndices  = np.array(range((foldIdx-1)*oneFoldAmount, numOfSamples))
        trainingIndices = np.array(range(0, (foldIdx-1)*oneFoldAmount))


    if not isKFolderUponTestSet:
        tempIndices = testingIndices
        testingIndices = trainingIndices
        trainingIndices = tempIndices

    foldDir = foldDirs[foldIdx-1]  
    for AETypeIdx in range(numOfAETypes):

        AEType = AETypes[AETypeIdx]
        curExprDir = os.path.join(foldDir, AEType)
        curPredictionResultDir = os.path.join(predictionResultDir, AEType)
        createDirSafely(curExprDir)

        print("Evaluating AE type: "+AEType)
        predProbAE  = np.load(os.path.join(curPredictionResultDir, "predProb.npy"))
        predLogitAE = np.load(os.path.join(curPredictionResultDir, "predLogit.npy"))


        predProbAETr   = predProbAE[:, trainingIndices, :]
        predLogitsAETr = predLogitAE[:, trainingIndices, :]
        labelsTr       = labels[trainingIndices]

        predProbAETe   = predProbAE[1:, testingIndices, :]
        predLogitsAETe = predLogitAE[1:, testingIndices, :]
        labelsTe       = labels[testingIndices]


        # Clustering-and-voting based defenses
        # 2: AE-accuracy, AE-time cost
        #testResults : (numOfCVDefenses, 2)
        print("\t==== clustering and voting based defenses ====")
        AETestResultsCAV, BSTestResultCAV = clusteringDefensesEvaluation(
            curExprDir,
            predProbAETr,
            labelsTr,
            predProbAETe,
            labelsTe,
            predLCBS,
            labelsBS)

        # Weighted-confidence based defenses
        print("\t==== weighted-confidence based defenses")
        AETestResultsWC, BSTestResultsWC = weightedConfDefenseEvaluation(
                curExprDir,
                predProbAETr,
                predLogitsAETr,
                labelsTr,
                predProbAETe,
                predLogitsAETe,
                labelsTe,
                transformationList,
                predProbBS,
                predLogitBS,
                labelsBS)

postAnalysis(
        experimentRootDir,
        kFold,
        predictionResultDir,
        foldDirs,
        sampleTypes)


