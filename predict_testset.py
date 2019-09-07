import os
import sys
import time

import numpy as np

from config import *
from util import *

def usage():
    print("====================================================================================================================")
    print("python <this script> samplesDir experimentRootDir modelsDir numOfSamples testResultFoldName datasetName numOfClasses sampleType")
    print("====================================================================================================================")

if len(sys.argv) != 9:
    usage()
    exit(1)


samplesDir          = sys.argv[1]
experimentRootDir   = sys.argv[2]
modelsDir           = sys.argv[3]
numOfSamples        = int(sys.argv[4])
testResultFoldName  = sys.argv[5]
datasetName         = sys.argv[6]
numOfClasses        = int(sys.argv[7])
sampleType          = sys.argv[8]

DATA.set_current_dataset_name(datasetName)
AETypes = ATTACK.get_AETypes()

# Basic parameters for k-fold experiment setup
architecture = MODEL.ARCHITECTURE
testDir = os.path.join(experimentRootDir, testResultFoldName)
createDirSafely(testDir)

sampleTypes =["BS"]
sampleTypes.extend(AETypes)
print("[sample types]: {}\n".format(len(sampleTypes)))
print(sampleTypes)

targetModelName = "clean"
transformConfig = TRANSFORMATION()
transformationList = transformConfig.supported_types() 

predictionResultDir = os.path.join(testDir, "prediction_result")
createDirSafely(predictionResultDir)

# Prediction : needs a new prediction function
predictionForTest0(
        predictionResultDir,
        datasetName,
        architecture,
        numOfClasses,
        targetModelName,
        modelsDir,
        samplesDir,
        numOfSamples,
        sampleTypes,
        transformationList)

labels = np.load(os.path.join(samplesDir, "Label-"+datasetName+"-clean.npy"))
labels = np.argmax(labels, axis=1)

for sampleType in sampleTypes:
    predDir = os.path.join(predictionResultDir, sampleType)
    predProb = np.load(os.path.join(predDir, "predProb.npy"))
    modelsAcc = calAccuracyAllSingleModels(labels, predProb)
    with open(os.path.join(predictionResultDir, sampleType+"-accuracy.txt"), "w") as fp:
        for i in range(len(transformationList)):
            fp.write("{}\t{}\t{}\n".format(i, transformationList[i], modelsAcc[i]))


with open(os.path.join(predictionResultDir, "singleModelAccuracy.txt"), "w") as fp:
    nST = len(sampleTypes)
    nMs = len(transformationList) # clean model corresponds to index 0
    accs = np.zeros((nMs, nST))
    for sIdx in range(nST):
        sampleType = sampleTypes[sIdx]
        predDir = os.path.join(predictionResultDir, sampleType)
        predProb = np.load(os.path.join(predDir, "predProb.npy"))
        accs[:, sIdx] = calAccuracyAllSingleModels(labels, predProb)

    for i in range(nMs):
        fp.write("{}\t{}\t".format(i, transformationList[i]))
        for j in range(nST):
            fp.write("{}\t".format(accs[i,j]))
        fp.write("\n")


