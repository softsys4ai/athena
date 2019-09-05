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

# Basic parameters for k-fold experiment setup
architecture = MODEL.ARCHITECTURE
testDir = os.path.join(experimentRootDir, testResultFoldName)


sampleTypes =[sampleType]

targetModelName = "clean"
transformConfig = TRANSFORMATION()
transformationList = transformConfig.supported_types() 

# Create fold directories for evaluation
predictionResultDir = os.path.join(testDir, "prediction_result")


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


