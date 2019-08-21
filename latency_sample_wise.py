import os
import sys
import time

import numpy as np
from scipy.stats import ttest_ind
import matplotlib
matplotlib.use('Agg')



from config import *
from util import *
from transformation import transform_images

DEBUG = True #MODE.DEBUG


# Basic parameters
timeStamp=time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
experimentRootDir=os.path.join("experiment/latency", timeStamp)
createDirSafely(experimentRootDir)

numOfSamples  = 500
modelsDir = "models"
sampleDir     = "testing_samples"
datasetName = DATA.mnist
architecture = "cnn"
modelFilenameTag = datasetName+"-"+architecture
transformConfig = TRANSFORMATION()
transformationList = transformConfig.supported_types() 

#transformationList = ["clean", "vertical_flip", "quant_2_clusters"]
# Load the original clean model and the list of tranformation based model
models, logitsModels = loadModels(modelsDir, modelFilenameTag, transformationList)

# load AEs and their original labels
EPS = [0.25] #ATTACK.FGSM_EPS
attackApproach = "fgsm"
AETypes = []
epsCnt=0
for eps in EPS:
    epsInt = int(1000*eps)
    AETypes.append(attackApproach+"_eps"+str(epsInt))
numOfAETypes = len(AETypes)
targetModel = "clean"
AEFilenameTag = datasetName+"-"+architecture+"-"+targetModel+"-"

numOfModels = len(models)



BSs = np.load(os.path.join(sampleDir, "BS-mnist-clean.npy"))    
AEs = np.load(os.path.join(sampleDir, "AE-"+AEFilenameTag+AETypes[0]+".npy"))

# Prediction: save both probabilities and logits as prediction result
predTCAE = np.zeros((numOfModels, numOfSamples, 3)) # 3 - transformation, predProb and predLogits
predTCBS = np.zeros((numOfModels, numOfSamples, 3)) # 3 - transformation, predProb and predLogits


def getTimeCost(transformType, model, logitsModel, sample):
    # transformation cost
    startTime = time.monotonic()
    tranSample = transform_images(sample, transformType)
    endTime = time.monotonic()
    transCost = endTime - startTime

    # BS - model prediction cost - using probability-based defense
    startTime = time.monotonic()
    model.predict(tranSample)
    endTime = time.monotonic()
    probPredCost = endTime - startTime

    # BS - model prediction cost - using logits-based defense
    startTime = time.monotonic()
    logitsModel.predict(tranSample)
    endTime = time.monotonic()
    logitPredCost = endTime - startTime

    return np.array([transCost, probPredCost, logitPredCost])


for modelID in range(numOfModels):
    
    transformType = transformationList[modelID]
    print("Predict with model {} - {}".format(modelID, transformType))
    for idx in range(numOfSamples):
        print("Processing image "+str(idx+1))
        curBS = np.expand_dims(BSs[idx], axis=0)
        predTCBS[modelID, idx, :] = getTimeCost(transformType, models[modelID], logitsModels[modelID], curBS)
        curAE = np.expand_dims(AEs[idx], axis=0)   
        predTCAE[modelID, idx, :] = getTimeCost(transformType, models[modelID], logitsModels[modelID], curAE)

np.save(os.path.join(experimentRootDir, "predTCBS.npy"), predTCBS)
np.save(os.path.join(experimentRootDir, "predTCAE.npy"), predTCAE)

excludePercent=0.05
startIdx = int(numOfSamples * excludePercent)
endIdx   = int(numOfSamples * (1-excludePercent))

xLabels = ["BS-Tran", "BS-Prob", "BS-Logit", "AE-Tran", "AE-Prob", "AE-Logit"]
yLabel = "time cost in ms"
xstickSize = 8
rotationDegrees=45
tTestResult = np.zeros((numOfModels, 3, 2)) # 2: 0 - t value, 1 - probability
for modelID in range(numOfModels):
    print("Process model ID "+str(modelID))
    for idx in range(3):
        tTestResult[modelID, idx, 0], tTestResult[modelID, idx, 1] = ttest_ind(
                predTCBS[modelID, startIdx:endIdx, idx],
                predTCAE[modelID, startIdx:endIdx, idx])

    title = transformationList[modelID]
    saveFP = os.path.join(experimentRootDir, title+".pdf")
    data = np.hstack((predTCBS[modelID, :, :], predTCAE[modelID, :, :]))
    boxPlot(data*1000, title, xLabels, yLabel, saveFP, xstickSize, rotationDegrees)

np.save(os.path.join(experimentRootDir, "tTestResult.npy"), tTestResult)
valueTypes=["T_Value", "Probability"]
with open(os.path.join(experimentRootDir, "tTestResult.txt"), "w") as fp:
    sf = "{}\t{}\t{}\t{}\t{}\n"
    fp.write(sf.format("Model_ID", "Value_Type", "Transformation", "Inference_Prob", "Inference_Logit"))
    for modelID in range(numOfModels):
        for idx in range(2):
            fp.write(sf.format(
                transformationList[modelID],
                valueTypes[idx],
                tTestResult[modelID, 0, idx],
                tTestResult[modelID, 1, idx],
                tTestResult[modelID, 2, idx]))




