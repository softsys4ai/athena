import os
import sys
import time

import numpy as np

from utils.config import *
from utils.ensemble_utils import *

def usage():
    print("====================================================================================================================")
    print("python <this script> samplesDir experimentRootDir modelsDir numOfSamples testResultFoldName datasetName numOfClasses")
    print("====================================================================================================================")

if len(sys.argv) != 8:
    usage()
    exit(1)


samplesDir          = sys.argv[1]
experimentRootDir   = sys.argv[2]
modelsDir           = sys.argv[3]
numOfSamples        = int(sys.argv[4])
testResultFoldName  = sys.argv[5]
datasetName         = sys.argv[6]
numOfClasses        = int(sys.argv[7])

DATA.set_current_dataset_name(datasetName)

# Basic parameters for k-fold experiment setup
architecture = MODEL.ARCHITECTURE
testDir = os.path.join(experimentRootDir, testResultFoldName)

AETypes = ATTACK.get_AETypes()


numOfAETypes = len(AETypes)
sampleTypes =["BS"]
sampleTypes.extend(AETypes)
numOfSampleTypes = numOfAETypes + 1

targetModelName = "clean"
transformConfig = TRANSFORMATION()
transformationList = transformConfig.supported_types() 

# Create fold directories for evaluation
predictionResultDir = os.path.join(testDir, "prediction_result")


# Prediction : needs a new prediction function
predictionForTest(
        predictionResultDir,
        datasetName,
        architecture,
        numOfClasses,
        targetModelName,
        modelsDir,
        samplesDir,
        numOfSamples,
        AETypes,
        transformationList)


numOfTrans = len(transformationList) - 1
numOfModels = 1 + numOfTrans # clean model + transform models


# Evaluation: training and testing
labels      = np.load(os.path.join(samplesDir, "Label-"+datasetName+"-"+targetModelName+".npy"))
labels      = np.argmax(labels, axis=1)

trainModelDir = os.path.join(experimentRootDir, "train_models")

numOfDefenses = numOfCVDefenses+2*numOfWCDefenses

acc1Model = np.zeros((numOfAETypes+1, numOfModels))

# accuracies of clean model, random defense and upper bound
rdCleanUPAcc = np.zeros((numOfAETypes+1, 3))
clusters=[]
for tmID in range(numOfTrans):
    clusters.append([tmID])

sampleTypes = ["BS"]
sampleTypes.extend(AETypes)
numOfSampleTypes = len(sampleTypes)


# the 1st dimension maps to a kind of ensemble model trained on the specific type of AE
# numOfAETypes: each type of AE builds a group of ensemble models
defenseAccs = np.zeros((numOfAETypes, numOfSampleTypes,  numOfDefenses))
defenseTCs = np.zeros((numOfAETypes, numOfSampleTypes, numOfDefenses))


# Test each ensemble model trained by each type of AEs
for sampleTypeIdx in range(numOfSampleTypes):
    sampleType = sampleTypes[sampleTypeIdx]
    curPredictionResultDir = os.path.join(predictionResultDir, sampleType)

    print("Testing sample type: "+sampleType)
    predProb  = np.load(os.path.join(curPredictionResultDir, "predProb.npy"))
    predLogit = np.load(os.path.join(curPredictionResultDir, "predLogit.npy"))
    predProbLC = np.zeros((numOfModels, numOfSamples, 2))
    predProbLC[:, :, 0] = np.argmax(predProb, axis=2)
    predProbLC[:, : ,1] = np.max(predProb, axis=2)

    # accuracy of AE on clean model and all transform models
    acc1Model[sampleTypeIdx, :] = calAccuracyAllSingleModels(labels, predProb)

    # accuracy of clean model
    rdCleanUPAcc[sampleTypeIdx, 0] = acc1Model[sampleTypeIdx, 0]
    # accuracy of random defense
    rdCleanUPAcc[sampleTypeIdx, 1] = np.mean(acc1Model[sampleTypeIdx, 1:])
    # upper-bound accuracy
    rdCleanUPAcc[sampleTypeIdx, 2] = getUpperBoundAccuracy(
            predProbLC[1:, :, :],
            clusters,
            labels)

    # test the sample on each ensemble model
    for AETypeIdx in range(numOfAETypes):
        AEType = AETypes[AETypeIdx]
        curTrainModelDir = os.path.join(trainModelDir, AEType)

        # accuracy of clustering-and-voting based defenses
        for defenseIdx in range(numOfCVDefenses):
            defenseName = cvDefenseNames[defenseIdx] 
            clusters = loadCAVModel(os.path.join(curTrainModelDir, defenseName+".txt"))

            # testing
            votedResults, defenseTCs[AETypeIdx, sampleTypeIdx, defenseIdx] = votingAsDefense(
                    predProbLC[1:, :, :],
                    clusters,
                    vsac=cvDefenseNames[defenseIdx],
                    measureTC=True)
            defenseAccs[AETypeIdx, sampleTypeIdx, defenseIdx] = calAccuracy(votedResults[:, 0], labels)


        # accuracy of weithed-confidence based defenses
        for defenseIdx in range(numOfWCDefenses):
            defenseName = wcDefenseNames[defenseIdx]
            for plIdx in range(2):
                wcMatFilename = defenseName+"_EM.npy"
                mIDsFilename  = defenseName+"_modelIDs.npy"
                pred = predProb[1:, :, :]
                if plIdx == 1: # predict logit instead of probability
                    wcMatFilename = "LG_" + wcMatFilename
                    mIDsFilename  = "LG_" +  mIDsFilename
                    pred = predLogit[1:, :, :]

                wcMat = np.load(os.path.join(curTrainModelDir, wcMatFilename))
                # ID of transform models: starts from 0.
                mIDs  = np.load(os.path.join(curTrainModelDir, mIDsFilename))
               
                curPred = pred[mIDs] 
                dIdx = numOfCVDefenses + plIdx * numOfWCDefenses + defenseIdx

                # testing
                predLabels,  defenseTCs[AETypeIdx, sampleTypeIdx, dIdx] = wcdefenses(
                        curPred, wcMat, defenseName, measureTC=True)
                defenseAccs[AETypeIdx, sampleTypeIdx, dIdx] = calAccuracy(predLabels, labels)

         
# Report accuracy data
# accuracies of random defense, clean model and upper bound
# rdCleanUPAcc = np.zeros((numOfAETypes+1, 3))
# defenseAccBSs, defenseAccAEs: (numOfAETypes, numofDefenses)
rdCleanUPAccFP = os.path.join(testDir, "acc_randomDefense_cleanModel_upperBound.txt")
with open(rdCleanUPAccFP, "w") as fp:
    sformat = "{}\t{}\t{}\t{}\n"
    fp.write(sformat.format("Type", "Clean_Model", "Rrandom_Defense", "Upper_Bound"))
    for sampleTypeIdx in range(numOfSampleTypes):
        fp.write(sformat.format(
            sampleTypes[sampleTypeIdx],
            rdCleanUPAcc[sampleTypeIdx, 0],
            rdCleanUPAcc[sampleTypeIdx, 1],
            rdCleanUPAcc[sampleTypeIdx, 2]))

def saveAccTable(sampleTypes, accMat, filepath):
    with open(filepath, "w") as fp:
        sformat = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
        fp.write(sformat.format(
            "SampleType",
            "CV_Maj",
            "CV_Max",
            "1s_Mean",
            "EM_Mean",
            "EM_MXMV",
            "1s_Mean_L",
            "EM_Mean_L",
            "EM_MXMV_L"))
        for sampleTypeIdx in range(len(sampleTypes)):
            fp.write(sformat.format(
                sampleTypes[sampleTypeIdx],
                accMat[sampleTypeIdx, 0],
                accMat[sampleTypeIdx, 1],
                accMat[sampleTypeIdx, 2],
                accMat[sampleTypeIdx, 3],
                accMat[sampleTypeIdx, 4],
                accMat[sampleTypeIdx, 5],
                accMat[sampleTypeIdx, 6],
                accMat[sampleTypeIdx, 7]))


for AETypeIdx in range(numOfAETypes):
    AEType = AETypes[AETypeIdx]
    defenseAccsFP = os.path.join(testDir, "acc_of_ensembles_built_on_"+AEType+".txt")
    saveAccTable(sampleTypes, defenseAccs[AETypeIdx], defenseAccsFP)

meanDefenseAccs = np.round(defenseAccs.mean(axis=0), decimals=4)
stdDefenseAccs  = np.round(defenseAccs.std(axis=0), decimals=4)

saveAccTable(sampleTypes, meanDefenseAccs, os.path.join(testDir, "mean_acc_of_ensembles.txt"))
saveAccTable(sampleTypes, stdDefenseAccs, os.path.join(testDir, "std_acc_of_ensembles.txt"))


# averaging defense time cost across different sample types
defenseTCs = defenseTCs.mean(axis=1) 

# Report latency
# defenseTCs : (numOfAETypes, numofDefenses)
# predTCs: (numOfSampleTypes, numOfModels, 3)
predTCs = np.load(os.path.join(predictionResultDir, "predTCs.npy"))
predAndTransTCs = np.zeros((predTCs.shape[0], predTCs.shape[1], 2))
predAndTransTCs[:, :, 0] = predTCs[:, :, 0] + predTCs[:, :, 1]
predAndTransTCs[:, :, 1] = predTCs[:, :, 0] + predTCs[:, :, 2]

maxTCTransModels = np.argmax(predAndTransTCs[:, 1:, :], axis=1)
maxTCTransModelsFP = os.path.join(testDir, "maxTCTransModels.txt")
with open(maxTCTransModelsFP, "w") as fp:
    sformat="{}\t{}\t{}\n"
    fp.write(sformat.format(
        "Type",
        "ProbPred",
        "LogitPred"))
    fp.write(sformat.format(
        "BS",
        maxTCTransModels[0, 0],
        maxTCTransModels[0, 1]))
    for AETypeIdx in range(numOfAETypes):
        fp.write(sformat.format(
            AETypes[AETypeIdx],
            maxTCTransModels[1+AETypeIdx, 0],
            maxTCTransModels[1+AETypeIdx, 1]))

predAndTransTCs = np.max(predAndTransTCs[:, 1:, :], axis=1) # find the largest time cost of transformation and inference across models
CAVEnsembleTCs = np.zeros((numOfAETypes, 2))
CAVEnsembleTCs[:, 0] = predAndTransTCs[1:, 0] + defenseTCs[:, 0]
CAVEnsembleTCs[:, 1] = predAndTransTCs[1:, 0] + defenseTCs[:, 1]
WCEnsemblesTCs = np.zeros((numOfAETypes, 6))
WCEnsemblesTCs[:, 0] = predAndTransTCs[1:, 0] + defenseTCs[:, 2]
WCEnsemblesTCs[:, 1] = predAndTransTCs[1:, 0] + defenseTCs[:, 3]
WCEnsemblesTCs[:, 2] = predAndTransTCs[1:, 0] + defenseTCs[:, 4]
WCEnsemblesTCs[:, 3] = predAndTransTCs[1:, 1] + defenseTCs[:, 5]
WCEnsemblesTCs[:, 4] = predAndTransTCs[1:, 1] + defenseTCs[:, 6]
WCEnsemblesTCs[:, 5] = predAndTransTCs[1:, 1] + defenseTCs[:, 7]

# probability inference on clean model
# defense time costs
totalTCs = np.zeros((numOfAETypes, 1 + numOfDefenses))
totalTCs[:, 0]   = predTCs[1:, 0, 1]
totalTCs[:, 1:3] = CAVEnsembleTCs
totalTCs[:, 3:]  = WCEnsemblesTCs
totalTCFP = os.path.join(testDir, "time_cost_of_each_ensemble_model.txt") 
with open(totalTCFP, "w") as fp:
    sformat = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
    fp.write(sformat.format(
        "TrainDataType",
        "Clean",
        "CV_Maj",
        "CV_Max",
        "1s_Mean",
        "EM_Mean",
        "EM_MXMV",
        "1s_Mean_L",
        "EM_Mean_L",
        "EM_MXMV_L"))
    for AETypeIdx in range(numOfAETypes):
        fp.write(sformat.format(
            AETypes[AETypeIdx],
            totalTCs[AETypeIdx, 0],
            totalTCs[AETypeIdx, 1],
            totalTCs[AETypeIdx, 2],
            totalTCs[AETypeIdx, 3],
            totalTCs[AETypeIdx, 4],
            totalTCs[AETypeIdx, 5],
            totalTCs[AETypeIdx, 6],
            totalTCs[AETypeIdx, 7],
            totalTCs[AETypeIdx, 8]))
ensembleModelNames = [
        "CV_Maj",
        "CV_Max",
        "1s_Mean",
        "EM_Mean",
        "EM_MXMV",
        "1s_Mean_L",
        "EM_Mean_L",
        "EM_MXMV_L"]
xLabel = ["Clean"]
xLabel.extend(ensembleModelNames)
yLabel = "Latency (ms)"
title = "Latency of clean model and ensemble models"
saveFP = os.path.join(testDir, "latency.pdf")
xtickSize = 8
boxPlot(totalTCs*1000, title, xLabel, yLabel, saveFP, xtickSize, 45)

relativeTotTC = totalTCs / totalTCs[:, 0][:, None]
relativeTotTCFP = os.path.join(testDir, "relative_time_cost_of_each_ensemble_model.txt") 
with open(relativeTotTCFP, "w") as fp:
    sformat = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
    fp.write(sformat.format(
        "TrainDataType",
        "CV_Maj",
        "CV_Max",
        "1s_Mean",
        "EM_Mean",
        "EM_MXMV",
        "1s_Mean_L",
        "EM_Mean_L",
        "EM_MXMV_L"))
    for AETypeIdx in range(numOfAETypes):
        fp.write(sformat.format(
            AETypes[AETypeIdx],
            relativeTotTC[AETypeIdx, 1],
            relativeTotTC[AETypeIdx, 2],
            relativeTotTC[AETypeIdx, 3],
            relativeTotTC[AETypeIdx, 4],
            relativeTotTC[AETypeIdx, 5],
            relativeTotTC[AETypeIdx, 6],
            relativeTotTC[AETypeIdx, 7],
            relativeTotTC[AETypeIdx, 8]))

xLabel = ensembleModelNames
yLabel = "Latency Percentage"
title = "Latency of ensemble models relative to clean model"
saveFP = os.path.join(testDir, "relative_latency.pdf")
boxPlot(relativeTotTC[:, 1:], title, xLabel, yLabel, saveFP, xtickSize, 45)

# backup raw data of time cost
np.save(os.path.join(testDir, "defenseTCs.npy"), defenseTCs)

# backup accuracy of BS and AEs on clean models and all transform models
np.save(os.path.join(testDir, "acc1Model.npy"), acc1Model)
