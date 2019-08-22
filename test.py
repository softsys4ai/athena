import os
import sys
import time

import numpy as np

from config import *
from util import *

def usage():
    print("=============================================================================================")
    print("python <this script> samplesDir experimentRootDir modelsDir numOfSamples testResultFoldName")
    print("=============================================================================================")

if len(sys.argv) != 6:
    usage()
    exit(1)


samplesDir          = sys.argv[1]
experimentRootDir   = sys.argv[2]
modelsDir           = sys.argv[3]
numOfSamples        = int(sys.argv[4])
testResultFoldName  = sys.argv[5]

# Basic parameters for k-fold experiment setup
datasetName = DATA.mnist
architecture = MODEL.ARCHITECTURE
numOfClasses = 10
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
predProbBS  = np.load(os.path.join(predictionResultDir, "BS/predProb.npy"))
#predProbBS  = predProbBS[1:]
predLogitBS = np.load(os.path.join(predictionResultDir, "BS/predLogit.npy"))
#predLogitBS = predLogitBS[1:]
labels      = np.load(os.path.join(samplesDir, "Label-"+datasetName+"-"+targetModelName+".npy"))
labels      = np.argmax(labels, axis=1)
predLCBS    = np.zeros((predProbBS.shape[0], predProbBS.shape[1], 2))
predLCBS[:, :, 0] = np.argmax(predProbBS, axis=2)
predLCBS[:, :, 1] = np.max(predProbBS, axis=2)
labelsBS    = labels

trainModelDir = os.path.join(experimentRootDir, "train_models")

numOfDefenses = numOfCVDefenses+2*numOfWCDefenses

acc1Model = np.zeros((numOfAETypes+1, numOfModels))
acc1Model[0,:] = calAccuracyAllSingleModels(labelsBS, predProbBS)

# the 1st dimension maps to a kind of ensemble model trained on the specific type of AE
defenseAccAEs = np.zeros((numOfAETypes, numOfDefenses))
defenseAccBSs = np.zeros((numOfAETypes, numOfDefenses))
defenseTCAEs = np.zeros((numOfAETypes, numOfDefenses))
defenseTCBSs = np.zeros((numOfAETypes, numOfDefenses))
# accuracies of clean model, random defense and upper bound
rdCleanUPAcc = np.zeros((numOfAETypes+1, 3))
clusters=[]
for tmID in range(numOfTrans):
    clusters.append([tmID])
# BS - accuracy of clean model, random defense and upper bound
# accuracy of clean model
rdCleanUPAcc[0, 0] = acc1Model[0, 0]
# accuracy of random defense
rdCleanUPAcc[0, 1] = np.mean(acc1Model[0, 1:])
# upper-bound accuracy
rdCleanUPAcc[0, 2] = getUpperBoundAccuracy(
        predLCBS[1:, :, :],
        clusters,
        labelsBS)



# Test each ensemble model trained by each type of AEs
for AETypeIdx in range(numOfAETypes):
    AEType = AETypes[AETypeIdx]
    curTrainModelDir = os.path.join(trainModelDir, AEType)
    curPredictionResultDir = os.path.join(predictionResultDir, AEType)

    print("Evaluating AE type: "+AEType)
    predProbAE  = np.load(os.path.join(curPredictionResultDir, "predProb.npy"))
    predLogitAE = np.load(os.path.join(curPredictionResultDir, "predLogit.npy"))
    predProbLC = np.zeros((numOfModels, numOfSamples, 2))
    predProbLC[:, :, 0] = np.argmax(predProbAE, axis=2)
    predProbLC[:, : ,1] = np.max(predProbAE, axis=2)

    # accuracy of AE on clean model and all transform models
    acc1Model[AETypeIdx+1, :] = calAccuracyAllSingleModels(labels, predProbAE)

    # accuracy of clean model
    rdCleanUPAcc[AETypeIdx+1, 0] = acc1Model[AETypeIdx+1, 0]
    # accuracy of random defense
    rdCleanUPAcc[AETypeIdx+1, 1] = np.mean(acc1Model[AETypeIdx+1, 1:])
    # upper-bound accuracy
    rdCleanUPAcc[AETypeIdx+1, 2] = getUpperBoundAccuracy(
            predProbLC[1:, :, :],
            clusters,
            labels)

    # accuracy of clustering-and-voting based defenses
    for defenseIdx in range(numOfCVDefenses):
        defenseName = cvDefenseNames[defenseIdx] 
        clusters = loadCAVModel(os.path.join(curTrainModelDir, defenseName+".txt"))

        # testing AE
        votedResults, defenseTCAEs[AETypeIdx, defenseIdx] = votingAsDefense(
                predProbLC[1:, :, :],
                clusters,
                vsac=cvDefenseNames[defenseIdx],
                measureTC=True)
        defenseAccAEs[AETypeIdx, defenseIdx] = calAccuracy(votedResults[:, 0], labels)

        # tesing BS
        votedResults, defenseTCBSs[AETypeIdx, defenseIdx] = votingAsDefense(
                predLCBS[1:, :, :],
                clusters,
                vsac=cvDefenseNames[defenseIdx],
                measureTC=True)
        defenseAccBSs[AETypeIdx, defenseIdx] = calAccuracy(votedResults[:, 0], labelsBS)


    # accuracy of weithed-confidence based defenses
    for defenseIdx in range(numOfWCDefenses):
        defenseName = wcDefenseNames[defenseIdx]
        for plIdx in range(2):
            wcMatFilename = defenseName+"_EM.npy"
            mIDsFilename  = defenseName+"_modelIDs.npy"
            predAE = predProbAE[1:, :, :]
            predBS = predProbBS[1:, :, :]
            if plIdx == 1: # predict logit instead of probability
                wcMatFilename = "LG_" + wcMatFilename
                mIDsFilename  = "LG_" +  mIDsFilename
                predAE = predLogitAE[1:, :, :]
                predBS = predLogitBS[1:, :, :]

            wcMat = np.load(os.path.join(curTrainModelDir, wcMatFilename))
            # ID of transform models: starts from 0.
            mIDs  = np.load(os.path.join(curTrainModelDir, mIDsFilename))
           
            curPredAE = predAE[mIDs] 
            curPredBS = predBS[mIDs]
            dIdx = numOfCVDefenses + plIdx * numOfWCDefenses + defenseIdx

            # testing AE
            predLabels,  defenseTCAEs[AETypeIdx, dIdx] = wcdefenses(
                    curPredAE, wcMat, defenseName, measureTC=True)
            defenseAccAEs[AETypeIdx, dIdx] = calAccuracy(predLabels, labels)

            # testing BS
            predLabels,  defenseTCBSs[AETypeIdx, dIdx] = wcdefenses(
                    curPredBS, wcMat, defenseName, measureTC=True)
            defenseAccBSs[AETypeIdx, dIdx] = calAccuracy(predLabels, labels)


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

defenseAccAEsFP = os.path.join(testDir, "acc_AEs_ensembles.txt")
with open(defenseAccAEsFP, "w") as fp:
    sformat = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
    fp.write(sformat.format(
        "Type",
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
            defenseAccAEs[AETypeIdx, 0],
            defenseAccAEs[AETypeIdx, 1],
            defenseAccAEs[AETypeIdx, 2],
            defenseAccAEs[AETypeIdx, 3],
            defenseAccAEs[AETypeIdx, 4],
            defenseAccAEs[AETypeIdx, 5],
            defenseAccAEs[AETypeIdx, 6],
            defenseAccAEs[AETypeIdx, 7]))

defenseAccBSsFP = os.path.join(testDir, "acc_BSs_ensembles.txt")
with open(defenseAccBSsFP, "w") as fp:
    sformat = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
    fp.write(sformat.format(
        "Type",
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
            defenseAccBSs[AETypeIdx, 0],
            defenseAccBSs[AETypeIdx, 1],
            defenseAccBSs[AETypeIdx, 2],
            defenseAccBSs[AETypeIdx, 3],
            defenseAccBSs[AETypeIdx, 4],
            defenseAccBSs[AETypeIdx, 5],
            defenseAccBSs[AETypeIdx, 6],
            defenseAccBSs[AETypeIdx, 7]))


# Report latency
# defenseTCBSs , defenseTCAEs : (numOfAETypes, numofDefenses)
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
CAVEnsembleTCs[:, 0] = predAndTransTCs[1:, 0] + defenseTCAEs[:, 0]
CAVEnsembleTCs[:, 1] = predAndTransTCs[1:, 0] + defenseTCAEs[:, 1]
WCEnsemblesTCs = np.zeros((numOfAETypes, 6))
WCEnsemblesTCs[:, 0] = predAndTransTCs[1:, 0] + defenseTCAEs[:, 2]
WCEnsemblesTCs[:, 1] = predAndTransTCs[1:, 0] + defenseTCAEs[:, 3]
WCEnsemblesTCs[:, 2] = predAndTransTCs[1:, 0] + defenseTCAEs[:, 4]
WCEnsemblesTCs[:, 3] = predAndTransTCs[1:, 1] + defenseTCAEs[:, 5]
WCEnsemblesTCs[:, 4] = predAndTransTCs[1:, 1] + defenseTCAEs[:, 6]
WCEnsemblesTCs[:, 5] = predAndTransTCs[1:, 1] + defenseTCAEs[:, 7]

# probability inference on clean model
# defense time costs
totalTCsAE = np.zeros((numOfAETypes, 1 + numOfDefenses))
totalTCsAE[:, 0]   = predTCs[1:, 0, 1]
totalTCsAE[:, 1:3] = CAVEnsembleTCs
totalTCsAE[:, 3:]  = WCEnsemblesTCs
totalTCAEFP = os.path.join(testDir, "time_cost_of_each_ensemble_model.txt") 
with open(totalTCAEFP, "w") as fp:
    sformat = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
    fp.write(sformat.format(
        "Type",
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
            totalTCsAE[AETypeIdx, 0],
            totalTCsAE[AETypeIdx, 1],
            totalTCsAE[AETypeIdx, 2],
            totalTCsAE[AETypeIdx, 3],
            totalTCsAE[AETypeIdx, 4],
            totalTCsAE[AETypeIdx, 5],
            totalTCsAE[AETypeIdx, 6],
            totalTCsAE[AETypeIdx, 7],
            totalTCsAE[AETypeIdx, 8]))
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
boxPlot(totalTCsAE*1000, title, xLabel, yLabel, saveFP, xtickSize, 45)

relativeTotTCAE = totalTCsAE / totalTCsAE[:, 0][:, None]
relativeTotTCAEFP = os.path.join(testDir, "relative_time_cost_of_each_ensemble_model.txt") 
with open(relativeTotTCAEFP, "w") as fp:
    sformat = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
    fp.write(sformat.format(
        "Type",
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
            relativeTotTCAE[AETypeIdx, 1],
            relativeTotTCAE[AETypeIdx, 2],
            relativeTotTCAE[AETypeIdx, 3],
            relativeTotTCAE[AETypeIdx, 4],
            relativeTotTCAE[AETypeIdx, 5],
            relativeTotTCAE[AETypeIdx, 6],
            relativeTotTCAE[AETypeIdx, 7],
            relativeTotTCAE[AETypeIdx, 8]))

xLabel = ensembleModelNames
yLabel = "Latency Percentage"
title = "Latency of ensemble models relative to clean model"
saveFP = os.path.join(testDir, "relative_latency.pdf")
boxPlot(relativeTotTCAE[:, 1:], title, xLabel, yLabel, saveFP, xtickSize, 45)

# backup raw data of time cost
np.save(os.path.join(testDir, "defenseTC_BS.npy"), defenseTCBSs)
np.save(os.path.join(testDir, "defenseTC_AE.npy"), defenseTCAEs)

# backup accuracy of BS and AEs on clean models and all transform models
np.save(os.path.join(testDir, "acc1Model.npy"), acc1Model)
