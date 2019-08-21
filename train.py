import os
import sys
import time

import numpy as np

from config import *
from util import *

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
timeStamp=time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
experimentRootDir=os.path.join(rootDir,timeStamp)
createDirSafely(experimentRootDir)
with open("current_experiment_root_dir_name.txt", "w") as fp:
    fp.write(experimentRootDir)

#kFold = 5
isKFolderUponTestSet=True
datasetName = DATA.mnist
architecture = MODEL.ARCHITECTURE
numOfClasses = 10

AETypes = ATTACK.get_fgsm_AETypes()

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

trainModelDir = os.path.join(experimentRootDir, "train_models")

for AETypeIdx in range(numOfAETypes):
    bestAccCAV = np.zeros((numOfCVDefenses))
    bestAccWC = np.zeros((2, numOfWCDefenses)) # 0 - probability, 2 - logit
    bestClusters = []
    bestEMModel = {}

    AEType = AETypes[AETypeIdx]
    for foldIdx in range(1, 1+kFold):
        # foldIndices: testing samples' indices for this fold
        print("fold " + str(foldIdx))
        if kFold == 1:
            trainingIndices = np.array(range(0, numOfSamples))
            testingIndices = np.copy(trainingIndices)
        else:
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
        curExprDir = os.path.join(foldDir, AEType)
        curPredictionResultDir = os.path.join(predictionResultDir, AEType)
        createDirSafely(curExprDir)

        print("Evaluating AE type: "+AEType)
        predProbAE  = np.load(os.path.join(curPredictionResultDir, "predProb.npy"))
        predLogitAE = np.load(os.path.join(curPredictionResultDir, "predLogit.npy"))

        print(trainingIndices)
        print(trainingIndices.shape)
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
        AETestResultsCAV, BSTestResultCAV, optimalClusters = clusteringDefensesEvaluation(
            curExprDir,
            predProbAETr,
            labelsTr,
            predProbAETe,
            labelsTe,
            predLCBS,
            labelsBS)


        # Weighted-confidence based defenses
        print("\t==== weighted-confidence based defenses")
        AETestResultsWC, BSTestResultsWC, WCModels = weightedConfDefenseEvaluation(
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

        if foldIdx == 1:
            bestAccCAV = AETestResultsCAV[:, 0]
            bestClusters = optimalClusters
            bestAccWC = AETestResultsWC[:, :, 0]
            bestWCModels = WCModels
            
        else:
            # update best CAV models : clusters
            for defenseIdx in range(numOfCVDefenses):
                if bestAccCAV[defenseIdx] < AETestResultsCAV[defenseIdx, 0]:
                    bestAccCAV[defenseIdx] = AETestResultsCAV[defenseIdx, 0]
                    bestClusters[defenseIdx] = optimalClusters[defenseIdx]

            # update best WC models
            for defenseIdx in range(numOfWCDefenses):
                defenseName = wcDefenseNames[defenseIdx]
                for plIdx in range(2):
                    if bestAccWC[plIdx, defenseIdx] < AETestResultsWC[plIdx, defenseIdx, 0]:
                        bestAccWC[plIdx, defenseIdx] = AETestResultsWC[plIdx, defenseIdx, 0]
                        bestWCModels[plIdx][defenseName] = WCModels[plIdx][defenseName]






    # Dump out models
    curTrainModelDir = os.path.join(trainModelDir, AEType)
    createDirSafely(curTrainModelDir)
    for defenseIdx in range(numOfCVDefenses):
        defenseName = cvDefenseNames[defenseIdx]
        numOfClusters = len(bestClusters[defenseIdx])
        clusters = bestClusters[defenseIdx]
        with open(os.path.join(curTrainModelDir, defenseName+".txt"), "w") as fp:
            for cluster in clusters:
                for tranModelID in cluster:
                    fp.write(str(tranModelID)+" ")
                fp.write("\n")

    # WCModels: dictionary. wc_defense_name / (expertise mattrix, array of model IDs)
    for plIdx in range(2):
        for defenseIdx in range(numOfWCDefenses):
            defenseName = wcDefenseNames[defenseIdx]
            WCModel = bestWCModels[plIdx][defenseName]
            expertiseModelFP = defenseName+"_EM.npy"
            modelIDsFP       = defenseName+"_modelIDs.npy"
            if plIdx == 1:
                expertiseModelFP = "LG_"+expertiseModelFP
                modelIDsFP       = "LG_"+modelIDsFP
            np.save(os.path.join(curTrainModelDir, expertiseModelFP), WCModel[0])
            np.save(os.path.join(curTrainModelDir, modelIDsFP), WCModel[1])
            

# post analysis
postAnalysis(
        experimentRootDir,
        kFold,
        predictionResultDir,
        foldDirs,
        sampleTypes)


