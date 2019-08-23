import os
import sys
import time

import numpy as np
from scipy.spatial import distance_matrix

import seaborn as sns; sns.set()

import util
from config import *
from util import *

def usage():
    print("=============================================================================================")
    print("python <this script> samplesDir rootDir modelsDir numOfSamples testResultFoldName")
    print("=============================================================================================")

if len(sys.argv) != 6:
    usage()
    exit(1)


samplesDir          = sys.argv[1]
rootDir             = sys.argv[2]
modelsDir           = sys.argv[3]
numOfSamples        = int(sys.argv[4])
testResultFoldName  = sys.argv[5]

# Basic parameters for k-fold experiment setup
timeStamp=time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
experimentRootDir=os.path.join(rootDir,timeStamp)
createDirSafely(experimentRootDir)

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
labels      = np.load(os.path.join(samplesDir, "Label-"+datasetName+"-"+targetModelName+".npy"))
labels      = np.argmax(labels, axis=1)

trainModelDir = os.path.join(experimentRootDir, "train_models")

numOfDefenses = numOfCVDefenses+2*numOfWCDefenses

acc1Model = np.zeros((numOfAETypes+1, numOfModels))

sampleTypes = ["BS"]
sampleTypes.extend(AETypes)
numOfSampleTypes = len(sampleTypes)



def majorityVote(participants):
    '''
        Input:
            participants: a list of opinions. Each element in the list is a numpy array, N X 2.
                            N is the number of events. The second dimension contains (opinion/label, confidence)
        Output:
            voteResult  : a numapy array NX2 represents opinion and confidence across N events 
    '''
    numOfEvents = participants[0].shape[0]
    voteResult = np.zeros((numOfEvents, 2))
    misCount = 0
    for eventID in range(numOfEvents):
        countDict={}
        # counting
        for participant in participants:
            vote = participant[eventID][0] 
            if vote in countDict:
                countDict[vote] = (1+countDict[vote][0], participant[eventID][1]+countDict[vote][1])
            else:
                countDict[vote] = (1, participant[eventID][1])

        # voting
        votingResult=None
        count = 0
        for key, value in countDict.items():
            if (value[0] > count) or (value[0]==count and (value[1]/value[0] > countDict[votingResult][1] / count)):
                count = value[0]
                votingResult = key
        
        voteResult[eventID, 0] = votingResult
        voteResult[eventID, 1] = count 
    return voteResult

def saveDATable(sampleTypes, dA, vA, cA, filepath):
    with open(filepath, "w") as fp:
        sf = "{}\t{}\t{}\t{}\n"
        fp.write(sf.format("SampleType", "DetectionAcc", "VotingAcc", "Acc(Clean)"))
        for sIdx in range(len(sampleTypes)):
            fp.write(sf.format(
                sampleTypes[sIdx],
                dA[sIdx],
                vA[sIdx],
                cA[sIdx]))

# calculate the accuracy of each model for all type of samples
accEachModelEachSampleType = np.zeros((numOfModels, numOfSampleTypes))
for sampleTypeIdx in range(numOfSampleTypes):
    sampleType = sampleTypes[sampleTypeIdx]
    curPredictionResultDir = os.path.join(predictionResultDir, sampleType)
    print("calculating accuracy of each model for sample type: "+sampleType)
    predProb  = np.load(os.path.join(curPredictionResultDir, "predProb.npy"))
    accEachModelEachSampleType[:, sampleTypeIdx] = calAccuracyAllSingleModels(labels, predProb)
with open(os.path.join(testDir, "acc_each_model_each_sample_type.txt"), "w") as fp:
    fp.write("Model")
    for sampleType in sampleTypes:
        fp.write("\t"+sampleType)
    fp.write("\n")

    for mIdx in range(numOfModels):
        fp.write(transformationList[mIdx])
        for sampleTypeIdx in range(numOfSampleTypes):
            fp.write("\t"+str(accEachModelEachSampleType[mIdx, sampleTypeIdx]))
        fp.write("\n")


    

thresholdRatios = [0.5, 0.6, 0.7, 0.8]

sampleKinds = [True]
for _ in AETypes:
    sampleKinds.append(False)

for thresholdRatio in thresholdRatios:
    threshold = int(thresholdRatio*numOfModels)
    print("Threshold Ratio: "+str(thresholdRatio))
    # the 1st dimension maps to a kind of ensemble model trained on the specific type of AE
    detectionAccs = np.zeros((numOfSampleTypes))
    votingAccs = np.zeros((numOfSampleTypes))
    cleanAccs = np.zeros((numOfSampleTypes))
    # Test each ensemble model trained by each type of AEs
    for sampleTypeIdx in range(numOfSampleTypes):
        sampleType = sampleTypes[sampleTypeIdx]
        sampleKind = sampleKinds[sampleTypeIdx] # True - BS, False - AE
        curPredictionResultDir = os.path.join(predictionResultDir, sampleType)
        
        print("\tTesting sample type: "+sampleType)
        predProb  = np.load(os.path.join(curPredictionResultDir, "predProb.npy"))
        predLogit = np.load(os.path.join(curPredictionResultDir, "predLogit.npy"))
        predProbLC = np.zeros((numOfModels, numOfSamples, 2))
        predProbLC[:, :, 0] = np.argmax(predProb, axis=2)
        predProbLC[:, : ,1] = np.max(predProb, axis=2)

        # clean model accuracy
        cleanAccs[sampleTypeIdx] = accEachModelEachSampleType[0, sampleTypeIdx]

        # majority voting as detection - threshold 80% of transform models
        participants = []
        for mIdx in range(numOfModels):
            participants.append(predProbLC[mIdx, :, :])
        votedResult2 = util.majorityVote(participants)
        votingAccs[sampleTypeIdx] = calAccuracy(labels, votedResult2[:, 0]) 

        votedResult = majorityVote(participants)
        detectedAECnt = 0
        for sIdx in range(numOfSamples):
            if votedResult[sIdx, 1] < threshold:
                detectedAECnt += 1
        if sampleKind:
            detectionAccs[sampleTypeIdx] = 1 - (detectedAECnt)/numOfSamples
        else:
            detectionAccs[sampleTypeIdx] = (detectedAECnt)/numOfSamples
    filepath = os.path.join(testDir, "detection_acc_"+str(thresholdRatio)+".txt")
    saveDATable(sampleTypes, detectionAccs, votingAccs, cleanAccs, filepath)
        # distance matrix - heatmap
    print("\tThreshold of number of Models that have the same vote: "+str(threshold))
    print("\t"+str(sampleTypes))
    print("\tdetection accuracy: "+str(detectionAccs))
    print("\tvoting accuracy: "+str(votingAccs))
    print("\tclean model accuracy: "+str(cleanAccs))
    print("\n")

def distMatrx(groupOfPoints, metric="l2"):
    '''
        nPoints X nGroup X nDim
    '''
    nGroups = groupOfPoints[0].shape[0]
    nPoints = groupOfPoints.shape[0]
    dms = np.zeros((nGroups, nPoints, nPoints))
    for gIdx in range(nGroups):
        points = groupOfPoints[:, gIdx, :]
        dms[gIdx]=distance_matrix(points, points, p=2)
    meanDM = dms.mean(axis=0)
    stdDM = dms.std(axis=0)
    return meanDM, stdDM

def dumpMat(mat, filepath):
    with open(filepath, "w") as fp:
        for rIdx in range(mat.shape[0]):
            for cIdx in range(mat.shape[1]):
                fp.write(str(mat[rIdx, cIdx])+"\t")
            fp.write("\n")

for sampleTypeIdx in range(numOfSampleTypes):
    sampleType = sampleTypes[sampleTypeIdx]
    sampleKind = sampleKinds[sampleTypeIdx] # True - BS, False - AE
    curPredictionResultDir = os.path.join(predictionResultDir, sampleType)
    
    print("[DM] Testing sample type: "+sampleType)
    predProb  = np.load(os.path.join(curPredictionResultDir, "predProb.npy"))
    predLogit = np.load(os.path.join(curPredictionResultDir, "predLogit.npy"))

    meanDM, stdDM = distMatrx(predProb, metric="l2")
    dumpMat(meanDM, os.path.join(testDir, "meanDM_"+sampleType+"_Prob.txt"))
    dumpMat(stdDM, os.path.join(testDir, "stdDM_"+sampleType+"_Prob.txt"))

    ax = sns.heatmap(meanDM) 
    fig = ax.get_figure()
    fig.savefig(os.path.join(testDir, "DM_"+sampleType+"_Prob.pdf"))
    fig.clf()


    meanDM, stdDM = distMatrx(predLogit, metric="l2")
    dumpMat(meanDM, os.path.join(testDir, "meanDM_"+sampleType+"_Logit.txt"))
    dumpMat(stdDM, os.path.join(testDir, "stdDM_"+sampleType+"_Logit.txt"))

    ax = sns.heatmap(meanDM) 
    fig = ax.get_figure()
    fig.savefig(os.path.join(testDir, "DM_"+sampleType+"_Logit.pdf"))
    fig.clf()
