import os
import time

import matplotlib.pyplot as plt

import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

from config import *
from transformation import transform_images


def randomChoiceBasedDefense(predProb, measureTC=False):
    '''
        Input:
            predProb: (numOfModels, numOfSamples, numOfClasses)
    '''
    timeCost = -1

    if measureTC:
        startTime = time.monotonic()


    numOfModels     = predProb.shape[0]
    numOfSamples    = predProb.shape[1]
    numOfClasses    = predProb.shape[2]

    modelIndices = np.random.random_integers(
            low=0,
            high=(numOfModels-1),
            size=numOfSamples)

    predProbResult = np.zeros((numOfSamples, numOfClasses))

    for sampleIdx in range(numOfSamples):
        predProbResult[sampleIdx, :] = predProb[modelIndices[sampleIdx], sampleIdx, :]

    predLC=np.zeros((numOfSamples, 2))
    predLC[:, 0] = np.argmax(predProbResult, axis=1)
    predLC[:, 1] = np.max(predProbResult, axis=1)

    if measureTC:
        timeCost = (time.monotonic() - startTime) / numOfSamples

    return predLC, timeCost

def clusteringBasedDefesTrainPre(
        curExprDir,
        numOfModels,
        numOfAEs,
        numOfTrans,
        maxNumOfClusters,
        NC, 
        AEPredLC,
        trueLabels):
    '''
        Run for a specific type of AEs
        Input:
            AEPredLC: (numOfModels, 2). 0 - label, 1 - confidence
            NC: list of numbers of cluters to run with KMeans

    '''

    # Compute model-sample vector for transform models and AEs
    msv = np.zeros((numOfTrans, numOfAEs))
    for modelID in range(1, numOfModels): # clean model's ID is 0 
        for aeID in range(numOfAEs):
            if AEPredLC[modelID, aeID, 0] == trueLabels[aeID]:
                msv[modelID-1, aeID] = 1
            else:
                msv[modelID-1, aeID] = 0

    np.save(os.path.join(curExprDir, "msv.npy"), msv)
    with open(os.path.join(curExprDir, "msv.txt"), "w") as fp:
        for transID in range(numOfTrans):
            for aeID in range(numOfAEs):
                fp.write(str(msv[transID, aeID])+",")
            fp.write("\n")
         

    # Clustering using KMeans with Squared Euclidean distance
    clusteringResultDir = os.path.join(curExprDir, kmeansResultFoldName)
    createDirSafely(clusteringResultDir)

    ubAccs = np.zeros((maxNumOfClusters))
    for numOfClusters in NC:
        # clustering into c groups
        kmeans = None
        inertia = np.inf
        # Run KMeans 10 times and return the clustering result with
        # smallest sum of squared distances of samples to their closest cluster center.
        numOfTries = 10
        for _ in range(numOfTries):
            cur_kmeans = KMeans(n_clusters=numOfClusters).fit(msv)
            if cur_kmeans.inertia_ < inertia:
                inertia = cur_kmeans.inertia_
                kmeans  = cur_kmeans
        # Write the clustering result into a file. Each line represents a cluster. 
        # Note: in KMeans, cluster ID starts from 0. 
        with open(os.path.join(clusteringResultDir, "C"+str(numOfClusters)+".txt"), "w") as fp:
            clusters = []
            for c in range(numOfClusters):
                # transform model ID starts at 1.
                cluster = 1 + np.where(kmeans.labels_==c)[0]
                clusters.append(cluster)
                for tranModelID in cluster:
                    fp.write(str(tranModelID)+" ")
                fp.write("\n")

        # Compute upper-bound accuracy
        ubAccs[numOfClusters-1] = getUpperBoundAccuracy(
                AEPredLC,
                clusters,
                trueLabels)

    np.save(os.path.join(curExprDir, "upper_bound_accuracy.npy"), ubAccs)

def clusteringBasedDefesTrain(
        curExprDir,
        maxNumOfClusters,
        NC,
        AEPredLC,
        labels):

    '''
        Train with a specific set of AE type
        Input:
            AEPredLC: (numOfTrans, numOfAEs, 2)
            labels    : true labels
        Output:
            trainResult: numOfCVDefenses X 3 numpy tuple.
                        0th column: num of clusters
                        1st column: the corresponding accuracy of classifying AEs
                        2nd column: the corresponding accuracy of classifying BSs
    '''
    clusteringResultDir = os.path.join(curExprDir, kmeansResultFoldName)

    accsAEClustering = np.zeros((maxNumOfClusters, numOfCVDefenses))
 
    # Find the optimal number of clusters 
    for numOfClusters in NC:
        clusters = loadClusteringResult(clusteringResultDir, numOfClusters)
        for defenseIdx in range(numOfCVDefenses):
            defenseName = cvDefenseNames[defenseIdx]
            votedResultAE, timeCostAE = votingAsDefense(
                    AEPredLC,
                    clusters,
                    vsac=defenseName)
            accsAEClustering[numOfClusters-1, defenseIdx] = calAccuracy(votedResultAE[:, 0], labels)


    bestNumOfClusters = np.argmax(accsAEClustering, axis=0)
    bestAccsAE = accsAEClustering[bestNumOfClusters].diagonal() 

    trainResult = np.hstack((
        1+bestNumOfClusters.reshape(numOfCVDefenses, 1),
        bestAccsAE.reshape(numOfCVDefenses, 1)))

    np.save(os.path.join(curExprDir, "ensemble_models_clustering_based_defenses.npy"), trainResult)
    return trainResult




def clusteringBasedDefesTest(
        curExprDir,
        predLC,
        labels,
        numOfClusters):

    '''
        Input:
            predLC: (numOfTrans, numOfSamples, 2)
            labels    : true labels
            numOfClusters: 1D numpy tuple of numOfCVDefenses elements  
        Iutput:
            votedResults: numOfSamples X 2. 2 - label and confidence
            accuracyAndTCs: numOfCVDefenses X 2 numpy array. 2 - accuracy and time cost
    '''
    numOfSamples = predLC.shape[1]
    clusteringResultDir = os.path.join(curExprDir, kmeansResultFoldName)
    votedResults = np.zeros((numOfCVDefenses, numOfSamples, 2))
    accuracyAndTCs   = np.zeros((numOfCVDefenses, 2))
    for defenseIdx in range(numOfCVDefenses):
        clusters = loadClusteringResult(clusteringResultDir, numOfClusters[defenseIdx])

        votedResults[defenseIdx, :, :], timeCost = votingAsDefense(
                predLC,
                clusters,
                vsac=cvDefenseNames[defenseIdx],
                measureTC=True)
        accuracyAndTCs[defenseIdx, 1] = timeCost
        accuracyAndTCs[defenseIdx, 0] = calAccuracy(votedResults[defenseIdx, :, 0], labels)

    return votedResults, accuracyAndTCs


def clusteringDefensesEvaluation(
        curExprDir,
        AEPredProbTrain,
        labelsTrain,
        AEPredProbTest,
        labelsTest,
        BSPredLC,
        BSLabels):

    '''
        Run for a specific type of AEs
        Note:
            1. numOfTrans is the number of tranform models
            2. numOfModels is numOfTrans + 1 (the original model without transformation applied)
        Input:
            AEPredProbTrain : (numOfModels, numOfAEs, numOfClasses)
            labelsTrain     : true labels for training set
            AEPredProbTest  : (numOfTrans, numOfAEs, numOfClasses)
            labelsTest      : labels for testing set
            BSPredLC        : (numOfTrans, numOfBSs, numOfClasses)
            BSLabels        : true labels

    '''

    # TRAIN
    print("\t\t==== Training ====")
    numOfModels = AEPredProbTrain.shape[0]
    numOfTrainingSamples = AEPredProbTrain.shape[1] # number of AEs = number of BSs
    numOfTrans = numOfModels-1
    maxNumOfClusters = numOfTrans
    NC=list(range(1, maxNumOfClusters+1)) # list of numbers of clusters

    AEPredLCTrain = np.zeros((numOfModels, numOfTrainingSamples, 2))

    AEPredLCTrain[:, :, 0] = np.argmax(AEPredProbTrain, axis=2)
    AEPredLCTrain[:, :, 1] = np.max(AEPredProbTrain, axis=2)

    clusteringBasedDefesTrainPre(
            curExprDir,
            numOfModels,
            numOfTrainingSamples,
            numOfTrans,
            maxNumOfClusters,
            NC,
            AEPredLCTrain,
            labelsTrain)

    # use the prediction from the transform models for training
    trainingResult = clusteringBasedDefesTrain(
            curExprDir,
            maxNumOfClusters,
            NC,
            AEPredLCTrain[1:, :, :],
            labelsTrain)


    # TEST
    print("\t\t==== Testing ====")
    numOfTestingSamples = AEPredProbTest.shape[1]
    AEPredLCTest = np.zeros((numOfTrans, numOfTestingSamples, 2))

    AEPredLCTest[:, :, 0] = np.argmax(AEPredProbTest, axis=2)
    AEPredLCTest[:, :, 1] = np.max(AEPredProbTest, axis=2)

    numOfClusters = trainingResult[:, 0].astype("int")
    # voted label and confidence for AEs
    #testVotes   : (numOfCVDefenses, numOfTestingSamples, 2)
    # 2: AE-accuracy, AE-time cost
    #testResults : (numOfCVDefenses, 2)
    testVotesAE, testResultsAE = clusteringBasedDefesTest(
            curExprDir,
            AEPredLCTest,
            labelsTest,
            numOfClusters)

    testVotesBS, testResultsBS = clusteringBasedDefesTest(
            curExprDir,
            BSPredLC,
            BSLabels,
            numOfClusters)

    np.save(os.path.join(curExprDir, "AE_testVotes_ClusteringAndVote.npy"), testVotesAE)
    np.save(os.path.join(curExprDir, "AE_testResults_ClusteringAndVote.npy"), testResultsAE)

    np.save(os.path.join(curExprDir, "BS_testVotes_ClusteringAndVote.npy"), testVotesBS)
    np.save(os.path.join(curExprDir, "BS_testResults_ClusteringAndVote.npy"), testResultsBS)


    return testResultsAE, testResultsBS



def wcdefenses(pred, expertiseMat, defenseName, measureTC=False):
    '''
        Input:
            pred: numOfModels X numOfSamples X numOfClasses
            expertiseMat: numOfModels X numOfClasses
            defenseName: 1s_SM, EM_SM, or EM_MMV

        Output:
            predLabels: 1D numpy array - numOfSamples labels
    '''
    timeCost = -1
    if measureTC:
        numOfSamples = pred.shape[1]
        startTime = time.monotonic()

    if defenseName == "1s_SM":
        ones = np.ones((expertiseMat.shape))
        predLabels = wc_based_defense(pred, ones)
    elif defenseName == "EM_SM":
        predLabels = wc_based_defense(pred, expertiseMat)
    elif defenseName == "EM_MMV":
        predLabels = wc_mv_defense(pred, expertiseMat)
    else:
        errMsg = "Unknown weighted-confidence based defense - defense name is {}.\n".format(defenseName)
        supDefsMsg = "Currently, only support defenses names) - 1s_SM, EM_SM and EM_MMV."
        raise ValueError(errMsg+supDefsMsg)

    if measureTC:
        timeCost = (time.monotonic() - startTime) / numOfSamples

    return predLabels, timeCost

def weightedConfBasedDefsTrainPre(
        curExprDir,
        AEPredProb,
        labels,
        transformationList):

    '''
        Input:
            AEPredProb: (numOfModels, numOfAEs, numOfClasses)
        Output:
            modelsAcc:  a numOfModels X 2 numpy array represents the accuracy of
                        each premitive model including the original/clean model.
            expertiseMat: a numOfTrans X numOfClasses numpy array represents
                            how good each transform model could recover each classes of AEs.
                            TopK is a number between 1 and the num of all transform models
    '''
    numOfModels = AEPredProb.shape[0]
    numOfAEs = AEPredProb.shape[1]
    numOfClasses = AEPredProb.shape[2]
    numOfTrans = len(transformationList) - 1 

    AEPredLC = np.zeros((numOfModels, numOfAEs, 2))
    AEPredLC[:, :, 0] = np.argmax(AEPredProb, axis=2)
    AEPredLC[:, :, 1] = np.max(AEPredProb, axis=2)


    # Compute accuracy for each single model
    printFormat = "{:2}\t{:30}\t{:<6}\n"
    modelsAcc = calAccuracyAllSingleModels(labels, AEPredProb)
    np.save(
        os.path.join(curExprDir, "accuracy_each_single_model_train.npy"),
        modelsAcc)
    with open(os.path.join(curExprDir, "accuracy_each_single_model_train.txt"), "w") as fp:
        fp.write(printFormat.format(
            "ID",
            "Model Name",
            "Acc(AE)"))
        for modelID in range(numOfModels):
            transformType = transformationList[modelID]
            AEAcc = modelsAcc[modelID] 
            fp.write(printFormat.format(
                modelID,
                transformType,
                AEAcc))

    # Count number of samples in each classes in the testing set
    classesCnt=[]
    for i in range(numOfClasses):
        classesCnt.append(len(np.where(labels==i)[0]))
    classesCnt = np.array(classesCnt).reshape((1, numOfClasses))
    np.save(
        os.path.join(curExprDir, "classCount_train.npy"),
        classesCnt)

    # Comptue expertise matrix for transform models
    expertiseMat = np.zeros((numOfTrans, numOfClasses)) 
    for modelID in range(1, numOfModels): # clean model's ID is 0 
        for aeID in range(numOfAEs):
            if AEPredLC[modelID, aeID, 0] == labels[aeID]:
                expertiseMat[modelID-1, labels[aeID]] += 1

    expertiseMat = np.round(expertiseMat/classesCnt, 4)

    np.save(
        os.path.join(curExprDir, "expertiseMat.npy"),
        expertiseMat)

    with open(os.path.join(curExprDir, "expertiseMat.txt"), "w") as fp:
        for transID in range(numOfTrans):
            for labelID in range(numOfClasses):
                fp.write(str(expertiseMat[transID, labelID])+",")
            fp.write("\n")

    return modelsAcc, expertiseMat

def weightedConfBasedDefsTrain(
        curExprDir,
        useLogits,
        AEPred,
        labels,
        modelsAcc,
        expertiseMat):

    '''
        Input:
            AEPred: (numOfTrans, numOfAEs, numOfClasses)
            expertiseMat: (numOfTrans, numOfClasses)
            modelsAcc: accuracy of all transform models under an attack
        Output:
            bestTopKEM : a dictionary - key is wc defense name, value is a tuple of the best expertise mattrix and an array of best model IDs
    '''
    numOfTrans = AEPred.shape[0]
   

    # Find the best TopK expertise matrix that will be used for online testing
    sortedIndices = np.argsort(-modelsAcc) 

    bestTopKEM = {}

    for defenseIdx in range(numOfWCDefenses):
        defenseName = wcDefenseNames[defenseIdx]

        ## numOfWCBasedDefenses is 3
        AEAccsWC = np.zeros((numOfTrans))

        for num in range(numOfTrans):
            topModelIndices = sortedIndices[0:num+1]

            curAEPred = AEPred[topModelIndices, :, :]

            curExpertiseMat = expertiseMat[topModelIndices, :]

            predLabelsAE, timeCostAE = wcdefenses(
                    curAEPred, curExpertiseMat, defenseName, measureTC=False)
             
            AEAccsWC[num] = calAccuracy(predLabelsAE, labels)


        bestTopK = np.argmax(AEAccsWC)
        bestAccsAndTopK = np.hstack((AEAccsWC[bestTopK], bestTopK))

        bestModelIDs = sortedIndices[0:bestTopK+1]
        bestTopKExpertiseMat = expertiseMat[bestModelIDs, :]

        bestTopKEM[defenseName] = (bestTopKExpertiseMat, bestModelIDs)

        logitPrefix=""
        if useLogits:
            logitPrefix = "LG_"
        allAccsFP = os.path.join(
                curExprDir, logitPrefix+"train_all_accuracies_"+defenseName+".npy")
        bestAccTopKFP = os.path.join(
                curExprDir, logitPrefix+"train_best_accuracy_"+defenseName+".npy")
        bestTopKEMFP = os.path.join(
                curExprDir, logitPrefix+"train_topK_expertise_mattrix_"+defenseName+".npy")
        bestModelIDsFP = os.path.join(
                curExprDir, logitPrefix+"train_topK_model_IDs_"+defenseName+".npy")

        
        np.save(allAccsFP, AEAccsWC)
        np.save(bestAccTopKFP, bestAccsAndTopK)
        np.save(bestTopKEMFP, bestTopKExpertiseMat)
        np.save(bestModelIDsFP, bestModelIDs)

    return bestTopKEM


def weightedConfBasedDefsTest(
        curExprDir,
        pred,
        labels,
        EMModels):

    '''
        Input:
            AEPred: (numOfTrans, numOfSamples, numOfClasses)
            EMModels: dictionary. wc_defense_name / (expertise mattrix, array of model IDs)
        Output:
            testResult: (numOfWCDefenses, 2)
            votedResults: (numOfWCDefenses, numOfSamples).
    '''
    numOfSamples = pred.shape[1]
    votedResults = np.zeros((numOfWCDefenses, numOfSamples))   
    # 3: 0-Accuracy, 1-Time cost per sample in seconds
    testResult = np.zeros((numOfWCDefenses, 2))

    for defenseIdx in range(numOfWCDefenses):
        defenseName = wcDefenseNames[defenseIdx]

        curExpertiseMat = EMModels[defenseName][0]
        topModelIndices = EMModels[defenseName][1] 

        curPred = pred[topModelIndices, :, :]


        predLabels, timeCost = wcdefenses(
                curPred, curExpertiseMat, defenseName, measureTC=True)

        testResult[defenseIdx, 0] = calAccuracy(predLabels, labels)
        testResult[defenseIdx, 1] = timeCost

        votedResults[defenseIdx, :] = predLabels

    return testResult, votedResults


def weightedConfDefenseEvaluation(
        curExprDir,
        predProbAETr,
        predLogitsAETr,
        labelsTr,
        predProbAETe,
        predLogitsAETe,
        labelsTe,
        transformationList,
        predProbBS,
        predLogitsBS,
        labelsBS):

    '''
        Input:
            predProbAETr: numOfModels X numOfSamples X 10
            predProbAETe: numOfTrans X numOfSamples X  10
            predProbBS  : numOfTrans X numOfSamples X 10
            predLogitsAETr: numOfModels X numOfSamples X 10
            predLogitsAETe: numOfTrans X numOfSamples X  10
            predLogitsBS  : numOfTrans X numOfSamples X 10
    '''

    # TRAIN
    # include the original model
    # modelsAcc: (numOfModels)
    print("\t\t=== pre-train ====")
    modelsAcc, expertiseMat = weightedConfBasedDefsTrainPre(
            curExprDir,
            predProbAETr,
            labelsTr,
            transformationList)

    # For TESTING
    numOfGroups = 2
    numOfTestingSamples = predProbAETe.shape[1]
    AETestResults = np.zeros((numOfGroups, numOfWCDefenses, 2))
    AEVotedResults = np.zeros((numOfGroups, numOfWCDefenses, numOfTestingSamples))

    numOfBSSamples = predProbBS.shape[1]
    BSTestResults = np.zeros((numOfGroups, numOfWCDefenses, 2))
    BSVotedResults = np.zeros((numOfGroups, numOfWCDefenses, numOfBSSamples))

    # testResults[2, numOfWCDefenses-1] is non-sense. Same to other three arraies.
    for plIdx, useLogits, AEPredTr, AEPredTe, BSPred in zip(
            list(range(numOfGroups)),
            [False, True],
            [predProbAETr, predLogitsAETr],
            [predProbAETe, predLogitsAETe],
            [predProbBS, predLogitsBS]):

        predictionType = "probability" if not useLogits else "logit"
        # TRAIN
        # modelsAcc[1:]: exclude the original model at the index 0 in modelsAcc
        print("\t\t==== Training with {} ====".format(predictionType))
        EMModels = weightedConfBasedDefsTrain(
                curExprDir,
                useLogits,
                AEPredTr[1:],
                labelsTr,
                modelsAcc[1:],
                expertiseMat)


        # TEST
        print("\t\t==== Testing with {} ====".format(predictionType))
        # AE
        AETestResults[plIdx, :, :], AEVotedResults[plIdx, :, :] = weightedConfBasedDefsTest(
                curExprDir,
                AEPredTe,
                labelsTe,
                EMModels)

        # BS
        BSTestResults[plIdx, :, :], BSVotedResults[plIdx, :, :] = weightedConfBasedDefsTest(
                curExprDir,
                BSPred,
                labelsBS,
                EMModels)


    votesFP = os.path.join(curExprDir, "AE_testVotes_WeightedConfDefenses.npy")
    testResultsFP = os.path.join(curExprDir, "AE_testResults_WeightedConfDefenses.npy")
    np.save(votesFP, AEVotedResults)
    np.save(testResultsFP, AETestResults)

    votesBSFP = os.path.join(curExprDir, "BS_testVotes_WeightedConfDefenses.npy")
    testResultsBSFP = os.path.join(curExprDir, "BS_testResults_WeightedConfDefenses.npy")
    np.save(votesBSFP, BSVotedResults)
    np.save(testResultsBSFP, BSTestResults)
 

    return AETestResults, BSTestResults

def curvePlot(xs, ys, xlabel, ylabel, labelFontSize, tickFontSize, lineColor, marker, outputfile, title):
        #print("\t[curvePlot] ploting " + outputfile)

        plt.plot(xs, ys, marker+lineColor)
        plt.xlabel(xlabel, fontsize=labelFontSize)
        plt.ylabel(ylabel, fontsize=labelFontSize)
        plt.xticks(fontsize=tickFontSize)
        plt.yticks(fontsize=tickFontSize)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outputfile)
        plt.close()
        #print("\tploting finished!")

def drawUBCurve(curDir, predLC, labels, AEType, modelsAcc):
    '''
        predLC:
            numOfTrans X numOfSamples X 2
    '''
    numOfTrans = predLC.shape[0]
    accsUB = np.zeros((numOfTrans))
    numOfTopKModels = np.zeros((numOfTrans))

    sortedIndices = np.argsort(-modelsAcc) 

    for numOfWeakModelsToExclude in range(numOfTrans):
        numOfTopKModelsToKeep = numOfTrans - numOfWeakModelsToExclude
        curModelIDs = sortedIndices[:numOfTopKModelsToKeep]
        clusters = []
        for modelID in range(numOfTopKModelsToKeep):
            clusters.append([modelID])
        accsUB[numOfWeakModelsToExclude] = getUpperBoundAccuracy(predLC[curModelIDs], clusters, labels)
        numOfTopKModels[numOfWeakModelsToExclude] = numOfTrans - numOfWeakModelsToExclude

    np.save(os.path.join(curDir, "upper_bound_accuracy_curver_data.npy"), np.vstack((numOfTopKModels, accsUB))) 
    
    curvePlot(
            numOfTopKModels,
            accsUB,
            "Number of Top K Transform Models",
            "Upper-Bound Accuracy",
            18,
            16,
            "c",
            "-x",
            os.path.join(curDir, "upper_bound_accuracy_vs_topk_models.pdf"),
            AEType)

def boxPlot(data, title, xLabels, yLabel, saveFP):
    green_diamond = dict(markerfacecolor='g', marker='D')
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(data, notch=True, flierprops=green_diamond, labels = xLabels)
    ax.set_ylabel(yLabel)
    fig.savefig(saveFP, bbox_inches='tight')
    plt.close(fig)

def postAnalysis(
        experimentRootDir,
        kFold,
        predictionResultDir,
        foldDirs,
        sampleTypes):

    AETypes = sampleTypes[1:]
    # for weighted-condience based defenses,
    # there are two groups. One uses probability, the other uses logit.
    numOfDefenses = numOfCVDefenses + numOfWCDefenses*2 
    numOfAETypes = len(AETypes)
    
    TrainAccs   = np.zeros((kFold, numOfAETypes, numOfDefenses))
    TestAccsAE  = np.zeros((kFold, numOfAETypes, numOfDefenses))
    TestAccsBS  = np.zeros((kFold, numOfAETypes, numOfDefenses))

    postAnaDir=os.path.join(experimentRootDir, "result_of_post_analysis")
    createDirSafely(postAnaDir)

    defenseTCs = np.zeros((kFold, numOfAETypes, numOfDefenses))
    for foldIdx in range(kFold):
        for AETypeIdx in range(numOfAETypes):
            AEType = AETypes[AETypeIdx]
            curExprDir = os.path.join(foldDirs[foldIdx], AEType)

            # clustering-and-voting based defenses
            ensembleModelsCVDefenses = np.load(os.path.join(curExprDir, "ensemble_models_clustering_based_defenses.npy"))
            TrainAccs[foldIdx, AETypeIdx, 0:numOfCVDefenses] = ensembleModelsCVDefenses[:, 1]

            AETestResultsCAV = np.load(os.path.join(curExprDir, "AE_testResults_ClusteringAndVote.npy"))
            TestAccsAE[foldIdx, AETypeIdx, 0:numOfCVDefenses] = AETestResultsCAV[:, 0]

            BSTestResultsCAV = np.load(os.path.join(curExprDir, "BS_testResults_ClusteringAndVote.npy"))
            TestAccsBS[foldIdx, AETypeIdx, 0:numOfCVDefenses] = BSTestResultsCAV[:, 0]

            defenseTCs[foldIdx, AETypeIdx, 0:numOfCVDefenses] = AETestResultsCAV[:, 1]
            
            # weighted-confidence baased defenses
            TrainAccs[foldIdx, AETypeIdx, numOfCVDefenses]   = np.load(os.path.join(curExprDir, "train_best_accuracy_1s_SM.npy"))[0]
            TrainAccs[foldIdx, AETypeIdx, numOfCVDefenses+1] = np.load(os.path.join(curExprDir, "train_best_accuracy_EM_SM.npy"))[0]
            TrainAccs[foldIdx, AETypeIdx, numOfCVDefenses+2] = np.load(os.path.join(curExprDir, "train_best_accuracy_EM_MMV.npy"))[0]
            TrainAccs[foldIdx, AETypeIdx, numOfCVDefenses+3] = np.load(os.path.join(curExprDir, "LG_train_best_accuracy_1s_SM.npy"))[0]
            TrainAccs[foldIdx, AETypeIdx, numOfCVDefenses+4] = np.load(os.path.join(curExprDir, "LG_train_best_accuracy_EM_SM.npy"))[0]
            TrainAccs[foldIdx, AETypeIdx, numOfCVDefenses+5] = np.load(os.path.join(curExprDir, "LG_train_best_accuracy_EM_MMV.npy"))[0]

            AETestResultWCD = np.load(os.path.join(curExprDir, "AE_testResults_WeightedConfDefenses.npy"))
            TestAccsAE[foldIdx, AETypeIdx, numOfCVDefenses:] = np.hstack((AETestResultWCD[0, :, 0], AETestResultWCD[1, :, 0]))
            BSTestResultWCD = np.load(os.path.join(curExprDir, "BS_testResults_WeightedConfDefenses.npy"))
            TestAccsBS[foldIdx, AETypeIdx, numOfCVDefenses:] = np.hstack((BSTestResultWCD[0, :, 0], BSTestResultWCD[1, :, 0]))

            defenseTCs[foldIdx, AETypeIdx, numOfCVDefenses:] = np.hstack((AETestResultWCD[0, :, 1], AETestResultWCD[1, :, 1]))
        curDefenseTCsFP = os.path.join(foldDirs[foldIdx], "defenseTimeCost_fold"+str(foldIdx+1)+"_in_ms.txt")
        create2DTable(
                np.round(defenseTCs[foldIdx, :, :]*1000, decimals=6),
                defensesList,
                AETypes,
                curDefenseTCsFP)



    np.save(os.path.join(postAnaDir, "TestLatency_DefenseTimeCost.npy"), defenseTCs)
    msDefenseTCs = defenseTCs*1000 # ms: millium seconds
    meanDefenseTCsFP = os.path.join(experimentRootDir, "meanDefenseTimeCostInMS.txt")
    stdDefenseTCsFP = os.path.join(experimentRootDir, "stdDefenseTimeCostInMS.txt")
    create2DTable(msDefenseTCs.mean(axis=0).round(decimals=6), defensesList, AETypes, meanDefenseTCsFP)
    create2DTable(msDefenseTCs.std(axis=0).round(decimals=6), defensesList, AETypes, stdDefenseTCsFP)



    np.save(os.path.join(postAnaDir, "TrainAccs_fold_AEType.npy"), np.round(TrainAccs, decimals=4))
    np.save(os.path.join(postAnaDir, "TestAccsAE_fold_AEType.npy"), np.round(TestAccsAE, decimals=4))
    np.save(os.path.join(postAnaDir, "TestAccsBS_fold_AEType.npy"), np.round(TestAccsBS, decimals=4))


    # 0 - accuracy of clean model, 1 - upper-bound accuracy
    AccCMAndUB = np.zeros((numOfAETypes, 2))
    AccRD = np.zeros((numOfAETypes))
    labels = np.load(os.path.join(predictionResultDir, "labels.npy"))
    for AETypeIdx in range(numOfAETypes):
        AEType = AETypes[AETypeIdx]
        curPredDir = os.path.join(predictionResultDir, AEType)
        curDir = os.path.join(postAnaDir, AEType)
        createDirSafely(curDir)

        predProb = np.load(os.path.join(curPredDir, "predProb.npy"))
        numOfModels = predProb.shape[0]
        numOfSamples = predProb.shape[1]
        numOfTrans = numOfModels-1

        predProbCM = predProb[0]
        AccCMAndUB[AETypeIdx, 0] = calAccProb(predProbCM, labels)

        predLC = np.zeros((numOfModels, numOfSamples, 2))
        predLC[:, :, 0] = np.argmax(predProb, axis=2)
        predLC[:, :, 1] = np.max(predProb, axis=2)
        clusters = []
        for modelIdx in range(numOfTrans): # only consider transform models when calculting upper-bound accuracy
            clusters.append([modelIdx])
        AccCMAndUB[AETypeIdx, 1] = getUpperBoundAccuracy(predLC[1:], clusters, labels)

        modelsAcc = np.zeros((numOfTrans))
        for idx in range(numOfTrans):
            modelsAcc[idx] = calAccuracy(predLC[idx, :, 0], labels)
        AccRD[AETypeIdx] = np.round(modelsAcc.mean(), decimals=4)
       
        drawUBCurve(curDir, predLC[1:], labels, AEType, modelsAcc)


    np.round(AccCMAndUB, decimals=4)
    np.save(os.path.join(postAnaDir, "accuracy_clean_model_and_upper_bound.npy"), AccCMAndUB)

    # create accuracy tables: mean and std
    # for Training part
    aveTrainAccsFP=os.path.join(postAnaDir, "ave_train_accs_table.txt")
    create2DTable(
            np.round(TrainAccs.mean(axis=0), decimals=4),
            defensesList,
            AETypes,
            aveTrainAccsFP)
    stdTrainAccsFP=os.path.join(postAnaDir, "std_train_accs_table.txt")
    create2DTable(
            np.round(TrainAccs.std(axis=0), decimals=4),
            defensesList,
            AETypes,
            stdTrainAccsFP)

    # for Testing AEs
    aveTestAEAccsFP=os.path.join(postAnaDir, "ave_test_AEs_accs_table.txt")
    TestAccsAE2 = np.hstack((np.round(TestAccsAE.mean(axis=0), decimals=4), AccRD.reshape((numOfAETypes, 1)), AccCMAndUB))
    colHeaders = defensesList.copy()
    colHeaders.extend(["Random Defense", "Clean Model", "Upper Bound"])
    create2DTable(
            TestAccsAE2,
            colHeaders,
            AETypes,
            aveTestAEAccsFP)
    stdTestAEAccsFP=os.path.join(postAnaDir, "std_test_AEs_accs_table.txt")
    create2DTable(
            np.round(TestAccsAE.std(axis=0), decimals=4),
            defensesList,
            AETypes,
            stdTestAEAccsFP)

    # for Testing BSs
    aveTestBSAccsFP=os.path.join(postAnaDir, "ave_test_BSs_accs_table.txt")
    create2DTable(
            np.round(TestAccsBS.mean(axis=0), decimals=4),
            defensesList,
            AETypes,
            aveTestBSAccsFP)
    stdTestBSAccsFP=os.path.join(postAnaDir, "std_test_BSs_accs_table.txt")
    create2DTable(
            np.round(TestAccsBS.std(axis=0), decimals=4),
            defensesList,
            AETypes,
            stdTestBSAccsFP)


    # plot latency
    predTCs = np.load(os.path.join(predictionResultDir, "predTCs.npy"))
    predTCs = predTCs.mean(axis=1)
    xLabels = ["Transformation", "Inference (Probability)", "Inference (Logit)"]
    yLabel = "Time Cost in Millium Seconds"
    for sampleTypeIdx in range(len(sampleTypes)):
        sampleType = sampleTypes[sampleTypeIdx]
        saveFP = os.path.join(predictionResultDir, sampleType+"_latency.pdf")
        boxPlot(predTCs[sampleTypeIdx]*1000, sampleType, xLabels, yLabel, saveFP)

def create2DTable(mat, colHeaders, rowHeaders, filename):
    '''
        mat: a m X n 2D array
        colHeaders: n elements 1D array
        rowHeaders: m elements 1D array
        filename: file path to save the table
    '''

    with open(filename, "w") as fp:
        # dump the headers
        fp.write("\t")
        for header in colHeaders:
            fp.write(header+"\t")
        fp.write("\n")

        # dump each row
        for rIdx in range(len(rowHeaders)):
            fp.write(rowHeaders[rIdx]+"\t")
            for cIdx in range(len(colHeaders)):
                fp.write(str(mat[rIdx, cIdx])+"\t")
            fp.write("\n")

def calAccuracyAllSingleModels(
        labels,
        predProb):
    '''
        Input:
            predProb: (numOfModels X numOfSamples)
        Output:
            modelsAcc: (numOfModels).
    '''
    numOfModels = predProb.shape[0]
    modelsAcc = np.zeros((numOfModels))
    for modelID in range(numOfModels):
        modelsAcc[modelID] = calAccProb(predProb[modelID, :, :], labels)       
    return modelsAcc
 

def calAccuracyAllSingleModels0(
        curExprDir,
        numOfModels,
        labels,
        AEPredProb,
        BSPredProb,
        transformationList):
    '''
        Output:
            modelsAcc: (numOfModels, 2). 2 - AE and BS
    '''
    modelsAcc = np.zeros((numOfModels, 2))
    printFormat = "{:2}\t{:30}\t{:<6}\t{:<6}\n"
    with open(os.path.join(curExprDir, "accuracy_each_single_model.txt"), "w") as fp:
        fp.write(printFormat.format(
            "ID",
            "Model Name",
            "Acc(AE)",
            "Acc(BS)"))
        for modelID in range(numOfModels):
            transformType = transformationList[modelID]
            AEAcc = calAccProb(AEPredProb[modelID, :, :], labels)       
            BSAcc = calAccProb(BSPredProb[modelID, :, :], labels)  
            fp.write(printFormat.format(
                modelID,
                transformType,
                AEAcc,
                BSAcc))
            modelsAcc[modelID, 0] = AEAcc
            modelsAcc[modelID, 1] = BSAcc
    np.save(
        os.path.join(curExprDir, "accuracy_each_single_model.npy"),
        modelsAcc)
    return modelsAcc
 

def kFoldPredictionSetup(
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
        isKFoldUponTestSet=True):
    '''
        Input:
            isKFoldUponTestSet = False means to take a fold of samples for training instead of testing.

        Output:
    '''
    # Load models and create models to output logits
    modelFilenamePrefix = datasetName+"-"+architecture
    models, logitsModels = loadModels(modelsDir, modelFilenamePrefix, transformationList)
    numOfModels = len(models) # include the clean model, positioning at index 0

    # connect input images with predicted results 
    kFoldImgIndices = np.array(range(numOfSamples))
    np.random.shuffle(kFoldImgIndices)
    np.save(os.path.join(experimentRootDir, "kFoldImgIndices.npy"), kFoldImgIndices)

    # Load sample dataset
    sampleFilenameTag = datasetName+"-"+architecture+"-"+targetModelName
    labels_raw = np.load(os.path.join(samplesDir, "Label-"+datasetName+"-"+targetModelName+".npy"))
    labels = np.argmax(labels_raw, axis=1)
    labels = labels[kFoldImgIndices]

    sampleTypes =["BS"]
    sampleTypes.extend(AETypes)
    numOfAETypes = len(AETypes)
    numOfSampleTypes = 1+numOfAETypes
    # average time cost in seconds 
    # averaing upon samples
    predTCs = np.zeros((numOfSampleTypes, kFold, numOfModels, 3))

    oneFoldAmount = int(numOfSamples/kFold)

    for sampleTypeIdx, sampleType in zip(list(range(numOfSampleTypes)), sampleTypes):
        print("Sample type: "+sampleType)
        if sampleType == "BS":
            sampleFilename = "BS-"+datasetName+"-"+targetModelName+".npy"
        else:
            sampleFilename = "AE-"+sampleFilenameTag+"-"+sampleType+".npy"

        curExprDir = os.path.join(predictionResultDir, sampleType)
        createDirSafely(curExprDir)

        samples = np.load(os.path.join(samplesDir, sampleFilename))
        samples = samples[kFoldImgIndices]

        for foldIdx in range(1, 1+kFold):
            print("\tPrediction on {} on fold {}".format(sampleType, foldIdx))
            if foldIdx != kFold:
                if isKFoldUponTestSet:
                    foldIndices  = np.array(range((foldIdx-1)*oneFoldAmount, foldIdx*oneFoldAmount))
                else:
                    foldIndices = np.hstack((
                        np.array(range(0, (foldIdx-1)*oneFoldAmount)),
                        np.array(range(foldIdx*oneFoldAmount, numOfAEs))))
            else:
                if isKFoldUponTestSet:
                    foldIndices  = np.array(range((foldIdx-1)*oneFoldAmount, numOfSamples))
                else:
                    foldIndices = np.array(range(0, (foldIdx-1)*oneFoldAmount))

            curSamples = samples[foldIndices]
            numOfCurSamples = curSamples.shape[0] 

            # 0 - Transform TC, 1 - Prediction (Prob) TC 2 - Prediction (Logit) TC
            curPredTCs  = np.zeros((numOfModels, 3))
            predShape = (numOfModels, numOfCurSamples, numOfClasses)
            curPredProb   = np.zeros(predShape)
            curPredLogits = np.zeros(predShape)

            for modelID in range(numOfModels):
                transformType = transformationList[modelID]

                # Transformation cost
                startTime = time.monotonic()
                tranSamples = transform_images(curSamples, transformType)
                endTime = time.monotonic()
                curPredTCs[modelID, 0] = endTime - startTime

                # model prediction cost - using probability-based defense
                curPredProb[modelID, :, :],   curPredTCs[modelID, 1] = prediction(
                        tranSamples,
                        models[modelID])
                # model prediction cost - using logits-based defense
                curPredLogits[modelID, :, :], curPredTCs[modelID, 2] = prediction(
                        tranSamples,
                        logitsModels[modelID])

       
            predTCs[sampleTypeIdx, foldIdx-1, : ,:] = curPredTCs / numOfCurSamples

            # stack up the result of the current fold
            if foldIdx == 1:
                predProb      = curPredProb
                predLogits    = curPredLogits
            else:
                predProb      = np.hstack((predProb, curPredProb))
                predLogits    = np.hstack((predLogits, curPredLogits))

        np.save(os.path.join(curExprDir, "predProb.npy"), predProb)
        np.save(os.path.join(curExprDir, "predLogit.npy"), predLogits)
    
    np.save(os.path.join(predictionResultDir, "predTCs.npy"), predTCs)
    np.save(os.path.join(predictionResultDir, "labels.npy"), labels)
       




def kFolderPredictionSetup0(
        experimentRootDir,
        kFolder,
        folderDirs,
        predictionResultDir,
        datasetName,
        architecture,
        numOfClasses,
        targetModelName,
        modelsDir,
        AEDir,
        numOfAEs,
        AETypes,
        transformationList,
        isKFolderUponTestSet=True):
    '''
        Input:
            isKFolderUponTestSet = False means to take a folder of samples for training instead of testing.

        Output:
    '''
    # Load models and create models to output logits
    modelFilenamePrefix = datasetName+"_"+architecture
    models, logitsModels = loadModels(modelsDir, modelFilenamePrefix, transformationList)
    numOfModels = len(models)

    # connect input images with predicted results 
    kFolderImgIndices = np.array(range(numOfAEs))
    np.random.shuffle(kFolderImgIndices)
    np.save(os.path.join(experimentRootDir, "kFolderImgIndices.npy"), kFolderImgIndices)
    oneFolderAmount = int(numOfAEs/kFolder)

    sampleFilenamePrefix = datasetName+"-"+architecture+"-"+targetModelName
    numOfAETypes = len(AETypes)
    # 0 - time cost of the corresponding transformation
    # 1 - time cost of model inference for probability
    # 2 - time cost of model inference for logits
    predTCs = np.zeros((kFolder, numOfAETypes, numOfModels, 3))
    for AETypeIdx, AEType in zip(list(range(numOfAETypes)), AETypes):
        curExprDir = os.path.join(predictionResultDir, sampleFilenamePrefix+"_"+AEType)
        createDirSafely(curExprDir)

        LBs_vec = np.load(os.path.join(AEDir, "label_mnist_cnn_clean_"+AEType+".npy"))
        LBs = np.argmax(LBs_vec, axis=1)
        AEs = np.load(os.path.join(AEDir, "adv_mnist_cnn_clean_"+AEType+".npy"))
        BSs = np.load(os.path.join(AEDir, "orig_mnist_cnn_clean_"+AEType+".npy"))    
       
        predProbAE, predProbBS, predLogitsAE, predLogitsBS, newLBs = None, None, None, None, None

        for folderIdx in range(1, 1+kFolder):
            print("\tPrediction on {} on fold {}".format(AEType, folderIdx))
            if folderIdx != kFolder:
                if isKFolderUponTestSet:
                    folderIndices  = np.array(range((folderIdx-1)*oneFolderAmount, folderIdx*oneFolderAmount))
                else:
                    folderIndices = np.hstack((
                        np.array(range(0, (folderIdx-1)*oneFolderAmount)),
                        np.array(range(folderIdx*oneFolderAmount, numOfAEs))))
            else:
                if isKFolderUponTestSet:
                    folderIndices  = np.array(range((folderIdx-1)*oneFolderAmount, numOfAEs))
                else:
                    folderIndices = np.array(range(0, (folderIdx-1)*oneFolderAmount))

            curAEs = AEs[kFolderImgIndices[folderIndices]]
            curBSs = BSs[kFolderImgIndices[folderIndices]]
            curLBs = LBs[kFolderImgIndices[folderIndices]]

            predTCPerModel  = np.zeros((numOfModels, 2, 3))
            predShape = (numOfModels, numOfAEs, numOfClasses)
            curPredProbAE   = np.zeros(predShape)
            curPredProbBS   = np.zeros(predShape)
            curPredLogitsAE = np.zeros(predShape)
            curPredLogitsBS = np.zeros(predShape)

            for modelID in range(numOfModels):
                curAEs = np.copy(AEs)
                curBSs = np.copy(BSs)
                
                transformType = transformationList[modelID]
                print("Predict AE({}) and BS with model {} - {}".format(AEType, modelID, transformType))

                # AE - transformation cost
                startTime = time.monotonic()
                curAEs = transform_images(curAEs, transformType)
                endTime = time.monotonic()
                predTCPerModel[modelID, 0, 0] = endTime - startTime


                # AE - model prediction cost - using probability-based defense
                curPredProbAE[modelID, :, :],   predTCPerModel[modelID, 0, 1] = prediction(
                        curAEs,
                        models[modelID])
                
                # AE - model prediction cost - using logits-based defense
                curPredLogitsAE[modelID, :, :], predTCPerModel[modelID, 0, 2] = prediction(
                        curAEs,
                        logitsModels[modelID])

                # BS - transformation cost
                startTime = time.monotonic()
                curBSs = transform_images(curBSs, transformType)
                endTime = time.monotonic()
                predTCPerModel[modelID, 1, 0] = endTime - startTime
                           
                # BS - model prediction cost - using probability-based defense
                curPredProbBS[modelID, :, :],   predTCPerModel[modelID, 1, 1] = prediction(
                        curBSs,
                        models[modelID])
                
                # BS - model prediction cost - using logits-based defense
                curPredLogitsBS[modelID, :, :], predTCPerModel[modelID, 1, 2] = prediction(
                        curBSs,
                        logitsModels[modelID])
         
            # modelID = 0 => original model
            predTCs[folderIdx-1, AETypeIdx, : ,:] = predTCPerModel.sum(axis=1)

            # stack up the result of the current folder
            if folderIdx == 1:
                predProbAE      = curPredProbAE
                predProbBS      = curPredProbBS
                predLogitsAE    = curPredLogitsAE
                predLogitsBS    = curPredLogitsBS
                newLBs          = curLBs
            else:
                predProbAE      = np.hstack((predProbAE, curPredProbAE))
                predProbBS      = np.hstack((predProbBS, curPredProbBS))
                predLogitsAE    = np.hstack((predLogitsAE, curPredLogitsAE))
                predLogitsBS    = np.hstack((predLogitsBS, curPredLogitsBS))
                newLBs          = np.hstack((newLBs, curLBs)) # 1D numpy array => hstack

        np.save(os.path.join(curExprDir, "predProbAE.npy"), predProbAE)
        np.save(os.path.join(curExprDir, "predProbBS.npy"), predProbBS)
        np.save(os.path.join(curExprDir, "predLogitsAE.npy"), predLogitsAE)
        np.save(os.path.join(curExprDir, "predLogitsBS.npy"), predLogitsBS)
        np.save(os.path.join(curExprDir, "labels.npy"), newLBs)

    np.save(os.path.join(predictionResultDir, "predTCs.npy"), predTCs)
    return predictionResultDir, numOfModels, folderDirs

def loadModels(modelsDir, modelFilenamePrefix, transformationList):
    models=[]
    logitsModels=[]
    for transformType in transformationList:
        modelName = modelFilenamePrefix+"-"+transformType
        modelNameFP = os.path.join(modelsDir, modelName+".h5")
        print("load model {}".format(modelName))
        model = load_model(modelNameFP)
        models.append(model)
        # Create corresponding model for outputing logits
        layerName=model.layers[-2].name
        logitsModel = Model(
                inputs=model.input,
                outputs=model.get_layer(layerName).output)
        logitsModels.append(logitsModel)

    return models, logitsModels


def prediction(X, model):

    startTime   = time.monotonic()
    pred        = model.predict(X)
    timeCost    = time.monotonic() - startTime

    return pred, timeCost
               

def calLatency(totalPredTC, totalEvalClusteringTC, totalEvalWCTC, useLogits):
    # totalPredTC            : (numOfAETypes, numOfModels, 2, 2)
    #                           2 - AE and BS. 2 - predProb and predLogits
    # totalEvalClusteringTC  : (numOfAETypes, numOfCVDefenses)
    # totalEvalWCTC          : (numOfAETypes, numOfWCBasedDefenses)

    numOfAETypes = totalPredTC.shape[0]
    numOfTrans = totalPredTC.shape[1] - 1
    numOfClusteringDefenses = totalEvalClusteringTC.shape[1]
    numOfWCDefenses         = totalEvalWCTC.shape[1]
    latencies = np.zeros((numOfAETypes, numOfClusteringDefenses+numOfWCDefenses+1)) # 1: original latency
    
    for AETypeIdx in range(numOfAETypes):
        # classification time cost for 2 * numOfAEs
        originalPredTC = totalPredTC[AETypeIdx, 0, :, 0].sum()

        # time cost of using eacn transformation-based model
        # and defense approaches
        ## prediction cost
        predTC = np.max(totalPredTC[AETypeIdx, 1:, :, 0], axis=0).sum()
        if useLogits:
            predTC = np.max(totalPredTC[AETypeIdx, 1:, :, 1], axis=0).sum()
        ## defense cost
        defenseTCs = np.hstack((totalEvalClusteringTC[AETypeIdx, :], totalEvalWCTC[AETypeIdx, :])) 
        ## entire cost
        end2EndTCs = predTC + defenseTCs

        latencies[AETypeIdx, :] = np.hstack((end2EndTCs, originalPredTC))

    return latencies

def createDirSafely(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def calAccuracy(predLabels, trueLabels):
    return round(
            1.0 * len(np.where(trueLabels == predLabels)[0]) / len(trueLabels),
            4)

def calAccProb(predProb, trueLabels):
    '''
        Input:
            predProb: (numOfSamples,  numOfClasses)
        Output:
            predLabels: (numOfSamples)
    '''
    predLabels = np.argmax(predProb, axis=1)
    return calAccuracy(predLabels, trueLabels)


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
        voteResult[eventID, 1] = countDict[votingResult][1]/count 

    return voteResult

def maxConfidenceVote(participants):
    '''
        Input:
            participants: a list of opinions. Each element in the list is a numpy array, N X 2.
                            N is the number of events. The second dimension contains (opinion/label, confidence)
        Output:
            voteResult  : a numapy array NX2 represents opinion and confidence across N events 
    '''
    numOfEvents = participants[0].shape[0]
    voteResult = np.zeros((numOfEvents, 2))
    for eventID in range(numOfEvents):
        label = None
        maxConf = -np.inf
        for participant in participants:
            if maxConf < participant[eventID][1]:
                label = participant[eventID][0]
        voteResult[eventID, 0] = label
        voteResult[eventID, 1] = maxConf

    return voteResult

def upperBoundAccuracy(opinions, labels):
    '''
        opinions: a list of each voter's opinions across all events. Each element in the list is a numpy array, numOfEvents X 2
        labels  : a 1D numpy array
    '''
    ret = 0.0
    numOfEvents = opinions[0].shape[0]
    cnt = 0
    for eID in range(numOfEvents):
        for opinion in opinions:
            if labels[eID] == opinion[eID][0]:
                cnt+=1
                break
    ret = round(cnt/numOfEvents, 4)
    return ret

def votingAsDefense(predLC, clusters, vsac="Majority", measureTC=False):
    '''
        Input:
            predLC  :   numOfModels X numOfSamples X 2 numpy array. The 3dr dimension: (label, confidence)
            clusters:   a 2D lists where each element is a list of models' ID, representing a cluster
            acvs    :   across-clusters voting strategy, default stragety is majority voting.
                        Note: voting strategy inside cluster is majority voting.
            measureTC: a boolean parameter tells if time cost is measured
        Output:
            timeCost: defense time cost in second per sample. -1 means time cost is not measured.
            votedResult: numOfSamples X 2 numpy array. The 1st column represents the classified label while the 2nd column tells the confidence about this label.
    '''
    timeCost=-1

    if measureTC:
        numOfSamples = predLC.shape[1]
        startTime = time.monotonic()
    clusterRepresentatives = []
    for cluster in clusters:
        insideClusterOpinions = []
        for modelID in cluster: # transform model ID start from 1
            insideClusterOpinions.append(predLC[modelID-1])
        votedResult = majorityVote(insideClusterOpinions)
        clusterRepresentatives.append(votedResult)

    if vsac == "Majority":
        votedResult = majorityVote(clusterRepresentatives)
    elif vsac == "Max":
        votedResult = maxConfidenceVote(clusterRepresentatives)
    else:
        raise ValueError("The given across-clusters voting strategy, {}, is not supported. By now, only support 'Majority' and 'Max'".format(vsac))

    if measureTC:
        timeCost = (time.monotonic() - startTime) / numOfSamples

    return votedResult, timeCost
 


def getUpperBoundAccuracy(predLC, clusters, trueLabels): # cover all clusters
    '''
        Note: a sample will be considered as classfied correctly as long as it is correctly classified by some cluster based on majority voting
        Input:
            predLC  :   numOfModels X numOfSamples X 2 numpy array. The 3dr dimension: (label, confidence)
            clusters:   a 2D lists where each element is a list of models' ID, representing a cluster

        Output:
            upper bound accuracy for the clustering result represented by "clusters"
    '''
    clusterRepresentatives = []
    for cluster in clusters:
        insideClusterOpinions = []
        for modelID in cluster:
            insideClusterOpinions.append(predLC[modelID, :, :])
        clusterRepresentatives.append(majorityVote(insideClusterOpinions))

    return upperBoundAccuracy(clusterRepresentatives, trueLabels)

def loadClusteringResult(clusteringResultDir, numOfClusters):
    '''
        Output:
            clusters:   a 2D list where each element is a list of model IDs, represents a cluster
    '''
    # Note: model IDs appear in the clustering result file is 1-based.
    filepath = os.path.join(clusteringResultDir, "C"+str(numOfClusters)+".txt")
    clusters = []
    with open(filepath) as fp:
        for line in fp:
            line = line.rstrip()
            if line == "":
                continue

            parts = line.split()
            cluster = []
            for modelID in parts:
                cluster.append(int(modelID))
            clusters.append(cluster)
    return clusters

def loadAllClusteringResult(clusteringResultDir, maxNumOfClusters):
    '''
        Output:
            clusteringResult:   a dictionary where the key is the numOfClusters, value is the corresponding FCM clustering result, a 2D list.
    '''
    clusteringResult = {}
    for numOfClusters in range(1, maxNumOfClusters+1):
        clusteringResult[numOfClusters] = loadClusteringResult(clusteringResultDir, numOfClusters)
    return clusteringResult


# Defense approach based weighted sum of each model's initial confidences
def wc_based_defense(pred, modelWeights=None):
    '''
        Input:
            pred: numOfModels X numOfSamples X 10. Either probability or logits
            modelWeights: 2D numpy array - numOfTranModels X 10
        Output:
            predLabels: 1D numpy array - numOfSamples X 1
    '''
    numOfModels = pred.shape[0]
    numOfSamples = pred.shape[1]
    numOfClasses = pred.shape[2]
    weightedPredProb = np.zeros(pred.shape)
    if modelWeights is None:
        modelWeights = np.ones((numOfModels, numOfClasses)) # each model is 100% trusted
   
    predLabels = np.zeros((numOfSamples))
    for sampleID in range(numOfSamples):

        wm = np.multiply(pred[:,sampleID,:], modelWeights)
        mean_wm = wm.mean(axis=0)
        predLabels[sampleID] = np.argmax(mean_wm)

    return predLabels

# Calculate the accuracy of the ensemble model
# based max trustworthness and majority voting
def wc_mv_defense(pred, modelWeights):
    '''
        Input:
            pred: numOfModels X numOfSamples X 10. Either probability or logits
            modelWeights: 2D numpy array - numOfModels X 10
        Output:
            PredLabels: 1D numpy array - numOfSamples X 1
    '''
    numOfModels = pred.shape[0]
    numOfSamples = pred.shape[1]
    numOfClasses = pred.shape[2]
    weightedPredProb = np.zeros(pred.shape)
   
    predLabels = np.zeros((numOfSamples))
    for sampleID in range(numOfSamples):
        # max trustworthness
        weightedProb = np.multiply(pred[:,sampleID,:], modelWeights)
        predLabelsPerSample = np.argmax(weightedProb, axis=1)
        confidencesPerSample = np.max(weightedProb, axis=1)

        # majority voting
        voteDict={}
        for pl, conf in zip(predLabelsPerSample, confidencesPerSample):
            if pl in voteDict:
                voteDict[pl] = (1+voteDict[pl][0], conf+voteDict[pl][1])
            else:
                voteDict[pl] = (1, conf)
        predLabel = None
        maxCount  = 0
        for pl, value in voteDict.items():
            if (maxCount < value[0]) or (maxCount == value[0] and voteDict[predLabel][1] < value[1]):
                maxCount = value[0]
                predLabel = pl

        predLabels[sampleID] = predLabel

    return predLabels


def createKFoldDirs(experimentRootDir, kFold):
    foldDirs=[]
    for foldIdx in range(1, kFold+1):
        foldDir=os.path.join(experimentRootDir, str(foldIdx))
        createDirSafely(foldDir)
        foldDirs.append(foldDir)
    return foldDirs
