import os
import numpy as np

def calAccuracy(predLabels, trueLabels):
    return round(
            1.0 * len(np.where(trueLabels == predLabels)[0]) / len(trueLabels),
            4)


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

def votingAsDefense(predLC, clusters, vsac="Majority"):
    '''
        Input:
            predLC  :   numOfModels X numOfSamples X 2 numpy array. The 3dr dimension: (label, confidence)
            clusters:   a 2D lists where each element is a list of models' ID, representing a cluster
            acvs    :   across-clusters voting strategy, default stragety is majority voting.
                        Note: voting strategy inside cluster is majority voting.
        Output:
            votedResult: numOfSample X 2 numpy array. The 1st column represents the classified label while the 2nd column tells the confidence about this label.
    '''

    clusterRepresentatives = []
    for cluster in clusters:
        insideClusterOpinions = []
        for modelID in cluster:
            insideClusterOpinions.append(predLC[modelID, :, :])
        clusterRepresentatives.append(majorityVote(insideClusterOpinions))

    if vsac == "Majority":
        return majorityVote(clusterRepresentatives)
    elif vsac == "Max":
        return maxConfidenceVote(clusterRepresentatives)
    else:
        raise ValueError("The given across-clusters voting strategy, {}, is not supported. By now, only support 'Majority' and 'Max'".format(acvs))



def getUpperBoundAccuracy(predLC, clusters, trueLabels): # cover all clusters
    '''
        Note: a sample will be considered as classfied correctly as long as it is correctly classified by some cluster based on majority voting
        Input:
            predLC  :   numOfModels X numOfSamples X 2 numpy array. The 3dr dimension: (label, confidence)
            clusters:   a 2D lists where each element is a list of models' ID, representing a cluster

        Output:
            votedResult: numOfSample X 2 numpy array. The 1st column represents the classified label while the 2nd column tells the confidence about this label.
    '''
    clusterRepresentatives = []
    for cluster in clusters:
        insideClusterOpinions = []
        for modelID in cluster:
            insideClusterOpinions.append(predLC[modelID, :, :])
        clusterRepresentatives.append(majorityVote(insideClusterOpinions))

    return upperBoundAccuracy(clusterRepresentatives, trueLabels)

def loadClusteringResult(resultDir, numOfClusters):
    '''
        Output:
            clusters:   a 2D list where each element is a list of model IDs, represents a cluster
    '''
    # Note: model IDs appear in the clustering result file is 1-based.
    filepath = os.path.join(resultDir, "C"+str(numOfClusters)+".txt")
    clusters = []
    with open(filepath) as fp:
        for line in fp:
            line = line.rstrip()
            parts = line.split()
            cluster = []
            for modelID in parts:
                cluster.append(int(modelID))
            clusters.append(cluster)
    return clusters

def loadAllClusteringResult(resultDir, numOfModels):
    '''
        Output:
            clusteringResult:   a dictionary where the key is the numOfClusters, value is the corresponding FCM clustering result, a 2D list.
    '''
    clusteringResult = {}
    for numOfClusters in range(1, numOfModels+1):
        clusteringResult[numOfClusters] = loadClusteringResult(resultDir, numOfClusters)
    return clusteringResult

# Calculate the accuracy of the ensemble model
# based weighted sum of each model's initial confidences
def calAccu(predProb, trueLabels, modelWeights=None):
    '''
        Input:
            predProb: numOfModels X numOfAEs X 10
            trueLabels: 1D numpy array - numOfAEs
            modelWeights: 2D numpy array - numOfModels X 10
        Output:
            accuracy
    '''
    numOfModels = predProb.shape[0]
    numOfAEs = predProb.shape[1]
    numOfClasses = predProb.shape[2]
    weightedPredProb = np.zeros(predProb.shape)
    if modelWeights is None:
        modelWeights = np.ones((numOfModels, numOfClasses)) # each model is 100% trusted
    
    cnt = 0

    for aeID in range(numOfAEs):
        predLabel = np.argmax(np.multiply(predProb[:,aeID,:], modelWeights).sum(axis=0))
        if predLabel == trueLabels[aeID]:
            cnt+=1

    return round(1.0*cnt/numOfAEs, 4)

# Calculate the accuracy of the ensemble model
# based max trustworthness and majority voting
def calAccu_MTMV(predProb, trueLabels, modelWeights=None):
    '''
        Input:
            predProb: numOfModels X numOfAEs X 10
            trueLabels: 1D numpy array - numOfAEs
            modelWeights: 2D numpy array - numOfModels X 10
        Output:
            accuracy
    '''
    numOfModels = predProb.shape[0]
    numOfAEs = predProb.shape[1]
    numOfClasses = predProb.shape[2]
    weightedPredProb = np.zeros(predProb.shape)
    if modelWeights is None:
        modelWeights = np.ones((numOfModels, numOfClasses)) # each model is 100% trusted
    
    cnt = 0

    for aeID in range(numOfAEs):
        # max trustworthness
        weightedProb = np.multiply(predProb[:,aeID,:], modelWeights)
        predLabels = np.argmax(weightedProb, axis=1)
        confidences = np.max(weightedProb, axis=1)

        # majority voting
        voteDict={}
        for pl, conf in zip(predLabels, confidences):
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

        if predLabel == trueLabels[aeID]:
            cnt+=1

    return round(1.0*cnt/numOfAEs, 4)


