import os
import sys

from sklearn.cluster import KMeans
import numpy as np

from tensorflow.keras.models import load_model
from transformation import IMG_TRANSFORMATIONS, transform_images

from kmeans import k_means

modelsDir="./trans_models"
AEDir    ="./AEs"
predVecFP="./vectors.txt"


# load AEs and their original labels
numOfEPS = 4
listEPS = [10,15,20,25]

numOfModels = len(IMG_TRANSFORMATIONS)
numOfAEs    = 10000

# list of tranformation based model
modelFPList=[]
models=[]
for imgTranID in range(numOfModels):
    modelName = IMG_TRANSFORMATIONS[imgTranID]
    modelNameFP = os.path.join(modelsDir, "mnist_cnn_"+modelName+".h5")
    modelFPList.append(modelNameFP)
    print("load model {} - {}".format(imgTranID, modelName))
    models.append(load_model(modelNameFP))


predVec = np.zeros((numOfModels, numOfAEs))

with open(predVecFP) as fp:
    row = 0
    for line in fp:
        line = line.rstrip()
        if len(line) == 0:
            continue

        parts = line.split(',')
        col = 0
        for elem in parts:
            predVec[row, col] = int(elem)
            col+=1
        row+=1
predVecList=predVec.tolist()

# Clustering
NC=list(range(3, 4))#numOfModels+1)) # list of numbers of clusters
accuracies = np.zeros((numOfEPS, len(NC)))
clusteringResult = {}
for numOfClusters in NC:
    clusteringResult[numOfClusters] = []

with open("clustering_result.txt", "w") as fp:
    for numOfClusters in NC:
        # clustering into c groups
        print("Clustering: {} clusters".format(numOfClusters))
    #    kmeans = KMeans(n_clusters=numOfClusters, random_state=0).fit(predVec)
    #    for c in range(numOfClusters):
    #        clusteringResult[numOfClusters].append(np.where(kmeans.labels_ == c)[0])
    #        print(np.where(kmeans.labels_ == c)[0])

        assignments = k_means(predVec, numOfClusters, "L2", "ZerosFarAway")
        fp.write("## number of clusters: "+str(numOfClusters)+"\n")
        for c in range(numOfClusters):
            cluster = np.where(assignments==c)[0]
            clusteringResult[numOfClusters].append(cluster)
            print(cluster)
            fp.write("\t"+str(cluster)+"\n")
        fp.write("\n")

def vote1(participants):
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



for epsID in range(numOfEPS):
    eps = listEPS[epsID]
    AEs   =np.load(os.path.join(AEDir, "adv_mnist_cnn_clean_fgsm_eps"+str(eps)+".npy"))    
    labels=np.load(os.path.join(AEDir, "label_mnist_cnn_clean_fgsm_eps"+str(eps)+".npy"))
    labels=np.argmax(labels, axis=1)
    print("eps:{}, # of AEs:{}, # of labels:{}".format(eps, AEs.shape, labels.shape))

    predResult = np.zeros((numOfModels, numOfAEs, 2)) # the 3rd dimensions: label and confidence
    # load pretrained models to compose ensemble models for testing
    for modelID in range(numOfModels):
        #print("Use model {}".format(M[modelID]))
        model = models[modelID] #load_model(modelFPList[modelID])
        print("Predict AE (eps={}) with model {} - {}".format(eps, modelID, IMG_TRANSFORMATIONS[modelID]))
        predProb = model.predict(AEs)
        predLabels = np.argmax(predProb, axis=1)
        predConfs  = np.amax(predProb, axis=1)
        predLandC = np.vstack((predLabels, predConfs)).T
        predResult[modelID, :, :] = predLandC

    print("Use ensemble models")
    for cID in range(len(NC)):
    # test the ensemble model represented by the c groups
        numOfClusters = NC[cID]
        clusters = clusteringResult[numOfClusters]
        level1Opinions = []
        level2Opinions = []
        for cluster in clusters:
            for modelID in cluster:
                level2Opinions.append(predResult[modelID, :, :])
            level1Opinions.append(vote1(level2Opinions))
        votingResult = vote1(level1Opinions)
        accuracy = 1.0 * len(np.where(labels == votingResult[:, 0])[0]) / len(labels)
        accuracies[epsID, cID] = round(accuracy, 4)
        print("EPS: {}, numOfCluster: {}, Accuracy: {}".format(eps, numOfClusters, round(accuracy, 4)))
   

with open("accuracy_clustering_KMeans.txt", "w") as fp:
    fp.write("#ofCs\t")
    for epsID in range(numOfEPS):
        fp.write("["+str(listEPS[epsID])+"%]\t")

    fp.write("\n")

    for cID in range(len(NC)):
        fp.write(str(NC[cID])+"\t")
        for epsID in range(numOfEPS):
            fp.write(str(accuracies[epsID, cID])+"\t")
        fp.write("\n")
