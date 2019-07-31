import os
import sys

from sklearn.cluster import KMeans
import numpy as np

from tensorflow.keras.models import load_model
from transformation import IMG_TRANSFORMATIONS, transform_images

#from kmeans import k_means
from fcmeans import FCM

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
NC=list(range(1, 11))#numOfModels+1)) # list of numbers of clusters
accuracies = np.zeros((numOfEPS, len(NC)))
clusteringResult = {}

with open("FCM_clustering_result.txt", "w") as fp:
    for numOfClusters in NC:
        # clustering into c groups
        print("Clustering: {} clusters".format(numOfClusters))
        fcm = FCM(n_clusters=numOfClusters)
        fcm.fit(predVec)
        clusteringResult[numOfClusters] = fcm.u
        fp.write("\n## number of clusters: "+str(numOfClusters)+"\n")
        fp.write(str(fcm.u)+"\n")
        fp.write("\n")

def vote2(pred, mem):
    '''
        pred: numOfModels X numOfAEs X 2
        mem : numOfModels X numOfClusters
        voteResult: numOfAEs X 2
    '''
    numOfModels = pred.shape[0]
    numOfAEs = pred[0].shape[0]
    voteResult = np.zeros((numOfAEs, 2))
    numOfClusters = mem.shape[1]
    for aeID in range(numOfAEs):
        countDict1={}
        # counting
        for cID in range(numOfClusters):
            countDict2={}
            for modelID in range(numOfModels):
                label = pred[modelID, aeID, 0]
                if label in countDict2:
                    countDict2[label] = countDict2[label] + pred[modelID, aeID, 1]*mem[modelID, cID]
                else:
                    countDict2[label] = pred[modelID, aeID, 1]*mem[modelID, cID]
            votedLabel = None
            maxWC = -np.inf
            for label, weightedConf in countDict2.items():
                if maxWC < weightedConf:
                    votedLabel = label
                    maxWC = weightedConf

            if votedLabel in countDict1:
                countDict1[votedLabel] = countDict1[votedLabel] + maxWC
            else:
                countDict1[votedLabel] = maxWC
        finalVote = None
        maxAccWC = -np.inf
        for label, wc in countDict1.items():
            if maxAccWC < wc:
                finalVote = label
                maxAccWC = wc
        voteResult[aeID, 0] = finalVote
        voteResult[aeID, 1] = maxAccWC

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
        mem = clusteringResult[numOfClusters]

        votingResult = vote2(predResult, mem)
        accuracy = 1.0 * len(np.where(labels == votingResult[:, 0])[0]) / len(labels)
        accuracies[epsID, cID] = round(accuracy, 4)
        print("EPS: {}, numOfCluster: {}, Accuracy: {}".format(eps, numOfClusters, round(accuracy, 4)))
   

with open("accuracy_clustering_FCM.txt", "w") as fp:
    fp.write("#ofCs\t")
    for epsID in range(numOfEPS):
        fp.write("["+str(listEPS[epsID])+"%]\t")

    fp.write("\n")

    for cID in range(len(NC)):
        fp.write(str(NC[cID])+"\t")
        for epsID in range(numOfEPS):
            fp.write(str(accuracies[epsID, cID])+"\t")
        fp.write("\n")
