import os
import sys

from sklearn.cluster import KMeans
import numpy as np

from tensorflow.keras.models import load_model
import cv2

from transformation import IMG_TRANSFORMATIONS, transform_images

AEDir    ="./AEs"

# load AEs and their original labels
numOfEPS = 4
numOfModels = 25
numOfAEs    = 10000


listEPS = [10,15,20,25]

#modelacc = np.zeros((numOfModels, 4))
numOfTrans = len(IMG_TRANSFORMATIONS)


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



for epsID in range(len(listEPS)):
    advPredProb=np.zeros((numOfModels, numOfAEs, 10))

    eps = listEPS[epsID]
    # advPredProb: numOfModels X numOfAEs X numOfClasses
    advPredProb  = np.load("adv_pred_prob_eps"+str(eps)+".npy")
    origPredProb = np.load("orig_pred_prob_eps"+str(eps)+".npy")
    labels_raw   =np.load(os.path.join(AEDir, "label_mnist_cnn_clean_fgsm_eps"+str(eps)+".npy"))
    labels=np.argmax(labels_raw, axis=1)
    modelsExpertise = np.load("eps"+str(eps)+"_modelExpertise.npy")

    advPredLabels = np.argmax(advPredProb, axis=2)
    modelAccs = np.zeros((numOfModels))
    for modelID in range(numOfModels):
        modelAccs[modelID] = round(1.0 * len(np.where(advPredLabels[modelID, :] == labels)[0]) / numOfAEs, 4)
        #print("model {} - Acc: {}".format(modelID+1, modelAccs[modelID]))
    print("Sort models in descending order based on their accuracies")
    sortedIndices = np.argsort(-modelAccs)
    for sID in sortedIndices:
        print("model {} - Acc: {}".format(sID+1, modelAccs[sID]))

    accuracies = np.zeros((numOfModels, 6))
    print("#ofTM\tAE-1\tBS-1\tAE-W\tBS-W\tAE-MV\tBS-MV")
    with open("accuracies_weighted_confidences_eps"+str(eps)+".txt", "w") as fp:
        fp.write("#ofTM\tAE-1\tBS-1\tAE-W\tBS-W\tAE-MV\tBS-MV\n")
        for num in range(numOfModels):
            topModelIndices = sortedIndices[0:num+1]
            curAdvPredProb = advPredProb[topModelIndices, :, :]
            curOrigPredProb = origPredProb[topModelIndices, :, :]
            curModelsExpertise = modelsExpertise[topModelIndices, :]

            advAcc  = calAccu(curAdvPredProb, labels)
            oriAcc  = calAccu(curOrigPredProb, labels)
            advAccW = calAccu(curAdvPredProb, labels, curModelsExpertise)
            oriAccW = calAccu(curOrigPredProb, labels, curModelsExpertise)
            advAccMTMV = calAccu_MTMV(curAdvPredProb, labels, curModelsExpertise)
            oriAccMTMV = calAccu_MTMV(curOrigPredProb, labels, curModelsExpertise)

            accuracies[num, :] = np.array([advAcc, oriAcc, advAccW, oriAccW, advAccMTMV, oriAccMTMV])
            print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(num+1, advAcc, oriAcc, advAccW, oriAccW, advAccMTMV, oriAccMTMV))
            fp.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(num+1, advAcc, oriAcc, advAccW, oriAccW, advAccMTMV, oriAccMTMV))

        bestAccs = np.max(accuracies, axis=0)
        bestAccMNs = np.argmax(accuracies, axis=0)
        print("\n[Best Accuracies]")
        print("Type\tAE-1\tBS-1\tAE-W\tBS-W\tAE-MV\tBS-MV")
        print("Accu\t{:.2%}\t{:.2%}\t{:.2%}\t{:.2%}\t{:.2%}\t{:.2%}".format(bestAccs[0], bestAccs[1], bestAccs[2], bestAccs[3], bestAccs[4], bestAccs[5]))
        print("#OfTM\t{:^5d}\t{:^5d}\t{:^5d}\t{:^5d}\t{:^5d}\t{:^5d}".format(bestAccMNs[0], bestAccMNs[1], bestAccMNs[2], bestAccMNs[3], bestAccMNs[4], bestAccMNs[5]))
        
        fp.write("\n[Best Accuracies]\n")
        fp.write("Type\tAE-1\tBS-1\tAE-W\tBS-W\tAE-MV\tBS-MV\n")
        fp.write("Accu\t{:.2%}\t{:.2%}\t{:.2%}\t{:.2%}\t{:.2%}\t{:.2%}\n".format(bestAccs[0], bestAccs[1], bestAccs[2], bestAccs[3], bestAccs[4], bestAccs[5]))
        fp.write("#OfTM\t{:^5d}\t{:^5d}\t{:^5d}\t{:^5d}\t{:^5d}\t{:^5d}\n".format(bestAccMNs[0], bestAccMNs[1], bestAccMNs[2], bestAccMNs[3], bestAccMNs[4], bestAccMNs[5]))
 


    np.save("accuracies_weighted_confidences_eps"+str(eps)+".npy", accuracies)

