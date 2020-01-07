import os
import sys

import numpy as np
import cv2
from utils.config import *
from utils.util import *

# Basic parameters
experimentRootDir="experiment/2019-07-28_08-51-00"
AEDir    ="./AEs"

listEPS = ATTACK.FGSM_EPS
numOfEPS = len(listEPS)
transformConfig = TRANSFORMATION()
IMG_TRANSFORMATIONS = transformConfig.supported_types() 
numOfModels = len(IMG_TRANSFORMATIONS) - 1 # exclude the clean model and only consider the transformation-based model
maxNumOfClusters = numOfModels
numOfAEs    = 10000



#modelacc = np.zeros((numOfModels, 4))

allBestAccsAE=np.zeros((numOfEPS, 3)) # 3 defense approaches
for epsID in range(numOfEPS):
    eps = int(1000*listEPS[epsID])
    curExprDir = os.path.join(experimentRootDir, "eps"+str(eps))
    print("Process EPS "+str(eps))


    AEPredProbFP = os.path.join(curExprDir, "AE_pred_prob_eps"+str(eps)+".npy")
    # AEPredProb: (numOfModels, numOfAEs, numOfClasses)
    AEPredProb = np.load(AEPredProbFP)
    BSPredProb = np.load(os.path.join(curExprDir, "BS_pred_prob_eps"+str(eps)+".npy"))
    labels_raw   =np.load(os.path.join(AEDir, "label_mnist_cnn_clean_fgsm_eps"+str(eps)+".npy"))
    labels=np.argmax(labels_raw, axis=1)

    # modelsExpertise: numOfModels X numOfClasses
    modelsExpertise = np.load(os.path.join(curExprDir, "modelExpertise_eps"+str(eps)+".npy"))

    AEPredLabels = np.argmax(AEPredProb, axis=2)

    modelAccs = np.zeros((numOfModels))
    for modelID in range(numOfModels):
        modelAccs[modelID] = round(1.0 * len(np.where(AEPredLabels[modelID, :] == labels)[0]) / numOfAEs, 4)
        #print("model {} - Acc: {}".format(modelID+1, modelAccs[modelID]))
    print("Sort models in descending order based on their accuracies")
    sortedIndices = np.argsort(-modelAccs)
    for sID in sortedIndices:
        print("model {} - Acc: {}".format(sID+1, modelAccs[sID]))

    accuracies = np.zeros((numOfModels, 6))
    print("#ofTM\tAE-1\tBS-1\tAE-W\tBS-W\tAE-MV\tBS-MV")
    with open(os.path.join(curExprDir, "accuracies_weighted_confidences_eps"+str(eps)+".txt"), "w") as fp:
        fp.write("#ofTM\tAE-1\tBS-1\tAE-W\tBS-W\tAE-MV\tBS-MV\n")
        for num in range(numOfModels):
            topModelIndices = sortedIndices[0:num+1]
            curAdvPredProb = AEPredProb[topModelIndices, :, :]
            curOrigPredProb = BSPredProb[topModelIndices, :, :]
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
        allBestAccsAE[epsID, :] = np.array([bestAccs[0], bestAccs[2], bestAccs[4]])
        bestAccMNs = np.argmax(accuracies, axis=0)
        print("\n[Best Accuracies]")
        print("Type\tAE-1\tBS-1\tAE-W\tBS-W\tAE-MV\tBS-MV")
        print("Accu\t{:.2%}\t{:.2%}\t{:.2%}\t{:.2%}\t{:.2%}\t{:.2%}".format(bestAccs[0], bestAccs[1], bestAccs[2], bestAccs[3], bestAccs[4], bestAccs[5]))
        print("#OfTM\t{:^5d}\t{:^5d}\t{:^5d}\t{:^5d}\t{:^5d}\t{:^5d}".format(bestAccMNs[0], bestAccMNs[1], bestAccMNs[2], bestAccMNs[3], bestAccMNs[4], bestAccMNs[5]))
        
        fp.write("\n[Best Accuracies]\n")
        fp.write("Type\tAE-1\tBS-1\tAE-W\tBS-W\tAE-MV\tBS-MV\n")
        fp.write("Accu\t{:.2%}\t{:.2%}\t{:.2%}\t{:.2%}\t{:.2%}\t{:.2%}\n".format(bestAccs[0], bestAccs[1], bestAccs[2], bestAccs[3], bestAccs[4], bestAccs[5]))
        fp.write("#OfTM\t{:^5d}\t{:^5d}\t{:^5d}\t{:^5d}\t{:^5d}\t{:^5d}\n".format(bestAccMNs[0], bestAccMNs[1], bestAccMNs[2], bestAccMNs[3], bestAccMNs[4], bestAccMNs[5]))
 


    np.save(os.path.join(curExprDir, "accuracies_weighted_confidences_eps"+str(eps)+".npy"), accuracies)

np.save(os.path.join(experimentRootDir, "allBestAccuracy.npy"), allBestAccsAE)
with open(os.path.join(experimentRootDir, "allBestAccuracy.txt"), "w") as fp:
    strFormat="{:<4}\t{:<8}\t{:<8}\t{:<8}\n"
    fp.write(strFormat.format("EPS", "WC_ONES", "WC_EM_MX", "WC_EM_MV"))
    for epsID in range(numOfEPS):
        fp.write(strFormat.format(listEPS[epsID], allBestAccsAE[epsID][0], allBestAccsAE[epsID][1], allBestAccsAE[epsID][2]))

    '''
    advAcc  = calAccu(AEPredProb, labels)
    oriAcc  = calAccu(BSPredProb, labels)
    advAccW = calAccu(AEPredProb, labels, modelsExpertise)
    oriAccW = calAccu(BSPredProb, labels, modelsExpertise)
    advAccMTMV = calAccu_MTMV(AEPredProb, labels, modelsExpertise)
    oriAccMTMV = calAccu_MTMV(BSPredProb, labels, modelsExpertise)
 
    print("EPS: {}, advAcc: {}, oriAcc: {}, advAccW: {}, oriAccW: {}, advAccMTMV: {}, oriAccMTMV: {}.".format(eps, advAcc, oriAcc, advAccW, oriAccW, advAccMTMV, oriAccMTMV))
    '''


    '''
    # count each classes
    classesCnt=[]
    for i in range(10):
        classesCnt.append(len(np.where(labels==i)[0]))
    print("num of AES: {}".format(np.array(classesCnt).sum()))
    #print("eps: {}, shape of labels: {} \n\tclassCnt: {}".format(eps, labels.shape,classesCnt))
    #np.save("eps"+str(eps)+"_classCount.npy", np.array(classesCnt))

    '''

    '''
    for modelID in range(numOfModels):
        for aeID in range(numOfAEs):
            if predResult[modelID, aeID, 0] == labels[aeID]:
                modelExpertise[modelID, labels[aeID]] += 1
    modelacc[:, epsID] = np.transpose(modelExpertise.sum(axis=1)/np.array(classesCnt).sum())
    '''

    '''
    # normalize
    modelExpertise = np.round(modelExpertise/np.array(classesCnt), 4)
    np.save("eps"+str(eps)+"_modelExpertise.npy", modelExpertise)
    with open("eps"+str(eps)+"_modelExpertise.txt", "w") as fp:
        for modelID in range(numOfModels):
            for labelID in range(10):
                fp.write(str(modelExpertise[modelID, labelID])+",")
            fp.write("\n")
    '''
'''
with open("transModelAcc.txt", "w") as fp:
    for modelID in range(numOfModels):
        for epsID in range(4):
            fp.write(str(modelacc[modelID, epsID])+",")
        fp.write("\n")
np.save("transModelAcc.npy", modelacc)
'''
