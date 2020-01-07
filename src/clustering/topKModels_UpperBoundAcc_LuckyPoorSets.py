import os
import numpy as np
from utils.config import *
from utils.util import *


# Basic parameters
experimentRootDir="experiment/2019-07-28_08-51-00"
AEDir = "AEs"

distMetrics=["sqeuclidean", "cityblock", "cosine", "correlation"]
dmDirs=[]
ubAccsAllDM =[]
for distMetric in distMetrics:
    dmDir = os.path.join(experimentRootDir, distMetric)
    dmDirs.append(dmDir)
    if not os.path.exists(dmDir):
        os.makedirs(dmDir)

# load AEs and their original labels
listEPS = ATTACK.FGSM_EPS
numOfEPS = len(listEPS)

numOfModels = 36

# Calculate upper-bound accuracy
for distID in range(len(distMetrics)):
    # the first column represent the accuracy for benign samples
    ubAccs = np.zeros((numOfModels, 1+numOfEPS))

    # Table header for upper-bound accuracies
    ubAccsTableFP = os.path.join(dmDirs[distID], "upper_bound_accuracy_table.txt")
    with open(ubAccsTableFP, "a+") as fp:
        fp.write("{:<4}".format("#ofC"))
        fp.write("\t{:<8}".format("Benign"))
        for epsID in range(len(listEPS)):
            fp.write("\t{:<8}".format("EPS"+str(round(listEPS[epsID], 3))))
        fp.write("\n")

    distMetric = distMetrics[distID]
    print("Process with distance metric "+ distMetric)
    # for different AEs
    for epsID in range(len(listEPS)):
        eps = int(1000*listEPS[epsID])
        curExprDir = os.path.join(experimentRootDir, "eps"+str(eps))
        print("Process EPS "+str(eps))

        clusteringResultDir = os.path.join(curExprDir, distMetric)
        clusteringResult = loadAllClusteringResult(clusteringResultDir, numOfModels)

        AEPredProbFP = os.path.join(curExprDir, "AE_pred_prob_eps"+str(eps)+".npy")
        # AEPredProb: (numOfModels, numOfAEs, numOfClasses)
        AEPredProb = np.load(AEPredProbFP)
        AEPredLC = np.zeros((AEPredProb.shape[0], AEPredProb.shape[1], 2))
        AEPredLC[:, :, 0] = np.argmax(AEPredProb, axis=2)
        AEPredLC[:, :, 1] = np.max(AEPredProb, axis=2)

        labels_raw = np.load(
                os.path.join(AEDir, "label_mnist_cnn_clean_fgsm_eps"+str(eps)+".npy"))
        labels = np.argmax(labels_raw, axis=1)

        for numOfClusters in range(1, 1+numOfModels):
            ubAccs[numOfClusters-1, epsID+1] = getUpperBoundAccuracy(
                    AEPredLC,
                    clusteringResult[numOfClusters],
                    labels)

    # for Benign samples
    BSPredProb = np.load(os.path.join(experimentRootDir, "eps5/BS_pred_prob_eps5.npy"))  
    BSPredLC = np.zeros((BSPredProb.shape[0], BSPredProb.shape[1], 2))
    BSPredLC[:, :, 0] = np.argmax(BSPredProb, axis=2)
    BSPredLC[:, :, 1] = np.max(BSPredProb, axis=2)

    labels_raw = np.load(
            os.path.join(AEDir, "label_mnist_cnn_clean_fgsm_eps5.npy"))
    labels = np.argmax(labels_raw, axis=1)
    for numOfClusters in range(1, 1+numOfModels):
        ubAccs[numOfClusters-1, 0] = getUpperBoundAccuracy(
                BSPredLC,
                clusteringResult[numOfClusters],
                labels)

    ubAccsFP = os.path.join(dmDirs[distID], "upper_bound_accuracies.npy")
    np.save(ubAccsFP, ubAccs)
    ubAccsAllDM.append(ubAccs)

    # dump the table for upper bound accuracy
    with open(ubAccsTableFP, "a+") as fp:
        for numOfClusters in range(1, 1+numOfModels):
            fp.write("{:<4}".format(numOfClusters))
            fp.write("\t{:<8}".format(ubAccs[numOfClusters-1, 0]))
            for epsID in range(len(listEPS)):
                fp.write("\t{:<8}".format(ubAccs[numOfClusters-1, epsID+1]))
            fp.write("\n")


# Accuracy of each model in the set of  clean model and top k transform models
# Upper-Bound accuracy
topK=5
with open(os.path.join(experimentRootDir, "TopKTransModels.txt"), "a+") as fp:
        fp.write("{:<4}".format("EPS"))
        tempStr = "Clean(AccAE/AccBS)"
        fp.write("\t{:<20}".format(tempStr))
        for tID in range(topK):
            tempStr = "T"+ str(tID+1)+"(AccAE/AccBS)"
            fp.write("\t{:<20}".format(tempStr))
        # for upper_bound accuracy for all distance metrics
        tempStr = "UpperBoundAcc"
        fp.write("\t{:<20}".format(tempStr))
        fp.write("\n")



for epsID in range(len(listEPS)):
    eps = int(1000*listEPS[epsID])
    curExprDir = os.path.join(experimentRootDir, "eps"+str(eps))
    print("Process EPS "+str(eps))

    # Compute accuracy for each single model
    # the 2nd dimention in modelAccs is the classification accuracy under Attack (AE) and Non-Attack (Benign Sample)
    modelAccs = np.load(os.path.join(curExprDir, "accuracy_each_single_model.npy"))
    topKTMs = np.argsort(-modelAccs[1:, 0])[0:topK]
    with open(os.path.join(experimentRootDir, "TopKTransModels.txt"), "a+") as fp:
        fp.write("{:<4}".format(eps))
        tempStr = "M0"+"("+str(modelAccs[0, 0])+"/"+str(modelAccs[0, 1])+")"
        fp.write("\t{:<20}".format(tempStr))
        for transID in topKTMs:
            tempStr = "M"+ str(transID+1)+"("+str(modelAccs[transID+1, 0])+"/"+str(modelAccs[transID+1, 1])+")"
            fp.write("\t{:<20}".format(tempStr))
        # The upper_bound accuracy is the same for all cases of distance metrics
        ubacc = ubAccsAllDM[0][numOfModels-1, epsID+1]
        fp.write("\t{:<20}".format(ubacc))
        fp.write("\n")

