import os
import sys

import numpy as np
from utils.util import *
from utils.config import *
from models.transformation import transform_images


# Basic parameters
experimentRootDir="experiment/2019-07-28_08-51-00"
modelsDir="./trans_models"
AEDir    ="./AEs"

listEPS = ATTACK.FGSM_EPS
numOfEPS = len(listEPS)
transformConfig = TRANSFORMATION()
IMG_TRANSFORMATIONS = transformConfig.supported_types() 
numOfModels = len(IMG_TRANSFORMATIONS) - 1 # exclude the clean model and only consider the transformation-based model
maxNumOfClusters = numOfModels
numOfAEs    = 10000

distMetrics=["sqeuclidean", "cityblock", "cosine", "correlation"]
dmDirs=[]
for distMetric in distMetrics:
    dmDir = os.path.join(experimentRootDir, distMetric)
    dmDirs.append(dmDir)
    if not os.path.exists(dmDir):
        os.makedirs(dmDir)

#weighted_confidence_approaches = ["wc_ones", "wc_em_mx", "wc_em_mv"]
#clustering_based_approaches = ["mv_mv", "ma_mv", "mcd_mv_c1", "mcd_mv_c3", "mcd_mv_ck"]
defenseList = ["mv_mv", "ma_mv"]
numOfDefenceApproaches = len(defenseList)

for distID in range(len(distMetrics)):
    distMetric = distMetrics[distID]
    print("Process "+distMetric)
    bestAccs = np.zeros((numOfEPS, 2*numOfDefenceApproaches))
    for epsID in range(numOfEPS):
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

        labels=np.load(os.path.join(AEDir, "label_mnist_cnn_clean_fgsm_eps"+str(eps)+".npy"))
        labels=np.argmax(labels, axis=1)

        defenseAccs = np.zeros((maxNumOfClusters, 2))
        for numOfClusters in range(1, 1+maxNumOfClusters):
            print("Process with a clustering result of "+str(numOfClusters)+" clusters")
            votedResult_maj = votingAsDefense(
                    AEPredLC,
                    clusteringResult[numOfClusters],
                    vsac="Majority")
            defenseAccs[numOfClusters-1, 0] = calAccuracy(votedResult_maj[:, 0], labels)

            votedResult_max = votingAsDefense(
                    AEPredLC,
                    clusteringResult[numOfClusters],
                    vsac="Max")

            defenseAccs[numOfClusters-1, 1] = calAccuracy(votedResult_max[:, 0], labels)
 
        bestAccs[epsID, 0:2] = np.max(defenseAccs, axis=0) # best accuracy
        bestAccs[epsID, 2:4] = 1+np.argmax(defenseAccs, axis=0) # best # of clusters
        defenseAccsFilename = "defense_accuracy_eps"+str(eps)+"_"+distMetric
        np.save(os.path.join(curExprDir, defenseAccsFilename+".npy"), defenseAccs)
        with open(os.path.join(curExprDir, defenseAccsFilename+".txt"), "w") as fp:
            fp.write("{:<4}\t{:<6}\t{:<6}\n".format("#ofC", "MV_MV", "MA_MV"))
            for clusterID in range(maxNumOfClusters):
                fp.write("{:<4}\t{:<6}\t{:<6}\n".format(
                    clusterID+1,
                    defenseAccs[clusterID][0],
                    defenseAccs[clusterID][1]))

    # create eps-vs-defense_approach table
    np.save(os.path.join(dmDirs[distID], "best_accuracy_of_clustering_based_defenses.npy"), bestAccs)
    with open(os.path.join(dmDirs[distID], "best_accuracy_of_clustering_based_defense.txt"), "w") as fp:
        strFormat = "{:<5}\t{:<9}\t{:<9}\t{:<6}\t{:<6}\t{:<6}\n"
        fp.write(strFormat.format("EPS", "MV_MV/#C", "MA_MV/#C", "CUA", "BSM", "UBA"))
        for epsID in range(numOfEPS):
            eps = int(1000*listEPS[epsID])
            curExprDir = os.path.join(experimentRootDir, "eps"+str(eps))
            # (1+numOfModels, 2): 1 is for the clean model. 2 - Acc(AE) Acc(BS)
            accuracyEachSingleModel = np.load(os.path.join(curExprDir, "accuracy_each_single_model.npy"))
            cua = accuracyEachSingleModel[0, 0] # accuracy of clean model under attack
            bsm = np.max(accuracyEachSingleModel[1:1+numOfModels, 0]) # best single model

            ubAccsFP = os.path.join(dmDirs[distID], "upper_bound_accuracies.npy")
            ubAccs = np.load(ubAccsFP)
            uba = ubAccs[numOfModels-1, epsID+1] # upper bound accuracy
            fp.write(strFormat.format(
                listEPS[epsID],
                str(bestAccs[epsID, 0])+"/"+str(int(bestAccs[epsID, 2])),
                str(bestAccs[epsID, 1])+"/"+str(int(bestAccs[epsID, 3])),
                cua,
                bsm,
                uba))


