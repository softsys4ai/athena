import os
import sys
import time

from sklearn.cluster import KMeans
import numpy as np

from tensorflow.keras.models import load_model
import cv2

from config import *
from transformation import transform_images

DEBUG = True #MODE.DEBUG

def calAccu(predProb, trueLabels):
    predLabels = np.argmax(predProb, axis=1)
    accuracy = 1.0*len(np.where(predLabels==trueLabels)[0]) / len(trueLabels)
    if DEBUG:
        print("accu: {}".format(accuracy))
    return round(accuracy, 4)


# Basic parameters
timeStamp=time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
experimentRootDir=os.path.join("experiment",timeStamp)
os.makedirs(experimentRootDir)

numOfAEs  = 10000
modelsDir = "models"
AEDir     = "AEs"
transformConfig = TRANSFORMATION()
IMG_TRANSFORMATIONS = transformConfig.supported_types() 

# Load the original clean model and the list of tranformation based model
#models=[load_model(os.path.join(modelsDir, "mnist_cnn_clean.h5"))]
models=[]
for modelName in IMG_TRANSFORMATIONS:
    modelNameFP = os.path.join(modelsDir, "mnist_cnn_"+modelName+".h5")
    print("load model {}".format(modelName))
    models.append(load_model(modelNameFP))

# load AEs and their original labels
listEPS = ATTACK.FGSM_EPS
numOfEPS = len(listEPS)
numOfModels = len(models)
numOfTrans = len(IMG_TRANSFORMATIONS) - 1 
numOfClasses = DATA.NB_CLASSES
topK=5
with open(os.path.join(experimentRootDir, "TopKTransModels.txt"), "a+") as fp:
        fp.write("{:<4}\t".format("EPS"))
        for tID in range(topK):
            tempStr = "T"+ str(tID+1)+"(AccAE/AccBS)"
            fp.write("{:<20}".format(tempStr))
        fp.write("\n")


for epsID in range(len(listEPS)):
    eps = int(1000*listEPS[epsID])
    curExprDir = os.path.join(experimentRootDir, "eps"+str(eps))
    os.makedirs(curExprDir)

    AEPredProb = np.zeros((numOfModels, numOfAEs, numOfClasses))
    BSPredProb = np.zeros((numOfModels, numOfAEs, numOfClasses))

    AEs = np.load(os.path.join(AEDir, "adv_mnist_cnn_clean_fgsm_eps"+str(eps)+".npy"))
    BSs = np.load(os.path.join(AEDir, "orig_mnist_cnn_clean_fgsm_eps"+str(eps)+".npy"))    
    labels_raw = np.load(os.path.join(AEDir, "label_mnist_cnn_clean_fgsm_eps"+str(eps)+".npy"))
    labels = np.argmax(labels_raw, axis=1)

    # Prediction
    #numOfAugTrans = len(TRANSFORMATION.AUGMENT)
    #labelsForAugmentTransAEs = np.zeros((numOfAugTrans, numOfAEs)) 
    #labelsForAugmentTransBSs = np.zeros((numOfAugTrans, numOfAEs)) 
    #augmentTransIdx = 0
    for modelID in range(numOfModels):
        curAEs = np.copy(AEs)
        curBSs = np.copy(BSs)

        transformType = IMG_TRANSFORMATIONS[modelID]
        print("\nTransform: "+transformType)
        '''
        if transformType in TRANSFORMATION.AUGMENT:
            curAEs, newLabelsAE = transform_images(AEs, transformType)
            curBSs, newLablesBS = transform_images(BSs, transformType)
            labelsForAugmentTransAEs[augmentTransIdx, :] = newLabelsAE
            labelsForAugmentTransBSs[augmentTransIdx, :] = newLabelsBS
            augmentTransIdx += 1
        else:
        '''
        curAEs = transform_images(curAEs, transformType)
        curBSs = transform_images(curBSs, transformType)

        print("Predict AE-eps{} with model {} - {}".format(eps, modelID, transformType))
        AEPredProb[modelID, :, :] = models[modelID].predict(curAEs)
        print("Predict BS with model {} - {}".format( modelID, transformType))
        BSPredProb[modelID, :, :] = models[modelID].predict(curBSs)
 

    np.save(os.path.join(curExprDir, "AE_pred_prob_eps"+str(eps)+".npy"), AEPredProb)
    np.save(os.path.join(curExprDir, "BS_pred_prob_eps"+str(eps)+".npy"), BSPredProb)
    
    #np.save(
    #        os.path.join(curExprDir, "AE_aug_trans_new_labels_eps"+str(eps)+".npy"),
    #        labelsForAugmentTransAEs)
    #np.save(
    #        os.path.join(curexprDir, "BS_aug_trans_new_labels_eps"+str(eps)+".npy"),
    #        labelsForAugmentTransBSs)
    


    # Compute accuracy for each single model
    modelAccs = np.zeros((numOfModels, 2))
    with open(os.path.join(curExprDir, "accuracy_each_single_model.txt"), "w") as fp:
        fp.write("ID\t{:30}\t{:6}\t{:6}\n".format(
            "Model Name",
            "Acc(AE)",
            "Acc(BS)"))
        #augmentTransIdx = 0
        for modelID in range(numOfModels):
            transformType = IMG_TRANSFORMATIONS[modelID]
            curLabelsAE = np.copy(labels)
            curLabelsBS = np.copy(labels)
            AEAcc = calAccu(AEPredProb[modelID, :, :], curLabelsAE)       
            BSAcc = calAccu(BSPredProb[modelID, :, :], curLabelsBS)  
            fp.write("{}\t{:30}\t{:<6}\t{:<6}\n".format(
                modelID,
                transformType,
                AEAcc,
                BSAcc))
            modelAccs[modelID, 0] = AEAcc
            modelAccs[modelID, 1] = BSAcc
    np.save(
        os.path.join(curExprDir, "accuracy_each_single_model.npy"),
        modelAccs)
    
    topKTMs = np.argsort(-modelAccs[1:, 0])[0:topK]
    with open(os.path.join(experimentRootDir, "TopKTransModels.txt"), "a+") as fp:
        fp.write("{:<4}\t".format(eps))
        for transID in topKTMs:
            tempStr = "M"+ str(transID+1)+"("+str(modelAccs[transID+1, 0])+"/"+str(modelAccs[transID+1, 1])+")"
            fp.write("{:<20}".format(tempStr))
        fp.write("\n")

    # Count number of samples in each classes in the testing set
    classesCnt=[]
    for i in range(numOfClasses):
        classesCnt.append(len(np.where(labels==i)[0]))
    if DEBUG:
        print("eps: {}, shape of labels: {} \n\tclassCnt: {}".format(
            eps,
            labels.shape,classesCnt))
    classesCnt = np.array(classesCnt)
    np.save(
        os.path.join(curExprDir, "classCount_eps"+str(eps)+".npy"),
        classesCnt)

    # Comptue expertise matrix for transform models
    # Compute model-sample vector for transform models and AEs
    msv = np.zeros((numOfTrans, numOfAEs))
    modelExpertise = np.zeros((numOfTrans, numOfClasses)) 
    for modelID in range(1, numOfModels):
        for aeID in range(numOfAEs):
            if np.argmax(AEPredProb[modelID, aeID, :]) == labels[aeID]:
                modelExpertise[modelID-1, labels[aeID]] += 1
                msv[modelID-1, aeID] = 1
            else:
                msv[modelID-1, aeID] = 0

    modelExpertise = np.round(modelExpertise/classesCnt, 4)
    np.save(
        os.path.join(curExprDir, "modelExpertise_eps"+str(eps)+".npy"),
        modelExpertise)
    np.save(os.path.join(curExprDir, "msv_eps"+str(eps)+".npy"), msv)
    with open(os.path.join(curExprDir, "modelExpertise_eps"+str(eps)+".txt"), "w") as fp, \
        open(os.path.join(curExprDir, "msv_eps"+str(eps)+".txt"), "w") as msv_fp:
        for transID in range(numOfTrans):
            for aeID in range(numOfAEs):
                msv_fp.write(str(msv[transID, aeID])+",")
            msv_fp.write("\n")
            for labelID in range(numOfClasses):
                fp.write(str(modelExpertise[transID, labelID])+",")
            fp.write("\n")



