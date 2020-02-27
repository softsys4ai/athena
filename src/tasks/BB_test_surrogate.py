import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model, Model


def testOneData(
        datasetFilePath,
        model,
        trueLabels,
        ):
    data = np.load(datasetFilePath)
    data = np.clip(data, 0, 1) # ensure its values inside [0, 1]

    print("Prediction...")
    rawPred = model.predict(data)
    labels = np.argmax(rawPred, axis=1)
    acc = round(accuracy_score(trueLabels, labels), 4)

    return acc

def testOneTargetModel(
        resultDir,
        BBDir,
        targetName,
        AETypes,
        budgets,
        trueLabels):

    rows, cols = len(AETypes), len(budgets)
    curAEDir = os.path.join(BBDir, "AE/"+targetName)
    modelDir = os.path.join(BBDir, "surrogate_models")
    accs = np.zeros((rows, cols))
    nAEs = 1000
    misClassifiedIndices = np.ones((rows, cols, nAEs))

    for cIdx in range(cols):
        budget = budgets[cIdx]
        modelFN = "model_{}Samples_{}-mnist-cnn-clean.h5".format(budget, targetName) 
        model = load_model(os.path.join(modelDir, modelFN))
        for rIdx in range(rows):
            AETag = AETypes[rIdx]
            print("budget {}: {}".format(budget, AETag))

            fn =  "BB_model_{}Samples_{}_AE-mnist-cnn-clean-{}.npy".format(
                    budget, targetName, AETag)
            datasetFilePath = os.path.join(curAEDir, fn)

            data = np.load(datasetFilePath)
            data = np.clip(data, 0, 1) # ensure its values inside [0, 1]
            pred = model.predict(data)
            labels = np.argmax(pred, axis=1)
            accs[rIdx, cIdx] = round(accuracy_score(trueLabels, labels), 4)

            misClassifiedIndices[rIdx, cIdx, np.where(labels!=trueLabels)[0]] = 0 


            print("AETag: {}, nSamples: {}, ACC: {}".format(AETag, budget, accs[rIdx, cIdx]))

    np.save(os.path.join(resultDir, "BB_AE_Classification_surrogate_target_"+targetName+".npy"), misClassifiedIndices)

    return accs


def saveResultPerTarget(resultDir, accs, targetName, budgets, AENames):
    np.save(
            os.path.join(resultDir, "BB_AE_testAcc_surrogate_target_"+targetName+".npy")
            , accs)
    with open(os.path.join(resultDir, "BB_AE_testAcc_surrogate_target_"+targetName+".csv"), "w") as fp:
        fp.write("Query Budget")
        for budget in budgets:
            fp.write("\t{}".format(budget))
        fp.write("\n")

        for ridx in range(len(AENames)):
            fp.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
                AENames[ridx],
                accs[ridx, 0],
                accs[ridx, 1],
                accs[ridx, 2],
                accs[ridx, 3],
                accs[ridx, 4]))


BBDir = sys.argv[1]
resultDir=sys.argv[2]


AETypes =[ 
        "bim_ord2_nbIter100_eps2000",
        "bim_ordinf_nbIter100_eps300",
        "cw_l2_lr100_maxIter100",
        "deepfool_l2_overshoot0",  
        "fgsm_eps300", 
        "jsma_theta50_gamma70",
        "mim_eps300_nbIter1000",
        "onepixel_pxCount30_maxIter30_popsize100",
        "pgd_eps300"]


trueLabels = np.load(os.path.join(BBDir, "BS/BB_AE_Sources/BS_1k_label_For_AE.npy"))

budgets=[10, 50, 100, 500, 1000]
targetNames=["RD", "MV", "AVEP", "T2MV", "AVEL", "UM"]


for targetName in targetNames:
    print("targetName: "+targetName)
    accs = testOneTargetModel(
        resultDir,
        BBDir,
        targetName,
        AETypes,
        budgets,
        trueLabels)

    saveResultPerTarget(resultDir, accs, targetName, budgets, AETypes)
   

