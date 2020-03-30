import os
import sys
import gc
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

from utils.config import TRANSFORMATION
from models.ensemble import load_models, prediction,  ensemble_defenses_util


def testOneData(
        datasetFilePath,
        models,
        nClasses,
        transformationList,
        EnsembleIDs,
        trueLabels,
        useLogit=False
        ):
    accs = []
    stds = []
    data = np.load(datasetFilePath)
    data = np.clip(data, 0, 1) # ensure its values inside [0, 1]

    print("Prediction...")
    rawPred, _, _ = prediction(data, models, nClasses, transformationList)

    if not useLogit:
        ensembleID = EnsembleIDs[0]
        # use probability
        if ensembleID != 0:
            misClassifiedIndices = np.ones(1000)
            print("Processing ensembleID {} using probability".format(ensembleID))
            labels = ensemble_defenses_util(rawPred, ensembleID)
            accs.append(round(accuracy_score(trueLabels, labels), 4))
            stds.append(0)
            misClassifiedIndices[np.where(labels!=trueLabels)[0]] = 0

        else:
            misClassifiedIndices = np.ones((1000, 100))
            for run in range(100):
                print("[Run {}]".format(run))
                print("Processing ensembleID {} using probability".format(ensembleID))
                labels = ensemble_defenses_util(rawPred, ensembleID)
                accs.append(round(accuracy_score(trueLabels, labels), 4))
                misClassifiedIndices[np.where(labels!=trueLabels)[0], run] = 0

            accs = np.array(accs)
            accs_mean = accs.mean()
            accs_std  = accs.std()
            accs = []
            accs.append(accs_mean)
            stds.append(accs_std)
    else:
        # use logit and EnsembleID 2
        misClassifiedIndices = np.ones(1000)
        ensembleID=2
        print("Processing ensembleID {} using logit".format(ensembleID))
        labels = ensemble_defenses_util(rawPred, ensembleID)
        accs.append(round(accuracy_score(trueLabels, labels), 4))
        stds.append(0)
        misClassifiedIndices[np.where(labels!=trueLabels)[0]] = 0



    return np.array(accs), np.array(stds), misClassifiedIndices

def testOneEnsemble(
        resultDir,
        BBDir,
        ensembleTag,
        testEnsembleID,
        AETypes,
        budgets,
        trueLabels,
        nClasses,
        transformationList,
        models,
        useLogit):

    rows, cols = len(AETypes), len(budgets)
    targetName = ensembleTag
    curAEDir = os.path.join(BBDir, "AE/"+targetName)
    testEnsembleIDs = [testEnsembleID]
    accs = np.zeros((rows, cols))
    stds = np.zeros((rows, cols))
    nAEs = 1000
    if targetName == "RD":
        nRuns = 100
        misClassifiedIndices = np.ones((rows, cols, nAEs, nRuns))
    else:
        misClassifiedIndices = np.ones((rows, cols, nAEs))

    for rIdx in range(rows):
        AETag = AETypes[rIdx]
        for cIdx in range(cols):
            budget = budgets[cIdx]

            fn =  "BB_model_{}Samples_{}_AE-mnist-cnn-clean-{}.npy".format(
                    budget, targetName, AETag)
            datasetFilePath = os.path.join(curAEDir, fn)

            accs[rIdx, cIdx], stds[rIdx, cIdx], misClassifiedIndices[rIdx, cIdx] = testOneData(
                    datasetFilePath,
                    models,
                    nClasses,
                    transformationList,
                    testEnsembleIDs,
                    trueLabels,
                    useLogit=useLogit
                    )
            print("AETag: {}, nSamples: {}, ACC: {}".format(AETag, budget, accs[rIdx, cIdx]))

    np.save(
            os.path.join(
                resultDir,
                "BB_AE_Classification_target_{}.npy".format(targetName)),
            misClassifiedIndices)

    return accs, stds

def testUndefendedModel(
        resultDir,
        BBDir,
        targetName,
        AETypes,
        budgets,
        trueLabels,
        model):

    targetName = "UM"
    rows, cols = len(AETypes), len(budgets)
    curAEDir = os.path.join(BBDir, "AE/"+targetName)
    accs = np.zeros((rows, cols))

    nAEs = 1000
    misClassifiedIndices = np.ones((rows, cols, 1000))
    for rIdx in range(rows):
        AETag = AETypes[rIdx]
        for cIdx in range(cols):
            budget = budgets[cIdx]

            fn =  "BB_model_{}Samples_{}_AE-mnist-cnn-clean-{}.npy".format(
                    budget, targetName, AETag)
            datasetFilePath = os.path.join(curAEDir, fn)

            data = np.load(datasetFilePath)
            data = np.clip(data, 0, 1) # ensure its values inside [0, 1]
            pred = model.predict(data)
            labels = np.argmax(pred, axis=1)
            accs[rIdx, cIdx] = round(accuracy_score(trueLabels, labels), 4)

            misClassifiedIndices[np.where(labels!=trueLabels)[0]] = 0
            print("AETag: {}, nSamples: {}, ACC: {}".format(AETag, budget, accs[rIdx, cIdx]))

    np.save(
            os.path.join(
                resultDir,
                "BB_AE_Classification_target_{}.npy".format(targetName)),
            misClassifiedIndices)

    return accs


def saveOneTypeResultPerTarget(resultDir, result, resultType, targetName, budgets, AENames):
    np.save(
            os.path.join(resultDir, "BB_AE_test{}_target_{}.npy".format(resultType, targetName))
            , accs)
    with open(os.path.join(resultDir, "BB_AE_test{}_target_{}.csv".format(resultType, targetName)), "w") as fp:
        fp.write("Query Budget")
        for budget in budgets:
            fp.write("\t{}".format(budget))
        fp.write("\n")

        for ridx in range(len(AENames)):

            fp.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
                AENames[ridx],
                result[ridx, 0],
                result[ridx, 1],
                result[ridx, 2],
                result[ridx, 3],
                result[ridx, 4]))

def saveResultPerTarget(resultDir, accs, targetName, budgets, AENames, stds=None):
    saveOneTypeResultPerTarget(resultDir, accs, "Acc", targetName, budgets, AENames)
    if stds is not None:
        saveOneTypeResultPerTarget(resultDir, stds, "STD", targetName, budgets, AENames)


BBDir = sys.argv[1]
modelsDir=sys.argv[2]
resultDir=sys.argv[3]


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


nClasses = 10 #trueLabelVec.shape[1]

EnsembleIDs=[0,1,2,3]
modelFilenamePrefix="mnist-cnn" # dataset name and network architecture

# include "clean" type: no transformation.
# transformationList[0] is "clean"
transformationList=TRANSFORMATION.supported_types()
# remove "clean" because the correspondingly model will not be used in ensemble
transformationList.remove("clean")
nTrans = len(transformationList)

trueLabels = np.load(os.path.join(BBDir, "BS/BB_AE_Sources/BS_1k_label_For_AE.npy"))

budgets=[10, 50, 100, 500, 1000]
EnsembleTags=["RD", "MV", "AVEP", "T2MV", "AVEL"]



# the undefended model
targetName="UM"
undefendedModel = load_model(os.path.join(modelsDir, "model-mnist-cnn-clean.h5"))
print("##Testing {}".format(targetName))
accs = testUndefendedModel(
        resultDir,
        BBDir,
        targetName,
        AETypes,
        budgets,
        trueLabels,
        undefendedModel)

print("Saving the result of testing {}".format(targetName))
saveResultPerTarget(resultDir, accs, targetName, budgets, AETypes)


# the MV, T2MV and AVEP ensembles
useLogit = False
print("Loading prob transformation models")
models = load_models(modelsDir, modelFilenamePrefix, transformationList, convertToLogit=useLogit)

for testEnsembleID, ensembleTag in zip(range(len(EnsembleTags)), EnsembleTags):
    if ensembleTag == "AVEL" or ensembleTag == "RD":
        continue

    print("## Testing {}".format(ensembleTag))
    accs, _ = testOneEnsemble(
        resultDir,
        BBDir,
        ensembleTag,
        testEnsembleID,
        AETypes,
        budgets,
        trueLabels,
        nClasses,
        transformationList,
        models,
        useLogit)

    print("Saving the result of testing {}".format(ensembleTag))
    saveResultPerTarget(resultDir, accs, ensembleTag, budgets, AETypes)

# the RD ensemble
for testEnsembleID, ensembleTag in zip(range(len(EnsembleTags)), EnsembleTags):
    if ensembleTag != "RD":
        continue

    print("## Testing {}".format(ensembleTag))
    accs, stds = testOneEnsemble(
        resultDir,
        BBDir,
        ensembleTag,
        testEnsembleID,
        AETypes,
        budgets,
        trueLabels,
        nClasses,
        transformationList,
        models,
        useLogit)

    print("Saving the result of testing {}".format(ensembleTag))
    saveResultPerTarget(resultDir, accs, ensembleTag, budgets, AETypes, stds=stds)

del models
gc.collect()

# the AVEL ensembles
useLogit = True
print("Loading logit transformation models")
models = load_models(modelsDir, modelFilenamePrefix, transformationList, convertToLogit=useLogit)

for testEnsembleID, ensembleTag in zip(range(len(EnsembleTags)), EnsembleTags):
    if ensembleTag != "AVEL":
        continue

    print("## Testing {}".format(ensembleTag))
    accs, _ = testOneEnsemble(
        resultDir,
        BBDir,
        ensembleTag,
        testEnsembleID,
        AETypes,
        budgets,
        trueLabels,
        nClasses,
        transformationList,
        models,
        useLogit)

    print("Saving the result of testing {}".format(ensembleTag))
    saveResultPerTarget(resultDir, accs, ensembleTag, budgets, AETypes)

del models
gc.collect()


