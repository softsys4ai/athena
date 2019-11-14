import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score

from utils.config import TRANSFORMATION
from utils.ensemble import load_models, prediction,  ensemble_defenses_util 


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
    data = np.load(datasetFilePath)
    data = np.clip(data, 0, 1) # ensure its values inside [0, 1]

    print("Prediction...")
    rawPred, transTCs, predTCs = prediction(data, models, nClasses, transformationList)

    ensembleTCs = []
    if not useLogit:
        # use probability
        for ensembleID in EnsembleIDs:
            print("Processing ensembleID {} using probability".format(ensembleID))
            start_time = time.time()
            labels = ensemble_defenses_util(rawPred, ensembleID)
            ensembleTCs.append(time.time() - start_time)
            accs.append(round(accuracy_score(trueLabels, labels), 4))
    else:
        # use logit and EnsembleID 2
        ensembleID=2
        print("Processing ensembleID {} using logit".format(ensembleID))
        start_time = time.time()
        labels = ensemble_defenses_util(rawPred, ensembleID)
        ensembleTCs.append(time.time() - start_time)
        accs.append(round(accuracy_score(trueLabels, labels), 4))


    return np.array(accs), np.array(transTCs), np.array(predTCs), np.array(ensembleTCs)

BSLabelFP=sys.argv[1]
samplesDir=sys.argv[2]
modelsDir=sys.argv[3]


AETypes = {
        "biml2": ["bim_ord2_nbIter100_eps1000", "bim_ord2_nbIter100_eps250", "bim_ord2_nbIter100_eps500"],
        "bimli":["bim_ordinf_nbIter100_eps100", "bim_ordinf_nbIter100_eps90", "bim_ordinf_nbIter100_eps75"],
        "cwl2":["cw_l2_lr350_maxIter100", "cw_l2_lr500_maxIter100", "cw_l2_lr700_maxIter100"],
        "dfl2":["deepfool_l2_overshoot20", "deepfool_l2_overshoot30", "deepfool_l2_overshoot50"],  
        "fgsm":["fgsm_eps100", "fgsm_eps250", "fgsm_eps300"], 
        "jsma":["jsma_theta30_gamma50", "jsma_theta50_gamma50", "jsma_theta50_gamma70"],
        "mim":["mim_eps20_nbIter1000", "mim_eps30_nbIter1000", "mim_eps50_nbIter1000"],
        "op":["onepixel_pxCount15_maxIter30_popsize100", "onepixel_pxCount30_maxIter30_popsize100", "onepixel_pxCount5_maxIter30_popsize100"],
        "pgd":["pgd_eps250", "pgd_eps100", "pgd_eps300"]
        }

sampleSubDirs=[
        "legitimates"#, "fgsm"
        #"biml2",  "bimli",  "cwl2", "dfl2"
        #"fgsm",  "jsma", "mim",  "op",  "pgd"
        ]

# (nSamples, <sample dimension>, nChannels)
# (nClasses)
trueLabelVec=np.load(BSLabelFP)
trueLabels = np.argmax(trueLabelVec, axis=1)
nClasses = trueLabelVec.shape[1]

EnsembleIDs=[0,1,2,3]
rows=0
cols=1+len(EnsembleIDs)
if "legitimates" in sampleSubDirs:
    rows=1+3*(len(sampleSubDirs) - 1)
else:
    rows=3*len(sampleSubDirs)
accs = np.zeros((rows, cols))

modelFilenamePrefix="mnist-cnn" # dataset name and network architecture

# include "clean" type: no transformation.
# transformationList[0] is "clean"
transformationList=TRANSFORMATION.supported_types()
# remove "clean" because the correspondingly model will not be used in ensemble
transformationList.remove("clean")
nTrans = len(transformationList)

transTCs_Prob = np.zeros((rows, nTrans))
transTCs_Logit = np.zeros((rows, nTrans))
predTCs_Prob = np.zeros((rows, nTrans))
predTCs_Logit = np.zeros((rows, nTrans))
ensembleTCs = np.zeros((rows, 5))

rowIdx=0
rowHeaders=[]
AEFilenamePrefix="test_AE-mnist-cnn-clean"
datasetFilePaths = []

for subDirName in sampleSubDirs:
    if subDirName == "legitimates": # BS
        datasetFilePaths.append(
                os.path.join(os.path.join(samplesDir, subDirName), "test_BS-mnist-clean.npy"))
        rowHeaders.append("BS")
    else: # AE
        AETags = AETypes[subDirName]
        for AETag in AETags:
            datasetFilePaths.append(
                    os.path.join(os.path.join(samplesDir, subDirName), AEFilenamePrefix+"-"+AETag+".npy"))
            rowHeaders.append(AETag)

useLogit = False
print("Loading prob models")
models = load_models(modelsDir, modelFilenamePrefix, transformationList, convertToLogit=useLogit)

for datasetFilePath in datasetFilePaths:
    accs[rowIdx, 0:4], transTCs_Prob[rowIdx], predTCs_Prob[rowIdx], ensembleTCs[rowIdx, 0:4] = testOneData(
            datasetFilePath,
            models,
            nClasses,
            transformationList,
            EnsembleIDs,
            trueLabels,
            useLogit=useLogit
            )
    rowIdx+=1
del models

useLogit=True
print("Loading logit models")
logitModels = load_models(modelsDir, modelFilenamePrefix, transformationList, convertToLogit=useLogit)

for datasetFilePath in datasetFilePaths:
    accs[rowIdx, 4], transTCs_Logit[rowIdx], predTCs_Logit[rowIdx], ensembleTCs[rowIdx, 4] = testOneData(
            datasetFilePath,
            logitModels,
            nClasses,
            transformationList,
            EnsembleIDs,
            trueLabels,
            useLogit=useLogit
            )
    rowIdx+=1
del logitModels


np.save("acc_ensemble_test.npy", accs)
with open("acc_ensemble_test.txt", "w") as fp:
    fp.write("Acc\tRD\tMV\tAVEP\tT2MV\tAVEL\n")
    for ridx in range(len(rowHeaders)):
        fp.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
            rowHeaders[ridx],
            accs[ridx, 0],
            accs[ridx, 1],
            accs[ridx, 2],
            accs[ridx, 3],
            accs[ridx, 4]))

transTCs = (transTCs_Prob + transTCs_Logit)/2

np.save("transTCs.npy", transTCs)
np.save("predTCs_Prob.npy", predTCs_Prob)
np.save("predTCs_Logit.npy", predTCs_Logit)
np.save("ensembleTCs.npy", ensembleTCs)
