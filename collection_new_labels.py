import os
import sys

import numpy as np

from transformation import transform
from utils.config import TRANSFORMATION
from utils.ensemble import load_models, prediction, ensemble_defenses, ensemble_defenses_util 


BSDataFP=sys.argv[1]
BSLabelFP=sys.argv[2]
newLabelDir=sys.argv[3]
modelsDir=sys.argv[4]

# (nSamples, <sample dimension>, nChannels)
# BSData = np.load(BSDataFP)
# (nClasses)
BSLabelVec=np.load(BSLabelFP)
BSLabels = np.argmax(BSLabelVec, axis=1)
nClasses = BSLabelVec.shape[1]

newLabelFNList=[]
EnsembleIDs=[0,1,2,3]


modelFilenamePrefix="mnist-cnn" # dataset name and network architecture

# include "clean" type: no transformation.
# transformationList[0] is "clean"
transformationList=TRANSFORMATION.supported_types()
# remove "clean" because the correspondingly model will not be used in ensemble
transformationList.remove("clean")

datasetFilePath = BSDataFP

data = np.load(datasetFilePath)
data = np.clip(data, 0, 1) # ensure its values inside [0, 1]


useLogit = False
print("Loading prob models")
models = load_models(modelsDir, modelFilenamePrefix, transformationList, convertToLogit=useLogit)
print("Prediction...")
rawPred = prediction(data, models, nClasses, transformationList)

# use probability
for ensembleID in EnsembleIDs:
    print("Processing ensembleID {} using probability".format(ensembleID))
    labels = ensemble_defenses_util(rawPred, ensembleID)

    labelsFN = "label_EnsembleID{}_prob.npy".format(ensembleID)
    labelsFP=os.path.join(newLabelDir, labelsFN)
    np.save(labelsFP, labels)
    newLabelFNList.append(labelsFN)
del models
del rawPred

# use logit and EnsembleID 2
ensembleID=2
useLogit=True
print("Loading logit models")
models = load_models(modelsDir, modelFilenamePrefix, transformationList, convertToLogit=useLogit)
print("Prediction...")
rawPred = prediction(data, models, nClasses, transformationList)

print("Processing ensembleID {} using logit".format(ensembleID))
labels = ensemble_defenses_util(rawPred, ensembleID)
labelsFN = "label_EnsembleID{}_logit.npy".format(ensembleID)
labelsFP=os.path.join(newLabelDir, labelsFN)
np.save(labelsFP, labels)
newLabelFNList.append(labelsFN)

with open(os.path.join(newLabelDir, "nameList.txt"), "w") as fp:
    for name in newLabelFNList:
        fp.write(name+"\n")


