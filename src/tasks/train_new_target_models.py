import sys

import keras

from utils.ensemble_utils import *
from tasks.train_models import train_models_with_newLabels

def usage():
    print("==============================================================================")
    print("python <this script> inputImagesFP  experimentRootDir datasetName numOfClasses")
    print("==============================================================================")

if len(sys.argv) != 5:
    usage()
    exit(1)


inputImagesFP = sys.argv[1]
experimentRootDir = sys.argv[2]
datasetName = sys.argv[3]
numOfClasses = int(sys.argv[4])


DATA.set_current_dataset_name(datasetName)
# Basic parameters for k-fold experiment setup
architecture = MODEL.ARCHITECTURE
AETypes = ATTACK.get_AETypes()
numOfAETypes = len(AETypes)

targetModelName = "clean"
transformConfig = TRANSFORMATION()
transformationList = transformConfig.supported_types() 
transformationList = transformationList[1:] # exclude the 'clean' transformation - no transformation
numOfTrans = len(transformationList)


# Prediction
inputSamples = np.load(inputImagesFP)
numOfSamples = inputSamples.shape[0]
sampleIndices = np.array(range(numOfSamples))
np.random.shuffle(sampleIndices)


newLabelsDir = os.path.join(experimentRootDir, "newLabels")
newTargetModelsDir = os.path.join(experimentRootDir, "newTargetModels")
createDirSafely(newTargetModelsDir)

numOfDefenses = numOfCVDefenses+2*numOfWCDefenses

# used when training new target models with cifar_9 dataset
validationRate=0.2
newTargetModelType="clean"
trainSetSizeList = [10, 100, 1000, 5000, numOfSamples]
needAugment = False
if (datasetName == DATA.cifar_10):
    needAugment = True




# Test each ensemble model trained by each type of AEs
for AETypeIdx in range(numOfAETypes):
    AEType = AETypes[AETypeIdx]
    curNewLabelsDir = os.path.join(newLabelsDir, AEType)

    print("Collecting new labels from the ensemble models built upon "+AEType)

    # accuracy of clustering-and-voting based defenses
    for defenseIdx in range(numOfCVDefenses):
        defenseName = cvDefenseNames[defenseIdx] 
        ensembleType = defenseName
        newLabelsFilename = defenseName+"_newLabels.npy"

        newLabels = np.load(os.path.join(curNewLabelsDir, newLabelsFilename))
        newLabels = keras.utils.to_categorical(newLabels, numOfClasses)

        # Train a list of new target models with different number of traning samples
        for numOfTrainSamples in trainSetSizeList:
            sIndices = sampleIndices[0:numOfTrainSamples]
            train_models_with_newLabels(
                    datasetName,
                    AEType,
                    ensembleType,
                    newTargetModelType,
                    numOfTrainSamples,
                    inputSamples[sIndices],
                    newLabels[sIndices],
                    validation_rate=validationRate,
                    need_argument=needAugment)


    # accuracy of weithed-confidence based defenses
    for defenseIdx in range(numOfWCDefenses):
        defenseName = wcDefenseNames[defenseIdx]
        for plIdx in range(2):
            wcMatFilename = defenseName+"_EM.npy"
            mIDsFilename  = defenseName+"_modelIDs.npy"
            newLabelsFilename = defenseName+"_newLabels.npy"
            ensembleType = defenseName
            if plIdx == 1: # predict logit instead of probability
                wcMatFilename = "LG_" + wcMatFilename
                mIDsFilename  = "LG_" +  mIDsFilename
                newLabelsFilename = "LG_" + newLabelsFilename
                ensembleType = "LG_" + ensembleType
            
          

            newLabels = np.load(os.path.join(curNewLabelsDir, newLabelsFilename))
            newLabels = keras.utils.to_categorical(newLabels, numOfClasses)


            # Train a list of new target models with different number of traning samples
            for numOfTrainSamples in trainSetSizeList:
                sIndices = sampleIndices[0:numOfTrainSamples]
                train_models_with_newLabels(
                        datasetName,
                        AEType,
                        ensembleType,
                        newTargetModelType,
                        numOfTrainSamples,
                        inputSamples[sIndices],
                        newLabels[sIndices],
                        validation_rate=validationRate,
                        need_argument=needAugment)


