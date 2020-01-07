import os
import time
import numpy as np

from collections import Counter
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model, Model

from models.transformation import transform

def load_models(modelsDir, modelFilenamePrefix, transformationList, convertToLogit=False):
    models=[]
    print("Number of transformations: {}".format(len(transformationList)))
    for tIdx in range(len(transformationList)):
        transformType = transformationList[tIdx]
        modelName = "model-"+modelFilenamePrefix+"-"+transformType
        print("loading model {}".format(modelName))
        # Create corresponding model for outputing logits

        modelNameFP = os.path.join(modelsDir, modelName+".h5")
        model = load_model(modelNameFP)

        if convertToLogit:
            layerName=model.layers[-2].name
            logitsModel = Model(
                    inputs=model.input,
                    outputs=model.get_layer(layerName).output)
            models.append(logitsModel)
        else:
            models.append(model)
    print("Number of loaded models: {}".format(len(models)))
    return models

def prediction(data, models, nClasses, transformationList):
    '''
        input:
            data: nSamples X <Sample Dimension>
            models: a list of classification models
        output:
            prediction matrix M - nWeakModels X nSamples X nClasses.
    '''
    nSamples, nWeakModels = data.shape[0], len(models)
    rawPred = np.zeros((nWeakModels, nSamples, nClasses))
    transTCs = []
    predTCs = []
    for mIdx in range(nWeakModels):
        testData = data.copy() # some transformation will change the data.

        startTime = time.time()
        transformationType = transformationList[mIdx]
        testData = transform(testData, transformationType)
        transTCs.append(time.time()-startTime)

        startTime = time.time()
        rawPred[mIdx] = models[mIdx].predict(testData)
        predTCs.append(time.time() - startTime)

    return rawPred, transTCs, predTCs


# ensemble_ID = 0
def ensemble_random_defense(rawPred):
    '''
        input:
            rawPred: nWeakModels X nSamples X nClasses
        output:
            predLabels: nSamples
    '''
    inputShape = rawPred.shape
    nSamples, nWeakModels = inputShape[1], inputShape[0] 
    predLabels = np.array([-1]*nSamples)

    for sIdx in range(nSamples):
        weakModelIdx = np.random.choice(nWeakModels)
        predLabels[sIdx] = np.argmax(rawPred[weakModelIdx, sIdx, :])

    return predLabels

# ensemble_ID = 1
def ensemble_majority_voting(rawPred):
    '''
        input:
            rawPred: nWeakModels X nSamples X nClasses
        output:
            predLabels: nSamples
    '''
    inputShape = rawPred.shape
    nSamples, nWeakModels = inputShape[1], inputShape[0]
    predLabels = np.array([-1]*nSamples)

    for sIdx in range(nSamples):
        labels = np.argmax(rawPred[:, sIdx, :], axis=1)
        c = Counter(labels.tolist())
        majorityVote, count = c.most_common()[0]
        predLabels[sIdx] = majorityVote

    return predLabels

# ensemble_ID = 2
# confidence: probability or logit
# derive two ensemble models
def ensemble_ave_confidence(rawPred):
    '''
        input:
            rawPred: nWeakModels X nSamples X nClasses
        output:
            predLabels: nSamples
    '''
    inputShape = rawPred.shape
    nSamples, nWeakModels = inputShape[1], inputShape[0]
    predLabels = np.array([-1]*nSamples)

    for sIdx in range(nSamples):
        means = rawPred[:, sIdx, :].mean(axis=0)
        predLabels[sIdx] = np.argmax(means)

    return predLabels


# ensemble_ID = 3
def ensemble_top2labels_majority_voting(rawPred):
    '''
        input:
            rawPred: nWeakModels X nSamples X nClasses
        output:
            predLabels: nSamples
    '''
    inputShape = rawPred.shape
    nSamples, nWeakModels = inputShape[1], inputShape[0]
    predLabels = np.array([-1]*nSamples)

    for sIdx in range(nSamples):
        ind = np.argpartition(rawPred[:, sIdx, :], -2, axis=1)[:, -2:]
        labels = np.ravel(ind)
        c = Counter(labels.tolist())
        majorityVote, count = c.most_common()[0]
        predLabels[sIdx] = majorityVote

    return predLabels

def ensemble_defenses_util(rawPred, ensembleID):
    '''
        input:
            rawPred: nWeakModels X nSamples X nClasses
            ensembleID
        output:
            labels: nSamples
    '''

    if ensembleID == 0:
        return ensemble_random_defense(rawPred)
    elif ensembleID == 1:
        return ensemble_majority_voting(rawPred)
    elif ensembleID == 2:
        return ensemble_ave_confidence(rawPred)
    elif ensembleID == 3:
        return ensemble_top2labels_majority_voting(rawPred)

def ensemble_defenses(
        modelsDir,
        modelFilenamePrefix,
        transformationList,
        datasetFilePath,
        nClasses,
        ensembleID,
        useLogit=False,
        checkTimeCost=False):
    '''
        input:
            modelFilenamePrefix and transformationList are used to obtain the filename of models.
            Assume modle's filename has the format, model-<modelFilenamePrefix-<transformType>.h5
            If this assumption changes, please change the corresponding in load_models().
        output:
            labels: nSamples
    '''
    if useLogit:
        convertToLogit = True
    else:
        convertToLogit = False
    models = load_models(modelsDir, modelFilenamePrefix, transformationList, convertToLogit=convertToLogit)

    data = np.load(datasetFilePath)
    data = np.clip(data, 0, 1) # ensure its values inside [0, 1]
    rawPred, transTCs, predTCs = prediction(data, models, nClasses, transformationList)

    return ensemble_defenses_util(rawPred, ensembleID)

def evaluate_ensemble_defenses(
        modelsDir,
        modelFilenamePrefix,
        transformationList,
        datasetFilePath,
        trueLabelFilePath,
        nClasses,
        ensembleID,
        useLogit=False,
        checkTimeCost=False):
    '''
        input:
            modelFilenamePrefix and transformationList are used to obtain the filename of models.
            Assume modle's filename has the format, model-<modelFilenamePrefix-<transformType>.h5
            If this assumption changes, please change the corresponding in load_models().
        output:
            labels: nSamples
    '''
    predLabels = ensemble_defenses(
            modelsDir,
            modelFilenamePrefix,
            transformationList,
            datasetFilePath,
            nClasses,
            ensembleID,
            useLogit=useLogit,
            checkTimeCost=False)

    trueLabels = np.load(trueLabelPath)

    return round(accuracy_score(trueLabels, predLabels), 4)
