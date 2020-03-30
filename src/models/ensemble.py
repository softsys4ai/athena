import os
import time
import numpy as np

from collections import Counter
import operator
from sklearn.metrics import accuracy_score
# from tensorflow.keras.models import load_model, Model

from models.transformation import transform
import utils.data_utils as data_utils

'''
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
'''

def prediction(data, models, nClasses, transformationList, batch_size=32, channel_last=True):
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
    data = np.float32(data)
    for mIdx in range(nWeakModels):
        startTime = time.time()
        transformationType = transformationList[mIdx]
        testData = transform(data, transformationType)
        transTCs.append(time.time()-startTime)

        if not channel_last:
            # input shape of cnn model is <n_samples, n_channels, rows, cols>
            testData = data_utils.set_channels_first(testData)
        startTime = time.time()
        rawPred[mIdx] = models[mIdx].predict(testData, batch_size=batch_size)
        predTCs.append(time.time() - startTime)

    return rawPred, transTCs, predTCs


def get_topK(arr, topK=1):
    return arr.argsort()[-topK:][::-1]


# ensemble_ID = 0
def ensemble_random_defense(rawPred, topK=1):
    '''
        input:
            rawPred: nWeakModels X nSamples X nClasses
        output:
            predLabels: nSamples
    '''
    inputShape = rawPred.shape
    nSamples, nWeakModels = inputShape[1], inputShape[0] 
    predLabels = []

    for sIdx in range(nSamples):
        weakModelIdx = np.random.choice(nWeakModels)
        preds = np.array(rawPred[weakModelIdx, sIdx, :])
        predLabels.append(get_topK(preds, topK))

    return np.asarray(predLabels)


# ensemble_ID = 1
def ensemble_majority_voting(rawPred, topK=1):
    '''
        input:
            rawPred: nWeakModels X nSamples X nClasses
        output:
            predLabels: nSamples
    '''
    inputShape = rawPred.shape
    nSamples, nWeakModels = inputShape[1], inputShape[0]
    predLabels = []

    for sIdx in range(nSamples):
        labels = np.argmax(rawPred[:, sIdx, :], axis=1)
        c = sorted(Counter(labels.tolist()).items(), key=operator.itemgetter(1), reverse=True)
        labels = np.array(k[0] for k in c)
        predLabels.append(labels[:topK])

    return np.asarray(predLabels)

# ensemble_ID = 2
# confidence: probability or logit
# derive two ensemble models
def ensemble_ave_confidence(rawPred, topK=1):
    '''
        input:
            rawPred: nWeakModels X nSamples X nClasses
        output:
            predLabels: nSamples
    '''
    inputShape = rawPred.shape
    nSamples, nWeakModels = inputShape[1], inputShape[0]
    predLabels = []

    for sIdx in range(nSamples):
        means = rawPred[:, sIdx, :].mean(axis=0)
        predLabels.append(get_topK(means, topK))

    return np.asarray(predLabels)


# ensemble_ID = 3
def ensemble_top2labels_majority_voting(rawPred, topK=1):
    '''
        input:
            rawPred: nWeakModels X nSamples X nClasses
        output:
            predLabels: nSamples
    '''
    inputShape = rawPred.shape
    nSamples, nWeakModels = inputShape[1], inputShape[0]
    predLabels = []

    for sIdx in range(nSamples):
        ind = np.argpartition(rawPred[:, sIdx, :], -2, axis=1)[:, -2:]
        labels = np.ravel(ind)
        c = sorted(Counter(labels.tolist()).items(), key=operator.itemgetter(1), reverse=True)
        labels = np.array([k[0] for k in c])
        predLabels.append(labels[:topK])

    return np.asarray(predLabels)


def ensemble_defenses_util(rawPred, ensembleID, topK=1):
    '''
        input:
            rawPred: nWeakModels X nSamples X nClasses
            ensembleID
        output:
            labels: nSamples
    '''

    if ensembleID == 0:
        return ensemble_random_defense(rawPred, topK)
    elif ensembleID == 1:
        return ensemble_majority_voting(rawPred, topK)
    elif ensembleID == 2:
        return ensemble_ave_confidence(rawPred, topK)
    elif ensembleID == 3:
        return ensemble_top2labels_majority_voting(rawPred, topK)

def ensemble_defenses(
        # modelsDir,
        models,
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
    # models = load_models(modelsDir, modelFilenamePrefix, transformationList, convertToLogit=convertToLogit)

    data = np.load(datasetFilePath)
    data = data_utils.rescale(data) # ensure its values inside [0, 1]
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
            checkTimeCost=checkTimeCost)

    trueLabels = np.load(trueLabelFilePath)

    return round(accuracy_score(trueLabels, predLabels), 4)
