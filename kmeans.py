from collections import defaultdict
from random import uniform
from math import sqrt
import numpy as np
import random
from copy import deepcopy


DIST_METRIC=["L1", "L2", "INF", "Zeros"]

def distanceToCenter(points, center, metric='L2', diffType=None):
    diff=points-center
    if diffType != None:
        pShape=points.shape
        for nIdx in range(pShape[0]):
            for dIdx in range(pShape[1]):
                if points[nIdx][dIdx] == center[dIdx] and center[dIdx] == 0:
                    diff[nIdx][dIdx] = 2

    if metric == "L1":
        return np.linalg.norm(diff,ord=1, axis=1)
    elif metric == "L2":
        return np.linalg.norm(diff,ord=2, axis=1)
    elif metric == "INF":
        return np.linalg.norm(diff,ord='inf', axis=1)



def k_means(data, k, distType="L2", diffType=None):
    '''
        Input:
            data: a NXD numpy array. N points
            k   : number of clusters
        Output:
            assignments: a 1D numpy array
    '''

    np.random.seed(0)

    # Number of training data
    n = data.shape[0]
    # Number of features in the data
    c = data.shape[1]

    # Generate random centers, here we use sigma and mean to ensure it represent the whole data
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    centers = np.random.randn(k,c)*std + mean

    centers_old = np.zeros(centers.shape) # to store old centers
    centers_new = deepcopy(centers) # Store new centers

    data.shape
    clusters = np.zeros(n)
    distances = np.zeros((n,k))

    error = np.linalg.norm(centers_new - centers_old)

    # When, after an update, the estimate of that center stays the same, exit loop
    while error != 0:
        # Measure the distance to every center
        for i in range(k):
            #distances[:,i] = np.linalg.norm(data - centers[i], axis=1)
            distances[:,i] = distanceToCenter(data, centers[i], distType, diffType)
        # Assign all training data to closest center
        clusters = np.argmin(distances, axis = 1)
        
        centers_old = deepcopy(centers_new)
        # Calculate mean for every cluster and update the center
        for i in range(k):
            centers_new[i] = np.mean(data[clusters == i], axis=0)

        error = np.linalg.norm(centers_new - centers_old)
    return clusters
