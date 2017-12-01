from matplotlib import pyplot as plt
import numpy as np
import random

def generateData(num_points=30):
    data = np.concatenate((np.random.normal(loc=1.0, scale=1.0, size=[num_points, 2]) , np.random.normal(loc=5.0, scale=1.0, size=[num_points, 2])))
    plt.plot(data[:, 0], data[:, 1], 'o')
    # plt.show()
    return data

def equal(arr1, arr2):
    len1, len2 = np.shape(arr1)
    for i in range(len1):
        for j in range(len2):
            if arr1[i][j] != arr2[i][j]:
                return False
    return True

def kernel(a, b, gemma=0.01):
    result = np.exp(-gemma * np.sum(np.square(a - b), axis=0))
    if len(np.shape(result)) < 1:
        return np.expand_dims(result, axis=0)
    return result

def argmax_Second(arr):
    idx_1st = np.argmin(arr)
    val_1st = np.min(arr)
    for i in range(len(arr)):
        if arr[i] > val_1st:
            val_1st = arr[i]
            idx_1st = i
    idx_2nd = np.argmin(arr)
    val_2nd = np.min(arr)
    for i in range(len(arr)):
        if arr[i] > val_2nd and arr[i] < val_1st:
            val_2nd = arr[i]
            idx_2nd = i
    return idx_2nd