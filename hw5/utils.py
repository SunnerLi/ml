from matplotlib import pyplot as plt
import numpy as np
import random

def generateData(num_points=30):
    data = np.concatenate((np.random.normal(loc=1.0, scale=1.0, size=[num_points, 2]) , np.random.normal(loc=5.0, scale=1.0, size=[num_points, 2])))
    plt.plot(data[:, 0], data[:, 1], 'o')
    # plt.show()
    return data

def loadData(data_path = './test1_data.txt', tag_path = './test1_ground.txt'):
    contain = open(data_path, 'r').readlines()
    data = np.ndarray([2, len(contain)])
    for i in range(len(contain)):
        data[0][i] = float(contain[i].split(' ')[0])
        data[1][i] = float(contain[i].split(' ')[1])

    contain = open(tag_path, 'r').readlines()
    contain = [int(num[:-1]) for num in contain]
    tag = np.zeros([max(contain) + 1, len(contain)])
    for i in range(len(contain)):
        tag[contain[i]][i] = 1
    return data, tag

def equal(arr1, arr2):
    len1, len2 = np.shape(arr1)
    for i in range(len1):
        for j in range(len2):
            if arr1[i][j] != arr2[i][j]:
                return False
    return True

def kernel(a, b, gemma=0.0001):
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

def exist(pos, _list):
    for i in range(len(_list)):
        repeated = True
        for j in range(np.shape(pos)[0]):
            if pos[j] != _list[i][j]:
                repeated = False
        if repeated:
            return True
    return False 