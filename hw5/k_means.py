from matplotlib import pyplot as plt
from utils import *
import numpy as np
import random

def draw(data_arr, tag_arr, centers, title, k_cluster):
    plt.figure(1)
    for i in range(k_cluster):
        sub_data = data_arr[:, tag_arr == i]
        plt.plot(sub_data[0], sub_data[1], 'o', label='group_%s'%str(i))
    if not type(centers) == type(None):
        plt.plot(centers[0], centers[1], 'o', label='center')
    plt.title(title)
    plt.legend()
    plt.savefig(title)
    plt.gcf().clear()

def K_Means_Init(data_arr, k_cluster):
    num_points = np.shape(data_arr)[1]
    distance_arr = np.ndarray([num_points])
    centers = [data_arr[:, random.randint(0, num_points - 1)]]
    for i in range(k_cluster-1):
        for j in range(num_points):
            _dist = 0
            for k in range(len(centers)):
                _dist += np.sum(np.square(data_arr[:, j:j+1] - centers[k]))
            distance_arr[j] = _dist

        # Deal with duplicated distance
        idx = argmax_Second(distance_arr)
        new_center = data_arr[:, idx:idx+1]
        if exist(new_center, centers):
            idx = random.randint(0, np.shape(data_arr)[1])
            
        centers.append(data_arr[:, idx:idx+1])
    res_center = None
    for i in range(k_cluster):
        if res_center is None:
            res_center = np.expand_dims(centers[i], axis=-1)
        else:
            res_center = np.concatenate((res_center, centers[i]), axis=-1)
    plt.figure(1)
    plt.plot(data_arr[0], data_arr[1], 'o')
    plt.plot(res_center[0], res_center[1], 'o')                
    plt.savefig('initial')
    plt.gcf().clear()
    print(res_center)
    return res_center

def K_Means(data_arr, k):
    """
        data_arr = (2 * N)
    """
    num_points = np.shape(data_arr)[1]

    # Random Define the center
    idx = np.asarray(range(num_points))
    random.shuffle(idx)
    centers = data_arr[:, idx[:k]]
    tags = np.zeros([num_points])        

    # Use K-means++ idea to define the center
    centers = K_Means_Init(data_arr, k)

    # Clustering
    previous_center = None
    stop_counter = 0
    while True:        
        # E-step
        tags = np.empty([num_points])
        for i in range(num_points):
            tags[i] = np.argmin(np.sqrt(np.sum(np.square(centers - np.expand_dims(data_arr[:, i], axis=-1)), axis=0)))

        # M-step
        for i in range(k):
            sub_data = data_arr[:, tags == i]
            new_center = np.mean(sub_data, axis=-1)
            centers[:, i:i+1] = np.expand_dims(new_center, axis=-1)

        # Draw
        draw(data_arr, tags, centers, 'iter_%s'%str(stop_counter), k)

        # Break
        if not type(previous_center) == type(None):
            if equal(centers, previous_center) == True or stop_counter > 20:
                draw(data_arr, tags, centers, 'final', k)
                break
        
        previous_center = np.copy(centers)
        stop_counter += 1
    
        
if __name__ == '__main__':
    # Random try
    # data_arr = generateData()
    # data_arr = data_arr.T
    # K_Means(data_arr)

    # HW5 data
    data_arr, tag_arr = loadData(data_path = './test2_data.txt', tag_path = './test2_ground.txt')
    K_Means(data_arr, 2)