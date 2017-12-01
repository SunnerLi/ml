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
    plt.show()

def K_Means(data_arr, k):
    """
        data_arr = (2 * N)
    """
    data_arr = data_arr.T
    num_points = np.shape(data_arr)[1]

    # Random Define the center
    idx = np.asarray(range(num_points))
    random.shuffle(idx)
    centers = data_arr[:, idx[:k]]
    tags = np.zeros([num_points])        

    # Clustering
    previous_center = None
    stop_counter = 0
    while True:        
        # E-step
        tags = np.empty([num_points])
        for i in range(num_points):
            tags[i] = np.argmin(np.sqrt(np.sum(np.square(centers - data_arr[:, i]), axis=0)))

        # M-step
        for i in range(k):
            sub_data = data_arr[:, tags == i]
            new_center = np.mean(sub_data, axis=-1)
            centers[i] = new_center
        centers = centers.T
        # print('new center: ', centers)

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
    K_Means(generateData(), 2)