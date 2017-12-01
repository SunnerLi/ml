from scipy.cluster.hierarchy import fcluster, linkage
from matplotlib import pyplot as plt
from k_means import K_Means_Init
from utils import *
import numpy as np
import random

def draw(data_arr, tag_arr, title, k_cluster):
    plt.figure(1)
    for k in range(k_cluster):
        sub_tag = tag_arr[k]
        sub_data = data_arr[:, sub_tag == 1]
        plt.plot(sub_data[0], sub_data[1], 'o', label='group_%s'%str(k))
    plt.title(title)
    plt.legend()
    plt.show()    

def Kernel_K_Means(data_arr, k_cluster):
    """
        data_arr = (2 * N)
    """
    data_arr = data_arr.T
    num_points = np.shape(data_arr)[1]
    tag_arr = np.zeros([k_cluster, num_points])
    distance_arr = np.ndarray([k_cluster, num_points])

    # Random initialize tag
    for i in range(num_points):
        tag_arr[random.randint(0, k_cluster - 1)][i] = 1

    # Use K-means++ idea to define the center
    tag_arr = np.zeros([k_cluster, num_points])
    centers = K_Means_Init(data_arr, k_cluster)
    for i in range(num_points):
        tag_arr[np.argmin(np.sqrt(np.sum(np.square(centers - data_arr[:, i]), axis=0)))][i] = 1


    previous_distance_arr = None
    stop_counter = 0
    print('tag_arr: ', tag_arr)
    while True:   
        # Calculate distance
        for k in range(k_cluster):     
            pi_k = np.shape(data_arr[:, tag_arr[k] == 1])[1]
            third_term = None
            for p in range(num_points):
                for q in range(num_points):
                    if type(third_term) == type(None):
                        third_term = tag_arr[k][p] * tag_arr[k][q] * kernel(data_arr[:, p:p+1], data_arr[:, q:q+1])
                    else:
                        third_term += tag_arr[k][p] * tag_arr[k][q] * kernel(data_arr[:, p:p+1], data_arr[:, q:q+1])
            third_term /= (pi_k ** 2)
            
            for j in range(num_points):
                first_term = kernel(data_arr[:, j:j+1], data_arr[:, j:j+1])
                second_term = np.dot(kernel(data_arr, data_arr[:, j:j+1]), tag_arr[k:k+1, :].T)
                second_term /= pi_k
                distance_arr[k][j] = first_term[0] - 2 * second_term[0] + third_term[0]

        # Determine class
        tag_arr = np.zeros([k_cluster, num_points])
        for j in range(num_points):
            tag_arr[np.argmin(distance_arr[:, j])][j] = 1

        # Draw
        draw(data_arr, tag_arr, 'iter_%s'%str(stop_counter), k_cluster)

        # Stop or not
        if not previous_distance_arr is None:
            if equal(previous_distance_arr, distance_arr) == True or stop_counter > 20:
                draw(data_arr, tag_arr, 'final', k_cluster)
                break
        previous_distance_arr = distance_arr
        stop_counter += 1
    return tag_arr
        
if __name__ == '__main__':
    Kernel_K_Means(generateData(), 2)