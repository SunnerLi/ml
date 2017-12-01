from kernel_k_means import Kernel_K_Means
from matplotlib import pyplot as plt
from scipy.linalg import eigh, eig
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

def K_Means(data_arr, k_cluster):
    num_points = np.shape(data_arr)[1]
    print(num_points)
    tag_arr = np.zeros([k_cluster, num_points])
    distance_arr = np.ndarray([k_cluster, num_points])

    for i in range(1):
        # E step 
        if i == 0:
            for j in range(num_points):
                tag_arr[random.randint(0, k_cluster-1)][j] = 1
        else:
            tag_arr = np.zeros([k_cluster, num_points])
            for j in range(num_points):
                tag_arr[np.argmin(distance_arr[:, j:j+1])][j] = 1
        
        # M step
        for j in range(num_points):
            norm = kernel(data_arr, data_arr[:, j:j+1])
            distance_arr[:, j] = np.dot(tag_arr, norm)
        print(distance_arr)
    
    print(tag_arr)

def SpectralWrapper(data_arr, k_cluster):
    """
        data_arr = (2 * N)
    """
    data_arr = data_arr.T
    num_points = np.shape(data_arr)[1]

    plt.figure(1)
    plt.plot(data_arr[0], data_arr[1], 'o')
    plt.show()

    # Compute W matrix and D matrix by RBF kernel
    W = np.ndarray([num_points, num_points])
    for i in range(num_points):
        W[i] = kernel(data_arr, data_arr[:, i:i+1])
    D = np.diag(np.sum(W, axis=1))
    print(W, '\n')
    print(D)

    # Compute unnormalized graph laplacian
    L = W - D
    print(L)

    # Compute first k generalized eigenvectors
    eigen_values, eigen_vectors = eig(L, D)
    print('vector shape: ', np.shape(eigen_vectors))
    U = eigen_vectors[:, :k_cluster].T

    print(U)
    print('U shape: ', np.shape(U))


    plt.figure(1)
    plt.plot(U[0, :3], U[1, :3], 'o')
    plt.plot(U[0, 3:], U[1, 3:], 'o')
    plt.show()

    # k-means
    tag_arr = Kernel_K_Means(U.T, k_cluster)

    # Plot result
    draw(data_arr, tag_arr, 'Spectral Clustering result', k_cluster)

if __name__ == '__main__':
    SpectralWrapper(generateData(), 2)