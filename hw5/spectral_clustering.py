from scipy.linalg import eigh
from utils import *
import numpy as np

def SpectralWrapper(data_arr, k_cluster):
    """
        data_arr = (2 * N)
    """
    data_arr = data_arr.T
    num_points = np.shape(data_arr)[1]

    # Compute W matrix and D matrix by RBF kernel
    W = np.ndarray([num_points, num_points])
    for i in range(num_points):
        W[i] = kernel(data_arr, data_arr[:, i:i+1])
    D = np.diag(np.sum(W, axis=1))
    print(W, '\n')
    print(D)

    # Compute unnormalized graph laplacian
    L = W - D

    # Compute first k generalized eigenvectors
    eigen_values, eigen_vectors = eigh(L, D)
    print(np.shape(eigen_vectors))


if __name__ == '__main__':
    SpectralWrapper(generateData(), 2)