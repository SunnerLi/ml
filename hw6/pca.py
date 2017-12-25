from matplotlib import pyplot as plt
import numpy as np

data = np.asarray([
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0],
    [2, 2],
    [-0.5, -0.5],
    [1.5, 1.5],
    [-1, -1],
    [0.8, 0.2],
    [0.2, 0.8]
])

W = None

def show(data):
    plt.plot(data[:, 0], data[:, 1], 'o')
    plt.show()

def pca(X):
    global W

    if W is None:
        center_X = X - np.mean(X, axis=0)
        whiten_X = center_X - np.std(center_X, axis=0)
        S = whiten_X.T.dot(whiten_X) / np.shape(whiten_X)[0] - 1
        Lambda, W = np.linalg.eigh(S)
        idx = np.argsort(Lambda)[::-1]
        Lambda = Lambda[idx]
        W = W[:, idx]
    Z = np.dot(X, W)
    return Z

def pca_reverse(Z):
    global W
    if W is None:
        print('W is None...')
        exit()
    return Z.dot(np.linalg.inv(W))

if __name__ == '__main__':
    data = pca(data)
    show(data)