#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.
from draw import * 
import numpy as np
import pylab


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.

        Ret:    1. Shannon entropy, size is (1,)
                2. Probability in high dim, size is (2500,)
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.

        Calculate the conditional probability in high dimensional space
        Ret:    Conditional probability matrix, size is (2500, 2500)
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(
            np.add(
                -2 * np.dot(X, X.T), sum_X
            ).T, sum_X
        )                           # shape: (2500, 2500)
    P = np.zeros((n, n))            # shape: (2500, 2500)
    beta = np.ones((n, 1))          # shape: (2500, 1)
    logU = np.log(perplexity)       # shape: ()

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]       # shape: (2500,)
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 1:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.

        Ret:    The projected array whose size is (N, no_dims)
                e.g. (2500, 50)
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(origin_X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.

        Arg:    no_dims     - The number of dimension you want to project
                
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = origin_X.copy()
    X = pca(X, initial_dims).real       # X shape after PCAL (2500, 50)
    (n, d) = X.shape
    max_iter = 100
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)     # (2500, 2)
    dY = np.zeros((n, no_dims))         # (2500, 2) 
    iY = np.zeros((n, no_dims))         # (2500, 2)
    gains = np.ones((n, no_dims))       # (2500, 2)
    cost_list = []

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.				# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))   # Student-T equation
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        # ---------------------------------------------------
        # Both P, Q and PQ size is (2500, 2500)
        # ---------------------------------------------------
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update (early exaggeration)
        # ---------------------------------------------------
        # iY is Y(t-1) - Y(t-2)
        # ---------------------------------------------------
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))
            cost_list.append(C)

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y, cost_list, P, Q


def sne(origin_X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs symmetric SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.

        Arg:    no_dims     - The number of dimension you want to project
                
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = origin_X.copy()
    X = pca(X, initial_dims).real       # X shape after PCAL (2500, 50)
    (n, d) = X.shape
    max_iter = 100
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)     # (2500, 2)
    dY = np.zeros((n, no_dims))         # (2500, 2) 
    iY = np.zeros((n, no_dims))         # (2500, 2)
    gains = np.ones((n, no_dims))       # (2500, 2)

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.				# early exaggeration
    P = np.maximum(P, 1e-12)
    cost_list = []

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        # num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))   # Student-T equation
        num = np.exp(-np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        # ---------------------------------------------------
        # Both P, Q and PQ size is (2500, 2500)
        # ---------------------------------------------------
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update (early exaggeration)
        # ---------------------------------------------------
        # iY is Y(t-1) - Y(t-2)
        # ---------------------------------------------------
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))
            cost_list.append(C)

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y, cost_list, P, Q

if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")
    X = X[:500, ]
    labels = labels[:500, ]
    # print(labels)

    # Q1 & Q2
    # -------------------------------------------------------
    # Adopt T-SNE
    # -------------------------------------------------------
    Y, cost_list, P_tsne, Q_tsne = tsne(X, 2, 50, 20.0)
    drawScatter(Y, labels, 'tsne_result.png')
    drawCost(cost_list, 'tsne_cost_curve.png')
    # print(P_tsne)   
    print(np.min(P_tsne), np.max(P_tsne))
    P_tsne += (np.eye(np.shape(P_tsne)[0]) * np.max(P_tsne))
    drawCorrelationMatrix(P_tsne, 'P_tsne_before.png')
    P_tsne = formCorrelationMatrix(P_tsne, labels)
    Q_tsne = formCorrelationMatrix(Q_tsne, labels)
    # print(P_tsne)
    # for i in range(np.shape(P_tsne)[0]):
    #     P_tsne[i][i] = 1.
    #     break

    # -------------------------------------------------------
    # Adopt Symmetric SNE
    # -------------------------------------------------------
    """
    Y, cost_list, P_sne, Q_sne = sne(X, 2, 50, 20.0)
    drawScatter(Y, labels, 'sne_result.png')
    drawCost(cost_list, 'sne_cost_curve.png')
    """

    # Q3
    #P_tsne = formCorrelationMatrix(P_tsne, labels)
    #Q_tsne = formCorrelationMatrix(Q_tsne, labels)
    drawCorrelationMatrix(P_tsne, 'P_tsne_after.png')
    drawCorrelationMatrix(Q_tsne, 'Q_tsne.png')

    """
    P_sne = formCorrelationMatrix(P_sne, labels)
    Q_sne = formCorrelationMatrix(Q_sne, labels)
    drawCorrelationMatrix(P_sne, 'P_sne.png')
    drawCorrelationMatrix(Q_sne, 'Q_sne.png')
    """

    # Q4
    # Adopt different perplexity
    """
    Y, cost_list, _, __ = tsne(X, 2, 50, 5.0)
    drawScatter(Y, labels, 'tsne_result_5.png')
    Y, cost_list, _, __ = tsne(X, 2, 50, 100.0)
    drawScatter(Y, labels, 'tsne_result_100.png')
    """