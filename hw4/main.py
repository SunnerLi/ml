from matplotlib import pyplot as plt
import numpy as np
import random

def mul(x, w):
    """ x : [2 , 1]
        w : [3, 1]
    """
    op_x = np.concatenate((np.asarray([[1]]), x), axis=-1)
    return op_x.dot(w)

def generate(mx, vx, my, vy, n):
    return np.random.normal(mx, vx, n), np.random.normal(my, vy, n)

def logisticRegression(A, y_arr):
    w = np.asarray([[1], [1], [1]], dtype=float)
    for i in range(1000):
        # SGD
        for j in range(len(A)):
            x = A[j:j+1]
            y = y_arr[j]
            diff = 1 / (1 + np.exp(-mul(x, w))) - y
            gradient = mul(x, diff[0][0]).T
            w -= 0.001 * gradient

        # L2 distance measure
        loss_sum = 0
        for j in range(len(A)):
            x = A[j:j+1]
            y = y_arr[j]
            diff = 1 / (1 + np.exp(-mul(x, w))) - y
            loss_sum += 0.5 * pow(diff[0][0], 2)
        print('iter: ', i, '\tl2 loss: ', loss_sum)

if __name__ == '__main__':
    # Define parameters of generator
    n = 10      # number of data point, D
    mx1 = 0.0   # The mean of the x in first class
    vx1 = 1.0   # The variance of the x in first class
    my1 = 0.0   # The mean of the y in first class
    vy1 = 1.0   # The variance of the y in first class
    mx2 = 2.0   # The mean of the x in second class
    vx2 = 1.0   # The variance of the x in second class
    my2 = 2.0   # The mean of the y in second class
    vy2 = 1.0   # The variance of the y in second class

    # Show points
    x_list_1, y_list_1 = generate(mx1, vx1, my1, vy1, n)
    x_list_2, y_list_2 = generate(mx2, vx2, my2, vy2, n)
    plt.plot(x_list_1, y_list_1, 'o', label='1')
    plt.plot(x_list_2, y_list_2, 'o', label='2')
    plt.legend()
    # plt.show()

    # Form A
    A = np.empty([2 * n, 2])
    for i in range(2*n):
        for j in range(2):
            if j == 0:
                if i < n:
                    A[i][j] = x_list_1[i]
                else:
                    A[i][j] = x_list_2[i-n]
            if j == 1:
                if i < n:
                    A[i][j] = y_list_1[i]
                else:
                    A[i][j] = y_list_2[i-n]
    y_arr = np.concatenate((y_list_1, y_list_2), axis=-1).T
    logisticRegression(A, y_arr)