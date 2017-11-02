from matplotlib import pyplot as plt
import numpy as np
import random

def mul(x, w):
    """ x : [2 , 1]
        w : [3, 1]
    """
    op_x = np.concatenate((np.asarray([[1]]), x), axis=-1)
    return op_x.dot(w)

def phi_X(x):
    """
        x : [2, 20]
    """
    res = np.ones([1, np.shape(x)[1]])
    return np.concatenate((res, x), axis=0)

def generate(mx, vx, n):
    return np.random.normal(mx, vx, n)

def logisticRegression(A, y_arr, break_cond = 1e-4):
    w = np.asarray([[1], [0.1], [-0.01]], dtype=float)
    for i in range(20):
        # Steepest gradient decent
        y_ = 1 / (1 + np.exp(-w.T.dot(A)))
        diff = (y_ - y_arr.T).T
        gradient = A.dot(diff)
        w -= gradient

        # l2 loss
        y_ = 1 / (1 + np.exp(-w.T.dot(A)))
        diff = (y_ - y_arr.T).T
        loss = 0.5 * np.sum(np.square(diff))
        print('iter: ', i, '\tl2 loss: ', loss)

        # Break check
        if loss < break_cond:
            break
    return w


if __name__ == '__main__':
    # Define parameters of generator
    n = 100      # number of data point, D
    mx1 = 0.0   # The mean of the x in first class
    vx1 = 1.0   # The variance of the x in first class
    my1 = 0.0   # The mean of the y in first class
    vy1 = 1.0   # The variance of the y in first class
    mx2 = 2.0   # The mean of the x in second class
    vx2 = 1.0   # The variance of the x in second class
    my2 = 2.0   # The mean of the y in second class
    vy2 = 1.0   # The variance of the y in second class

    # Show points
    x_arr = np.ones([3, 2*n])
    x_list_1 = generate(mx1, vx1, n)
    y_list_1 = generate(my1, vy1, n)
    x_list_2 = generate(mx2, vx2, n)
    y_list_2 = generate(my2, vy2, n)
    for i in range(1, len(x_arr)):
        for j in range(2*n):
            if i == 1:
                if j < n:
                    x_arr[i][j] = x_list_1[j]
                else:
                    x_arr[i][j] = x_list_2[j-n]
            else:
                if j < n:
                    x_arr[i][j] = y_list_1[j]
                else:
                    x_arr[i][j] = y_list_2[j-n]
    plt.plot(x_list_1, y_list_1, 'o', label='1')
    plt.plot(x_list_2, y_list_2, 'o', label='2')
    plt.legend()
    # plt.show()

    # Do logistic regression
    y_list = [0] * n + [1] * n
    y_arr = np.asarray([y_list]).T
    w = logisticRegression(x_arr, y_arr)

    # Draw decision boundary
    min_range = np.min(x_arr)
    max_range = np.max(x_arr)
    sample_num = 100
    dummy_points = np.meshgrid(np.linspace(min_range, max_range, sample_num), np.linspace(min_range, max_range, sample_num))
    dummy_points_reshape = np.reshape(dummy_points, [2, -1])
    dummy_points_hyper = np.concatenate((np.ones([1, sample_num ** 2]), dummy_points_reshape), axis=0)

    y_ = 1 / (1 + np.exp(-w.T.dot(dummy_points_hyper)))
    diff = (y_.T - 0.5) ** 2
    idx = diff < 0.01
    x_boundary_list = []
    y_boundary_list = []
    for i in range(sample_num ** 2):
        if idx[i][0] == True:
            x_boundary_list.append(dummy_points_reshape[0][i])
            y_boundary_list.append(dummy_points_reshape[1][i])

    plt.plot(x_boundary_list, y_boundary_list)
    plt.show()
    