from matplotlib import pyplot as plt
from q1 import gaussian
import numpy as np
import argparse
import random

def phi(x, m):
    res = np.empty([1, m+1])
    for i in range(m+1):
        res[0][i] = x ** i
    return res.T

def Y(x, w):
    if type(w) != np.ndarray:
        w = np.asarray(w)
    return np.dot(w, x)

if __name__ == '__main__':
    # Deal with argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=2, dest='n', help='the order of basis function')
    parser.add_argument('--a', type=float, default=1.0, dest='a', help='the variance of error')
    parser.add_argument('--size', type=int, default=1, dest='size', help='the number of points you want to generate')
    parser.add_argument('--w', type=float, action='append', default=[1.0], dest='w', help='the weight of linear model')
    args = parser.parse_args()

    # Generate
    x_list = []
    y_list = []
    y_fuse_list = []
    for i in range(args.size):
        x = random.random() * 20 - 10
        x_hyper = phi(x, args.n)
        y = Y(x_hyper, args.w) 
        y_fuse = y + gaussian(0, args.a, size=1)
        x_list.append(x)
        y_list.append(y)
        y_fuse_list.append(y_fuse)
    
    # Plot
    plt.plot(x_list, y_list, 'o', label='mean of line')
    plt.plot(x_list, y_fuse_list, 'o', label='actual line')
    plt.legend()
    plt.show()