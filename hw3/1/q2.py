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

def generateX(_min, _max, order):
    res = np.empty([order])
    for i in range(order):
        res[i] = random.random() * (_max - _min) + _min
    return res

def Y(x, w):
    if type(w) != np.ndarray:
        w = np.asarray(w)
    return np.dot(w, x)

def generateLinearModel(n, a, size, w):
    x_list = []
    y_list = []
    for i in range(size):
        x = generateX(-10, 10, n)
        y = Y(x, w) 
        y_fuse = y + gaussian(0, a, size=1)
        x_list.append(x)
        y_list.append(y_fuse)
    return np.asarray(x_list), np.asarray(y_list)

if __name__ == '__main__':
    # Deal with argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=2, dest='n', help='the order of basis function')
    parser.add_argument('--a', type=float, default=1.0, dest='a', help='the variance of error')
    parser.add_argument('--size', type=int, default=1, dest='size', help='the number of points you want to generate')
    parser.add_argument('--w', type=float, action='append', default=[], dest='w', help='the weight of linear model')
    args = parser.parse_args()
    if args.n != len(args.w):
        print('you should make match between n and the weight vector...')
        exit()

    # Generate
    x_arr, y_arr = generateLinearModel(args.n, args.a, args.size, args.w)
    for i in range(len(x_arr)):
        print('x: ', x_arr[i], '\tt: ', y_arr[i])