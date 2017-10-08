from matplotlib import pyplot as plt
from utils import *
import numpy as np
import argparse

def load(path='./input.dat'):
    """
        Load the sequential training data

        Arg:    path    - The path of the training data
        Ret:    The 2-D array whose shape is [num_epoch, 2]
    """
    string = open(path, 'r').readlines()
    res = np.zeros([len(string), 2])
    for i in range(len(string)):
        for c in string[i]:
            if c == '0':
                res[i][0] += 1
            elif c == '1':
                res[i][1] += 1
    return res

def draw(x_list, y_list):
    """
        Draw the curve of whole bayesian movement

        Arg:    x_list  - The list of x linear space whose shape is [num_epoch, 3, num_sample_points]
                y_list  - The list of PDF whose shape is [num_epoch, 3, num_sample_points]
    """
    if len(x_list) != len(y_list):
        print('invalid length...')
        exit()
    plt.figure(1)
    for i in range(len(x_list)):
        title = [' prior', ' likelihood', ' posterior']
        for j in range(3):
            plt.subplot(len(x_list), 3, j+1+i*3)
            plt.plot(x_list[i][j], y_list[i][j])
            max_prob_x = [x_list[i][j][np.argmax(y_list[i][j])]] * 100
            max_prob_y = np.linspace(0, np.max(y_list[i][j]), 100)
            plt.plot(max_prob_x, max_prob_y, linestyle='--')
            plt.title('iter ' + str(i+1) + title[j])
    plt.show()

if __name__ == '__main__':
    # Parse the parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=int, default=2, dest='a', help='initial a of beta distribution')
    parser.add_argument('--b', type=int, default=2, dest='b', help='initial b of beta distribution')    
    args = parser.parse_args()
    a = args.a
    b = args.b

    # Train
    x_list = []
    y_list = []
    training_data = load(path='./input.dat')
    for i in range(len(training_data)):
        _a, _b = training_data[i][1], training_data[i][0]
        _x_list = []
        _y_list = []
        print('iter: ', i+1, end='\t')
        print('prior: ', round(bataDistribution_maxProb(a, b), 5), end='\t\t')
        x_curve, y_curve = bataDistribution_curve(a, b)
        _x_list.append(x_curve)
        _y_list.append(y_curve)
        # print('likelihood: ', round(binomialDistribution_maxProb(_a+_b, _b), 5), end='\t')
        print('likelihood: ', round(_a / (_a+_b), 5), end='\t')
        x_curve, y_curve = binomialDistribution_curve(_a+_b, _b)
        _x_list.append(x_curve)
        _y_list.append(y_curve)
        a += _a
        b += _b
        print('posterior: ', round(bataDistribution_maxProb(a, b), 5))
        x_curve, y_curve = bataDistribution_curve(a, b)
        _x_list.append(x_curve)
        _y_list.append(y_curve)
        x_list.append(_x_list)
        y_list.append(_y_list)
    draw(x_list, y_list)