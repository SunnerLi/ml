from collections import OrderedDict, Counter
from matplotlib import pyplot as plt
import numpy as np
import argparse
import random
import math

def __generate(miu, std):
    u = random.random()
    v = random.random()
    return pow(-2 * math.log(u), 0.5) * math.cos(2 * math.pi * v) * std + miu

def gaussian(miu, var, size=50):
    res = []
    return np.asarray([__generate(miu, pow(var, 0.5)) for i in range(size)])

def draw(freq_list):
    x_list = [x for x, y in freq_list]
    y_list = [y for x, y in freq_list]
    plt.plot(x_list, y_list)
    plt.show()

if __name__ == '__main__':
    # Deal with argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--mean', type=float, default=0.0, dest='mean', help='the mean of the gaussian')
    parser.add_argument('--var', type=float, default=1.0, dest='var', help='the variance of the gaussian')
    parser.add_argument('--size', type=int, default=1, dest='size', help='the number of points you want to generate')
    args = parser.parse_args()

    # Initialize
    mean = args.mean
    variance = args.var
    size = args.size
    num_list = []
    counter = {}
    
    # Generate
    num_list = gaussian(mean, variance, size=size)
    if size == 1:
        print('generated: ', num_list)
    else:
        num_list = np.round(num_list, 1)
        counter = Counter(num_list)
        print('mean: ', np.mean(num_list))
        print('var: ', np.var(num_list))
        counter = sorted((float(x), y) for x, y in counter.items())
        draw(counter)