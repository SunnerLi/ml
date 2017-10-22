import _init_path
from q1 import gaussian
import argparse
import math

"""
$ python online_learning.py --mean 3.0 --var 2.0 --size 500
"""

revised_mean = 0.0

def onlineLearning(mean, sample_var, popular_var, x, n):
    global revised_mean
    delta1 = x - mean
    new_mean = mean + delta1 / n
    delta2 = x - mean
    revised_mean += delta1 * delta2
    if n <= 1:
        new_sample_var = float('nan')
    else:
        new_sample_var = revised_mean / (n-1)
    new_popular_var = revised_mean / n 
    return new_mean, new_sample_var, new_popular_var

if __name__ == '__main__':
    # Deal with argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--mean', type=float, default=0.0, dest='mean', help='the mean of the generator')
    parser.add_argument('--var', type=float, default=1.0, dest='var', help='the variance of the generator')
    parser.add_argument('--size', type=int, default=5, dest='size', help='the number of points you want to generate')
    args = parser.parse_args()

    # Initialize
    mean = 0.0
    sample_var = 0.0
    popular_var = 0.0
    num_point = args.size

    # Online learning
    for i in range(1, num_point+1):
        n = gaussian(miu=args.mean, var=args.var, size=1)
        mean, sample_var, popular_var = onlineLearning(mean, sample_var, popular_var, n[0], i)
        print('n: ', i, '\tdata point: ', n[0], '\tmean: ', mean, '\tvar: ', popular_var)