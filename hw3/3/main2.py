import _init_path
from q2_2 import Y, generateLinearModel
from matplotlib import pyplot as plt
from q1 import gaussian
import numpy as np
import argparse
import random

"""
$ python main2.py --alpha 1.0 --beta 1.0 --size 100 --w 0.5 --w -0.4 --w 0 --w 0.001
"""

revised_mean = 0.0

def get_phi(x, m):
    res = np.empty([1, m+1])
    for i in range(m+1):
        res[0][i] = x ** i
    return res

def _dimensionProjection(x, m):
    """
        [
            1   1
            2   3
            4   9
            8   27
        ]
    """
    res = np.empty([m+1, len(x)])
    for i in range(m+1):
        res[i, :] = x ** i
    return res

if __name__ == '__main__':
    # Deal with argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=1.0, dest='alpha', help='the precision of the error term in generator')
    parser.add_argument('--beta', type=float, default=1.0, dest='beta', help='the precision of the prior')
    parser.add_argument('--size', type=int, default=5, dest='size', help='the number of points you want to do in online learning')
    parser.add_argument('--w', type=float, action='append', default=[], dest='w', help='the weight of generator')
    args = parser.parse_args()

    # Constants
    alpha = args.alpha              # The precision of the error term in generator
    beta = args.beta                # The precision of the prior
    generator_order = len(args.w)-1 # The order of the polynomial linear generator
    bayesian_order = len(args.w)-1  # The order of the basis function

    # Initalize w
    generator_weight = args.w
    m_0 = np.ones([bayesian_order+1])
    S_0 = np.eye(bayesian_order+1, dtype=float)

    # Train
    for i in range(1, args.size):
        x_arr, y_arr = generateLinearModel(generator_order, alpha, 1, generator_weight)
        x_arr = get_phi(x_arr, bayesian_order)
        S_N = np.linalg.inv(np.linalg.inv(S_0) + beta * x_arr.T.dot(x_arr))
        m_N = S_N.dot(np.linalg.inv(S_0).dot(m_0) + beta * x_arr.T.dot(y_arr))
        pred_mean = np.expand_dims(m_N.T, axis=-1)
        pred_vars = 1./beta + x_arr.dot(S_N)
        S_0 = S_N
        m_0 = m_N
        print('point: ', (x_arr[0][1], y_arr[0]), '\tmean: ', np.reshape(pred_mean, [-1]), '\tvar: ', np.reshape(pred_vars, [-1]))

    # Draw
    x_list = np.linspace(0, 10, num=500)
    y_list = np.asarray(generator_weight).dot(_dimensionProjection(x_list, generator_order))
    plt.plot(x_list, y_list, '-', label='generated')
    pred_list = np.reshape(pred_mean, [-1]).dot(_dimensionProjection(x_list, bayesian_order))
    plt.plot(x_list, pred_list, '-', label='bayesian')
    plt.legend()
    plt.show()