import _init_path
from q2_2 import Y, generateLinearModel
from matplotlib import pyplot as plt
from q1 import gaussian
import numpy as np
import argparse
import random

"""
$ python main.py --alpha 1.0 --beta 1.0 --size 100 --w 0.5 --w -0.4 --w 0 --w 0.001
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
    byaesian_mean = np.ones([bayesian_order+1, 1])
    bayesian_precision = np.eye(bayesian_order+1, dtype=float)
    m_0 = np.ones([bayesian_order+1])
    S_0 = np.eye(bayesian_order+1, dtype=float)

    # Train
    var_list = []
    for i in range(1, args.size+1):
        x_arr, y_arr = generateLinearModel(generator_order, alpha, 1, generator_weight)
        x_arr = get_phi(x_arr, bayesian_order)
        
        # Get posterior
        S_N = np.linalg.inv(np.linalg.inv(S_0) + beta * x_arr.T.dot(x_arr))
        m_N = S_N.dot(np.linalg.inv(S_0).dot(m_0) + beta * x_arr.T.dot(y_arr))

        # Get predictive distribution
        posterior_mean = m_N.dot(np.expand_dims(x_arr, -1))
        posterior_vars = 1./beta + x_arr.dot(S_N).dot(x_arr.T)

        # Update prior
        S_0 = S_N
        m_0 = m_N     

        # Pring parameter   
        print('point: ', (x_arr[0][1], y_arr[0]), '\tmean: ', np.reshape(posterior_mean, [-1]), '\tvar: ', np.reshape(posterior_vars, [-1]))
        var_list.append(posterior_vars[0][0])

    # Show trend of var
    plt.plot(range(len(var_list)), var_list, '-o')
    plt.show()