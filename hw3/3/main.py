import _init_path
from q2_2 import Y, generateLinearModel
from q1 import gaussian
import numpy as np
import random

revised_mean = 0.0

def phi(x, m):
    res = np.empty([1, m+1])
    for i in range(m+1):
        res[0][i] = x ** i
    return res

def onlineLearning(mean, sample_var, popular_var, x, n):
    global revised_mean
    delta1 = x - mean
    new_mean = mean + delta1 / n
    delta2 = x - mean
    revised_mean += delta1 * delta2
    print('delta1: ', delta1, '\tdelta2: ', delta2)
    if n <= 1:
        new_sample_var = float('nan')
    else:
        new_sample_var = revised_mean / (n-1)
    new_popular_var = revised_mean / n 
    return new_mean, new_sample_var, new_popular_var

if __name__ == '__main__':
    prior_precision = 1.0     
    generator_order = 2
    generator_error_var = 2.0
    generator_weight = [1.0, .5, -.4]
    bayesian_order = 3

    # Initalize w
    bayesian_weight_mean = np.ones([bayesian_order+1])
    bayesian_weight_var = np.eye(bayesian_order+1, dtype=float) / prior_precision

    # Train
    print('mean:\n', bayesian_weight_mean)
    print('var:\n', bayesian_weight_var)
    for i in range(1, 200):
        x_arr, y_arr = generateLinearModel(generator_order, generator_error_var, 1, generator_weight)
        x_arr = phi(x_arr, bayesian_order)
        bayesian_weight_mean = np.dot(np.dot( \
            np.linalg.inv(np.dot(x_arr.T, x_arr) + 
                (prior_precision / generator_error_var) * np.eye(np.shape(np.dot(x_arr.T, x_arr))[0])
            ), x_arr.T), y_arr)

        bayesian_weight_var = generator_error_var * np.dot(x_arr.T, x_arr) + prior_precision * np.eye(np.shape(np.dot(x_arr.T, x_arr))[0])
        
        # bayesian_weight_var = 1./generator_error_var + np.dot(x_arr.T, np.linalg.inv(inn))
        print('mean:\n', bayesian_weight_mean)
        print('var:\n', bayesian_weight_var)