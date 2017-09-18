import numpy as np
import argparse

def load(file_name='./data.dat'):
    x = []
    y = []
    with open(file_name, 'r') as f:
        data_points = f.readlines()
        for line in data_points:
            x.append(int(line.split(',')[0]))
            y.append(int(line.split(',')[1]))
    x = np.concatenate((np.expand_dims(np.asarray(x), axis=0), np.expand_dims(np.ones(len(x)), axis=0)))
    y = np.expand_dims(np.asarray(y), axis=0)
    return x.T, y.T

def linearRegression(x, y, m, _lambda):
    x = _dimensionProjection(x, m)
    print('proj: ', x)
    inverse_term = np.dot(x.T, x) + _lambda * np.eye(np.shape(np.dot(x.T, x))[0])
    loss_inv = np.linalg.pinv(inverse_term),
    return np.dot(np.dot(loss_inv, x.T), y)

def _dimensionProjection(x, m):
    """
        Raise the dimension as the demend
    """
    result = None
    for i in range(m, 0, -1):
        if type(result) == type(None):
            result = np.expand_dims(x.T[0] ** i, axis=0)
            print(result)
        else:
            result = np.concatenate((result, np.expand_dims(x.T[0] ** i, axis=0)), axis=0)
            print(result)
    result = np.concatenate((result, np.expand_dims(np.ones(np.shape(x.T[0])), axis=0)), axis=0)
    return result.T

def printOutModel(weights):
    weights = np.round(np.reshape(weights, [-1]), decimals=5)
    print('f(x) = ', end='')
    have_print_symbol = False
    for i in range(len(weights), 0, -1):
        if weights[len(weights) - i] != 0:
            if have_print_symbol == True:
                print(' + ', end='')
            print(weights[len(weights) - i], '* x^', i-1, end='')
            have_print_symbol = True
    print('')

if __name__ == '__main__':
    # Parse the arguement
    parser = argparse.ArgumentParser(description='Enter the input of the hw1')
    parser.add_argument('--name', type=str, default='./data.dat', dest='file_name', help='the data point pair file')
    parser.add_argument('--lead', type=int, default=2, dest='m', help='the leading number of polynomial basis')
    parser.add_argument('--lambda', type=float, default=1, dest='_lambda', help='the factor of regularization panalty')
    args = parser.parse_args()

    # Linear regression
    x, y = load(args.file_name)
    weight = linearRegression(x, y, args.m, args._lambda)
    printOutModel(weight)