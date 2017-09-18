from matplotlib import pyplot as plt
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
    inverse_term = np.dot(x.T, x) + _lambda * np.eye(np.shape(np.dot(x.T, x))[0])
    loss_inv = np.linalg.pinv(inverse_term),
    return np.dot(np.dot(loss_inv, x.T), y)

def _dimensionProjection(x, m):
    """
        Raise the dimension as the demend

        [
            [4, 1]
            [2, 1]
        ]


        [
            [16, 4, 1]
            [ 4, 2, 1]
        ]
    """
    result = None
    for i in range(m, 0, -1):
        if type(result) == type(None):
            result = np.expand_dims(x.T[0] ** i, axis=0)
        else:
            result = np.concatenate((result, np.expand_dims(x.T[0] ** i, axis=0)), axis=0)
    result = np.concatenate((result, np.expand_dims(np.ones(np.shape(x.T[0])), axis=0)), axis=0)
    return result.T

def printOutModel(weights):
    print('f(x) = ', end='')
    have_print_symbol = False
    for i in range(len(weights), 0, -1):
        if weights[len(weights) - i] != 0:
            if have_print_symbol == True:
                print(' + ', end='')
            print(weights[len(weights) - i], '* x^', i-1, end='')
            have_print_symbol = True
    print('')

def draw(x, y, m, weights):
    # Plot points
    x = x.T[0]
    y = y.T[0]
    plt.plot(x, y, 'o')

    # Plot curve
    curve_x = np.linspace(np.min(x), np.max(x))
    _curve_x = np.concatenate((np.expand_dims(curve_x, axis=0), np.expand_dims(np.ones(len(curve_x)), axis=0))).T
    _curve_x = _dimensionProjection(_curve_x, m)
    _curve_y = np.reshape(np.dot(_curve_x, weights), [-1])
    plt.plot(curve_x, _curve_y)

    # Show
    plt.show()

if __name__ == '__main__':
    # Parse the arguement
    parser = argparse.ArgumentParser(description='Enter the input of the hw1')
    parser.add_argument('--name', type=str, default='./data.dat', dest='file_name', help='the data point pair file')
    parser.add_argument('--lead', type=int, default=2, dest='m', help='the leading number of polynomial basis')
    parser.add_argument('--lambda', type=float, default=1, dest='_lambda', help='the factor of regularization panalty')
    args = parser.parse_args()

    # Linear regression
    x, y = load(args.file_name)
    weights = linearRegression(x, y, args.m, args._lambda)
    weights = np.round(np.reshape(weights, [-1]), decimals=5)
    printOutModel(weights)
    draw(x, y, args.m, weights)