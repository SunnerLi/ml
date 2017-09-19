from matplotlib import pyplot as plt
import numpy as np
import argparse
from scipy.linalg import lu, inv

# Example lambdas
lambdas = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]

def load(file_name='./data.dat'):
    """
        Load the data points from the file

        Arg:    file_name   - The name of the data points file
        Ret:    The arrays of coordinate x and y
    """
    x = []
    y = []
    with open(file_name, 'r') as f:
        data_points = f.readlines()
        for line in data_points:
            x.append(float(line.split(',')[0]))
            y.append(float(line.split(',')[1]))
    x = np.concatenate((np.expand_dims(np.asarray(x), axis=0), np.expand_dims(np.ones(len(x)), axis=0)))
    y = np.expand_dims(np.asarray(y), axis=0)
    return x.T, y.T

def invLU(_arr):
    """
        Get the inverse array by the LU decomposition

        Arg:    The array you want to get inverse
        Ret:    The inversed array
    """
    p, l, u = lu(_arr)
    u_inv = inv(u)
    l_inv = inv(l)
    arr_inv = np.dot(u_inv, l_inv)
    return np.dot(u_inv, l_inv)

def linearRegression(x, y, m, _lambda, use_lu=False):
    """
        Do the linear regression with regularization term

        For example, x = [[16, 4, 1], [4, 2, 1]]
        y = [[8], [4]]
        m = 2
        _lambda = 0 (without regularization)
        The result will become: [[[w0, w1, w2]]]

        Arg:    x       - The x array
                y       - The y array
                m       - The leading number of polynomial basis
                _lambda - The lambda (factor to control the influence of regularization panalty)
                use_lu  - If you want to use LU decomposition function to compute (default is False)
        Ret:    The optimal wights array
    """
    x = _dimensionProjection(x, m)
    inverse_term = np.dot(x.T, x) + _lambda * np.eye(np.shape(np.dot(x.T, x))[0])
    if use_lu:
        loss_inv = invLU(inverse_term)
    else:
        loss_inv = np.linalg.inv(inverse_term)
    return np.dot(np.dot(loss_inv, x.T), y)

def _dimensionProjection(x, m):
    """
        Raise the dimension as the demend
        For example, x = 
        [
            [4, 1]
            [2, 1]
        ]
        Then the result will be =
        [
            [16, 4, 1]
            [ 4, 2, 1]
        ]

        Arg:    x   - The x array
                m   - The leading number of polynomial basis
        Ret:    The x array with high dimension
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
    """
        Print out the equation of the polynomial model

        Arg:    weights - The weight array
    """
    print('f(x) = ', end='')
    have_print_symbol = False
    for i in range(len(weights), 0, -1):
        if weights[len(weights) - i] != 0:
            if have_print_symbol == True:
                print(' + ', end='')
            print('(', weights[len(weights) - i], ') * x^', i-1, end='')
            have_print_symbol = True
    print('')

def printOutMultiple(x, y, m, weights):
    """
        Print out the results of the linear regression.
        The result includes lambda, LSE error value and the equation of the polynomial model
        This function will print for multiple sub-plot

        Arg:    x       - The x array
                y       - The y array
                m       - The leading number of polynomial basis
                weights - The weights list which contain several weight array
    """
    for j in range(len(weights)):
        print('lambda: ', lambdas[j])
        printOutModel(weights[j])
        print('LSE: ', getError(x, y, m, weights[j]))
        print('')

def draw(x, y, m, weights):
    """
        Draw the result on the plane

        Arg:    x       - The x array
                y       - The y array
                m       - The leading number of polynomial basis
                weights - The weights array
    """
    # Plot points
    x = x.T[0]
    y = y.T[0]
    plt.plot(x, y, 'o')

    # Plot curve
    curve_x = np.linspace(np.min(x), np.max(x), num=50000)
    _curve_x = np.concatenate((np.expand_dims(curve_x, axis=0), np.expand_dims(np.ones(len(curve_x)), axis=0))).T
    _curve_x = _dimensionProjection(_curve_x, m)
    _curve_y = np.reshape(np.dot(_curve_x, weights), [-1])
    plt.plot(curve_x, _curve_y)

    # Show
    plt.show()

def drawMultiple(x, y, m, weights):
    """
        Draw the results on the plane
        The plane will contain 9 sub-figure

        Arg:    x       - The x array
                y       - The y array
                m       - The leading number of polynomial basis
                weights - The weights list which contain several weight array
    """
    # Plot curve
    curve_x = np.linspace(np.min(x.T[0]), np.max(x.T[0]), num=50000)
    _curve_x = np.concatenate((np.expand_dims(curve_x, axis=0), np.expand_dims(np.ones(len(curve_x)), axis=0))).T
    _curve_x = _dimensionProjection(_curve_x, m)

    # Plot each sub-figure
    plt.figure(1)
    for i in range(len(weights)):
        subplot_idx = '33' + str(i+1)
        plt.subplot(subplot_idx)
        plt.plot(x.T[0], y.T[0], 'o')

        # Plot curve
        _curve_y = np.reshape(np.dot(_curve_x, weights[i]), [-1])
        plt.plot(curve_x, _curve_y)
        plt.title('lambda=' + str(lambdas[i]) + '  LSE=' + str(getError(x, y, m, weights[i])))

    # Show
    plt.show()

def getError(x, y, m, weights):
    """
        Get the LSE for specific x, y and optimal weight array

        Arg:    x       - The x array
                y       - The y array
                m       - The leading number of polynomial basis
                weights - The weights array
    """
    x = _dimensionProjection(x, m)
    y_ = np.reshape(np.dot(x, weights), [-1])
    y = np.reshape(y, [-1])
    err = np.sum(np.square((y - y_))) / 2
    return err

if __name__ == '__main__':
    # Parse the arguement
    parser = argparse.ArgumentParser(description='Enter the input of the hw1')
    parser.add_argument('--name', type=str, default='./data.dat', dest='file_name', help='the data point pair file')
    parser.add_argument('--lead', type=int, default=2, dest='m', help='the leading number of polynomial basis')
    parser.add_argument('--lambda', type=float, default=-1, dest='_lambda', help='the factor of regularization panalty')
    args = parser.parse_args()
    x, y = load(args.file_name)

    # Do linear regression with single lambda or multiple
    if args._lambda != -1:
        weights = linearRegression(x, y, args.m, args._lambda, use_lu=True)
        weights = np.reshape(weights, [-1])
        printOutModel(weights)
        print('lambda: ', args._lambda, '\tLSE: ', getError(x, y, args.m, weights))
        draw(x, y, args.m, weights)
    else:
        weights = []
        for i in range(len(lambdas)):
            weights.append(np.reshape(linearRegression(x, y, args.m, lambdas[i]), [-1]))
        printOutMultiple(x, y, args.m, weights)
        drawMultiple(x, y, args.m, weights)