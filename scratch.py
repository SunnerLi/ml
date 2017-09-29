import numpy as np

"""
    There are some definitions of fundemental array operations here
    Written by SunnerLi (0656011)
"""

def zeros_like(a):
    """
        Create zero-like array just like numpy does

        Arg:    a   - The original array
        Ret:    The zeros array with the same shape
    """
    row, column = np.shape(a)
    res = np.zeros([row, column], dtype=float)
    return res

def T(a):
    """
        Return the transpose of the matrix

        Arg:    a   - The matrix you want to transpose
        Ret:    The transposed matrix
    """
    # res = zeros_like(a)
    row, column = np.shape(a)
    res = np.zeros([column, row])
    row, column = np.shape(res)
    for i in range(row):
        for j in range(column):
            res[i][j] = a[j][i]
    return res

def mul(a, b):
    """
        Return the multiply result of two matrix

        Arg:    a   - The 1st matrix you want to multiply
                b   - The 2nd matrix you want to multiply
        Ret:    The multiplied result
    """
    if len(np.shape(b)) < 2:
        b = np.expand_dims(b, axis=-1)
    res = np.zeros([np.shape(a)[0], np.shape(b)[1]])
    row, column = np.shape(res)
    for i in range(row):
        for j in range(column):
            for _a, _b in zip(a[i], T(b)[j]):
                res[i][j] += _a * _b
    return res

def argmax_from(_list, _start):
    """
        Find the index of maximun element
        Different from numpy, it find in the range of [_start, last_element)

        Arg:    _list   - The list you want to traversal
                _start  - The start position of the traversal range
        Ret:    The maximun index
    """
    _max = -1
    _max_index = 0
    for i in range(_start, len(_list)):
        if _list[i] > _max:
            _max = _list[i]
            _max_index = i
    return _max_index

def lu(arr):
    """
        LU Decomposition

        Arg:    arr - The array you want to do the LU decomposition
        Ret:    The upper triangle matrix and lower triangle matrix
    """
    # Dimension check
    row, column = np.shape(arr)
    if row != column:
        print("row isn't equal to column, exit...")
        return None, None

    # Permutate the row
    for j in range(column):
        max_index = argmax_from(T(arr)[j], j)
        if max_index != j:
            arr[[max_index, j]] = arr[[j, max_index]]

    # Gaussian elimination
    U = np.copy(arr)
    L = np.eye(row)
    for i in range(0, row-1):
        for j in range(i+1, row):
            scale = 1 * U[j][i] / U[i][i]
            U[j] += -scale * U[i]
            L[j][i] = scale
    return L, U

def inv(_arr):
    """
        Get the inverse of the matrix
        
        Arg:    _arr    - The array you want to find the inverse
        Ret:    The inverse of the matrix
    """
    n, n = np.shape(_arr)
    pivot = np.eye(n)
    _arr = np.concatenate((_arr, pivot), axis=1)
    for i in range(n):
        for j in range(n):
            if i != j:
                scale = _arr[j][i] / _arr[i][i]
                _arr[j, :] += -scale * _arr[i, :]
        _arr[i, :] /= _arr[i][i]
    return _arr[:, n:]

def inv_by_lu(arr):
    """
        Get the inverse by the scratch lu decomposition

        Arg:    arr - The array you want to find the inverse
        Ret:    The inverse of the matrix
    """
    l, u = lu(arr)
    return mul(inv(u), inv(l))