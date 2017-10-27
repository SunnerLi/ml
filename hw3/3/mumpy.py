import numpy as np
import math

"""
    There are some definitions of fundemental array operations here
    Written by SunnerLi (0656011)
"""

# Constant
pi = 3.1415926
ndarray = np.ndarray

def empty(_shape):
    """default function"""
    return np.empty(_shape)

def reshape(a, _shape):
    """default function"""
    return np.reshape(a, _shape)

def asarray(a):
    """default function"""
    return np.asarray(a)

def expand_dims(a, axis=-1):
    """default function"""
    return np.expand_dims(a, axis=-1)

def linspace(start, end, num=50):
    """illustration only"""
    return np.linspace(start, end, num=num)

def zeros_like(a):
    row, column = np.shape(a)
    res = np.zeros([row, column], dtype=float)
    return res

def T(a):
    # res = zeros_like(a)
    if len(shape(a)) == 1:
        row = len(a)
        res = np.zeros([1, row])
        for i in range(row):
            res[0][i] = a[i]
    else:
        row, column = shape(a)
        res = np.zeros([column, row])
        row, column = np.shape(res)
        for i in range(row):
            for j in range(column):
                res[i][j] = a[j][i]
    return res

def dot(a, b):
    print('a: ', a)
    print('b: ', b)
    if len(np.shape(a)) < 2:
        a = np.expand_dims(a, axis=-1)
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
    _max = -1
    _max_index = 0
    for i in range(_start, len(_list)):
        if _list[i] > _max:
            _max = _list[i]
            _max_index = i
    return _max_index

def argmax(_list):
    _max = -1
    _max_index = 0
    for i in range(len(_list)):
        if _list[i] > _max:
            _max = _list[i]
            _max_index = i
    return _max_index

def inv(_arr):
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

def max(arr):
    max_value = -2147483647
    for i in range(len(arr)):
        if arr[i] > max_value:
            max_value = arr[i]
    return max_value

def shape(arr):
    rank = 0
    shape_list = []
    break_cond = True
    _arr = arr
    if len(arr) == 0:
        return (0,)
    while break_cond == True:
        if type(_arr) == np.ndarray:
            _arr = _arr[0]
            rank += 1
        else:
            break_cond = False
    _arr = arr
    for i in range(rank):
        shape_list.append(len(_arr))
        _arr = _arr[0]
    return tuple(shape_list)

def rank(arr):
    return len(shape(arr))

def dtype(arr):
    _shape = shape(arr)
    _arr = arr
    counter = 0
    while counter < len(_shape):
        _arr = _arr[0]
        counter += 1
    return type(_arr)

def copy(arr):
    return np.asarray(list(arr))

def full(_shape, n):
    res = empty(_shape)
    res = reshape(res, [-1])
    for i in range(len(res)):
        res[i] = n
    return reshape(res, _shape)  

def _mean(arr):
    if type(arr[0]) == np.ndarray:
        res = empty(len(arr))
        for i in range(len(arr)):
            res[i] = _mean(arr[i])
        return res
    else:
        _sum = 0.0
        for i in range(len(arr)):
            _sum += arr[i]
        return _sum / len(arr)

def mean(arr, axis=0):
    m = 0
    if axis == 0:
        arr = T(arr)
        res = _mean(arr)        
        arr = T(arr)
    else:
        res = _mean(arr)
    return res

def _var(arr):
    miu = _mean(arr)
    if type(arr[0]) == np.ndarray:
        for i in range(len(arr)):
            res = empty(len(arr))
            for i in range(len(arr)):
                res[i] = _var(arr[i])
            return res
    else:
        _sum = 0
        for i in range(len(arr)):
            _sum += (arr[i] - miu) ** 2
        return _sum / len(arr)

def var(arr, axis=0):
    m = 0
    if axis == 0:
        arr = T(arr)
        res = _var(arr)        
        arr = T(arr)
    else:
        res = _var(arr)
    return res

def log(arr):
    origin_shape = shape(arr)
    arr = reshape(arr, [-1])
    for i in range(len(arr)):
        arr[i] = math.log(arr[i])
    return reshape(arr, origin_shape)

def sum(arr):
    _sum = 0
    for i in range(len(arr)):
        _sum += arr[i]
    return _sum

def logical_and(arr1, arr2):
    if len(arr1) == len(arr2):
        res = empty(shape(arr1))
        for i in range(len(arr1)):
            res[i] = arr1[i] & arr2[i]
        res = res.astype(dtype=bool)
        return res
    else:
        print("length isn't equal...")
        exit()

def _round(arr, digit):
    for i in range(len(arr)):
        arr[i] = round(arr[i], digit)
    return arr

def ones(_shape):
    res = empty(_shape)
    res = reshape(res, [-1])
    for i in range(len(res)):
        res[i] = 1
    return reshape(res, _shape)

def eye(_len, dtype):
    res = ones([_len, _len])
    for i in range(_len):
        for j in range(_len):
            if i != j:
                res[i][j] = 0
    return res
