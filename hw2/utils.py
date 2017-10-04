import numpy as np

def print2File(arr):
    with open('result.dat', 'w') as f:
        for i in range(len(arr)):
            f.write(str(arr[i]) + '\n')