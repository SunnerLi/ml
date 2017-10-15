import numpy as np

a = np.reshape(range(24), [4, 3, 2])
b = np.reshape(range(6), [3, 2])
print(a)
print(a - b)
np.logical_and()