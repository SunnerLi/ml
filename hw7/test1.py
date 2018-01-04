from draw import drawCorrelationMatrix
import numpy as np

C = np.asarray([[0, 0.1, 0.2, 0.8], [0.1, 0, 0.9, 0.2], [0.05, 0.9, 0, 0.1], [0.8, 0.1, 0.1, 0]])
labels = np.asarray([0, 1, 1, 0])

def swap_rows(C, var1, var2):
    D = C.copy()
    D[var2, :] = C[var1, :]
    D[var1, :] = C[var2, :]
    E = D.copy()
    E[:, var2] = D[:, var1]
    E[:, var1] = D[:, var2]
    return E

def formCorrelationMatrix(C, labels):
    #    for _label in np.unique(labels):
    change_pair = zip(range(len(labels)), np.argsort(labels))
    for before_idx, after_idx in change_pair:
        if before_idx < after_idx:
            print(before_idx, after_idx)
            C = swap_rows(C, before_idx, after_idx)
    return C

C_permu = formCorrelationMatrix(C, labels)
print(C_permu)
drawCorrelationMatrix(C_permu)
