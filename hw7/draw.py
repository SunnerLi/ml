from matplotlib import pyplot as plt
import numpy as np
import pylab

C = np.asarray([[0, 0.1, 0.2, 0.8], [0.1, 0, 0.9, 0.2], [0.05, 0.9, 0, 0.1], [0.8, 0.1, 0.1, 0]])
labels = np.asarray([0, 1, 1, 0])

def drawScatter(arr, label, fig_name):
    unique_labels = np.unique(label)
    for _label in unique_labels:
        idx = label == _label
        batch_arr = arr[idx]
        plt.plot(batch_arr[:, 0], batch_arr[:, 1], 'o', label=str(_label))
    plt.legend()
    if fig_name is None:
        print("figure name is None")
    else:
        plt.savefig(fig_name)
        plt.gca().clear()

def drawCost(cost_list, fig_name):
    plt.plot(range(len(cost_list)), cost_list)
    plt.savefig(fig_name)
    plt.gca().clear()

def swap_rows(C, var1, var2):
    D = C.copy()
    D[var2, :] = C[var1, :]
    D[var1, :] = C[var2, :]
    E = D.copy()
    E[:, var2] = D[:, var1]
    E[:, var1] = D[:, var2]
    return E

def formCorrelationMatrix(C, labels):
    """
        Ref: https://stats.stackexchange.com/questions/138325/clustering-a-correlation-matrix
    """
    change_pair = zip(range(len(labels)), np.argsort(labels))
    for before_idx, after_idx in change_pair:
        if before_idx < after_idx:
            C = swap_rows(C, before_idx, after_idx)
    return C

def drawCorrelationMatrix(C, fig_name):
    plt.figure()
    plt.imshow(C, interpolation='nearest')
    plt.savefig(fig_name)
    plt.gca().clear()


C_permu = formCorrelationMatrix(C, labels)
print(C_permu)
