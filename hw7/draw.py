from matplotlib import pyplot as plt
import numpy as np
import pylab

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
