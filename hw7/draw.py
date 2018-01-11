from matplotlib import pyplot as plt
from collections import Counter
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

def score(C, labels_num_list):
    '''
    Function to assign a score to an ordered covariance matrix.
    High correlations within a cluster improve the score.
    High correlations between clusters decease the score.
    '''
    n_variables = np.shape(C)[0]
    score = 0
    for digit in range(len(labels_num_list)):
        inside_cluster = np.arange(labels_num_list[digit]) + digit * labels_num_list[digit]
        outside_cluster = np.setdiff1d(range(n_variables), inside_cluster)

        # Belonging to the same cluster
        score += np.sum(C[inside_cluster, :][:, inside_cluster])

        # Belonging to different clusters
        score -= np.sum(C[inside_cluster, :][:, outside_cluster])
        score -= np.sum(C[outside_cluster, :][:, inside_cluster])

    return score

def formCorrelationMatrix(C, labels):
    """
        Ref: https://stats.stackexchange.com/questions/138325/clustering-a-correlation-matrix
    """
    # Make the neighbor as the same label
    change_pair = zip(range(len(labels)), np.argsort(labels))
    for before_idx, after_idx in change_pair:
        if before_idx < after_idx:
            C = swap_rows(C, before_idx, after_idx)

    # Get the number of each digit
    labels_num_list = []
    counter = Counter()
    for label in labels:
        if label not in counter:
            counter[int(label)] = 1
        else:
            counter[int(label)] += 1
    for i in range(int(np.max(labels))):
        labels_num_list.append(counter[i])

    # Try to switch the position to get higher score
    current_C = C
    n_variables = np.shape(current_C)[0]
    initial_ordering = np.arange(n_variables)
    current_ordering = initial_ordering
    initial_score = score(C, labels_num_list)
    current_score = initial_score
    if np.shape(current_C)[0] < 500:
        for i in range(1000):
            print(i)
            # Find the best row swap to make
            best_C = current_C
            best_ordering = current_ordering
            best_score = current_score
            for row1 in range(n_variables):
                for row2 in range(n_variables):
                    if row1 == row2:
                        continue
                    if np.argsort(labels)[row1] != np.argsort(labels)[row2]:
                        continue
                    option_ordering = best_ordering.copy()
                    option_ordering[row1] = best_ordering[row2]
                    option_ordering[row2] = best_ordering[row1]
                    option_C = swap_rows(best_C, row1, row2)
                    option_score = score(option_C, labels_num_list)
    
                    if option_score > best_score:
                        best_C = option_C
                        best_ordering = option_ordering
                        best_score = option_score
    
            for row1 in range(n_variables - 1, -1, -1):
                for row2 in range(n_variables - 1, -1, -1):
                    if row1 == row2:
                        continue
                    if np.argsort(labels)[row1] != np.argsort(labels)[row2]:
                        continue
                    option_ordering = best_ordering.copy()
                    option_ordering[row1] = best_ordering[row2]
                    option_ordering[row2] = best_ordering[row1]
                    option_C = swap_rows(best_C, row1, row2)
                    option_score = score(option_C, labels_num_list)
    
                    if option_score > best_score:
                        best_C = option_C
                        best_ordering = option_ordering
                        best_score = option_score
    
            if best_score > current_score:
                # Perform the best row swap
                current_C = best_C
                current_ordering = best_ordering
                current_score = best_score
            else:
                # No row swap found that improves the solution, we're done
                break
        
    return C

def drawCorrelationMatrix(C, fig_name):
    plt.figure()
    plt.imshow(C, interpolation='nearest')
    plt.savefig(fig_name)
    plt.gca().clear()


C_permu = formCorrelationMatrix(C, labels)
print(C_permu)
