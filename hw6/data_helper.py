import pandas as pd
import numpy as np

# data file name
x_train_name = 'X_train.csv'
y_train_name = 'T_train.csv'
x_test_name = 'X_test.csv'
y_test_name = 'T_test.csv'

def load_data():
    """
        Return the data as usual array type
    """
    x_train = pd.read_csv(x_train_name).values
    y_train = pd.read_csv(y_train_name).values
    x_test = pd.read_csv(x_test_name).values
    y_test = pd.read_csv(y_test_name).values
    return x_train.astype(np.float32), y_train - 1, x_test.astype(np.float32), y_test - 1

def shuffle(arr_x, arr_y):
    """
        Shuffle x and y with the same order
    """
    idx = np.random.shuffle(np.asarray(range(len(arr_x))))
    return arr_x[idx], arr_y[idx]

def to_svm_format(arr_x, arr_y):
    """
        Change two array to two list which follow LIBSVM format
    """
    list_x = []
    list_y = []
    for i, row in enumerate(arr_x):
        _dict = {}
        for j, feature in enumerate(row):
            _dict[j + 1] = feature
        list_x.append(_dict)
    for i, label in enumerate(arr_y):
        list_y.append(label[0])
    return list_x, list_y

def to_svm_file(list_x, list_y, file_name='train.txt'):
    contain = []
    for i, feature_dict in enumerate(list_x):
        _string = str(list_y[i]) + ' '
        for key, feature in feature_dict.items():
            _string = _string + str(key) + ':' + str(feature) + ' '
        _string += '\n'
        contain.append(_string)
    with open(file_name, 'w') as f:
        f.writelines(contain)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    print(np.shape(x_train))
    print(x_train[0])
    print(y_train[0])