from matplotlib import pyplot as plt
from _init_path import *
from data_helper import *
from svmutil import *
from svm import *
from pca import *
import subprocess
import os

# Ref:
# https://www.csie.ntu.edu.tw/~piaip/svm/svm_tutorial.html

color_code = ['b', 'g', 'r', 'c', 'm', 'y']

def doSVM(list_x, list_y, best_c, best_g, model_name = 'svm.model'):
    print('-' * 50, doSVM.__name__, '-'*50)
    if not os.path.exists(model_name):
        prob = svm_problem(list_y, list_x)
        param = svm_parameter('-s 0 -t 2 -c ' + str(best_c) + ' -g ' + str(best_g))
        model = svm_train(prob, param)
        print(type(model))
        svm_save_model(model_name, model) 

def GetSupportVector(list_y, model_name = 'svm.model'):
    m = svm_load_model(model_name)
    sv = m.get_SV()
    x_train, y_train, x_test, y_test = load_data()
    is_sv_list = np.zeros([len(list_y)])
    for i in range(m.get_nr_sv()):
        row = np.zeros([784])
        for key, feature in sv[i].items():
            row[key] = round(feature, 5)

        # Find the most approximate row
        min_value = 1000
        min_index = -1
        for j in range(len(x_train)):
            diff = np.sum(np.square(x_train[j] - row))
            # print(diff)
            if diff < min_value:
                min_value = diff
                min_index = j
                # print(min_index, min_value)
        is_sv_list[min_index] = 1
    return is_sv_list

def doPredict(input_file_name = 'border.txt', model_name = 'svm.model', output_file_name = 'border.out'):
    print('-' * 50, doPredict.__name__, '-'*50)
    args = [libsvm_path + 'svm-predict', input_file_name, model_name, output_file_name]
    cmd = " ".join(args)
    print(cmd)
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()

def doGrid(file_name = 'train.txt'):
    print('-' * 50, doGrid.__name__, '-'*50)
    args = ['grid.py', '-log2c', '-10,10,2', '-log2g', '-10,10,2', '-v', '10', '-svmtrain', './libsvm/svm-train', file_name]
    cmd = " ".join(args)
    print("python " + cmd)
    # _ = subprocess.call("python " + cmd, shell=True)
    proc = subprocess.Popen("python " + cmd, shell=True)
    proc.wait()

def doSubset(input_file_name = 'train.txt', output_file_name = 'subset.out', num = 50):
    print('-' * 50, doSubset.__name__, '-'*50)
    args = [tool_path + 'subset.py', '-s', '1', input_file_name, str(num), output_file_name]
    cmd = " ".join(args)
    _ = subprocess.call("python " + cmd, shell=True)

def getCoordinateList(xmin, xmax, ymin, ymax, period = 1):
    print('-' * 50, getCoordinateList.__name__, '-'*50)
    # Validate the parameters
    x_num = int((xmax - xmin) // period)
    y_num = int((ymax - ymin) // period)
    if xmin > xmax:
        xmax, xmin = xmin, xmax
    if ymin > ymax:
        ymax, ymin = ymin, ymax

    # Get coordinate grids
    x = np.linspace(xmin, xmax, x_num)
    y = np.linspace(ymin, ymax, y_num)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_list = np.ndarray([x_num * y_num, 2])
    print(np.shape(grid_x))

    # Fill into list
    counter = 0
    for x, y in zip(np.reshape(grid_x, [-1]), np.reshape(grid_y, [-1])):
        grid_list[counter][0] = x
        grid_list[counter][1] = y
        counter += 1
    return grid_list

def randomUpsampling(arr, extend_dim=784):
    """
        arr shape: (N, 2)
    """
    print('-' * 50, randomUpsampling.__name__, '-'*50)
    res = np.random.random([len(arr), extend_dim])
    cross = np.max(arr) - np.min(arr)
    shift = np.min(arr)
    res = res * cross + shift
    for i in range(len(arr)):
        res[i][0], res[i][1] = arr[i][0], arr[i][1]
    return res

if __name__ == '__main__':
    # Get data & shuffle
    x_train, y_train, x_test, y_test = load_data()
    # shuffle(x_train, y_train)

    # --------------------------------------------------------------------------------------
    # Q1: Do the SVM
    # --------------------------------------------------------------------------------------
    # write into .txt file and select subset
    list_x, list_y = to_svm_format(x_train, y_train)
    to_svm_file(list_x, list_y, file_name='train.txt')
    doSubset(input_file_name = 'train.txt', output_file_name = 'subset.out', num = 50)

    # Find best hyper-parameter
    doGrid(file_name='subset.out')   

    # Get parameter and train 
    best_c = float(open('param.dat', 'r').readlines()[0])
    best_g = float(open('param.dat', 'r').readlines()[1])
    doSVM(list_x, list_y, best_c, best_g)
    is_sv_list = GetSupportVector(list_y)

    # --------------------------------------------------------------------------------------
    # Q2 & Q3: PCA projection and plot support vector
    # --------------------------------------------------------------------------------------
    z_train = pca(x_train)[:, :2]
    print('-' * 50, ' Draw ', '-'*50)
    for i in range(5):
        num = 0
        for j, tag in enumerate(y_train):
            if tag == i:
                num += 1
        z_batch = []
        sv_batch = []
        for j in range(len(y_train)):
            if y_train[j] == i:
                _row = []
                for k in range(2):
                    _row.append(z_train[j, k])
                if is_sv_list[j] == 0:
                    z_batch.append(_row)
                else:
                    sv_batch.append(_row)

        z_batch = np.asarray(z_batch)
        sv_batch = np.asarray(sv_batch)
        plt.plot(z_batch[:, 0], z_batch[:, 1], 'o', color=color_code[i], label='digit_'+str(i))
        plt.plot(sv_batch[:, 0], sv_batch[:, 1], '^', color=color_code[i], label='digit_'+str(i))
    
    # --------------------------------------------------------------------------------------
    # Q4: Plot boundary
    # --------------------------------------------------------------------------------------
    # Generate grid file
    list_x, list_y = to_svm_format(z_train, y_train)
    to_svm_file(list_x, list_y, file_name='train.txt')
    doSVM(list_x, list_y, best_c, best_g, model_name = '2d.model')

    # Predict the label of generated points
    grid_list = getCoordinateList(-10, 5, -5, 5, period = 0.05)
    list_y = [[1]] * len(grid_list)
    list_x, list_y = to_svm_format(grid_list, list_y)
    to_svm_file(list_x, list_y, file_name='border.txt')
    doPredict(input_file_name = 'border.txt', model_name = '2d.model', output_file_name = 'border.out')

    # Plot the possible boundary
    border_predic_logits = np.asarray([ int(x[:-1]) for x in open('border.out', 'r').readlines()])
    border_list = []
    for i in range(1, len(border_predic_logits)):
        if border_predic_logits[i] != border_predic_logits[i-1]:
            if grid_list[i][0] != -10:
                border_list.append(grid_list[i])
    border_list = np.asarray(border_list)
    print('border list: ', np.shape(border_list))
    plt.plot(border_list[:, 0], border_list[:, 1], 'o', color=color_code[5], label='border')

    # Show
    plt.legend()
    plt.show()