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

color_code = ['b', 'g', 'r', 'c', 'm']

def doSVM(list_x, list_y, best_c, best_g, model_name = 'svm.model'):
    print('-' * 50, doSVM.__name__, '-'*50)
    if not os.path.exists(model_name):
        prob = svm_problem(list_y, list_x)
        param = svm_parameter('-s 0 -t 2 -c ' + str(best_c) + ' -g ' + str(best_g))
        model = svm_train(prob, param)
        print(type(model))
        svm_save_model(model_name, model)
    else:
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

"""
def doSVM(best_c, best_g, file_name = 'train.txt', model_name = 'svm.model'):
    print('-' * 50, doSVM.__name__, '-'*50)
    args = [libsvm_path + 'svm-train', '-s', '0', '-t', '2', '-c', str(best_c), '-g', str(best_g), file_name, model_name]
    cmd = " ".join(args)
    print(cmd)
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()
"""

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
    #doGrid(file_name='subset.out')   

    # Get parameter and train 
    best_c = float(open('param.dat', 'r').readlines()[0])
    best_g = float(open('param.dat', 'r').readlines()[1])
    is_sv_list = doSVM(list_x, list_y, best_c, best_g)
    # --------------------------------------------------------------------------------------
    # Q2: PCA projection
    # --------------------------------------------------------------------------------------
    z_train = pca(x_train)[:, :2]
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
    plt.legend()
    plt.show()