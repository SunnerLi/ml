from _init_path import *
from data_helper import *
from svmutil import *
from svm import *
import subprocess

# Ref:
# https://www.csie.ntu.edu.tw/~piaip/svm/svm_tutorial.html

def doSVM(list_x, list_y):
    prob = svm_problem(list_y, list_x)
    param = svm_parameter('-s 0 -t 2 -g 0.0001 -v 5')
    model = svm_train(prob, param)

def doGrid(file_name='train.txt'):
    # args = [grid_path + 'grid.py']
    args = [grid_path + 'grid.py', '-log2c', '-1,1,2', '-log2g', '-1,1,2', '-v', '100', '-svmtrain', './libsvm/svm-train', file_name]
    cmd = " ".join(args)
    print("python " + cmd)
    _, __ = subprocess.call("python " + cmd, shell=True)

if __name__ == '__main__':
    # x_train, y_train, x_test, y_test = load_data()

    # Train 
    # list_x, list_y = to_svm_format(x_train, y_train)
    # doSVM(list_x, list_y)
    # to_svm_file(list_x, list_y)
    doGrid()