from _init_path import *
from data_helper import *
from svmutil import *
from svm import *
import subprocess

# Ref:
# https://www.csie.ntu.edu.tw/~piaip/svm/svm_tutorial.html

def doSVM(list_x, list_y, best_c, best_g):
    print('-' * 50, doSVM.__name__, '-'*50)
    prob = svm_problem(list_y, list_x)
    param = svm_parameter('-s 0 -t 2 -c ' + str(best_c) + ' -g ' + str(best_g) + ' -v 5')
    model = svm_train(prob, param)

def doGrid(file_name='train.txt'):
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
    shuffle(x_train, y_train)

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