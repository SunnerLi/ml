import numpy as np

def print2File1D(arr):
    with open('result.dat', 'w') as f:
        for i in range(int(len(arr))):
            f.write(str(arr[i]) + '\n')

def print2File2D(arr):
    with open('result.dat', 'w') as f:
        for i in range(int(len(arr))):
            for j in range(int(len(arr[i]))):
                f.write(str(arr[i][j]) + ' ')
            f.write('\n')            

def printResult(predict, tag):
    batch, num_class = np.shape(predict)
    for i in range(batch):
        print('row', i, ':', end=' ')
        for j in range(num_class):
            if predict[i][j] < -1e+10:
                prob = '-infinity'
            else:
                prob = round(predict[i][j], 5)
            print(prob, end='\t')
        print('\tpredict: ', np.argmax(predict, axis=-1)[i], ' tag: ', np.asarray(tag, dtype=int)[i])
    correct = np.sum(np.argmax(predict, axis=-1) == np.asarray(tag, dtype=int))
    print('totally correction: ', correct, ' / ', batch)