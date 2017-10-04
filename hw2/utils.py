import pandas as pd
import numpy as np         

def printResult(predict, tag):
    """
        Print the result row by row

        Arg:    predict - The predict array result
                tag     - The original tag array
    """
    batch, num_class = np.shape(predict)
    predict_index_list = np.argmax(predict, axis=-1)
    tag_index_list = np.asarray(tag, dtype=int)

    # Show on terminal
    for i in range(batch):
        print('row', i, ':', end=' ')
        for j in range(num_class):
            if predict[i][j] < -1e+10:
                prob = '-infinity'
            else:
                prob = round(predict[i][j], 5)
            print(prob, end='\t')
        print('\tpredict: ', predict_index_list[i], ' tag: ', tag_index_list[i])
    correct = np.sum(predict_index_list == tag_index_list)
    print('totally correction: ', correct, ' / ', batch)
    print('error rate: ', float(batch - correct) / batch)

    # Write into csv
    _array = np.concatenate((np.reshape(range(batch), [batch, 1]), predict), axis=1)
    _array = np.concatenate((_array, np.expand_dims(predict_index_list, axis=-1)), axis=1)
    _array = np.concatenate((_array, np.expand_dims(tag_index_list, axis=-1)), axis=1)
    columns = ['row', 'prob of 0', 'prob of 1', 'prob of 2', 'prob of 3', 'prob of 4', 'prob of 5', 'prob of 6', 'prob of 7', 'prob of 8', 'prob of 9', 'predict', 'tag']
    pd.DataFrame(data=_array, columns=columns).to_csv('output.csv')