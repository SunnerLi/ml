import numpy as np

def load():
    """
        pattern: [5, 5]
        tag    : [5, 1]
    """
    # pattern = np.asarray(
        # [
            # [1, 1, 0, 1, 0],
            # [0, 0, 1, 1, 0],
            # [1, 0, 0, 0, 0],
            # [1, 0, 0, 1, 1],
            # [0, 1, 1, 0, 0]
        # ]
    # )
    # tag = np.asarray(
        # [[0], [1], [2], [1], [0]]
    # )
    pattern = 
    return pattern, tag

if __name__ == '__main__':
    x, y = load()
    """
        prob:   [3, 1]
    """
    prob = np.asarray(
        [[0.2], [0.7]]
    )    
    wx_table = np.ndarray([len(x), len(prob)], dtype=np.float)
    lambda_arr = np.ones([len(x), len(prob)], dtype=np.float)

    for epoch in range(2):
        # E step
        for i in range(len(x)):
            for j in range(len(prob)):
                _prob = 1.
                for k in range(np.shape(x)[1]):
                    _prob *= (prob[j][0] ** x[i][k]) * ((1 - prob[j][0]) ** (1 - x[i][k]))
                wx_table[i][j] = _prob ** lambda_arr[i][j]
        for i in range(len(x)):
            for j in range(len(prob)):
                lambda_arr[i][j] = wx_table[i][j] / np.sum(wx_table[i])
        print(lambda_arr)
        print(np.argmax(lambda_arr, axis=-1))

        # M step
        for i in range(len(prob)):
            new_prob = np.ndarray([len(x), 2], dtype=np.float)
            for j in range(len(x)):
                new_prob[j][0] = np.sum(x[j]) * lambda_arr[j][i]
                new_prob[j][1] = (np.shape(x)[1] - np.sum(x[j])) * lambda_arr[j][i]
            prob[i][0] = np.sum(new_prob[:, 0]) / np.sum(new_prob)
        print('iter: ', epoch, '\tprob: ', np.reshape(prob, [-1]))
