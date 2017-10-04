from data_helper import load_data
import numpy as np
import math

class NaiveBayseClassifier(object):
    def __init__(self, num_bin=32):
        self.num_class = -1
        self.num_bin = num_bin
        self.num_feature = -1
        self.prior = None
        self.distribution = None
        self.epsilon = 1e-9

    def fit(self, arr_x, arr_y):
        # Create the objects
        self.num_class = int(np.max(arr_y)) + 1
        batch, self.num_feature = np.shape(arr_x)
        self.prior = np.empty([self.num_class])
        self.distribution = np.full([self.num_class, self.num_feature, self.num_bin], self.epsilon)

        # Calculate the prior
        for i in range(self.num_class):
            self.prior[i] = float(len(arr_y[arr_y == float(i)])) / batch

        # Calculate the likelihood
        for i in range(self.num_class):
            _arr_x = arr_x[arr_y == float(i)].T
            for j in range(self.num_feature):
                x_single_feature = _arr_x[j]
                for k in range(self.num_bin):
                    idx = np.logical_and(x_single_feature >= k*32, x_single_feature < k*32+32)
                    self.distribution[i][j][k] += np.shape(_arr_x[j][idx])[0] / np.shape(x_single_feature)[0]

    def predict(self, arr_x):
        batch = len(arr_x)
        res = np.empty([batch, self.num_class])

        for i in range(batch):
            print(i)
            for j in range(self.num_class):
                _prob = 0
                for k in range(self.num_feature):
                    _prob += math.log(self.distribution[j][k][int(arr_x[i][k]/32)])
                res[i][j] = _prob + math.log(self.prior[j])
        print(np.argmax(res, axis=1))
        return res

"""
(train_x, train_y), (test_x, test_y) = load_data()
clf = NaiveBayseClassifier()
clf.fit(train_x, train_y)
clf.predict(train_x[300:310, :])
print(np.asarray(train_y[300:310], dtype=int))
"""