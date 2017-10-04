from data_helper import load_data
import numpy as np
import math

class NaiveBayseClassifier(object):
    def __init__(self):
        self.num_class = -1
        self.num_feature = -1
        self.prior = None
        self.miu = None
        self.sigma = None
        self.epsilon = 1e-9

    def fit(self, arr_x, arr_y):
        # Create the objects
        self.num_class = int(np.max(arr_y)) + 1
        batch, self.num_feature = np.shape(arr_x)
        self.prior = np.empty([self.num_class])
        self.miu = np.empty([self.num_class, self.num_feature])
        self.sigma = np.empty([self.num_class, self.num_feature])        
        
        # Calculate the prior
        for i in range(self.num_class):
            self.prior[i] = float(len(arr_y[arr_y == float(i)])) / batch

        # Calculate the likelihood
        for i in range(self.num_class):
            _arr_x = arr_x[arr_y == float(i)].T
            for j in range(self.num_feature):
                self.miu[i][j] = np.mean(_arr_x[j], axis=0)
                self.sigma[i][j] = np.var(_arr_x[j], axis=0)
        self.sigma += self.epsilon

    def predict(self, arr_x):
        batch = len(arr_x)
        res = np.empty([batch, self.num_class])

        for i in range(batch):
            for j in range(self.num_class):
                prob_log = -0.5 * (arr_x[i] - self.miu[j]) ** 2 / self.sigma[j]
                prob_log -= 0.5 * np.log(2 * np.pi * self.sigma[j])
                res[i][j] = np.sum(prob_log) + math.log(self.prior[j])
        return res

"""
(train_x, train_y), (test_x, test_y) = load_data()
clf = NaiveBayseClassifier()
clf.fit(train_x, train_y)
print(np.shape(train_x))
clf.predict(train_x[:10, :])
print(np.asarray(train_y[:10], dtype=int))
"""