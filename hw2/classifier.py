import numpy as np
import math

class NaiveBayseClassifier(object):
    def __init__(self):
        self.num_class = -1
        self.prior = []
        self.miu = []
        self.sigma = []

    def fit(self, arr_x, arr_y):
        # Calculate the prior
        self.num_class = int(np.max(arr_y) + 1)
        for i in range(self.num_class):
            _num = 0
            for j in range(len(arr_y)):
                if arr_y[j] == i:
                    _num += 1
            self.prior.append(float(_num) / len(arr_y))
        
        # Calculate the likelihood
        for i in range(self.num_class):
            features = arr_x[arr_y == float(i)]
            self.miu.append(np.mean(features, axis=0))
            self.sigma.append(np.std(features, axis=0))

    def predict(self, arr_x):
        batch = len(arr_x)
        res = np.zeros([batch, self.num_class])

        for i in range(1):
            scalar = 1 / self.sigma[i].T*(math.sqrt(2 * math.pi))
            exp_upper = np.square(np.subtract(arr_x, self.miu[i], dtype=float))
            exp_lower = 2 * np.square(self.sigma[i])
            exp_num = -exp_upper / exp_lower
            # print(np.shape(exp_upper), np.shape(exp_lower))
            prob = scalar * np.exp(exp_num)
            # print(np.shape(prob))
            prob_log = np.sum(np.log(prob), axis=1)
            print(np.shape(prob_log))