from data_helper import load_data
import numpy as np
import math

"""
    This program contain the scratch implementation of Naive Bayes classifier
    You should notice the mode integer in below:
        0 - discrete
        1 - contineous
    
    Author  : SunnerLi
    Date    : 2017/10/04
"""

class NaiveBayseClassifier(object):
    def __init__(self, mode=1, num_bin=32):
        """
            Constructor of Naive Bayes classifier
            The function would check if the number of bins is valid
            If it's invalid, error will raise and exit the program

            Arg:    mode    - The mode integer
                    num_bin - The number of bins
        """
        self.num_class = -1
        self.num_bin = num_bin
        self.num_feature = -1
        self.prior = None
        self.miu = None
        self.sigma = None
        self.distribution = None
        self.mode = mode
        self.epsilon = 1e-9

        # Check the number of bins is valid
        if 256 % self.num_bin != 0:
            print('error bin number!')
            exit()
        else:
            self.dis_width = 256 / self.num_bin

    def fit(self, arr_x, arr_y):
        """
            Fit the model by the given data

            Arg:    arr_x   - The array of training data
                    arr_y   - The array of training tag
        """
        # Create the objects
        self.num_class = int(np.max(arr_y)) + 1
        batch, self.num_feature = np.shape(arr_x)
        self.prior = np.empty([self.num_class])
        if self.mode == 1:
            self.miu = np.empty([self.num_class, self.num_feature])
            self.sigma = np.full([self.num_class, self.num_feature], self.epsilon)
        else:
            self.distribution = np.full([self.num_class, self.num_feature, self.num_bin], self.epsilon)
        
        # Calculate the prior
        for i in range(self.num_class):
            self.prior[i] = float(len(arr_y[arr_y == float(i)])) / batch

        # Calculate the likelihood
        for i in range(self.num_class):
            _arr_x = arr_x[arr_y == float(i)].T
            for j in range(self.num_feature):
                if self.mode == 1:
                    self.miu[i][j] = np.mean(_arr_x[j], axis=0)
                    self.sigma[i][j] += np.var(_arr_x[j], axis=0)
                else:
                    x_single_feature = _arr_x[j]
                    for k in range(self.num_bin):
                        idx = np.logical_and(x_single_feature >= k*self.dis_width, x_single_feature < k*self.dis_width + self.dis_width)
                        self.distribution[i][j][k] += np.shape(_arr_x[j][idx])[0] / np.shape(x_single_feature)[0]

    def predict(self, arr_x):
        """
            Predict the result for the given data array
            It will return the array which shape is [batch_num, class_num]
            You should do np.argmax() by yourself after receiving the result to see the integer prediction

            Arg:    arr_x   - The feature array want to predict
            Ret:    The predict array result
        """
        batch = len(arr_x)
        res = np.empty([batch, self.num_class])
        for i in range(batch):
            for j in range(self.num_class):
                if self.mode == 1:
                    prob_log = -0.5 * (arr_x[i] - self.miu[j]) ** 2 / self.sigma[j]
                    prob_log -= 0.5 * np.log(2 * np.pi * self.sigma[j])
                    res[i][j] = np.sum(prob_log) + math.log(self.prior[j])
                else:
                    _prob = 0
                    for k in range(self.num_feature):
                        _prob += math.log(self.distribution[j][k][int(arr_x[i][k] / self.dis_width)])
                    res[i][j] = _prob + math.log(self.prior[j])
        return res