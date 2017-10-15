from classifier import NaiveBayseClassifier
from data_helper import load_data
from utils import printResult
import numpy as np
import argparse

"""
    This program demonstrate the process of naive bayes classifier
    The usage of my scratch is just like sklearn
    Moreover, the style to load the data is just like keras

    Author  : SunnerLi
    Date    : 2017/10/04
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=1, dest='mode', help='discrete mode is 0, contineous mode is 1')
    args = parser.parse_args()

    print('load data...')
    (train_x, train_y), (test_x, test_y) = load_data()
    clf = NaiveBayseClassifier(mode=args.mode)
    print('train...')
    clf.fit(train_x, train_y)
    print('predict...')
    res = clf.predict(test_x)
    printResult(res, test_y)