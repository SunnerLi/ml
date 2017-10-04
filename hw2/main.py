from classifier2 import NaiveBayseClassifier
from data_helper import load_data
from utils import print2File2D, print2File1D, printResult
import numpy as np

(train_x, train_y), (test_x, test_y) = load_data()
clf = NaiveBayseClassifier()
clf.fit(train_x, train_y)
res = clf.predict(test_x)
printResult(res, test_y)