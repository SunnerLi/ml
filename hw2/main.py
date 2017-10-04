from classifier import NaiveBayseClassifier
from data_helper import load_data
from utils import print2File
import numpy as np

(train_x, train_y), (test_x, test_y) = load_data()
clf = NaiveBayseClassifier()
clf.fit(train_x, train_y)
clf.predict(train_x)