import numpy as np
import cv2
import sys
import os

def join(a, b):
    return a + '/' + b

prefix = './image/'
cluster_method = ['k_means', 'kernel_k_means', 'spectral_clustering']
test_folder = ['test1', 'test2']
experiment_diff = ['2', '3', '4', 'plus_plus']

cv2.imshow('k_means', np.ones([100, 100]))
cv2.waitKey(0)

for _cluster in cluster_method:
    for sub_fold in test_folder:
        for exp in experiment_diff:
            subfolder = join(join(join(prefix, _cluster), sub_fold), exp)
            img_list = sorted(os.listdir(subfolder))
            # img_list = os.listdir(subfolder)
            for image_name in img_list:
                img = cv2.imread(join(subfolder, image_name))
                cv2.imshow(_cluster, img)
                cv2.waitKey(250)
            cv2.waitKey(0)