# from tensorflow.examples.tutorials.mnist import input_data
from data_helper import load_data
import numpy as np

def load():
    mnist = input_data.read_data_sets('data/', one_hot=False)
    return mnist.train.images, mnist.train.labels
    # return np.ones([55000, 784]), np.ones([55000, ])

def EM(imgs, tags):
    """
        imgs: [784, 55000]
        tags: [1  , 55000]
    """
    # Random guess
    print(np.shape(imgs))
    print(np.shape(tags))

    dimension, img_num = np.shape(imgs)
    lambda_arr = np.ones([10, 1]) / 10
    prob_arr = np.ones([10, dimension]) / len(imgs)
    responsibility = np.empty([10, img_num])

    print(prob_arr[:, 400])

    for epoch in range(10):
        # E step
        for i in range(10):
            for j in range(img_num):
                sub_prob = 1
                _ = prob_arr[i, :] ** imgs[:, j] * (1 - prob_arr[i, :]) ** (1 - imgs[:, j])
                for k in range(dimension):
                    sub_prob *= (lambda_arr[i] * _[k])
                    if k % 150 == 0:
                        sub_prob *= 1e+150
                responsibility[i][j] = lambda_arr[i][0] * sub_prob
        for i in range(img_num):
            responsibility[:, i] /= np.sum(responsibility[:, i])

        print(np.unique(responsibility))

        # M step
        for i in range(10):
            lambda_arr[i][0] = np.sum(responsibility[i, :]) / img_num
        for i in range(dimension):
            for j in range(10):
                print(np.sum(responsibility[j, :] * imgs[i, :])Q)
                prob_arr[j][i] = np.sum(responsibility[j, :] * imgs[i, :]) / np.sum(responsibility[j, :])
        print(prob_arr[:, 400])

    # Evaluation
    print(tags)
    print(np.argmax(responsibility, axis=0))


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = load_data()

    # Binaryize
    train_x = train_x[:500]
    train_y = train_y[:500]
    for i in range(np.shape(train_x)[0]):
        for j in range(np.shape(train_x)[1]):
            if train_x[i][j] > 127.5:
                train_x[i][j] = 1.0
            else:
                train_x[i][j] = 0.0
        
    # EM
    EM(train_x.T / 255, train_y)