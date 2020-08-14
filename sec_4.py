import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

point_factor = 132
train_counts = 1000
test_percentage = 0.2
X_train = []
X_test = []
Y_train = []
Y_test = []


def train_wrath():
    X = []
    Y = []
    image = imread('wrath.jpg')
    height = len(image)
    width = len(image[0])
    for i in range(0, height):
        for j in range(0, width):
            if np.average(image[i][j]) < point_factor:
                X.append([j])
                Y.append(height - i)
    train_exs = random.sample(range(0, len(X)), train_counts)
    test_exs = random.sample(range(0, len(X)), int(train_counts * test_percentage))
    for index in train_exs:
        X_train.append(X[index])
        Y_train.append(Y[index])
    for index in test_exs:
        X_test.append(X[index])
        Y_test.append(Y[index])


def fit_wrath(X_train, X_test, Y_train, Y_test, size):
    mlp = MLPRegressor(hidden_layer_sizes=size, activation='relu', solver='lbfgs')
    mlp.fit(X_train, Y_train)
    predict = mlp.predict(X_test)
    draw_scatter(X_train, Y_train, X_test, Y_test, predict, size)


def draw_scatter(X_train, Y_train, X_test, Y_test, predict, size):
    plt.scatter(X_train, Y_train, c='r')
    plt.scatter(X_test, Y_test, c='b')
    plt.scatter(X_test, predict, c='g')
    plt.savefig("wrath_" + str(size) + ".jpg")
    plt.show()


# X, Y = train_wrath()
train_wrath()
for i in range(0, 10):
    fit_wrath(X_train, X_test, Y_train, Y_test, (2 ** i))
    fit_wrath(X_train, X_test, Y_train, Y_test, (2 ** i, 2 ** i))
