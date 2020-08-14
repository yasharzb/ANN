import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

a_heavy_side = -4
b_heavy_side = 4
a_poly_2 = -5
b_poly_2 = 5
a_homo = 0.01
b_homo = 4
a_sin = - math.pi
b_sin = math.pi
gen_data_counts = 100
test_percentage = 0.5


def rand_heavy_side(n: int):
    X = []
    Y = []
    for i in range(0, n):
        x = random.uniform(a_heavy_side, b_heavy_side)
        y = math.floor(1 + np.sign(x) * 0.5)
        if not [x] in X:
            X.append([x])
            Y.append(y)
        else:
            i -= 1
    return X, Y


# y = x
def rand_homo(n: int):
    X = []
    Y = []
    for i in range(0, n):
        x = random.uniform(a_homo, b_homo)
        y = 1 / x
        if not [x] in X:
            X.append([x])
            Y.append(y)
        else:
            i -= 1
    return X, Y


# x = 0
def rand_poly_2(n: int):
    X = []
    Y = []
    for i in range(0, n):
        x = random.uniform(a_poly_2, b_poly_2)
        y = x ** 2 + 2
        if not [x] in X:
            X.append([x])
            Y.append(y)
        else:
            i -= 1
    return X, Y


# y = 0
def rand_sin(n: int):
    X = []
    Y = []
    for i in range(0, n):
        x = random.uniform(a_sin, b_sin)
        y = math.sin(2 * x)
        if not [x] in X:
            X.append([x])
            Y.append(y)
        else:
            i -= 1
    return X, Y


def test_heavy_side():
    X_train, Y_train = rand_heavy_side(gen_data_counts)
    X_test, Y_test = rand_heavy_side(int(gen_data_counts * test_percentage))
    mlp = MLPRegressor(hidden_layer_sizes=(4,), activation='relu', solver='lbfgs')
    mlp.fit(X_train, Y_train)
    predict = mlp.predict(X_test)
    draw_scatter(X_train, Y_train, X_test, Y_test, predict)


def test_homo():
    X_train, Y_train = rand_homo(gen_data_counts)
    X_test, Y_test = rand_homo(int(gen_data_counts * test_percentage))
    mlp = MLPRegressor(hidden_layer_sizes=(4, 4), activation='relu', solver='lbfgs')
    mlp.fit(X_train, Y_train)
    predict = mlp.predict(X_test)
    draw_scatter(X_train, Y_train, X_test, Y_test, predict)


def test_poly_2():
    X_train, Y_train = rand_poly_2(gen_data_counts)
    X_test, Y_test = rand_poly_2(int(gen_data_counts * test_percentage))
    mlp = MLPRegressor(hidden_layer_sizes=(8,), activation='relu', solver='lbfgs')
    mlp.fit(X_train, Y_train)
    predict = mlp.predict(X_test)
    draw_scatter(X_train, Y_train, X_test, Y_test, predict)


def test_sin():
    X_train, Y_train = rand_sin(gen_data_counts)
    X_test, Y_test = rand_sin(int(gen_data_counts * test_percentage))
    mlp = MLPRegressor(hidden_layer_sizes=(16, 16), activation='relu', solver='lbfgs')
    mlp.fit(X_train, Y_train)
    predict = mlp.predict(X_test)
    draw_scatter(X_train, Y_train, X_test, Y_test, predict)


def draw_scatter(X_train, Y_train, X_test, Y_test, predict):
    plt.scatter(X_train, Y_train, c='r')
    plt.scatter(X_test, Y_test, c='b')
    plt.scatter(X_test, predict, c='g')
    plt.show()


test_heavy_side()
test_homo()
test_poly_2()
test_sin()
