import math
import random
import numpy as np
import operator
from sklearn.neural_network import MLPRegressor

a_heavy_side = -4
b_heavy_side = 4
a_poly_2 = -5
b_poly_2 = 5
a_homo = 0.01
b_homo = 4
a_sin = - math.pi
b_sin = math.pi
gen_data_counts = 1000
test_percentage = 0.5


# y = e ^ (x1 + x2)
def rand_heavy_side(n: int):
    X = []
    Y = []
    for i in range(0, n):
        x = [random.uniform(a_heavy_side, b_heavy_side), random.uniform(a_heavy_side, b_heavy_side)]
        y = math.e ** (x[0] + x[1])
        if not [x] in X:
            X.append(x)
            Y.append(y)
        else:
            i -= 1
    return X, Y


# y = x1 ^ 2 + x2 ^ 2 + x1 * x2
def rand_homo(n: int):
    X = []
    Y = []
    for i in range(0, n):
        x = [random.uniform(a_homo, b_homo), random.uniform(a_homo, b_homo)]
        y = x[0] ** 2 + x[1] ** 2 + x[0] * x[1]
        if not [x] in X:
            X.append(x)
            Y.append(y)
        else:
            i -= 1
    return X, Y


def test_heavy_side():
    X_train, Y_train = rand_heavy_side(gen_data_counts)
    X_test, Y_test = rand_heavy_side(int(gen_data_counts * test_percentage))
    mlp = MLPRegressor(hidden_layer_sizes=(8, 8), activation='relu', solver='lbfgs')
    mlp.fit(X_train, Y_train)
    predict = mlp.predict(X_test)
    get_error(predict, Y_test)


def test_homo():
    X_train, Y_train = rand_homo(gen_data_counts)
    X_test, Y_test = rand_homo(int(gen_data_counts * test_percentage))
    mlp = MLPRegressor(hidden_layer_sizes=(4, 4), activation='relu', solver='lbfgs')
    mlp.fit(X_train, Y_train)
    predict = mlp.predict(X_test)
    get_error(predict, Y_test)


def get_error(predict, Y_test):
    print(abs(100 * np.average(list(map(operator.sub, predict, Y_test))) / np.average(Y_test)))


test_heavy_side()
test_homo()
