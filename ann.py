from sympy import var
from sympy import sympify
import random
import numpy as np
import operator
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
import tkinter as tk


def guess_1D_function(func: str, n: int, a: float, b: float, noise_domain: float, test_percentage: float,
                      plot_name: str, sizes):
    def train(n: int):
        X_train = []
        Y_train = []
        for i in range(0, n):
            x = random.uniform(a, b)
            y = sympify(func).subs(var('x'), x)
            if not [x] in X_train:
                X_train.append([x])
                Y_train.append(y)
            else:
                i -= 1
        return X_train, Y_train

    X_train, Y_train = train(n)
    Y_train = list(map(operator.add, Y_train, np.random.normal(0, noise_domain, len(Y_train))))
    X_test, Y_test = train(int(n * (test_percentage / 100)))
    mlp = MLPRegressor(hidden_layer_sizes=sizes, activation='relu', solver='lbfgs')
    mlp.fit(X_train, Y_train)
    predict = mlp.predict(X_test)
    plt.scatter(X_train, Y_train, c='r')
    plt.scatter(X_test, Y_test, c='b')
    plt.scatter(X_test, predict, c='g')
    plt.savefig(plot_name + '.jpg')
    return 'File ' + plot_name + '.jpg created successfully'


def guess_MD_function(func: str, n: int, a: float, b: float, noise_domain: float, test_percentage: float, dim: int,
                      sizes: tuple, ):
    def train(n: int):
        X_train = []
        Y_train = []
        for i in range(0, n):
            x = [random.uniform(a, b) for i in range(0, dim)]
            y = sympify(func).subs(var('x_0'), x[0])
            for i in range(1, dim):
                y = sympify(func).subs(var('x_' + str(i)), x[i])
            if not [x] in X_train:
                X_train.append([x])
                Y_train.append(y)
            else:
                i -= 1
        return X_train, Y_train

    X_train, Y_train = train(n)
    Y_train = list(map(operator.add, Y_train, np.random.normal(0, noise_domain, len(Y_train))))
    X_test, Y_test = train(int(n * (test_percentage / 100)))
    mlp = MLPRegressor(hidden_layer_sizes=sizes, activation='relu', solver='lbfgs')
    mlp.fit(X_train, Y_train)
    predict = mlp.predict(X_test)
    print(abs(100 * np.average(list(map(operator.sub, predict, Y_test))) / np.average(Y_test)))


def wrath(path: str, point_factor: int, train_counts: int, test_percentage: float, plot_name: str, sizes: tuple):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    X = []
    Y = []
    image = imread(path)
    height = len(image)
    width = len(image[0])
    for i in range(0, height):
        for j in range(0, width):
            if np.average(image[i][j]) < point_factor:
                X.append([j])
                Y.append(height - i)
    train_exs = random.sample(range(0, len(X)), train_counts)
    test_exs = random.sample(range(0, len(X)), int(train_counts * (test_percentage / 100)))
    for index in train_exs:
        X_train.append(X[index])
        Y_train.append(Y[index])
    for index in test_exs:
        X_test.append(X[index])
        Y_test.append(Y[index])
    mlp = MLPRegressor(hidden_layer_sizes=sizes, activation='relu', solver='lbfgs')
    mlp.fit(X_train, Y_train)
    predict = mlp.predict(X_test)
    plt.scatter(X_train, Y_train, c='r')
    plt.scatter(X_test, Y_test, c='b')
    plt.scatter(X_test, predict, c='g')
    plt.savefig("wrath_" + plot_name + ".jpg")
    print("Done")
    return 'File wrath_' + plot_name + '.jpg created successfully'


def check_digit(sizes: tuple):
    X_train = []
    X_test = []
    y_test = []
    y_train = []
    train_indices = [1, 1195, 2200, 2931, 3589, 4241, 4797, 5461, 6106, 6648, 7291]
    test_indices = [1, 360, 624, 822, 988, 1188, 1348, 1518, 1665, 1831, 2008]
    for index in range(0, len(train_indices) - 1):
        for k in range(train_indices[index], train_indices[index + 1]):
            image = imread('usps/train/' + str(index) + '_' + str(k) + '.jpg')
            train_item = []
            for i in range(0, 16):
                for j in range(0, 16):
                    train_item.append(image[i][j])
            train_item = preprocessing.scale(train_item)
            X_train.append(train_item)
            y_train.append(index)
    mlp = MLPClassifier(hidden_layer_sizes=(sizes), activation='relu', solver='adam')
    mlp.fit(np.array(X_train), y_train)
    for index in range(0, len(test_indices) - 1):
        for k in range(test_indices[index], test_indices[index + 1]):
            image = imread('usps/test/' + str(index) + '_' + str(k) + '.jpg')
            test_item = []
            for i in range(0, 16):
                for j in range(0, 16):
                    test_item.append(image[i][j])
            test_item = preprocessing.scale(test_item)
            X_test.append(test_item)
            y_test.append(index)
    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)

    print(confusion_matrix(y_train, predict_train))
    print(classification_report(y_train, predict_train))

    print(confusion_matrix(y_test, predict_test))
    print(classification_report(y_test, predict_test))


def choose_op(command: str, inputs: list):
    ans = 'done'
    if command == '1D':
        ans = guess_1D_function(inputs[0], int(inputs[1]), float(inputs[2]), float(inputs[3]), float(inputs[4]),
                                float(inputs[5]), inputs[6], tuple(int(inputs[i]) for i in range(7, len(inputs))))
    if command == 'MD':
        guess_MD_function(inputs[0], int(inputs[1]), float(inputs[2]), float(inputs[3]), float(inputs[4]),
                          float(inputs[5]), int(inputs[6]), tuple(int(inputs[i]) for i in range(7, len(inputs))))
    if command == 'wrath':
        ans = wrath(inputs[0], int(inputs[1]), int(inputs[2]), float(inputs[3]), inputs[4],
                    tuple(int(inputs[i]) for i in range(5, len(inputs))))
    if command == 'CD':
        check_digit(tuple(int(inputs[i]) for i in range(0, len(inputs))))
    return ans


# command = input()
# inputs = list(input().strip().split())
# choose_op(command, inputs)
command = '1D'
inputs = ['sin(x)', '100', '-3.14', '3.14', '0', '20', 'plt_1', '16']


def call_func():
    ans = choose_op(str(command_entry.get()), str(input_entry.get()).strip().split())
    label = tk.Label(window, text=ans)
    label.pack()


window = tk.Tk()
help_label = tk.Label(window, text="Command : format")
help_label.pack()
help_lable_1D = tk.Label(window, text='1D : f(x) n a b noise_domain test_percentage plot_name sizes')
help_lable_1D.pack()
help_lable_MD = tk.Label(window, text='2D : f(x_0,x_1,...) n a b noise_domain test_percentage dimension_of_f sizes')
help_lable_MD.pack()
help_lable_wrath = tk.Label(window, text='wrath : image_path black_determination_factor training_counts test_percentage plot_name sizes')
help_lable_wrath.pack()
help_lable_CD = tk.Label(window, text='CD : sizes')
help_lable_CD.pack()
command_label = tk.Label(window, text='Command one of 1D, MD, wrath, CD')
command_label.pack()
command_entry = tk.Entry(window, width=300)
command_entry.pack()
command_label = tk.Label(window, text='Inputs seperated with spaces')
command_label.pack()
input_entry = tk.Entry(window, width=300)
input_entry.pack()
button = tk.Button(window, text="Solve", width=25, height=5, command=call_func)
button.pack()
window.mainloop()
