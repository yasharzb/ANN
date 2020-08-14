import numpy as np
from matplotlib.image import imread
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from skimage.util import *

X_train = []
X_test = []
y_test = []
test_ones = []
y_train = []
train_counts = 7291
train_indices = [1, 1195, 2200, 2931, 3589, 4241, 4797, 5461, 6106, 6648, 7291]
test_counts = 2007
test_indices = [1, 360, 624, 822, 988, 1188, 1348, 1518, 1665, 1831, 2008]


def train():
    for index in range(0, len(train_indices) - 1):
        for k in range(train_indices[index], train_indices[index + 1]):
            image = imread('usps/train/' + str(index) + '_' + str(k) + '.jpg')
            train_item = []
            for i in range(0, 16):
                for j in range(0, 16):
                    train_item.append(image[i][j])
            train_item = preprocessing.scale(train_item)
            train_item = random_noise(image=train_item, mode='s&p')
            X_train.append(train_item)
            y_train.append(index)


def test():
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


def init_mlp():
    mlp = MLPClassifier(hidden_layer_sizes=(64), activation='relu', solver='adam')
    mlp.fit(np.array(X_train), y_train)
    return mlp


def predict(mlp: MLPClassifier):
    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)

    print(confusion_matrix(y_train, predict_train))
    print(classification_report(y_train, predict_train))

    print(confusion_matrix(y_test, predict_test))
    print(classification_report(y_test, predict_test))


train()
test()
mlp = init_mlp()
predict(mlp)
