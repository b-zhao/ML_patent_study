import pandas as pd
import sys
import matplotlib
import matplotlib.pyplot as plt
import os
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from torch.utils.data import Dataset


def load_data_train():
    X = np.load('../data_preparation/X_train.npy')
    Y = np.load('../data_preparation/Y_train.npy').reshape((-1, 1))
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state=0)
    return x_train, y_train, x_test, y_test


def load_data_test():
    X = np.load('../data_preparation/X_test.npy')
    Y = np.load('../data_preparation/Y_test.npy').reshape((-1, 1))
    return X, Y




def load_data_with_convert_Y():
    X = np.load('data_preparation/X.npy')
    Y = np.load('data_preparation/Y_conv.npy')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.4, random_state=0)
    return x_train, y_train, x_test, y_test


class PatentDataset(Dataset):
    def __init__(self, data, label):
        self.x = data
        self.y = label
        self.data_len = len(data)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)