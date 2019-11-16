import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest as IF

class IsolationForest:
    def __init__(self, fileNameX = '../data_preparation/X.npy', fileNameY = '../data_preparation/Y.npy', trainSubset = 50, trainCount  = 10, threshold = 0.6, columnStart = 0, columnEnd = 13):
        print('Load X.npy')
        self.X = np.load(fileNameX)
        print('X shape:', self.X.shape)
        print('Load Y.npy')
        self.y = np.load(fileNameY)
        print('y shape:', self.y.shape)
        self.trainSubset = trainSubset
        self.trainCount  = trainCount
        self.threshold   = threshold
        self.columnStart = columnStart
        self.columnEnd   = columnEnd
        self.train()

    def train(self):
        XwithoutDummy = self.X[:, self.columnStart:self.columnEnd] 
        mask = None
        for i in range(self.trainCount):
            print('select a random subset of entries as training data for the', i + 1, 'time')   
            testX = XwithoutDummy[np.random.choice(self.X.shape[0], self.trainSubset, replace=False), :]
            clf = IF(behaviour='new', contamination='auto')
            clf.fit(testX)
            pred = clf.predict(XwithoutDummy)
            if mask is None:
                mask = pred 
            else:
                mask = mask + pred 
        threshold = self.threshold
        bo = mask >= threshold * 1 + (1 - threshold) * -1
        self.inlier_X  = self.X[bo]
        self.inlier_y  = self.y[bo]
        self.outlier_X = self.X[bo == False]
        self.outlier_y = self.y[bo == False]

    def getInlierX(self):
        return self.inlier_X

    def getOutlierX(self):
        return self.outlier_X

    def getInlierY(self):
        return self.inlier_y

    def getOutlierY(self):
        return self.outlier_y

    