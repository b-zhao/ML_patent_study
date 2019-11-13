import pandas as pd
import sys
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from scipy import stats

print('Load X.npy')
data = np.load('../data_preparation/X.npy')
print('Load Finish')
print('Data shape:', data.shape)
#100 means 1/100 
subset = 1
if subset != 1:
    data = data[np.random.choice(data.shape[0], data.shape[0] // subset, replace=False), :]
    print('Subset Data shape:', data.shape)
    

#10 means 1/10
trainSubset = 10
trainCount  = 10
mask = None
for i in range(trainCount):
    print('select a random subset of entries as training data for the', i + 1, 'time')   
    testData = data[np.random.choice(data.shape[0], trainSubset, replace=False), :]
    print('Start training isolation forest')
    clf = IsolationForest(behaviour='new', contamination='auto')
    clf.fit(testData)
    pred = clf.predict(data)
    if mask is None:
        mask = pred 
    else:
        mask = mask + pred 
threshold = 0.5
bo = mask >= threshold * 1 + (1 - threshold) * -1
inlier = data[bo]
print(inlier)
print(inlier.shape)
np.save('../data_preparation/X_inlier.npy', inlier )

