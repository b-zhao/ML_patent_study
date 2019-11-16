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

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors.nca import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error, r2_score

print('Load X.npy')
X = np.load('../data_preparation/X.npy')
print('Load Finish')
print('X shape:', X.shape)
print('Load Y.npy')
y = np.load('../data_preparation/Y.npy')
print('Load Finish')
print(X)

subset = 1
if subset != 1:
    choice = np.random.choice(X.shape[0], X.shape[0] // subset, replace=False)
    X = X[choice, :]
    y = y[choice]
    print('Subset X Data shape:', X.shape)
    print('Subset Y Data shape:', y.shape)

trainSubset = 50
trainCount  = 10

XwithoutDummy = X[:, 0:13]
#ASGNUM' 'CLAIMS' 'INVSEQ' 'NBCITE' 'NUMAPP' 'NUMPAT' 'PRIMINV' 'TEAM'
#'NUMPRIM' 'NUMLONE' 'NUMCOINV' 'TOTAPP' 'TEAMSIZE' 
mask = None
for i in range(trainCount):
    print('select a random subset of entries as training data for the', i + 1, 'time')   
    testX = XwithoutDummy[np.random.choice(X.shape[0], trainSubset, replace=False), :]
    print('Start training isolation forest')
    clf = IsolationForest(behaviour='new', contamination='auto')
    clf.fit(testX)
    pred = clf.predict(XwithoutDummy)
    if mask is None:
        mask = pred 
    else:
        mask = mask + pred 
threshold = 0.6
bo = mask >= threshold * 1 + (1 - threshold) * -1
inlier_X = X[bo]
inlier_y = y[bo]
outlier_X = X[bo == False]
outlier_y = y[bo == False]
print('new shape', inlier_y.size)

clf = svm.SVR()
X_train, X_test, y_train, y_test = train_test_split(inlier_X, inlier_y, test_size = 0.2, random_state=12345)

print('start training..')
clf.fit(X_train, y_train)

#%%

# Predict
# y_pred = clf.predict(dr_model.transform(X_test))
print('start predicting')
y_pred = clf.predict(X_test)

#%%

# Evaluate
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))


print("Actual grant time (days):")
print(y_test[0:20])
print("Predicted grant time (days):")
print(y_pred[0:20].astype(int))

p = y_test.astype(int)
q = y_pred.astype(int)


averageError = np.average(abs(p - q))
print('average error:', averageError)


