#%%

import pandas as pd
import sys
import matplotlib
import matplotlib.pyplot as plt
import os

from sklearn import svm
import numpy as np
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


#%%

# load data
data = pd.read_stata('Database_Patents_MLClass_Sample_Sep2019.dta')

# save to a csv file
if not os.path.exists('Database_Patents_MLClass_Sample_Sep2019.csv'):
    data.to_csv('Database_Patents_MLClass_Sample_Sep2019.csv')
print('data loaded ')
print(data.shape)

#%%

# Data pre-processing and extra examples

# data = data.dropna(subset=['APPDATE', 'CATEGORY', 'TEAMSIZE', 'GDATE'], how='any') # delete rows that have empty cells

# data = data.rename(columns = {'APPDATE': 'APPDATE'})
# data['APPDATE'] = data['APPDATE'].str.replace('-', '.')

# data.loc[data.classifier == 'Y', 'classifier'] = 1
# data.loc[data.classifier == 'N', 'classifier'] = 0


#%%

# Data normalization example

# scaler = MinMaxScaler()
# columns_to_norm = ['APPMONTH', 'APPYEAR']
# vals = data[columns_to_norm].values
# scaled_vals = scaler.fit_transform(vals)
# data_temp = pd.DataFrame(scaled_vals, columns = columns_to_norm, index = data.index)
# data[columns_to_norm] = data_temp

# data.head()

#%%

# Calculate days between application and grant
data['APPDATE'] = pd.to_datetime(data['APPDATE'])
data['GDATE'] = pd.to_datetime(data['GDATE'])
data['GTIME'] = (data['GDATE'] - data['APPDATE']).astype('timedelta64[D]')
data['NBCITE'] = pd.to_numeric(data['NBCITE'])

data = data.drop(['ABANDON_DATE', 'ABN_YEAR', 'APPNUM', 'APPTYPE',
       'ASGCITY', 'ASGCOUNTRY', 'ASGNUM', 'ASGSEQ', 'ASGSTATE',
       'ASSIGNEE', 'CLAIMS', 'CLASS', 'DISPOSAL_TYPE',
       'EXAMINER_ART_UNIT', 'EXAMINER_ID', 'FILING_DATE', 'FILING_YEAR',
       'FIRSTNAME', 'INVCITY', 'INVNUM', 'INVSEQ',
        'KIND', 'LASTNAME', 'NFCITE', 'NUMAPP', 'NUMPAT',
       'PATENT', 'RESIDENCE', 'SUBCLASS', 'ABN', 'DES', 'UTL', 'US', 'CAT',
       'PRIMINV', 'TEAM', 'NUMPRIM', 'LONE', 'NUMLONE', 'NUMCOINV', 'TOTAPP',
       'USINV', 'INVCOUNT'], 1)


# print(data.columns)


#%%

# One-hot encoding for non-numerical columns
data.CATEGORY = data.CATEGORY.astype(str)
CATEGORY_ohe = OneHotEncoder()
X = CATEGORY_ohe.fit_transform(data.CATEGORY.values.reshape(-1,1)).toarray()
dfOneHot = pd.DataFrame(X, columns = ["CATEGORY_"+str(int(i)) for i in range(X.shape[1])])
data = pd.concat([data, dfOneHot], axis=1)
data =data[~data.isin([np.inf, -np.inf]).any(1)]

data.INVCOUNTRY = data.INVCOUNTRY.astype(str)
INVCOUNTRY_ohe = OneHotEncoder()
X = INVCOUNTRY_ohe.fit_transform(data.INVCOUNTRY.values.reshape(-1,1)).toarray()
dfOneHot = pd.DataFrame(X, columns = ["INVCOUNTRY_"+str(int(i)) for i in range(X.shape[1])])
data = pd.concat([data, dfOneHot], axis=1)
data =data[~data.isin([np.inf, -np.inf]).any(1)]

data.INVSTATE = data.INVSTATE.astype(str)
INVSTATE_ohe = OneHotEncoder()
X = INVSTATE_ohe.fit_transform(data.INVSTATE.values.reshape(-1,1)).toarray()
dfOneHot = pd.DataFrame(X, columns = ["INVSTATE_"+str(int(i)) for i in range(X.shape[1])])
data = pd.concat([data, dfOneHot], axis=1)
data =data[~data.isin([np.inf, -np.inf]).any(1)]

data = data.dropna(subset=['APPDATE', 'CATEGORY', 'TEAMSIZE', 'GDATE', 'NBCITE'], how='any') # delete rows that have empty cells
print(data.head())

#%%

# Prepare training data
y = data['GTIME']
X = data.drop(['GDATE', 'INVCOUNTRY', 'INVSTATE', 'GTIME', 'CATEGORY', 'APPDATE', 'NBCITE'], 1)

data =data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=12345)


# For some reason there are some invalid values in the data, so we get rid of them.
idx = y_train.isin([np.nan, np.inf, -np.inf])
idx = np.nonzero(idx - 1)
X_train = X_train.iloc[idx]
y_train = y_train.iloc[idx]

idx = y_test.isin([np.nan, np.inf, -np.inf])
idx = np.nonzero(idx - 1)
X_test = X_test.iloc[idx]
y_test = y_test.iloc[idx]





#%%

# Define models

# clf = linear_model.LinearRegression()
# clf = linear_model.Ridge()
clf = svm.SVR()

# Some classification methods:
# clf = svm.SVC()
# clf = LogisticRegression()
# clf = RandomForestClassifier()
# clf = KNeighborsClassifier(n_neighbors=11)

# Dimension reduction
# dr_model = make_pipeline(StandardScaler(), PCA())
# dr_model = make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(n_components=10))
# dr_model =make_pipeline(StandardScaler(),NeighborhoodComponentsAnalysis())

#%%

# Train
# dr_model.fit(X_train, y_train)
# clf.fit(dr_model.transform(X_train), y_train)
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
print(y_test.to_numpy().flatten().astype(int)[0:20])
print("Predicted grant time (days):")
print(y_pred[0:20].astype(int))

p = y_test.to_numpy().flatten().astype(int)
q = y_pred.astype(int)


averageError = np.average(abs(p - q))
print('average error:', averageError)
#%%


