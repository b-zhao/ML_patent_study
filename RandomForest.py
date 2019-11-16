import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_stata('../Database_Patents_MLClass_Sample_Sep2019.dta')

# save to a csv file
if not os.path.exists('Database_Patents_MLClass_Sample_Sep2019.csv'):
    data.to_csv('Database_Patents_MLClass_Sample_Sep2019.csv')
print('data loaded ')
print(data.shape)

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

data=data[data['GTIME']>0]

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

rfr = RandomForestRegressor(n_estimators=250, max_depth=30, max_features=0.2, min_samples_leaf=5, oob_score=True)
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)

print("Root Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred)**(0.5))
print('Variance score: %.2f' % r2_score(y_test, y_pred))


print("Actual grant time (days):")
print(y_test.to_numpy().flatten().astype(int)[0:20])
print("Predicted grant time (days):")
print(y_pred[0:20].astype(int))