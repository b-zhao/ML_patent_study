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

def load_data():

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

    return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()


def load_data_train():
    X = np.load('data_preparation/X_train.npy')
    Y = np.load('data_preparation/Y_train.npy').reshape((-1, 1))
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state=0)
    return x_train, y_train, x_test, y_test


def load_data_test():
    X = np.load('data_preparation/X_test.npy')
    Y = np.load('data_preparation/Y_test.npy').reshape((-1, 1))
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