import pandas as pd
import sys
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np

# load data
print("Loading data...")
try:
    if not os.path.exists('../Database_Patents_MLClass_Sample_Sep2019.csv'):
        cfile = pd.read_stata('../Database_Patents_MLClass_Sample_Sep2019.dta')
        cfile.to_csv('../Database_Patents_MLClass_Sample_Sep2019.csv')
    data = pd.read_csv('../Database_Patents_MLClass_Sample_Sep2019.csv')
except:
    print("Loading failed. Make sure Database_Patents_MLClass_Sample_Sep2019.dta or *.csv is in the current directory")
    exit(1)

# try:
#     data = pd.read_csv('../data.csv')
# except:
#     print(
#         "Loading failed. Make sure Database_Patents_MLClass_Sample_Sep2019.dta or *.csv is in the current directory")
#     exit(1)



print("Total number of columns: " + str(len(data.columns)))
print(data.columns)

# drop useless columns
data = data.drop(['ABANDON_DATE', 'ABN_YEAR', 'APPMONTH',
       'APPNUM', 'APPTYPE', 'APPYEAR', 'ASGCITY', 'ASGCOUNTRY',
       'ASGSEQ', 'ASGSTATE',
       'DISPOSAL_TYPE', 'EXAMINER_ART_UNIT', 'EXAMINER_ID', 'FILING_DATE','NFCITE',
       'FILING_YEAR', 'INVCITY', 'INVSTATE', 'KIND', 'PATENT', 'RESIDENCE', 'ABN', 'DES', 'UTL', 'US',
       'CAT', 'LONE', 'USINV', 'INVCOUNT', 'SUBCLASS'], 1)

# data = data.drop(['AYM', 'ELAG_FLAG', 'GMONTH', 'GYEAR', 'GYM', 'PERDCAT', 'LNUMAPP', 'LNBCITE', 'LNFCITE', 'LCLAIMS', 'PRIM'], 1)
data = data.drop(['Unnamed: 0'], 1)

print("Number of columns we use: " + str(len(data.columns)))
print(data.columns)

# Delete tuples with empty values / nan (exception: CATEGORY and PRIMINV can be empty)
data = data.dropna(subset=['APPDATE', 'ASGNUM', 'ASSIGNEE', 'CLAIMS', 'CLASS',
       'FIRSTNAME', 'GDATE', 'INVCOUNTRY', 'INVNUM', 'INVSEQ', 'LASTNAME',
       'NBCITE' , 'NUMAPP', 'NUMPAT',
       'TEAM', 'NUMPRIM', 'NUMLONE', 'NUMCOINV', 'TOTAPP', 'TEAMSIZE'])

# Prepare Y (grant time = APPDATE - GDATE)
data['APPDATE'] = pd.to_datetime(data['APPDATE'])
data['GDATE'] = pd.to_datetime(data['GDATE'])
Y = np.array((data['GDATE'] - data['APPDATE']).astype('timedelta64[D]'))


# convert APPDATE and GDATE into int
data['APPDATE'] = data['APPDATE'].values.astype(int)
data['GDATE'] = data['GDATE'].values.astype(int)

data = data.drop(columns = ['APPDATE', 'INVNUM', 'GDATE'])

# normalization
numeric_data = data.dtypes[data.dtypes != 'object'].index
# for column in numeric_data:
#     data[column] = data[column].apply(
#         lambda x: (x - x.mean()) / x.st0d())

data[numeric_data] = data[numeric_data].apply(
    # lambda x: (x - x.mean()) / x.std())
    lambda x: (x - x.min()) / (x.max() - x.min()))
data[numeric_data] = data[numeric_data].fillna(0)




# Empty entries of CATEGORY are assigned "Others"
data.loc[data.CATEGORY.isnull(), 'CATEGORY'] = 'Others'
data.loc[data.CATEGORY == '', 'CATEGORY'] = 'Others'

# Empty entries of PRIMINV are assigned 1
data.loc[data.PRIMINV.isnull(), 'PRIMINV'] = '0'
data.loc[data.PRIMINV == '', 'PRIMINV'] = '0'

# Transforming PRIMINV to 0s and 1s
data.loc[data.PRIMINV == '0-Co-Inventor', 'PRIMINV'] = '0'
data.loc[data.PRIMINV != '0', 'PRIMINV'] = '1'
data['PRIMINV'] = pd.to_numeric(data['PRIMINV'])

# Transforming TEAM to 0s and 1s
data.loc[data.TEAM == '0-Lone Inventor', 'TEAM'] = '0'
data.loc[data.TEAM != '0', 'TEAM'] = '1'
data['TEAM'] = pd.to_numeric(data['TEAM'])


# Convert categorical features into numbers 
# 1. One-hot encoding
categories =['CATEGORY', 'INVCOUNTRY', 'CLASS']
def convert_onehot(data, feature):
    one_hot = pd.get_dummies(data[feature], prefix=feature)
    # concat to the data frame
    data = pd.concat([data, one_hot], axis=1)
    return data

for c in categories:
    print('Convert to one hot: Category ', c)
    data = convert_onehot(data, c)
    data = data.drop([c], 1)
    
# 2. factorization - convert strings into a unique number
# categories =['FIRSTNAME', 'LASTNAME', 'ASSIGNEE']
# for c in categories:
#     print('Factorization: Category ', c)
#     data[c], uniques = pd.factorize(data[c])

data = data.drop(columns=['FIRSTNAME', 'LASTNAME', 'ASSIGNEE'])

# saving pandas data frame to numpy array
col_names = np.array(data.columns)
for col in col_names:
    data[col] = pd.to_numeric(data[col], errors='coerce')
X = np.array(data)

def convert_Y(y):
    if y < 182:
        return 0
    elif y < 365:
        return 1
    elif y < 547:
        return 2
    elif y < 730:
        return 3
    elif y < 912:
        return 4
    elif y < 1095:
        return 5
    else:
        return 6

converter = np.vectorize(convert_Y)
Y_conv = converter(Y)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ohe.fit(np.array(range(7)).reshape((-1, 1)))

Y_conv = ohe.transform(Y_conv.reshape((-1, 1))).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)



print("Shape of X: " + str(np.shape(X)))
print("Shape of Y: " + str(np.shape(Y)))
print("Shape of Y_conv: " + str(np.shape(Y_conv)))

print("Whether there exists NAN in X: " + str(np.isnan(X).any()))

print("Saving to X.npy, Y.npy, and col_names.npy...")
np.save("X.npy", X)
np.save("col_names.npy", col_names)
np.save("Y.npy", Y)
np.save("Y_conv.npy", Y_conv)

np.save('X_train.npy', x_train)
np.save('X_test.npy', x_test)
np.save('Y_train.npy', y_train)
np.save('Y_test.npy', y_test)

print("Done! Use np.load(\"X.npy\") to load training data")










