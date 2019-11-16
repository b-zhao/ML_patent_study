import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from matplotlib import pyplot

from loader import load_data_train, load_data_test


xtrain, ytrain, xtest, ytest = load_data_train()
xevaluation, yevaluation = load_data_test()




column_name = np.load('data_preparation/col_names.npy')

test_report = []
def reg(max_depth, alpha, n_estimators, learning_rate):
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = learning_rate,
                    max_depth = max_depth, alpha = alpha, n_estimators = n_estimators)
    xg_reg.fit(xtrain,ytrain)

    preds = xg_reg.predict(xtest)
    rmse = np.sqrt(mean_squared_error(ytest, preds))
    print("RMSE: %f" % (rmse))

    abs_error = abs(preds - ytest)
    print("ABS: %f" % (np.average(abs_error)))

    record = {'depth': max_depth, 'alpha': alpha, 'n_estimator': n_estimators,
              'learning_rate':learning_rate, 'RMSE': rmse, 'abs': np.average(abs_error)}
    test_report.append(record)


m_d = [20, 25, 30,35,40]
alpha = [1]
n_e = [20]
lr = [0.1]

for a in m_d:
    for b in alpha:
        for c in n_e:
            for d in lr:
                reg(a, b, c, d)

print(test_report)
