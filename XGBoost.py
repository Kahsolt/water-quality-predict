#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/06 

import os
from argparse import ArgumentParser

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pylab as plt


''' Commandline '''
parser = ArgumentParser()
parser.add_argument('csv_fp', metavar='csv_fp', help='path to yoru .csv file')
parser.add_argument('--lag', default=360, type=int, help='window length')
parser.add_argument('--overwrite', action='store_true', help='force overwrite existing ckpt')
args = parser.parse_args()


''' Hparam '''
csv_fp = args.csv_fp
lag = args.lag
index = os.path.splitext(os.path.basename(csv_fp))[0]
save_fp = os.path.join('log', f'xgboost_{index}_lag={lag}.pkl')
os.makedirs('log', exist_ok=True)


''' RData'''
rdata = pd.read_csv(csv_fp, parse_dates=['monitorTime'], index_col='monitorTime', date_parser=pd.to_datetime)
ts = rdata['monitorAvgValue']


''' Data'''
# 知 lag 推 1
X, Y = [], []
for i in range(len(ts) - lag):
  X.append([ts[i + j] for j in range(lag)])
  Y.append(ts[i + lag])
X_train = np.asarray(X)
y_train = np.asarray(Y)


''' Train '''
if not os.path.exists(save_fp) or args.overwrite:
  ''' Model '''
  model = xgb.XGBRegressor() 
  model_gs = GridSearchCV(
    model,   
    {
      'max_depth': [3, 4, 5, 6, 7],
      'n_estimators': [20, 25, 30, 35, 50],
      'learning_rate': [0.1, 0.12, 0.125, 0.13, 0.15]
    },   
    cv=3,   
    verbose=2,
    n_jobs=-1,  
    scoring='neg_median_absolute_error',
    #scoring='neg_mean_absolute_error',
  )

  ''' Train '''
  model_gs.fit(X_train, y_train)
  print('Best: %f using %s' % (model_gs.best_score_, model_gs.best_params_))
  model = model_gs.best_estimator_

  ''' Save '''
  joblib.dump(model_gs, save_fp)
else:
  ''' Load '''
  model_gs = joblib.load(save_fp)
  print('Best: %f using %s' % (model_gs.best_score_, model_gs.best_params_))
  model = model_gs.best_estimator_


''' Infer '''
# predict seen history
pred = model.predict(X_train)
truth = y_train

mae = mean_absolute_error(truth, pred)
mse = mean_squared_error(truth, pred)
r2 = r2_score(truth, pred)
print('mae:', mae)
print('mse:', mse)
print('r2:', r2)

# predict near future
pred = pred.tolist()
x, y_pred = X_train[-1], pred[-1]
for _ in range(args.lag * 2):
  x = np.concatenate([x[1:], np.expand_dims(y_pred, 0)])
  y_pred = model.predict(np.expand_dims(x, 0))[0]
  pred.append(y_pred)

plt.plot(truth, color='blue', label='truth')
plt.plot(pred, color='red', label='predict')
plt.legend(loc='best')
plt.suptitle('seen history + near future')
plt.show()
