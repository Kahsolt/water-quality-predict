#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/06 

import joblib
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pylab as plt

from modules.util import save_metrics, save_figure
from modules.preprocess import *
from modules.typing import *

TASK_TYPE: ModelTask = Path(__file__).stem.split('_')[-1]


def init(params:ModelParams) -> GridSearchCV:
  model = XGBRegressor() 
  model_gs = GridSearchCV(model, **params)
  return model_gs


def train(model:GridSearchCV, dataset:Datasets):
  (X_train, y_train), _ = dataset
  X_train = X_train.squeeze(axis=-1)  # [N, I]
  y_train = y_train.squeeze(axis=-1)  # [N, O]
  model.fit(X_train, y_train)
  print('best: %f using %s' % (model.best_score_, model.best_params_))


def eval(model:GridSearchCV, dataset:Datasets, log_dp:Path):
  _, (X_test, y_test) = dataset
  X_test = X_test.squeeze(axis=-1)  # [N, I]
  y_test = y_test.squeeze(axis=-1)  # [N, O]
  pred = model.predict(X_test)      # [N, I] => [N, O]
  save_metrics(y_test, pred, fp=log_dp/'metrics.txt', task=TASK_TYPE)


def infer(model:GridSearchCV, seq:Seq, stats:Stats):
  breakpoint()
  pred = pred.tolist()
  x, y_pred = X_test[-1], pred[-1]
  for _ in range(8 * 2):
    x = np.concatenate([x[1:], np.expand_dims(y_pred, 0)])
    y_pred = model.predict(np.expand_dims(x, 0))[0]
    pred.append(y_pred)

  plt.clf()
  plt.plot(truth, color='blue', label='truth')
  plt.plot(pred, color='red', label='predict')
  plt.legend(loc='best')
  save_figure(log_dp / 'predict.png', title='seen history + near future')


def save(model:GridSearchCV, log_dp:Path):
  joblib.dump(model, log_dp / 'model.pkl')


def load(model:GridSearchCV, log_dp:Path) -> GridSearchCV:
  return joblib.load(log_dp / 'model.pkl')
