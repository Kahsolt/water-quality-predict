#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/06 

import joblib
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from modules.util import save_metrics
from modules.preprocess import *
from modules.typing import *

TASK_TYPE: ModelTask = Path(__file__).stem.split('_')[-1]


def init(params:ModelParams) -> GridSearchCV:
  model = XGBRegressor() 
  model_gs = GridSearchCV(model, **params)
  return model_gs


def train(model:GridSearchCV, dataset:Datasets):
  (X_train, y_train), _ = dataset
  assert X_train.shape[-1] == 1
  X_train = X_train.squeeze(axis=-1)  # [N, I]
  y_train = y_train.squeeze(axis=-1)  # [N, O]
  model.fit(X_train, y_train)
  print('best: %f using %s' % (model.best_score_, model.best_params_))


def eval(model:GridSearchCV, dataset:Datasets, log_dp:Path):
  _, (X_test, y_test) = dataset
  assert X_test.shape[-1] == 1
  X_test = X_test.squeeze(axis=-1)  # [N, I]
  y_test = y_test.squeeze(axis=-1)  # [N, O]
  pred = model.predict(X_test)      # [N, I] => [N, O]
  save_metrics(y_test, pred, fp=log_dp/'metrics.txt', task=TASK_TYPE)


def infer(model:GridSearchCV, x:Frame) -> Frame:
  assert x.shape[-1] == 1
  x = x.T                   # [I, D=1] => [N=1, I]
  y = model.predict(x)      # [N=1, O]
  y = y.T                   # [O, D=1]
  return y


def save(model:GridSearchCV, log_dp:Path):
  joblib.dump(model, log_dp / 'model.pkl')


def load(model:GridSearchCV, log_dp:Path) -> GridSearchCV:
  return joblib.load(log_dp / 'model.pkl')
