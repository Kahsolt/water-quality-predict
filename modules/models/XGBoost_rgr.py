#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/06 

import joblib
from pathlib import Path

import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from modules.util import get_metrics
from modules.preprocess import *
from modules.typing import *

TASK_TYPE: TaskType = TaskType(Path(__file__).stem.split('_')[-1])


def init(params:Params, logger:Logger=None) -> GridSearchCV:
  model: XGBRegressor = getattr(xgboost, params['model'])(
    objective=params['objective'],
  )
  model_gs = GridSearchCV(model, **params['gs_params'])
  return model_gs


def train(model:GridSearchCV, dataset:Datasets, params:Params, logger:Logger=None):
  (X_train, y_train), _ = dataset
  assert X_train.shape[-1] == 1
  X_train = X_train.squeeze(axis=-1)  # [N, I]
  y_train = y_train.squeeze(axis=-1)  # [N, O]
  model.fit(X_train, y_train)
  try: logger.info('best: %f using %s' % (model.best_score_, model.best_params_))
  except: pass


def eval(model:GridSearchCV, dataset:Datasets, params:Params, logger:Logger=None) -> EvalMetrics:
  _, (X_test, y_test) = dataset
  assert X_test.shape[-1] == 1
  X_test = X_test.squeeze(axis=-1)  # [N, I]
  y_test = y_test.squeeze(axis=-1)  # [N, O]
  pred = model.predict(X_test)      # [N, I] => [N, O]
  return get_metrics(y_test, pred, task=TASK_TYPE, logger=logger)


def infer(model:GridSearchCV, x:Frame, logger:Logger=None) -> Frame:
  assert x.shape[-1] == 1
  x = x.T                   # [I, D=1] => [N=1, I]
  y = model.predict(x)      # [N=1, O]
  y = y.T                   # [O, D=1]
  return y


def save(model:GridSearchCV, log_dp:Path, logger:Logger=None):
  fp = log_dp / 'model.pkl'
  if logger: logger.info(f'save model to {fp}')
  joblib.dump(model, fp)


def load(model:GridSearchCV, log_dp:Path, logger:Logger=None) -> GridSearchCV:
  fp = log_dp / 'model.pkl'
  if logger: logger.info(f'load model from {fp}')
  return joblib.load(fp)
