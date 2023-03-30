#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/06 

from pathlib import Path

import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from modules.util import get_metrics, device
from modules.preprocess import *
from modules.typing import *
from modules.models.XGBoost_rgr import train, save, load     # just proxy by

TASK_TYPE: TaskType = TaskType(Path(__file__).stem.split('_')[-1])


def init(params:Params, logger:Logger=None) -> GridSearchCV:
  model_cls: type[XGBClassifier] = getattr(xgboost, params['model'])
  if 'gs_params' in params:
    if device == 'cuda':
      model = model_cls(objective=params['objective'], tree_method='gpu_hist', gpu_id=0)
    else:
      model = model_cls(objective=params['objective'])
    model = GridSearchCV(model, **params['gs_params'])
  else:
    if device == 'cuda':
      model = model_cls(objective=params['objective'], tree_method='gpu_hist', gpu_id=0, **params['param'])
    else:
      model = model_cls(objective=params['objective'], **params['param'])
  return model


def infer(model:GridSearchCV, x:Frame, logger:Logger=None) -> Frame:
  assert x.shape[-1] == 1
  x = x.T                   # [I, D=1] => [N=1, I]
  y = model.predict(x)      # [N=1]
  y = np.expand_dims(y, axis=-1)  # [N=1, D=1]
  return y


def infer_prob(model:GridSearchCV, x:Frame, logger:Logger=None) -> Frame:
  assert x.shape[-1] == 1
  x = x.T                         # [I, D=1] => [N=1, I]
  y = model.predict_proba(x)      # [N=1, NC]
  return y


def eval(model:GridSearchCV, dataset:Datasets, params:Params, logger:Logger=None) -> EvalMetrics:
  _, (X_test, y_test) = dataset
  assert X_test.shape[-1] == 1
  X_test = X_test.squeeze(axis=-1)  # [N, I]
  y_test = y_test.squeeze(axis=-1)  # [N, O]
  pred = model.predict(X_test)      # [N, I] => [N, O]
  return get_metrics(y_test, pred, task=TASK_TYPE, logger=logger)
