#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/22

from pathlib import Path

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from modules.util import get_metrics
from modules.preprocess import *
from modules.typing import *
from modules.models.SVM_rgr import train, save, load      # just proxy by
from modules.models.XGBoost_clf import infer              # just proxy by

TASK_TYPE: TaskType = Path(__file__).stem.split('_')[-1]


def init(config:Config) -> GridSearchCV:
  model: SVC = getattr(svm, config['model'])()
  model_gs = GridSearchCV(model, **config['gs_params'])
  return model_gs


def eval(model:GridSearchCV, dataset:Datasets, config:Config) -> EvalMetrics:
  _, (X_test, y_test) = dataset
  assert X_test.shape[-1] == 1
  X_test = X_test.squeeze(axis=-1)  # [N, I]
  y_test = y_test.squeeze()         # [N]
  pred = model.predict(X_test)      # [N, I] => [N]
  return get_metrics(y_test, pred, task=TASK_TYPE)
