#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/15 

from logging import Logger

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from modules.typing import TaskType, EvalMetrics


def get_metrics(truth, pred, task:TaskType, logger:Logger=None) -> EvalMetrics:
  if task == TaskType.CLF:
    acc = accuracy_score(truth, pred)
    prec, recall, f1, supp = precision_recall_fscore_support(truth, pred, average='weighted')
    if logger:
      logger.info(f'acc:    {acc:.3%}')
      logger.info(f'prec:   {prec:.3%}')
      logger.info(f'recall: {recall:.3%}')
      logger.info(f'f1:     {f1:.3%}')
    return acc, prec, recall, f1

  elif task == TaskType.RGR:
    mae = mean_absolute_error(truth, pred)
    mse = mean_squared_error (truth, pred)
    r2  = r2_score           (truth, pred)
    if logger:
      logger.info(f'mae: {mae:.3f}')
      logger.info(f'mse: {mse:.3f}')
      logger.info(f'r2:  {r2:.3f}')
    return mae, mse, r2
