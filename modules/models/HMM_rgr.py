#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/14 

from pathlib import Path

from hmmlearn import hmm
from hmmlearn.hmm import BaseHMM
from sklearn.model_selection import GridSearchCV

from modules.util import get_metrics, get_logger
from modules.preprocess import *
from modules.typing import *
from modules.models.XGBoost_rgr import train, eval, save, load     # just proxy by

TASK_TYPE: ModelTask = Path(__file__).stem.split('_')[-1]


def init(config:Config) -> GridSearchCV:
  model: BaseHMM = getattr(hmm, [config['model']])()
  model_gs = GridSearchCV(model, **config['gs_params'])
  return model_gs


def infer(model:GridSearchCV, x:Frame) -> Frame:
  #prev, post = X[:cp, :], X[cp:, :]   # [T1, N], [T2, N]
  #n_samples = len(post)
  #_, prev_state = model.decode(prev)   # [T1]
  #pred, pred_state = model.sample(n_samples, currstate=prev_state[-1])   # [T2, order], [T2]
  breakpoint()
