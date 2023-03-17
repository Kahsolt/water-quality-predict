#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/22

from pathlib import Path

from pmdarima import AutoARIMA

from modules.util import get_metrics, get_logger
from modules.preprocess import *
from modules.typing import *
from modules.models.XGBoost_rgr import save, load

TASK_TYPE: TaskType = Path(__file__).stem.split('_')[-1]


def init(params:Params) -> AutoARIMA:
  return AutoARIMA(**params)


def train(model:AutoARIMA, seq:Seq, params:Params):
  seq = seq.squeeze()
  model.fit(seq)
  get_logger().info(model.summary())


def eval(model:AutoARIMA, seq:Seq, params:Params) -> EvalMetrics:
  seq = seq.squeeze()
  seqlen = len(seq)
  start = seqlen // 4
  pred = model.predict_in_sample(start=start, end=seqlen-1)
  return get_metrics(seq[:-start], pred, task=TASK_TYPE)


def infer(model:AutoARIMA, x:int) -> Frame:
  pred = model.predict_in_sample(start=x, end=x+1)
  return np.expand_dims(pred, axis=-1)
