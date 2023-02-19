#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/15 
# Update Time: 2022/10/05

import os
from re import compile as Regex
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import hparam as hp
from data import save_data


# '2022/6/1 01:00:00'
DATETIME_REGEX = Regex(r'(\d*)/(\d*)/(\d*) (\d*):(\d*):(\d*)')


def _parse_datetime_str(s:bytes, sep='-') -> str:
  m = DATETIME_REGEX.match(s.decode())
  year, month, day, hour, minute, second = [int(x) for x in m.groups()]
  dt = datetime(year, month, day, hour, minute, second)
  w = dt.weekday()
  h = dt.hour
  return f'{w}-{h}'     # '(weekday}-{hour}'


def preprocess(sep='-'):
  # load raw data
  CONVERTER = {
    0: lambda x: _parse_datetime_str(x, sep),
    1: lambda x: float(x),
  }
  def read_csv(fn):
    with open(os.path.join(hp.DATA_PATH, fn)) as fh:
      data = np.loadtxt(fh, delimiter=",", converters=CONVERTER, skiprows=1, dtype=object)
      ts, data = [x.squeeze() for x in np.split(data, 2, axis=-1)]
      return ts, data.astype(np.float32)
  
  ts1, COD = read_csv('COD.csv')
  ts2, PH  = read_csv('PH.csv')
  ts3, NH  = read_csv('氨氮.csv')

  # assure time consistency
  assert (ts1 == ts2).all() and (ts1 == ts3).all()
  weekday = [int(x.split(sep)[0]) for x in ts1]
  hour    = [int(x.split(sep)[1]) for x in ts1]

  # TODO: 在这里去除异常值、处理缺失值
  # COD = ... 
  # PH  = ...
  # NH  = ...

  # NOTE: 收集所有数值数据，以下做统一处理
  features = {
    'COD': COD, 
    'PH': PH,
    'NH': NH,
  }

  # minmax normalize
  def minmax_norm(x:np.array):
    x_min, x_max = x.min(), x.max()
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm, x_min, x_max

  tmp = {k: minmax_norm(v) for k, v in features.items()}
  features_n   = {k: v[0] for k, v in tmp.items()}
  features_min = {k: v[1] for k, v in tmp.items()}
  features_max = {k: v[2] for k, v in tmp.items()}
  features_qt  = {k: np.round(v * hp.QT_N_BIN).astype(np.int32) for k, v in features_n.items()}

  # necessary statistics during predicting
  stats = { k: (features_min[k], features_max[k]) for k in features_n.keys() }

  # collect data matrix, [N, D=5]
  fvmat    = np.stack([weekday, hour, *features_n .values()]).T
  fvmat_qt = np.stack([weekday, hour, *features_qt.values()]).T

  # save preprocessed data
  save_data(fvmat,    stats, hp.DATA_FILE)
  save_data(fvmat_qt, stats, hp.QTDATA_FILE)


if __name__ == "__main__":
  preprocess()
