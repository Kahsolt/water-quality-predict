#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/15 

import os
from re import compile as Regex
from datetime import datetime
os.chdir('C:/Users/Administrator/Desktop/水质预测模型_凯发_全_168')
import numpy as np

from scipy.interpolate import interp1d

import hparam as hp
from data import save_data


# '2022/6/1 01:00:00'
DATETIME_REGEX = Regex(r'(\d*)-(\d*)-(\d*) (\d*):(\d*):(\d*)')


def _parse_datetime_str(s:bytes, sep='-') -> str:
  m = DATETIME_REGEX.match(s.decode())
  year, month, day, hour, minute, second = [int(x) for x in m.groups()]
  dt = datetime(year, month, day, hour, minute, second)
  w = dt.weekday()
  h = dt.hour
  return f'{w}-{h}'     # '(weekday}-{hour}'

#3σ原则
def yichang(npArray):
    '''异常值处理与补值'''
    if len(npArray.shape) == 1: 
        npArray = npArray.copy().reshape(-1,1)
    else:
        npArray = npArray.copy()
    fArray = npArray[:,-1]
    avg=npArray.mean()
    st=np.std(npArray)
    npArray[:,-1] = np.array([e if e > 0 and (avg-3*st <= e <= 3*st+avg) else np.nan for e in fArray])
    X=np.array([i for i in range(len(npArray))])
    X=X.reshape(len(X),1)
    #首尾用临近值补值
    ValidDataIndex = X[np.where(np.isnan(npArray) == 0)]
    if ValidDataIndex[-1] < len(npArray) - 1: 
        npArray[ValidDataIndex[-1] + 1:,0] = npArray[ValidDataIndex[-1],0]  
    # 如果第一个正常数据的序号不是0，则表明第一个正常数据前存在缺失数据
    if ValidDataIndex[0] >= 1:
        npArray[:ValidDataIndex[0],0] = npArray[ValidDataIndex[0],0] 

    Y_0 =npArray[np.where(np.isnan(npArray) != 1)]
    X_0 = X[np.where(np.isnan(npArray) != 1)]
    IRFunction = interp1d(X_0, Y_0, kind = 'linear')
    Fill_X = X[np.where(np.isnan(npArray) == 1)]
    Fill_Y = IRFunction(Fill_X)
    npArray[Fill_X,0] = Fill_Y 
    return npArray


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
      return ts, data
  
  ts1, COD = read_csv('w01018_补值.csv')
  ts2, TN  = read_csv('w21001_补值.csv')
  ts3, NH  = read_csv('w21003_补值.csv')
  ts4, TP  = read_csv('w21011_补值.csv')
  ts5, PH  = read_csv('w01001_补值.csv')

  COD = COD.astype(np.float32)
  TN  = TN .astype(np.float32)
  NH  = NH .astype(np.float32)
  TP  = TP .astype(np.float32)
  PH  = PH .astype(np.float32)
  
  COD=yichang(COD).reshape(-1,)
  TN=yichang(TN).reshape(-1,)
  NH=yichang(NH).reshape(-1,)
  TP=yichang(TP).reshape(-1,)
  PH=yichang(PH).reshape(-1,)
  
  # assure time consistency
  assert (ts1 == ts2).all() and (ts1 == ts3).all() and (ts1 == ts4).all() and (ts1 == ts5).all()
  weekday = [int(x.split(sep)[0]) for x in ts1]
  hour    = [int(x.split(sep)[1]) for x in ts1]

  # TODO: do your data cleanning here
  
  
  # data normalize
  def minmax_norm(x:np.array):
    x_min, x_max = x.min(), x.max()
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm, x_min, x_max

  COD, COD_min, COD_max = minmax_norm(COD)
  TN,  TN_min,  TN_max  = minmax_norm(TN)
  NH,  NH_min,  NH_max  = minmax_norm(NH)
  TP,  TP_min,  TP_max  = minmax_norm(TP)
  PH,  PH_min,  PH_max  = minmax_norm(PH)
  # necessary statistics during predicting
  stats = {
    'COD': (COD_min, COD_max),
    'TN':  (TN_min,  TN_max),
    'NH':  (NH_min,  NH_max),
    'TP':  (TP_min,  TP_max),
    'PH':  (PH_min,  PH_max),
  }

  # collect data matrix, [N, D=6]
  fvmat = np.stack([weekday, hour, COD, TN, NH, TP, PH]).T

  # save preprocessed data
  save_data(fvmat, stats, hp.DATA_FILE)
  
def preprocess1(sep='-'):
  # load raw data
  CONVERTER = {
    0: lambda x: _parse_datetime_str(x, sep),
    1: lambda x: float(x),
  }
  def read_csv(fn):
    with open(os.path.join(hp.DATA_PATH, fn)) as fh:
      data = np.loadtxt(fh, delimiter=",", converters=CONVERTER, skiprows=1, dtype=object)
      ts, data = [x.squeeze() for x in np.split(data, 2, axis=-1)]
      return ts, data
  
  ts1, COD = read_csv('w01018_补值.csv')
  ts2, TN  = read_csv('w21001_补值.csv')
  ts3, NH  = read_csv('w21003_补值.csv')
  ts4, TP  = read_csv('w21011_补值.csv')
  ts5, PH  = read_csv('w01001_补值.csv')

  COD = COD.astype(np.float32)
  TN  = TN .astype(np.float32)
  NH  = NH .astype(np.float32)
  TP  = TP .astype(np.float32)
  PH  = PH .astype(np.float32)
  
  COD=yichang(COD).reshape(-1,)
  TN=yichang(TN).reshape(-1,)
  NH=yichang(NH).reshape(-1,)
  TP=yichang(TP).reshape(-1,)
  PH=yichang(PH).reshape(-1,)
  
  # assure time consistency
  assert (ts1 == ts2).all() and (ts1 == ts3).all() and (ts1 == ts4).all() and (ts1 == ts5).all()
  weekday = [int(x.split(sep)[0]) for x in ts1]
  hour    = [int(x.split(sep)[1]) for x in ts1]

  # TODO: do your data cleanning here
  # data normalize
  def minmax_norm(x:np.array):
    x_min, x_max = x.min(), x.max()
    x_norm = (x - x_min) / (x_max - x_min)
    return  x_min, x_max

  COD_min, COD_max = minmax_norm(COD)
  TN_min,  TN_max  = minmax_norm(TN)
  NH_min,  NH_max  = minmax_norm(NH)
  TP_min,  TP_max  = minmax_norm(TP)
  PH_min,  PH_max  = minmax_norm(PH)
  # necessary statistics during predicting
  stats = {
    'COD': (COD_min, COD_max),
    'TN':  (TN_min,  TN_max),
    'NH':  (NH_min,  NH_max),
    'TP':  (TP_min,  TP_max),
    'PH':  (PH_min,  PH_max),
  }
  

  fvmat = np.stack([weekday, hour, COD, TN, NH, TP, PH]).T

  # save preprocessed data
  save_data(fvmat,stats, hp.DATA_FILE1)
  
  
if __name__ == "__main__":
  preprocess()
  preprocess1()