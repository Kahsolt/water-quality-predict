#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/15 

from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.interpolate import interp1d
import pywt

from modules.typing import *


def to_hourly(df:DataFrame) -> DataFrame:
  return df[df.columns[1:]]     # drop T

def to_daily(df:DataFrame) -> DataFrame:
  T = df.columns[0]
  df[T] = df[T].map(lambda x: x.split(' ')[0])    # get date part from timestr
  df = df.groupby(T).mean().reset_index()
  return df[df.columns[1:]]     # drop T


def yichang(df:DataFrame) -> DataFrame:
  def process(npArray:Series) -> Series:
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

  return df.apply(process, axis=1)

def walvent(df:DataFrame, wavelet:str='db8', threshold:float=0.04) -> DataFrame:
  def process(npArrays:Series) -> Series:
    training_set_scaled=npArrays.reshape(-1)
    w = pywt.Wavelet(wavelet)                                             # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(training_set_scaled), w.dec_len)
    coeffs = pywt.wavedec(training_set_scaled, wavelet, level=maxlev)     # 将信号进行小波分解
    for i in range(1, len(coeffs)):
      coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))     # 将噪声滤波
    training_set_scaled = pywt.waverec(coeffs, wavelet)                   # 将信号进行小波重构
    npArrays=training_set_scaled.reshape(-1,1)
    return npArrays

  return df.apply(process, axis=1)


def log(df:DataFrame) -> Tuple[DataFrame, Stat]:
  return df.apply(lambda x: np.log(x + 1e-8), axis=1), tuple()

def std_norm(df:DataFrame) -> Tuple[DataFrame, Stat]:
  avg, std = df.mean(axis=1), df.std(axis=1)
  df_n = (df - avg) / std
  return df_n, (avg, std)

def minmax_norm(df:DataFrame) -> Tuple[DataFrame, Stat]:
  vmin, vmax = df.min(axis=1), df.max(axis=1)
  df_n = (df - vmin) / (vmax - vmin)
  return df_n, (vmin, vmax)

# ↑↑↑ above are valid preprocessors ↑↑↑


# ↓↓↓ below are innerly auto-called, DO NOT USE ↓↓↓

def read_csv(fp:str) -> DataFrame:
  return pd.read_csv(fp, encoding='utf-8')

def save_csv(df:DataFrame, fp:str) -> None:
  df.to_csv(fp, encoding='utf-8')


def log_inv(df:DataFrame) -> DataFrame:
  return df.apply(np.exp, axis=1)

def std_norm_inv(df:DataFrame, avg:Series, std:Series) -> DataFrame:
  return df * std + avg

def minmax_norm_inv(df:DataFrame, vmin:Series, vmax:Series) -> DataFrame:
  return df * (vmax - vmin) + vmin
