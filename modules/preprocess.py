#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/15 

from typing import Tuple

import numpy as np
from pandas import DataFrame, Series
from scipy.interpolate import interp1d
import pywt

from modules.typing import *


''' filter_T: 数据选择 '''
def ltrim_vacant(df:TDataFrame) -> TDataFrame:
  # TODO: 沿时间线向前回溯，若遇到长度大于 168(一周) 的数据空白期，则丢弃之前的所有数据
  return df


''' project: 分离时间轴和数据 '''
def to_hourly(df:TDataFrame) -> TimeAndData:
  return split_time_and_data(df)

def to_daily(df:TDataFrame) -> TimeAndData:
  T = df.columns[0]
  df[T] = df[T].map(lambda x: x.split(' ')[0])    # get date part from timestr
  df = df.groupby(T).mean().reset_index()
  return split_time_and_data(df)


''' filter_V: 数值修正 '''
def yichang(df:DataFrame) -> DataFrame:
  def process(seq:Series) -> Series:
    '''异常值处理与补值'''
    if len(seq.shape) == 1: 
      seq = seq.copy().reshape(-1,1)
    else:
      seq = seq.copy()
    fArray = seq[:,-1]
    avg=seq.mean()
    st=np.std(seq)
    seq[:,-1] = np.array([e if e > 0 and (avg-3*st <= e <= 3*st+avg) else np.nan for e in fArray])
    X=np.array([i for i in range(len(seq))])
    X=X.reshape(len(X),1)
    #首尾用临近值补值
    ValidDataIndex = X[np.where(np.isnan(seq) == 0)]
    if ValidDataIndex[-1] < len(seq) - 1: 
        seq[ValidDataIndex[-1] + 1:,0] = seq[ValidDataIndex[-1],0]  
    # 如果第一个正常数据的序号不是0，则表明第一个正常数据前存在缺失数据
    if ValidDataIndex[0] >= 1:
        seq[:ValidDataIndex[0],0] = seq[ValidDataIndex[0],0] 

    Y_0 =seq[np.where(np.isnan(seq) != 1)]
    X_0 = X[np.where(np.isnan(seq) != 1)]
    IRFunction = interp1d(X_0, Y_0, kind = 'linear')
    Fill_X = X[np.where(np.isnan(seq) == 1)]
    Fill_Y = IRFunction(Fill_X)
    seq[Fill_X,0] = Fill_Y 
    return seq

  return df.apply(process, axis=1)

def walvent(df:DataFrame, wavelet:str='db8', threshold:float=0.04) -> DataFrame:
  def process(seq:Series) -> Series:
    w = pywt.Wavelet(wavelet)                             # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(seq), w.dec_len)
    coeffs = pywt.wavedec(seq, wavelet, level=maxlev)     # 将信号进行小波分解
    for i in range(1, len(coeffs)):
      coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))     # 将噪声滤波
    seq = pywt.waverec(coeffs, wavelet)                   # 将信号进行小波重构
    return seq

  return df.apply(process, axis=1)

# ↑↑↑ above are valid preprocessors ↑↑↑


''' transform: 数值变换用于训练，必须是可逆的 '''
def log(seq:Seq) -> Tuple[Seq, Stat]:
  return log_apply(seq), tuple()

def std_norm(seq:Seq) -> Tuple[Seq, Stat]:
  avg = seq.mean(axis=0, keepdims=True)
  std = seq.std (axis=0, keepdims=True)
  seq_n = std_norm_apply(seq, avg, std)
  return seq_n, (avg, std)

def minmax_norm(seq:Seq) -> Tuple[Seq, Stat]:
  vmin = seq.min(axis=0, keepdims=True)
  vmax = seq.max(axis=0, keepdims=True)
  seq_n = minmax_norm_apply(seq, vmin, vmax)
  return seq_n, (vmin, vmax)

# ↑↑↑ above are valid transforms ↑↑↑


# ↓↓↓ below are innerly auto-called, SHOULD NOT USE in job.yaml ↓↓↓

def split_time_and_data(df:TDataFrame) -> TimeAndData:
  cols = df.columns
  return df[cols[0]], df[cols[1:]]


''' transform (apply): 数值变换 '''
def log_apply(seq:Seq) -> Seq:
  return np.log(seq + 1e-8)

def std_norm_apply(seq:Seq, avg:ndarray, std:ndarray) -> Seq:
  return (seq - avg) / std

def minmax_norm_apply(seq:Seq, vmin:ndarray, vmax:ndarray) -> Seq:
  return (seq - vmin) / (vmax - vmin)

''' transform (inverse): 数值逆变换 '''
def log_inv(seq:Seq) -> Seq:
  return np.exp(seq)

def std_norm_inv(seq:Seq, avg:ndarray, std:ndarray) -> Seq:
  return seq * std + avg

def minmax_norm_inv(seq:Seq, vmin:ndarray, vmax:ndarray) -> Seq:
  return seq * (vmax - vmin) + vmin
