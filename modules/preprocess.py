#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/15 

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pywt

from modules.util import get_logger
from modules.transform import *
from modules.typing import *


''' filter_T: 含时处理，数据选择 '''
def ticker_timer(df:TimeSeq) -> TimeSeq:
  df
  return df

def ltrim_vacant(df:TimeSeq) -> TimeSeq:
  logger = get_logger()

  def count_consecutive_nan(x:Series) -> Series:
    arr: List[float] = x.to_numpy()
    mask = np.isnan(arr).astype(int)
    for i in range(1, len(mask)):
      if mask[i]:
        mask[i] += mask[i-1]
    return Series(mask)

  tmstr = df.iloc[0][df.columns[0]]
  if ' ' in tmstr:    # hourly, '2021/1/1 00:00:00'
    limit = 168
  else:               # daily, '2021/1/1'
    limit = 7

  lendf = len(df)
  cnt = df[df.columns[1:]].apply(count_consecutive_nan)
  for i in range(lendf-1, -1, -1):
    if cnt.iloc[i].max() > limit:
      break
  df = df.iloc[i:]
  logger.info(f'  ltrim_vacant: {lendf} => {len(df)}')

  return df


''' project: 时间刻度投影，分离时轴与数值 '''
def to_hourly(df:TimeSeq) -> TimeAndValues:
  return split_time_and_values(df)

def to_daily(df:TimeSeq) -> TimeAndValues:
  T = df.columns[0]
  df[T] = df[T].map(lambda x: x.split(' ')[0])    # get date part from timestr
  df = df.groupby(T).mean().reset_index()
  return split_time_and_values(df)


''' filter_V: 不含时处理，数值修正 '''
def remove_outlier(df:Values) -> Values:
  logger = get_logger()

  def process(x:Series) -> Series:
    x = x.fillna(method='ffill')
    arr: Array = np.asarray(x)
    tmp = [(arr, 'original')]

    if 'map 3-sigma outliers to NaN':
      arr_n, (avg, std) = std_norm(log_apply(arr))
      L = avg - 3 * std
      H = avg + 3 * std
      logger.info(f'  outlier: [{L}, {H}]')
      arr_clip = arr_n.clip(L, H)
      arr = log_inv(std_norm_inv(arr_clip, avg, std))

      tmp.append((arr, '3-sigma outlier'))

    if 'padding by edge for NaNs at two endings':
      i = 0
      while np.isnan(arr[i]): i += 1
      j = len(arr) - 1
      while np.isnan(arr[j]): j -= 1

      arrlen = len(arr)
      arr = np.pad(arr[i: j+1], (i, arrlen - j - 1), mode='edge')
      assert len(arr) == arrlen

      tmp.append((arr, 'pad edge'))
    
    if 'interpolate for inner NaNs':
      mask = np.isnan(arr)
      X = range(len(arr))                         # gen dummy x-axis seq
      interp = interp1d(X, arr, kind='linear')    # fit the model
      arr_interp = interp(X)                      # predict
      arr = arr_interp * mask + arr * ~mask

      tmp.append((arr, 'interp'))

    if 'draw plot':
      plt.clf()
      n_fig = len(tmp)
      for i, (arr, title) in enumerate(tmp):
        plt.subplot(n_fig, 1, i+1)
        plt.plot(arr)
        plt.title(title)

    return Series(arr)

  return df.apply(process)

def wavlet_transform(df:Values, wavelet:str='db8', threshold:float=0.04) -> Values:
  def process(x:Series) -> Series:
    arr: Array = np.asarray(x)
    tmp = [(arr, 'original')]

    if 'wavlet transform':
      arrlen = len(arr)
      w = pywt.Wavelet(wavelet)                             # 选用Daubechies8小波
      maxlev = pywt.dwt_max_level(arrlen, w.dec_len)
      coeffs = pywt.wavedec(arr, wavelet, level=maxlev)     # 将信号进行小波分解
      for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))     # 将噪声滤波
      arr = pywt.waverec(coeffs, wavelet)                   # 将信号进行小波重构
      arr = arr.clip(min=0.0)     # positive fix
      arr = arr[:arrlen]          # length fix

      tmp.append((arr, 'wavlet'))

    if 'draw plot':
      plt.clf()
      n_fig = len(tmp)
      for i, (arr, title) in enumerate(tmp):
        plt.subplot(n_fig, 1, i+1)
        plt.plot(arr)
        plt.title(title)

    return Series(arr)

  return df.apply(process)

# ↑↑↑ above are valid preprocessors ↑↑↑


def split_time_and_values(df:TimeSeq) -> TimeAndValues:
  cols = df.columns
  return df[cols[0]], df[cols[1:]]
