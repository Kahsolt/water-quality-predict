#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/15 

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pywt

from modules.transform import *
from modules.typing import *


''' filter_T: 含时处理，数据选择 '''
def ticker_timer(df:TimeSeq) -> TimeSeq:
  T, vals = split_time_and_values(df)
  frms = vals.to_numpy()

  def get_leap_hours(now:str, before:str) -> int:
    s = datetime.fromisoformat(before)
    t = datetime.fromisoformat(now)
    return (t - s).seconds // 3600

  def back_n_hours(now:str, hours:int) -> str:
    s = datetime.fromisoformat(now)
    b = s - timedelta(hours=hours)
    return str(b)

  has_leap = False
  new_ts   = [T[0]]
  new_frms = [frms[0]]
  for i in range(1, len(T)):
    now = T[i]
    frm = frms[i]

    leap = get_leap_hours(now, new_ts[-1])
    while leap >= 2:
      has_leap = True
      tt = back_n_hours(now, leap-1)
      ff = np.ones_like(frm) * np.nan
      new_ts  .append(tt)
      new_frms.append(ff)
      leap -= 1
    new_ts  .append(now)
    new_frms.append(frm)

  if has_leap:
    newT = pd.Series   (np.stack(new_ts  , axis=0))
    newV = pd.DataFrame(np.stack(new_frms, axis=0), columns=vals.columns)
    return combine_time_and_values(newT, newV)
  else:
    return df

def ltrim_vacant(df:TimeSeq) -> TimeSeq:
  def count_consecutive_nan(x:Series) -> Series:
    arr: List[float] = x.to_numpy()
    mask = np.isnan(arr).astype(int)
    for i in range(1, len(mask)):
      if mask[i]:
        mask[i] += mask[i-1]
    return Series(mask)

  tmstr = df.iloc[0][df.columns[0]]
  if ' ' in tmstr:    # hourly, '2021/1/1 00:00:00'
    limit = 168       # FIXME: hard-coded magic number
  else:               # daily, '2021/1/1'
    limit = 7

  lendf = len(df)
  cnt = df[df.columns[1:]].apply(count_consecutive_nan)
  for i in range(lendf-1, -1, -1):
    if cnt.iloc[i].max() > limit:
      break
  df = df.iloc[i:]

  return df


''' project: 时间刻度投影，分离时轴与数值 '''
def to_hourly(df:TimeSeq) -> TimeAndValues:
  return split_time_and_values(df)

def to_daily(df:TimeSeq) -> TimeAndValues:
  T = df.columns[0]
  df[T] = df[T].map(lambda x: x.split(' ')[0])    # get date part from timestr
  grps = df.groupby(T)
  df = grps.mean()                  # avg daily
  mask = grps.count() > 12          # filter by thresh, FIXME: hard-coded magic number
  mask = mask.apply(lambda s: s.map(lambda e: e or np.nan))
  df = df * mask                    # map masked to NaN
  df.reset_index(inplace=True)
  return split_time_and_values(df)


''' filter_V: 不含时处理，数值修正 '''
def remove_outlier(df:Values) -> Values:
  def process(x:Series) -> Series:
    x = x.fillna(method='ffill')
    arr: Array = np.asarray(x)
    arrlen = len(arr)
    tmp = [(arr, 'original')]

    if 'map 3-sigma outliers to NaN':
      arr_v = arr[~np.isnan(arr)]
      _, (avg, std) = std_norm(arr_v)
      L = avg - 3 * std
      H = avg + 3 * std
      mask = (L < arr) & (arr < H)
      arr = np.asarray([arr[i] if m else np.nan for i, m in enumerate(mask)])

      assert len(arr) == arrlen
      tmp.append((arr, f'3-sigma outlier ({L.item()}, {H.item()})'))

    if 'draw plot':
      plt.clf()
      n_fig = len(tmp)
      for i, (arr, title) in enumerate(tmp):
        plt.subplot(n_fig, 1, i+1)
        plt.plot(arr)
        plt.title(title)

    return Series(arr)

  return df.apply(process)

def fill_nan(df:Values) -> Values:
  def process(x:Series) -> Series:
    x = x.fillna(method='ffill')
    arr: Array = np.asarray(x)
    arrlen = len(arr)
    tmp = [(arr, 'original')]

    if 'padding by edge for NaNs at two endings':
      X = np.arange(arrlen)
      idx_v = X[np.where(np.isnan(arr) == 0)]   # index of non-NaN values
      if len(idx_v):
        i, j = idx_v[0], idx_v[-1]
        if i != 0 or j != arrlen - 1:
          arr = np.pad(arr[i: j+1], (i, arrlen - j - 1), mode='edge')

      assert len(arr) == arrlen
      tmp.append((arr, 'pad edge'))
    
    if 'interpolate for inner NaNs':
      mask = np.isnan(arr)
      X = np.arange(arrlen)                # dummy x-axis
      X_s = X  [~mask]
      Y_s = arr[~mask]
      interp = interp1d(X_s, Y_s, kind='linear')    # fit the model
      arr = interp(X)                      # predict

      assert len(arr) == arrlen
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

      assert len(arr) == arrlen
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

def combine_time_and_values(T:Time, df:Values) -> TimeSeq:
  val_cols = list(df.columns)
  df['Time'] = T
  return df[['Time'] + val_cols]


def merge_csv():
  pass


if __name__ == '__main__':
  df = pd.read_csv('data/test.csv')

  df = ticker_timer(df)
  df = ltrim_vacant(df)
  T, df = to_daily(df)
  df = remove_outlier(df)
  df = wavlet_transform(df)
