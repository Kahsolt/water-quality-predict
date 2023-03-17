#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/19 

from random import shuffle

import torch
from torch.utils.data import DataLoader
import numpy as np

from modules.typing import *

def ex_thresh(seq:Seq, T:Time, **kwargs) -> Seq:
  '''
    当前 是否超标：
      否 -> 0
      是 -> 1
  '''

  thresh = kwargs['thresh']

  return (seq > thresh).astype(np.int32)    # [T, D=1]


def ex_thresh_3h(seq:Seq, T:Time, **kwargs) -> Seq:
  '''
    前 3h 超标情况：
      k 次 -> k   (0 <= k <= 3)
  '''

  thresh = kwargs['thresh']

  y = [(seq[i-3:i] > thresh).sum() for i in range(len(seq))]
  y = np.asarray(y, dtype=np.int32)   # [T]
  y = np.expand_dims(y, axis=-1)      # [T, D=1]
  return y


def ex_thresh_24h(seq:Seq, T:Time, **kwargs) -> Seq:
  '''
    自上一个 00:00 起，超标情况：
        0 次 -> 0
      1~3 次 -> 1
      4~5 次 -> 2
      6~  次 -> 3
  '''

  thresh = kwargs['thresh']

  seq = seq.squeeze(axis=-1)                  # [T]
  hours = T.map(lambda x:x.split(' ')[-1])    # [T]

  def get_score(i:int):
    cnt = 0
    while i > 0 and cnt < 6:
      if seq[i] > thresh: cnt += 1
      if hours[i] == '00:00:00': break
      i -= 1
    if        cnt == 0: return 0
    elif 1 <= cnt <= 3: return 1
    elif 4 <= cnt <= 5: return 2
    else:               return 3

  y = [get_score(j) for j in range(len(seq))]
  y = np.asarray(y, dtype=np.int32)   # [T]
  y = np.expand_dims(y, axis=-1)      # [T, D=1]
  return y


# ↑↑↑ above are valid encoders ↑↑↑

def encode_seq(x:Seq, T:Time, encode:Encode) -> Seq:
  label_func = globals()[encode['name']]
  label_func_params = encode.get('params') or {}
  return label_func(x, T, **label_func_params)

def split_frame_dataset(x:Seq, inlen:int=3, outlen:int=1, overlap:int=0, split:float=0.2, y:Seq=None) -> Datasets:
  ''' 在时间序列上滚动切片产生数据集样本，知 inlen 推 outlen 时间步 '''

  # x.shape: [T, D]
  assert len(x.shape) == 2

  seg_size = inlen + outlen - overlap
  rlim = len(x) - seg_size
  if y is None: y = x

  XY = []
  for r in range(rlim):
    seg_x = x[r:r+seg_size, :]
    seg_y = y[r:r+seg_size, :]
    XY.append((seg_x[:inlen],           # [I, D], FrameIn
               seg_y[inlen-overlap:]))  # [O, D], FrameOut
  shuffle(XY)

  X = [x for x, _ in XY]
  Y = [y for _, y in XY]
  X: Frames = np.stack(X, axis=0)     # [N, I, D]
  Y: Frames = np.stack(Y, axis=0)     # [N, O, D]

  cp = int(len(X) * split)
  X_eval, X_train = X[:cp, ...], X[cp:, ...]
  Y_eval, Y_train = Y[:cp, ...], Y[cp:, ...]

  return (X_train, Y_train), (X_eval, Y_eval)


class FrameDataset(torch.utils.data.Dataset):

  def __init__(self, dataset:Dataset):
    self.X: Frames = dataset[0]      # [N, I]
    self.Y: Frames = dataset[1]      # [N, O]

    assert len(self.X) == len(self.Y)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.Y[idx]
