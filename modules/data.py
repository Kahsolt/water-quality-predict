#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/19 

import torch
import numpy as np

from modules.typing import *


def resample_frame_dataset(x:np.ndarray, inlen:int=3, outlen:int=1, count:int=1000) -> Dataset:
  ''' 在时间序列上重采样切片，知 inlen 推 outlen '''

  # x.shape: [T, D=1]
  assert len(x.shape) == 2

  seg_size = inlen + outlen
  rlim = len(x) - seg_size

  X, Y = [], []
  for _ in range(count):
    r = np.random.randint(0, rlim)
    seg = x[r : r+seg_size, :]
    X.append(seg[:inlen])
    Y.append(seg[inlen:])
  X = np.stack(X, axis=0)     # [count, seg_size, D]
  Y = np.stack(Y, axis=0)     # [count, seg_size, D]

  return X, Y


class FrameDataset(torch.utils.data.Dataset):

  def __init__(self, X:np.ndarray, Y:np.ndarray):
    self.X = X      # [N, in]
    self.Y = Y      # [N, out]

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.Y[idx]
