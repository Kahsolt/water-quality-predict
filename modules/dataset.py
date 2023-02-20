#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/19 

import torch
import numpy as np

from modules.typing import *


def resample_frame_dataset(x:Seq, inlen:int=3, outlen:int=1, count:int=1000) -> Dataset:
  ''' 在时间序列上重采样切片，知 inlen 推 outlen '''

  # x.shape: [T, D=1]
  assert len(x.shape) == 2

  seg_size = inlen + outlen
  rlim = len(x) - seg_size

  X, Y = [], []
  for _ in range(count):
    r = np.random.randint(0, rlim)
    seg = x[r : r+seg_size, :]
    X.append(seg[:inlen])     # FrameIn
    Y.append(seg[inlen:])     # FrameOut
  X = np.stack(X, axis=0)     # [N, I, D]
  Y = np.stack(Y, axis=0)     # [N, O, D]

  return X, Y


class FrameDataset(torch.utils.data.Dataset):

  def __init__(self, dataset:Dataset):
    self.X = dataset[0]      # [N, I]
    self.Y = dataset[1]      # [N, O]

    assert len(self.X) == len(self.Y)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.Y[idx]
