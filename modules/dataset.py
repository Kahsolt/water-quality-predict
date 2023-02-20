#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/19 

import torch
import numpy as np

from modules.typing import *


def ex_thresh(seq:Seq, T:Series, inlen:int, outlen:int, **kwargs) -> Seq:
  '''
    当前是否超标：
      否 -> 0
      是 -> 1
  '''

  thresh = kwargs['thresh']

  breakpoint()

def ex_thresh_3h(seq:Seq, T:Series, inlen:int, outlen:int, **kwargs) -> Seq:
  '''
    前 3h 超标情况：
      k 次 -> k   (k <= 3)
  '''

  thresh = kwargs['thresh']

def ex_thresh_24h(seq:Seq, T:Series, inlen:int, outlen:int, **kwargs) -> Seq:
  '''
    自上一个 00:00 起，超标情况：
        0 次 -> 0
      1~3 次 -> 1
      4~5 次 -> 2
      6~  次 -> 3
  '''
  thresh = kwargs['thresh']


# ↑↑↑ above are valid encoders ↑↑↑


def resample_frame_dataset_and_encode(x:Seq, T:Series, encoder:Encoder, inlen:int=3, outlen:int=1, count:int=1000) -> Dataset:
  ''' 在时间序列上重采样切片并离散编码，知 inlen 推 outlen '''

  # x.shape: [T, D=1]
  assert len(x.shape) == 2
  assert x.shape[-1]  == 1

  label_func = globals()[encoder['name']]
  label_func_params = encoder.get('params') or {}

  seg_size = inlen + outlen
  rlim = len(x) - seg_size

  X, Y = [], []
  for _ in range(count):
    r = np.random.randint(0, rlim)
    seg_x = x[r:r+seg_size, :]
    seg_T = T[r:r+seg_size]
    X.append(seg_x[:inlen])   # [I, D=1], FrameIn
    Y.append(label_func(seg_x, seg_T, inlen, outlen, **label_func_params))  # [O, D=1], FrameOut
  X = np.stack(X, axis=0)     # [N, I, D=1]
  Y = np.stack(Y, axis=0)     # [N, O, D=1]

  return X, Y.astype(np.int32)

def resample_frame_dataset(x:Seq, inlen:int=3, outlen:int=1, count:int=1000) -> Dataset:
  ''' 在时间序列上重采样切片，知 inlen 推 outlen '''

  # x.shape: [T, D]
  assert len(x.shape) == 2

  seg_size = inlen + outlen
  rlim = len(x) - seg_size

  X, Y = [], []
  for _ in range(count):
    r = np.random.randint(0, rlim)
    seg = x[r:r+seg_size, :]
    X.append(seg[:inlen])     # [I, D], FrameIn
    Y.append(seg[inlen:])     # [O, D], FrameOut
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
