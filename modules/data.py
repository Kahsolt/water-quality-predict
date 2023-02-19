#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/15 

import pickle as pkl
from typing import Tuple

import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import hparam as hp


def load_data(fp:str) -> dict:
  with open(fp, 'rb') as fh:
    return pkl.load(fh)


def save_data(data:dict, fp:str):
  with open(fp, 'wb') as fh:
    pkl.dump(data, fh)


class ResampleDataset(Dataset):

  def __init__(self, fvmat, segment_size, count=5000):
    self.length = len(fvmat)               # 实际序列的总长度
    self.segment_size = segment_size       # 重采样样本的长度
    self.count = count                     # 重采样次数 (count个采样就算作一个数据集)

    self.weekdays   = fvmat[:, 0]    # [N]
    self.hours      = fvmat[:, 1]    # [N]
    self.COD_TN_NH_TP_PHs = fvmat[:, 2:]   # [N, D=5]

  def __len__(self):
    return self.count

  def __getitem__(self, idx):
    # 在总序列上随机截取一个分段
    w = self.weekdays  [idx : idx + self.segment_size]   .astype(np.int32)    # [segment_size]
    h = self.hours     [idx : idx + self.segment_size]   .astype(np.int32)    # [segment_size]
    d = self.COD_TN_NH_TP_PHs[idx : idx + self.segment_size, :].astype(np.float32)  # [segment_size, D=3]

    return w, h, d
