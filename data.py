#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/15 

import pickle as pkl
from typing import Tuple

import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import hparam as hp


def load_data(fp:str) -> Tuple[np.ndarray, dict]:
  with open(fp, 'rb') as fh:
    data = pkl.load(fh)

  return data['fvmat'], data['stats']


def save_data(fvmat:np.ndarray, stats:dict, fp:str):
  data = {
    'fvmat': fvmat,
    'stats': stats,
  }
  with open(fp, 'wb') as fh:
    pkl.dump(data, fh)


def inspect_data():
  fvmat, stats = load_data(hp.DATA_FILE)
  print(fvmat.shape)
  print(stats)

  COD = fvmat[:, 2].astype(np.float32)
  PH  = fvmat[:, 3].astype(np.float32)
  NH  = fvmat[:, 4].astype(np.float32)

  if not 'denorm':
    COD = np.asarray((COD + stats['COD'][0]) * (stats['COD'][1] - stats['COD'][0]))
    PH  = np.asarray((PH + stats['PH'][0])  * (stats['PH'][1]  - stats['PH'][0]))
    NH  = np.asarray((NH + stats['NH'][0])  * (stats['NH'][1]  - stats['NH'][0]))

  show_stats = lambda x, name: print(f'[{name}] min: {x.min()}, max: {x.max()}, mean: {x.mean()}, std: {x.std()}')
  show_stats(COD, 'COD')
  show_stats(PH,  'PH')
  show_stats(NH,  'NH')

  plt.subplot(311) ; plt.title('COD') ; plt.plot(COD)
  plt.subplot(312) ; plt.title('PH')  ; plt.plot(PH)
  plt.subplot(313) ; plt.title('NH')  ; plt.plot(NH)
  plt.show()

  plt.subplot(311) ; plt.title('COD') ; plt.hist(COD, bins=100)
  plt.subplot(312) ; plt.title('PH')  ; plt.hist(PH,  bins=100)
  plt.subplot(313) ; plt.title('NH')  ; plt.hist(NH,  bins=100)
  plt.show()


class ResampleDataset(Dataset):

  def __init__(self, fvmat, segment_size, count=5000):
    self.length = len(fvmat)               # 实际序列的总长度
    self.segment_size = segment_size       # 重采样样本的长度
    self.count = count                     # 重采样次数 (count个采样就算作一个数据集)

    self.weekdays   = fvmat[:, 0]    # [N]
    self.hours      = fvmat[:, 1]    # [N]
    self.features   = fvmat[:, 2:]   # [N, D]

  def __len__(self):
    return self.count

  def __getitem__(self, ignore):
    # 在总序列上随机截取一个分段
    idx = np.random.randint(0, self.length - self.segment_size - 1)

    w = self.weekdays[idx : idx + self.segment_size]   .astype(np.int32)    # [segment_size]
    h = self.hours   [idx : idx + self.segment_size]   .astype(np.int32)    # [segment_size]
    #d = self.features[idx : idx + self.segment_size, :].astype(np.float32)  # [segment_size, D]
    d = [                                                                   # D * [segment_size]
      self.features[idx : idx + self.segment_size, i].astype(np.int32)
        for i in range(self.features.shape[-1])
    ]

    # (np.array, np.array, List[np.array])
    return w, h, d


if __name__ == '__main__':
  inspect_data()
