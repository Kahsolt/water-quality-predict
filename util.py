#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/15 

import os
import sys
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging


def load_checkpoint(checkpoint_path, model, optimizer=None):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  iteration = checkpoint_dict['iteration']
  learning_rate = checkpoint_dict['learning_rate']
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
  state_dict = checkpoint_dict['model']
  new_state_dict= {}
  for k, v in model.state_dict().items():
    try:
      new_state_dict[k] = state_dict[k]
    except:
      logger.info("%s is not in the checkpoint" % k)
      new_state_dict[k] = v
  model.load_state_dict(new_state_dict)
  logger.info(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration})")
  return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
  logger.info(f"Saving model and optimizer state at iteration {iteration} to {checkpoint_path}")
  if hasattr(model, 'module'):
     #只保存模型参数
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save({'model': state_dict,
              'iteration': iteration,
              'optimizer': optimizer.state_dict(),
              'learning_rate': learning_rate}, checkpoint_path)


def get_logger(model_dir, filename="train.log"):
  global logger
  logger = logging.getLogger(os.path.basename(model_dir))
  logger.setLevel(logging.DEBUG)
  
  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  h = logging.FileHandler(os.path.join(model_dir, filename))
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)
  return logger


def plot_predict(ys:list, y_hats:list, save_fp:str='predict.png'):
  n_figs = len(ys)
  fig = plt.figure(figsize=(12,8))
  lst=['COD','TN','NH3-N','TP','PH']  #注意根据因子修改
  a=0
  for i, (y, y_hat) in enumerate(zip(ys, y_hats), start=1):
    plt.subplot(n_figs, 1, i)
    plt.plot(y_hat, 'r',label='forcast')
    plt.plot(y, 'b',label='true')
    plt.ylabel(lst[a])
    a+=1
    if a==1:
        legend = plt.legend(ncol=2,loc='best')
  fig.tight_layout(pad=0.5, w_pad=0, h_pad=0)
  print(f'[plot_predict] save to {save_fp}')
  plt.savefig(save_fp,dpi=500)

  plt.show()



def buzhi(npArray):
    if len(npArray.shape) == 1:
        npArray = npArray.copy().reshape(-1, 1)
    else:
        npArray = npArray.copy()
    fArray = npArray[:, -1]

    X = np.array([i for i in range(len(npArray))])
    X = X.reshape(len(X), 1)

    # 首尾用临近值补值
    ValidDataIndex = X[np.where(np.isnan(npArray) == 0)]
    if ValidDataIndex[-1] < len(npArray) - 1:
        npArray[ValidDataIndex[-1] + 1:, 0] = npArray[ValidDataIndex[-1], 0]
        # 如果第一个正常数据的序号不是0，则表明第一个正常数据前存在缺失数据
    if ValidDataIndex[0] >= 1:
        npArray[:ValidDataIndex[0], 0] = npArray[ValidDataIndex[0], 0]

    Y_0 = npArray[np.where(np.isnan(npArray) != 1)]
    X_0 = X[np.where(np.isnan(npArray) != 1)]
    IRFunction = interp1d(X_0, Y_0, kind='linear')
    Fill_X = X[np.where(np.isnan(npArray) == 1)]
    Fill_Y = IRFunction(Fill_X)
    npArray[Fill_X, 0] = Fill_Y
    return npArray
