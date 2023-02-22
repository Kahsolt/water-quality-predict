#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/21 

import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from modules.dataset import FrameDataset, DataLoader
from modules.typing import *


class CRNN(nn.Module):

  def __init__(self, 
    r_type:str, r_in:int, r_out:int, r_hidden=32, r_dropout:float=0.0, r_layers:int=1,
    c_layers:int=0, c_k:int=3, c_stride:int=2, c_in:int=32, c_out:int=32, c_hidden:int=32, c_act:str='leaky_relu',
  ):
    super().__init__()

    assert r_layers > 0
    self.use_cnn = c_layers > 0

    # CNN
    if self.use_cnn:
      self.convs = nn.ModuleList()
      self.convs.append(nn.Conv1d(c_in, c_hidden, c_k, c_stride))
      for _ in range(1, c_layers-1):
        self.convs.append(nn.Conv1d(c_hidden, c_hidden, c_k, c_stride))
      self.convs.append(nn.Conv1d(c_hidden, c_out, c_k, c_stride))

      self.act_fn = getattr(F, c_act)
      assert isinstance(self.act_fn, Callable)
      
    # RNN
    rnn_impl = getattr(nn, r_type)
    assert rnn_impl in [nn.LSTM, nn.GRU]

    self.rnn = rnn_impl(r_in, r_hidden, r_layers, dropout=r_dropout)
    self.linear = nn.Linear(r_hidden, r_out)

  def forward(self, x):
    # cnn
    if self.use_cnn:
      x = x.permute([0, 2, 1])    # [B, L, D] => [B, D, L]
      for layer in self.convs:
        x = layer(x)              # [B, D, L] => [B, C, L']
        x = self.act_fn(x)
      x = x.permute([0, 2, 1])    # [B, C, L'] => [B, L', C]

    # rnn
    x = x.permute([1, 0, 2])      # [B, L, D] => [L, B, D]
    x, _ = self.rnn(x)            # [L, B, I=1] => [L, B, H]
    x = x[-1, ...]                # last frame of outputs
    x = self.linear(x)            # [B, H] => [B, O]

    return x


def prepare_for_train(model:PyTorchModel, dataset:Datasets, config:Config):
  E  = config.get('epochs', 10)
  B  = config.get('batch_size', 32)
  O  = config.get('optimizer', 'Adam')
  lr = config.get('lr', 1e-3)
  wd = config.get('weight_decay', 1e-5)
  L  = config.get('loss', 'mse_loss')

  lr = float(lr)
  wd = float(wd)

  dataloader = DataLoader(FrameDataset(dataset[0]), batch_size=B, shuffle=True, pin_memory=True, drop_last=True)
  optimizer: Optimizer = getattr(optim, O)(model.parameters(), lr, weight_decay=wd)
  loss_fn: Callable = getattr(F, L)

  assert isinstance(optimizer, Optimizer)
  assert isinstance(loss_fn, Callable)

  return dataloader, optimizer, loss_fn, E


def prepare_for_eval(model:PyTorchModel, dataset:Datasets, config:Config):
  evalset = dataset[1]
  y_test = evalset[1]
  dataloader = DataLoader(FrameDataset(evalset), batch_size=1, shuffle=False, pin_memory=False)

  return dataloader, y_test
