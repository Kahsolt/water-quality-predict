#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/21 

import torch.nn as nn
from torch.nn.functional import *
from torch.optim import *

from modules.dataset import FrameDataset, DataLoader
from modules.typing import *


class RNN(nn.Module):

  def __init__(self, rnn_type:str, d_in:int, d_out:int, d_hidden=32, n_layers=1, dropout=0):
    super().__init__()

    rnn_impl = getattr(nn, rnn_type)
    assert rnn_impl in [nn.LSTM, nn.GRU]

    self.rnn = rnn_impl(d_in, d_hidden, n_layers, dropout=dropout)
    self.linear = nn.Linear(d_hidden, d_out)

  def forward(self, x):
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
  optimizer: Optimizer = globals()[O](model.parameters(), lr, weight_decay=wd)
  loss_fn: Callable = globals()[L]

  return dataloader, optimizer, loss_fn, E


def prepare_for_eval(model:PyTorchModel, dataset:Datasets, config:Config):
  _, evalset = dataset
  _, y_test = evalset
  dataloader = DataLoader(FrameDataset(evalset), batch_size=1, shuffle=False, pin_memory=False)

  return dataloader, y_test
