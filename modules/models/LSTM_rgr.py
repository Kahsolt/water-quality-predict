#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/21 

from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.functional import *
from torch.optim import *

from modules.dataset import FrameDataset, DataLoader
from modules.util import *
from modules.preprocess import *
from modules.typing import *

TASK_TYPE: ModelTask = Path(__file__).stem.split('_')[-1]


class LSTM(nn.Module):

  def __init__(self, d_in, d_out, d_hidden=128, n_layers=1):
    super().__init__()

    self.lstm = nn.LSTM(d_in, d_hidden, n_layers)
    self.linear = nn.Linear(d_hidden, d_out)

  def forward(self, x):
    breakpoint()
    x = x.view(len(x), 1, -1)
    x, _ = self.lstm(x)           # [L, B, I} => [L, B, H]
    x = self.linear(x)            # [L, B, H] => [L, B, O]
    return x


def init(config:Config):
  return globals()[config['model']](
    d_in=config['d_in'],
    d_out=config['d_out'],
    d_hidden=config['d_hidden'],
    n_layers=config['n_layers'],
  )


def train(model:LSTM, dataset:Datasets, config:Config):
  E  = config.get('epochs', 50)
  B  = config.get('batch_size', 32)
  O  = config.get('optimizer', 'Adam')
  lr = config.get('lr', 1e-3)
  wd = config.get('weight_decay', 1e-5)
  L  = config.get('loss', 'mse_loss')

  lr = float(lr)
  wd = float(wd)

  dataloader = DataLoader(FrameDataset(dataset[0]), batch_size=B, shuffle=True, pin_memory=True, drop_last=True)
  optimizer: Optimizer = globals()[O](model.parameters(), lr, weight_decay=wd)
  loss_fn = globals()[L]

  model.train()
  for i in range(E):
    for X, Y in dataloader:
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        breakpoint()
        loss.backward()
        optimizer.step()

    if i % 5 == 0:
      logger.info(f'[Epoch: {i}] loss:{loss.item():.7f}')


@torch.inference_mode()
def eval(model:LSTM, dataset:Datasets, config:Config):
  _, evalset = dataset
  _, y_test = evalset
  dataloader = DataLoader(FrameDataset(evalset), batch_size=1, shuffle=False, pin_memory=False)

  preds = []
  model.eval()
  for X, _ in dataloader:
    out = model(X)
    preds.append(out.numpy())
  pred: Frames = np.stack(preds, axis=0)

  get_metrics(y_test, pred, task=TASK_TYPE)


@torch.inference_mode()
def infer(model:LSTM, x:Frame) -> Frame:
  x = torch.from_numpy(x)
  y = model(x)
  y = y.numpy()
  return y


def save(model:LSTM, log_dp:Path):
  state_dict = model.state_dict()
  save_checkpoint(state_dict, log_dp / 'model.pth')


def load(model:LSTM, log_dp:Path) -> LSTM:
  state_dict = load_checkpoint(log_dp / 'model.pth')
  model.load_state_dict(state_dict)
  return model
