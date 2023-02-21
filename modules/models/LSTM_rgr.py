#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/21 

from pathlib import Path

import torch
from torch.nn.functional import *
from torch.optim import *

from modules.models.LSTM import LSTM
from modules.dataset import FrameDataset, DataLoader
from modules.util import *
from modules.preprocess import *
from modules.typing import *

TASK_TYPE: ModelTask = Path(__file__).stem.split('_')[-1]


def init(config:Config):
  model = globals()[config['model']](
    d_in=config['d_in'],
    d_out=config['d_out'],
    d_hidden=config['d_hidden'],
    n_layers=config['n_layers'],
  )
  return model.to(device)


def train(model:LSTM, dataset:Datasets, config:Config):
  global logger

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
  loss_fn = globals()[L]

  model.train()
  for i in range(E):
    for X, Y in dataloader:
      X = X.to(device)
      Y = Y.to(device)

      optimizer.zero_grad()
      y_hat = model(X)                    # [B=1, O=6]
      loss = loss_fn(y_hat, Y.squeeze(dim=-1))
      loss.backward()
      optimizer.step()
    logger.info(f'[Epoch: {i}] loss:{loss.item():.7f}')


@torch.inference_mode()
def eval(model:LSTM, dataset:Datasets, config:Config):
  _, evalset = dataset
  _, y_test = evalset
  dataloader = DataLoader(FrameDataset(evalset), batch_size=1, shuffle=False, pin_memory=False)

  preds = []
  model.eval()
  for X, _ in dataloader:
    X = X.to(device)
    Y = Y.to(device)

    y_hat = model(X)                        # [B=1, O=6]
    preds.append(y_hat.numpy())
  pred: Frames = np.stack(preds, axis=0)    # [N, O]
  pred = np.expand_dims(pred, axis=-1)      # [N, O, D=1]

  get_metrics(y_test, pred, task=TASK_TYPE)


@torch.inference_mode()
def infer(model:LSTM, x:Frame) -> Frame:
  x = torch.from_numpy(x)
  y = model(x)
  y = y.numpy()
  return y


def save(model:LSTM, log_dp:Path):
  save_checkpoint(model, log_dp / 'model.pth')


def load(model:LSTM, log_dp:Path) -> LSTM:
  return load_checkpoint(model, log_dp / 'model.pth')
