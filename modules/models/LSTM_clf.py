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
from modules.models.LSTM_rgr import init, save, load     # just proxy by

TASK_TYPE: ModelTask = Path(__file__).stem.split('_')[-1]


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
      Y = Y.squeeze().long().to(device)       # [B]
      assert len(Y.shape) == 1

      optimizer.zero_grad()
      logits = model(X)            # [B=32, NC=4]
      loss = loss_fn(logits, Y)
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
    Y = Y.squeeze().long().to(device)       # [B]
    assert len(Y.shape) == 1

    logits = model(X)                          # [B=1, O=6]
    pred = logits.argmax(dim=-1)
    preds.append(pred.numpy())
  pred: Frames = np.stack(preds, axis=0)    # [N, O]
  pred = np.expand_dims(pred, axis=-1)      # [N, O, D=1]

  get_metrics(y_test, pred, task=TASK_TYPE)


@torch.inference_mode()
def infer(model:LSTM, x:Frame) -> Frame:
  x = torch.from_numpy(x)
  y = model(x)
  y = y.numpy()
  return y
