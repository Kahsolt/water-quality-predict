#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/21 

from pathlib import Path

import torch

from modules.util import *
from modules.preprocess import *
from modules.typing import *
from modules.models.RNN import *

TASK_TYPE: ModelTask = Path(__file__).stem.split('_')[-1]


def init(config:Config) -> RNN:
  global logger

  model = RNN(
    rnn_type=config['model'],
    d_in    =config.get('d_in', 1),
    d_out   =config.get('d_out', 1),
    d_hidden=config.get('d_hidden', 32),
    n_layers=config.get('n_layers', 1),
    dropout =config.get('dropout', 0),
  )
  param_cnt = sum([p.numel() for p in model.parameters() if p.requires_grad])
  logger.info(f'  param_cnt: {param_cnt}')

  return model.to(device)


def train(model:RNN, dataset:Datasets, config:Config):
  global logger

  dataloader, optimizer, loss_fn, epochs = prepare_for_train(model, dataset, config)

  model.train()
  for i in range(epochs):
    for X, Y in dataloader:
      X = X.to(device)
      Y = Y.to(device)

      optimizer.zero_grad()
      y_hat = model(X)                    # [B=1, O=6]
      loss = loss_fn(y_hat, Y.squeeze(dim=-1))
      loss.backward()
      optimizer.step()
    logger.info(f'[Epoch: {i}] loss: {loss.item():.7f}')


@torch.inference_mode()
def eval(model:RNN, dataset:Datasets, config:Config):
  global logger

  dataloader, y_test = prepare_for_eval(model, dataset, config)

  preds = []
  model.eval()
  for X, _ in dataloader:
    X = X.to(device)      # [B=1, I, D]
    y_hat = model(X)      # [B=1, O=6]，知 I 推 O
    preds.append(y_hat.cpu().numpy())

  pred: Frames = np.concatenate(preds, axis=0)                  # [N, O=6]
  get_metrics(y_test.squeeze(axis=-1), pred, task=TASK_TYPE)    # [N, O=6]


@torch.inference_mode()
def infer(model:RNN, x:Frame) -> Frame:
  x = torch.from_numpy(x)
  x = x.to(device)          # [I=96, D=1]
  x = x.unsqueeze(axis=0)   # [B=1, I=96, D=1]
  y = model(x)              # [B=1, O=6]
  y = y.cpu().numpy()
  return y


def save(model:RNN, log_dp:Path):
  save_checkpoint(model, log_dp / 'model.pth')


def load(model:RNN, log_dp:Path) -> RNN:
  return load_checkpoint(model, log_dp / 'model.pth')
