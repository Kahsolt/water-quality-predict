#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/21 

from pathlib import Path

import torch

from modules.util import *
from modules.preprocess import *
from modules.typing import *
from modules.models.CRNN import *

TASK_TYPE: TaskType = TaskType(Path(__file__).stem.split('_')[-1])


def init(params:Params, logger:Logger=None) -> CRNN:
  model = CRNN(
    r_type    = params['rnn_type'],
    r_in      = params.get('rnn_in',      1),
    r_out     = params.get('rnn_out',     1),
    r_hidden  = params.get('rnn_hidden',  32),
    r_layers  = params.get('rnn_layers',  1),
    r_dropout = params.get('rnn_dropout', 0.0),
    c_layers  = params.get('cnn_layers',  0),
    c_k       = params.get('cnn_k',       3),
    c_stride  = params.get('cnn_stride',  2),
    c_in      = params.get('cnn_in',      32),
    c_out     = params.get('cnn_out',     32),
    c_hidden  = params.get('cnn_hidden',  32),
    c_act     = params.get('cnn_act',     'leaky_relu'),
  )
  param_cnt = sum([p.numel() for p in model.parameters() if p.requires_grad])
  if logger: logger.info(f'  param_cnt: {param_cnt}')

  return model.to(device)


def train(model:CRNN, dataset:Datasets, params:Params, logger:Logger=None):
  dataloader, optimizer, loss_fn, epochs = prepare_for_train(model, dataset, params)

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
    if logger: logger.info(f'[Epoch: {i}] loss: {loss.item():.7f}')


@torch.inference_mode()
def eval(model:CRNN, dataset:Datasets, params:Params, logger:Logger=None) -> EvalMetrics:
  dataloader, y_test = prepare_for_eval(model, dataset, params)

  preds = []
  model.eval()
  for X, _ in dataloader:
    X = X.to(device)      # [B=1, I, D]
    y_hat = model(X)      # [B=1, O=6]，知 I 推 O
    preds.append(y_hat.cpu().numpy())

  pred: Frames = np.concatenate(preds, axis=0)                  # [N, O=6]
  return get_metrics(y_test.squeeze(axis=-1), pred, task=TASK_TYPE, logger=logger)    # [N, O=6]


@torch.inference_mode()
def infer(model:CRNN, x:Frame, logger:Logger=None) -> Frame:
  x = torch.from_numpy(x).float()
  x = x.to(device)          # [I=96, D=1]
  x = x.unsqueeze(axis=0)   # [B=1, I=96, D=1]
  y = model(x)              # [B=1, O=6]
  y = y.T                   # [O=6, D=1]
  y = y.cpu().numpy()
  return y


def save(model:CRNN, log_dp:Path, logger:Logger=None):
  fp = log_dp / 'model.pth'
  if logger: logger.info(f'save model to {fp}')
  torch.save(model.state_dict(), fp)


def load(model:CRNN, log_dp:Path, logger:Logger=None) -> CRNN:
  fp = log_dp / 'model.pth'
  if logger: logger.info(f'load model from {fp}')
  state_dict = torch.load(fp, map_location=device)
  model.load_state_dict(state_dict)
  return model
