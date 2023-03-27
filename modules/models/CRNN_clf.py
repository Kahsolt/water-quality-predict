#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/21 

from pathlib import Path

import torch
import torch.nn.functional as F

from modules.util import *
from modules.preprocess import *
from modules.typing import *
from modules.models.CRNN import *
from modules.models.CRNN_rgr import init, save, load     # just proxy by

TASK_TYPE: TaskType = TaskType(Path(__file__).stem.split('_')[-1])


def train(model:CRNN, dataset:Datasets, params:Params, logger:Logger=None):
  dataloader, optimizer, loss_fn, epochs = prepare_for_train(model, dataset, params)

  model.train()
  for i in range(epochs):
    ok, cnt = 0, 0
    for X, Y in dataloader:
      X = X.to(device)
      Y = Y.squeeze().long().to(device)       # [B]
      assert len(Y.shape) == 1

      optimizer.zero_grad()
      logits = model(X)            # [B=32, NC=4]
      loss = loss_fn(logits, Y)
      loss.backward()
      optimizer.step()

      with torch.no_grad():
        pred = logits.argmax(dim=-1)
        ok += (Y == pred).sum().item()
        cnt += len(Y)

    if logger: logger.info(f'[Epoch: {i}] loss: {loss.item():.7f}, accuracy: {ok / cnt:.3%}')


@torch.inference_mode()
def eval(model:CRNN, dataset:Datasets, params:Params, logger:Logger=None) -> EvalMetrics:
  dataloader, y_test = prepare_for_eval(model, dataset, params)
  
  preds = []
  model.eval()
  for X, _ in dataloader:
    X = X.to(device)            # [B=1, I, D] 
    logits = model(X)           # [B=1, NC=4], 知 I 推 1
    pred = logits.argmax(dim=-1)
    preds.append(pred.cpu().numpy())

  pred: Frames = np.stack(preds, axis=0)                            # [N, O]
  return get_metrics(y_test.squeeze(), pred.squeeze(), task=TASK_TYPE, logger=logger)     # [N]


@torch.inference_mode()
def infer(model:CRNN, x:Frame, logger:Logger=None) -> Frame:
  x = torch.from_numpy(x).float().to(device)
  x = x.unsqueeze(axis=0)   # [B=1, I=96, D]
  y = model(x)              # [B=1, NC=4]
  y = y.argmax(dim=-1)      # [B=1]
  y = y.unsqueeze_(dim=-1)  # [B=O=1, D=1]
  y = y.cpu().numpy()
  return y


@torch.inference_mode()
def infer_prob(model:CRNN, x:Frame, logger:Logger=None) -> Frame:
  x = torch.from_numpy(x).float().to(device)
  x = x.unsqueeze(axis=0)   # [B=1, I=96, D]
  y = model(x)              # [B=1, NC=4]
  y = F.softmax(y, dim=-1)  # [B=1, NC=4]
  y = y.cpu().numpy()
  return y
