#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/15 

import os
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import hparam as hp
from CRNN import CRNN
from data import ResampleDataset, load_data
from modules.util import *


''' Env '''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(hp.RANDSEED)
torch.cuda.manual_seed(hp.RANDSEED)


def train(epoch):
  logger.info(f'>> Epoch: {epoch}')

  model.train()
  for batch_idx, (weekday, hour, features) in enumerate(train_loader):
    weekday   = weekday  .to(device, non_blocking=True).long()    # [B, T]
    hour      = hour     .to(device, non_blocking=True).long()    # [B, T]
    features  = [ft.to(device, non_blocking=True).long() for ft in features]    #  N * [B, T]

    prev_w = weekday [:, :-1]
    prev_h = hour    [:, :-1]
    prev_d = [ft[:, :-1] for ft in features]
    post_d = features[0][:, 1:]     # 只预测第一个指标, [B, T-1]

    optimizer.zero_grad()

    # (prev_w, prev_h, prev_d) => post_d
    post_d_hat = model(prev_w, prev_h, prev_d, post_d)
    breakpoint()
    loss = F.cross_entropy(post_d_hat, post_d)

    loss.backward()
    optimizer.step()

    if batch_idx % hp.LOG_INTERVAL == 0:
      logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {}'.format(
        epoch, batch_idx * len(hour), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
  
  sw.add_scalar('train/loss', loss.item(), global_step=epoch)
  logger.info(f'<< [Train]: mse_loss={loss.item()}')


def eval(epoch):
  losses = [ ]

  model.eval()
  with torch.inference_mode():
    for weekday, hour, COD_PH_NH in test_loader:
      weekday   = weekday  .to(device, non_blocking=True).long()
      hour      = hour     .to(device, non_blocking=True).long()
      COD_PH_NH = COD_PH_NH.to(device, non_blocking=True).float()

      prev_w = weekday  [:, :-1]
      prev_h = hour     [:, :-1]
      prev_d = COD_PH_NH[:, :-1]
      post_d = COD_PH_NH[:, 1:]

      post_d_hat = model.generate(prev_w, prev_h, prev_d)
      loss = F.mse_loss(post_d_hat, post_d)

      losses.append(loss)

  sw.add_scalar('test/loss', sum(losses), global_step=epoch)
  logger.info(f'<< [Eval]: mse_loss={sum(losses)}')


if __name__ == '__main__':
  ''' Logger '''
  logger = get_logger(hp.LOG_PATH)
  logger.info({k: getattr(hp, k) for k in dir(hp) if not k.startswith('__')})
  sw = SummaryWriter(hp.LOG_PATH)

  ''' Data '''
  #fvmat, stats = load_data(hp.DATA_FILE)
  fvmat, stats = load_data(hp.QTDATA_FILE)
  train_dataset = ResampleDataset(fvmat, hp.SEGMENT_SIZE, count=hp.TRAIN_DATA_COUNT)
  test_dataset  = ResampleDataset(fvmat, hp.SEGMENT_SIZE, count=hp.TEST_DATA_COUNT)
  train_loader  = DataLoader(train_dataset, batch_size=hp.BATCH_SIZE, pin_memory=True)
  test_loader   = DataLoader(test_dataset,  batch_size=hp.BATCH_SIZE, pin_memory=True)

  ''' Model & Optimzer '''
  model = CRNN().to(device)
  optimizer = optim.Adam(model.parameters(), lr=hp.LR, betas=hp.BETAS, weight_decay=hp.WEIGHT_DECAY)


  ''' Train & Eval '''
  for epoch in range(1, hp.EPOCHS + 1):
    train(epoch)
    eval(epoch)
    if epoch % hp.CKPT_INTERVAL == 0:
      save_checkpoint(model, optimizer, hp.LR, epoch, os.path.join(hp.LOG_PATH, f'model_{epoch}.pt'))

  save_checkpoint(model, optimizer, hp.LR, epoch, os.path.join(hp.LOG_PATH, 'model_final.pt'))
