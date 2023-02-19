#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/15 

import os
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
os.chdir('D:/数据F/泰州排口预警/凯发预测/水质预测模型_凯发_1')  #设置文件夹所在路径

import math
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
  for batch_idx, (weekday, hour, COD_TN_NH_TP) in enumerate(train_loader):
    weekday   = weekday  .to(device, non_blocking=True).long()
    hour      = hour     .to(device, non_blocking=True).long()
    COD_TN_NH_TP = COD_TN_NH_TP.to(device, non_blocking=True).float()

    prev_w = weekday  [:, :-1]
    prev_h = hour     [:, :-1]
    prev_d = COD_TN_NH_TP[:, :-1]
    post_d = COD_TN_NH_TP[:, 1:]

    optimizer.zero_grad()

    # (prev_w, prev_h, prev_d) => post_d
    post_d_hat = model(prev_w, prev_h, prev_d, post_d)
    loss = F.mse_loss(post_d_hat, post_d)

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
    for weekday, hour, COD_TN_NH_TP in test_loader:
      weekday   = weekday  .to(device, non_blocking=True).long()
      hour      = hour     .to(device, non_blocking=True).long()
      COD_TN_NH_TP = COD_TN_NH_TP.to(device, non_blocking=True).float()

      prev_w = weekday  [:, :-1]
      prev_h = hour     [:, :-1]
      prev_d = COD_TN_NH_TP[:, :-1]
      post_d = COD_TN_NH_TP[:, 1:]

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
  fvmat, stats = load_data(hp.DATA_FILE)
  train_len=int(len(fvmat)*hp.train_size)
  test_len=int((len(fvmat)-train_len)/2)
  fvmat_train=fvmat[:train_len,:]
  fvmat_test=fvmat[train_len:train_len+test_len,:]
  train_dataset = ResampleDataset(fvmat_train, hp.SEGMENT_SIZE, count=train_len-hp.SEGMENT_SIZE)
  #对数据进行重采样，从第1个开始滚动采样，每个样本包含168h的数据
  test_dataset  = ResampleDataset(fvmat_test, hp.SEGMENT_SIZE, count=test_len-hp.SEGMENT_SIZE)
  #对数据进行重采样，从第1个开始滚动采样，每个样本包含168h的数据
  train_loader  = DataLoader(train_dataset, batch_size=hp.BATCH_SIZE, pin_memory=True)
  #每次加载24个样本，相当于24*168h数据
  test_loader   = DataLoader(test_dataset,  batch_size=hp.BATCH_SIZE, pin_memory=True)

  ''' Model & Optimzer '''
  model = CRNN().to(device)
  optimizer = optim.Adam(model.parameters(), lr=hp.LR, betas=hp.BETAS, weight_decay=hp.WEIGHT_DECAY)
  #lr 学习率

  ''' Train & Eval '''
  for epoch in range(1, hp.EPOCHS + 1):
    train(epoch)
    eval(epoch)
    if epoch % hp.CKPT_INTERVAL == 0:
      save_checkpoint(model, optimizer, hp.LR, epoch, os.path.join(hp.LOG_PATH, f'model_{epoch}.pt'))
#保存模型参数
  save_checkpoint(model, optimizer, hp.LR, epoch, os.path.join(hp.LOG_PATH, 'model_final.pt'))
