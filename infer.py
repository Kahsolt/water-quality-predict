#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/15 

from argparse import ArgumentParser
from calendar import weekday

import numpy as np
import torch

import hparam as hp
from model import CRNN
from data import load_data
from util import *


''' Env '''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(hp.RANDSEED)
torch.cuda.manual_seed(hp.RANDSEED)


def infer(args):
  ''' Data '''
  # split & tensor
  fvmat, stats = load_data(hp.DATA_FILE)
  weekday   = torch.from_numpy(fvmat[:, 0] .astype(np.int32))  .to(device, non_blocking=True).long() .unsqueeze(0)   # [B=1, T]
  hour      = torch.from_numpy(fvmat[:, 1] .astype(np.int32))  .to(device, non_blocking=True).long() .unsqueeze(0)   # [B=1, T]
  COD_PH_NH = torch.from_numpy(fvmat[:, 2:].astype(np.float32)).to(device, non_blocking=True).float().unsqueeze(0)   # [B=1, T, D=3]

  # 离线预测：截取 segment_size-1 这么长的序列作为种子
  offset = np.random.randint(len(fvmat) - hp.INFER_STEPS - 1)
  prev_w = weekday  [:, offset : offset + hp.SEGMENT_SIZE]      # [B=1, N=segment_size]
  prev_h = hour     [:, offset : offset + hp.SEGMENT_SIZE]      # [B=1, N=segment_size]
  prev_d = COD_PH_NH[:, offset : offset + hp.SEGMENT_SIZE, :]   # [B=1, N=segment_size, D=3]
  
  CODt, PHt, NHt = [], [], []  # target/truth
  CODs, PHs, NHs = [], [], []  # source/predicted
  
  for point in prev_d[0, :, :]:
    point = point.cpu().numpy()
    CODt.append(point[0]) ; CODs.append(point[0])
    PHt .append(point[1]) ; PHs .append(point[1])
    NHt .append(point[2]) ; NHs .append(point[2])

  ''' Model '''
  model = CRNN().to(device)
  load_checkpoint(args.ckpt_fp, model)
  ckpt_name = os.path.splitext(os.path.basename(args.ckpt_fp))[0]

  ''' Predict '''
  model.eval()
  with torch.inference_mode():
    for i in range(hp.INFER_STEPS):
      if i % 10 == 0: print(f'>> gen {i} points')

      post_d_hat = model.generate(prev_w, prev_h, prev_d)

      # 整理预测值
      pred = post_d_hat[:, -1, :]      # 只取最后一帧
      COD, PH, NH = torch.unbind(pred.detach().cpu(), dim=-1)
      CODs.append(COD.item())
      PHs .append(PH .item())
      NHs .append(NH .item())

      # 整理真实值
      truth = COD_PH_NH[:, offset + hp.SEGMENT_SIZE + i + 1]
      COD, PH, NH = torch.unbind(truth.detach().cpu(), dim=-1)
      CODt.append(COD.item())
      PHt .append(PH .item())
      NHt .append(NH .item())

      # 抛弃第一帧
      prev_w = prev_w[:, 1:]
      prev_h = prev_h[:, 1:]
      prev_d = prev_d[:, 1:, :]
      # 将预测值追加为最后一帧
      prev_w = torch.cat([prev_w, weekday[:, offset+i+1].unsqueeze(1)], dim=1)
      prev_h = torch.cat([prev_h, hour   [:, offset+i+1].unsqueeze(1)], dim=1)
      prev_d = torch.cat([prev_d, pred                  .unsqueeze(1)], dim=1)

  plot_predict([CODt, PHt, NHt], [CODs, PHs, NHs], save_fp=os.path.join('log', f'{ckpt_name}.png'))

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('ckpt_fp', help='path to checkpoint file')
  args = parser.parse_args()
  infer(args)
