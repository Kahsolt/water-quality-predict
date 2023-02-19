#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/15 

#from argparse import ArgumentParser
import os 
#os.chdir('D:/数据F/泰州排口预警/凯发预测/水质预测模型_凯发_全_168')  #改路径
from calendar import weekday

import numpy as np
import torch

import hparam as hp
from CRNN import CRNN
from data import load_data
from util import *

from sklearn.metrics import mean_squared_error as mse,mean_absolute_error as mae,r2_score as r2
import pandas as pd

''' Env '''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(hp.RANDSEED)
torch.cuda.manual_seed(hp.RANDSEED)


def infer(offset):
  ''' Data '''
  # split & tensor
  fvmat, stats = load_data(hp.DATA_FILE)
  weekday   = torch.from_numpy(fvmat[:, 0] .astype(np.int32))  .to(device, non_blocking=True).long() .unsqueeze(0)   # [B=1, T]
  hour      = torch.from_numpy(fvmat[:, 1] .astype(np.int32))  .to(device, non_blocking=True).long() .unsqueeze(0)   # [B=1, T]
  COD_TN_NH_TP_PH = torch.from_numpy(fvmat[:, 2:].astype(np.float32)).to(device, non_blocking=True).float().unsqueeze(0)   # [B=1, T, D=3]

  # 离线预测：截取 segment_size-1 这么长的序列作为种子
  prev_w = weekday  [:, offset : offset + hp.SEGMENT_SIZE]      # [B=1, N=segment_size]
  prev_h = hour     [:, offset : offset + hp.SEGMENT_SIZE]      # [B=1, N=segment_size]
  prev_d = COD_TN_NH_TP_PH[:, offset : offset + hp.SEGMENT_SIZE, :]   # [B=1, N=segment_size, D=3]
  
  CODs, TNs, NHs, TPs, PHs = [], [], [], [], []  # source/predicted

  ''' Model '''
  model = CRNN().to(device)
  load_checkpoint('log\\model_200.pt', model)
  ckpt_name = os.path.splitext(os.path.basename('log\\model_200.pt'))[0]

  ''' Predict '''
  model.eval()
  with torch.inference_mode():
    for i in range(hp.INFER_STEPS):
      if i % 10 == 0: print(f'>> gen {i} points')

      post_d_hat = model.generate(prev_w, prev_h, prev_d)

      # 整理预测值
      pred = post_d_hat[:, -1, :]      # 只取最后一帧
      COD, TN, NH, TP, PH = torch.unbind(pred.detach().cpu(), dim=-1)
      CODs.append(COD.item())
      TNs .append(TN .item())
      NHs .append(NH .item())
      TPs .append(TP .item())
      PHs .append(PH .item())
      
      # 抛弃第一帧
      prev_w = prev_w[:, 1:]
      prev_h = prev_h[:, 1:]
      prev_d = prev_d[:, 1:, :]
      # 将预测值追加为最后一帧
      prev_w = torch.cat([prev_w, weekday[:, offset+i+1].unsqueeze(1)], dim=1)
      prev_h = torch.cat([prev_h, hour   [:, offset+i+1].unsqueeze(1)], dim=1)
      prev_d = torch.cat([prev_d, pred                  .unsqueeze(1)], dim=1)
  #plot_predict([CODt,TNt,NHt,TPt,PHt], [CODs,TNs,NHs,TPs,PHs])
  return CODs,TNs,NHs,TPs,PHs

  ''' Evaluate '''
def evaluate(true,forcast):
    '''指标计算'''
    RMSE = np.sqrt(mse(true,forcast)) #均方根误差
    MAE = mae(true,forcast) #平均绝对误差
    R2 = r2(true,forcast)
    return RMSE,MAE,R2

def re_norm(f_norm_vals,stat):
    '''数据反归一化'''
    max_v=stat[1]
    min_v=stat[0]
    '''数据反归一化'''
    return [(v*(max_v-min_v)+min_v) for v in f_norm_vals ]


if __name__ == '__main__':
  fvmat, stats = load_data(hp.DATA_FILE)
  train_len=int(len(fvmat)*hp.train_size)
  test_len=int((len(fvmat)-train_len)/2)
  infer_len=train_len+test_len-hp.SEGMENT_SIZE+2    # 13298
  cod_f=[]
  tn_f=[]
  nh_f=[]
  tp_f=[]
  ph_f=[]
  for i in range(infer_len,len(fvmat)-hp.SEGMENT_SIZE,hp.INFER_STEPS):
      if i+hp.SEGMENT_SIZE+2<len(fvmat):
          lst=[i+hp.SEGMENT_SIZE,i+hp.SEGMENT_SIZE+1,i+hp.SEGMENT_SIZE+2]
          print(lst)
          cod,tn,nh,tp,ph=infer(offset=i)
          #print(cod)
          cod_f.extend(cod[-3:])
          tn_f.extend(tn[-3:])
          nh_f.extend(nh[-3:])
          tp_f.extend(tp[-3:])
          ph_f.extend(ph[-3:])
      else:
          a=3-(i+hp.SEGMENT_SIZE+hp.INFER_STEPS-len(fvmat))
          lst=[(i+hp.SEGMENT_SIZE+s) for s in range(a)]
          print(lst)
          cod,tn,nh,tp,ph=infer(offset=i)
          cod_f.extend(cod[-3:-3+a])
          tn_f.extend(tn[-3:-3+a])
          nh_f.extend(nh[-3:-3+a])
          tp_f.extend(tp[-3:-3+a])
          ph_f.extend(ph[-3:-3+a])
          
  cod_t=fvmat[infer_len+hp.SEGMENT_SIZE:,2]  # (1495,)
  tn_t=fvmat[infer_len+hp.SEGMENT_SIZE:,3]
  nh_t=fvmat[infer_len+hp.SEGMENT_SIZE:,4]
  tp_t=fvmat[infer_len+hp.SEGMENT_SIZE:,5]
  ph_t=fvmat[infer_len+hp.SEGMENT_SIZE:,6]
  cod_rmse,cod_mae,cod_r2=evaluate(cod_t,cod_f)
  tn_rmse,tn_mae,tn_r2=evaluate(tn_t,tn_f)
  nh_rmse,nh_mae,nh_r2=evaluate(nh_t,nh_f)
  tp_rmse,tp_mae,tp_r2=evaluate(tp_t,tp_f)
  ph_rmse,ph_mae,ph_r2=evaluate(ph_t,ph_f)
  result=pd.DataFrame()
  result['因子']=['cod','tn','nh','tp','ph']
  result['RMSE']=[cod_rmse,tn_rmse,nh_rmse,tp_rmse,ph_rmse]
  result['MAE']=[cod_mae,tn_mae,nh_mae,tp_mae,ph_mae]
  result['R2']=[cod_r2,tn_r2,nh_r2,tp_r2,ph_r2]
  result.to_csv('result200.csv',index=False)
  
  #结果图输出
  codf=re_norm(cod_f, stats['COD'])
  tnf=re_norm(tn_f, stats['TN'])
  nhf=re_norm(nh_f, stats['NH'])
  tpf=re_norm(tp_f, stats['TP'])
  phf=re_norm(ph_f, stats['PH'])
  fvmat0, stats = load_data(hp.DATA_FILE1)
  codt=fvmat0[infer_len+hp.SEGMENT_SIZE:,2]
  tnt=fvmat0[infer_len+hp.SEGMENT_SIZE:,3]
  nht=fvmat0[infer_len+hp.SEGMENT_SIZE:,4]
  tpt=fvmat0[infer_len+hp.SEGMENT_SIZE:,5]
  pht=fvmat0[infer_len+hp.SEGMENT_SIZE:,6]
  
  plot_predict([codt,tnt,nht,tpt,pht], [codf,tnf,nhf,tpf,phf], save_fp=os.path.join('log', f'model_200.png'))

result_csv(codf,codt,os.path.join('log', f'cod.csv'))
result_csv(tnf,tnt,os.path.join('log', f'tn.csv'))
result_csv(nhf,nht,os.path.join('log', f'nh.csv'))
result_csv(tpf,tpt,os.path.join('log', f'tp.csv'))
result_csv(phf,pht,os.path.join('log', f'ph.csv')) 