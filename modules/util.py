#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/15 

import os
import sys
import random
import yaml
from pathlib import Path
from datetime import datetime
import pickle as pkl
import logging

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import precision_recall_fscore_support

from modules.typing import *

plt.rcParams['font.sans-serif'] = ['SimHei']    # 显示中文
plt.rcParams['axes.unicode_minus'] = False      # 正常显示负号

device = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging

def get_logger(name=None, log_dp=Path('.')) -> Logger:
  global logger

  if logger is logging:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    h = logging.FileHandler(log_dp / 'job.log')
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)

  return logger

def seed_everything(seed:int):
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print('>> extension "sklearn-intelex" enabled for speeding up, comment above line if you found it actually slows down or gets compatiblilty error')
  except ImportError:
    pass

def timestr() -> str:
  return f'{datetime.now()}'.replace(' ', 'T').replace(':', '-').split('.')[0]


def read_csv(fp:str) -> DataFrame:
  return pd.read_csv(fp, encoding='utf-8')

def save_csv(df:DataFrame, fp:str) -> None:
  df.to_csv(fp, encoding='utf-8')


def load_pickle(fp:Path) -> CachedData:
  if not fp.exists(): return
  logger.info(f'  load pickle from {fp}')
  with open(fp, 'rb') as fh:
    return pkl.load(fh)

def save_pickle(data:CachedData, fp:Path):
  logger.info(f'  save pickle to {fp}')
  with open(fp, 'wb') as fh:
    pkl.dump(data, fh)


def load_job(fp:Path) -> Job:
  with open(fp, 'r', encoding='utf-8') as fh:
    return yaml.safe_load(fh)

def save_job(job:Job, fp:Path) -> Job:
  with open(fp, 'w', encoding='utf-8') as fh:
    yaml.safe_dump(job, fh, sort_keys=False)


def get_metrics(truth, pred, task:ModelTask='clf') -> EvalMetrics:
  global logger

  assert task in ['clf', 'rgr'], ValueError(f'unknown task {task!r}')

  if   task == 'clf':
    prec, recall, f1, supp = precision_recall_fscore_support(truth, pred, average='macro')
    logger.info(f'prec:   {prec:.3%}')
    logger.info(f'recall: {recall:.3%}')
    logger.info(f'f1:     {f1:.3%}')
    return prec, recall, f1
  elif task == 'rgr':
    mae = mean_absolute_error(truth, pred)
    mse = mean_squared_error (truth, pred)
    r2  = r2_score           (truth, pred)
    logger.info(f'mae: {mae:.3f}')
    logger.info(f'mse: {mse:.3f}')
    logger.info(f'r2:  {r2:.3f}')
    return mae, mse, r2

def save_figure(fp:Path, title:str=None):
  if not plt.gcf().axes: return

  plt.tight_layout()
  plt.suptitle(title)
  plt.savefig(fp, dpi=400)
  logger.info(f'  save figure to {fp}')


def load_checkpoint(model:PyTorchModel, fp:Path, device=device):
  logger.info(f'load model from {fp}')
  state_dict = torch.load(fp, map_location=device)
  model.load_state_dict(state_dict)
  return model

def save_checkpoint(model:PyTorchModel, fp:Path):
  logger.info(f'save model to {fp}')
  torch.save(model.state_dict(), fp)
