#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/15 

import os
import sys
import random
import json
import yaml
import string
import subprocess
from time import time
from pathlib import Path
from datetime import datetime
from copy import deepcopy
import base64
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


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def get_logger(name, log_dp=Path('.')) -> Logger:
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)
  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  h = logging.FileHandler(log_dp / 'job.log')
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)
  return logger


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def fix_seed(seed:int) -> int:
  return seed if seed > 0 else random.randrange(np.iinfo(np.int32).max)

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
  return f'{datetime.now()}'.replace(' ', 'T').replace(':', '-')    # '2023-03-19T18-43-34.485700'

def ts_now() -> int:
  return int(time())

def timer(fn:Callable[..., Any]):
  def wrapper(*args, **kwargs):
    t = time()
    r = fn(*args, **kwargs)
    print(f'All things done in {time() - t:.3f}s')
    return r
  return wrapper

def rand_str(length=4) -> str:
  return ''.join(random.sample(string.ascii_uppercase, length))

def get_fullname(task:str, job:str) -> str:
  return f'{task}@{job}'

def enum_values(enum_cls:Enum) -> List[str]:
  return [e.value for e in enum_cls]


def read_csv(fp:Path, logger:Logger=None) -> DataFrame:
  if logger: logger.info(f'  read csv from {fp}')
  return pd.read_csv(fp, encoding='utf-8')

def save_csv(df:DataFrame, fp:str, logger:Logger=None):
  if logger: logger.info(f'  save csv to {fp}')
  df.to_csv(fp, encoding='utf-8')

def load_pickle(fp:Path, logger:Logger=None) -> CachedData:
  if not fp.exists(): return
  if logger: logger.info(f'  load pickle from {fp}')
  with open(fp, 'rb') as fh:
    return pkl.load(fh)

def save_pickle(data:CachedData, fp:Path, logger:Logger=None):
  if logger: logger.info(f'  save pickle to {fp}')
  with open(fp, 'wb') as fh:
    pkl.dump(data, fh)

def load_yaml(fp:Path, init=None):
  if init is None: assert fp.exists()

  if fp.exists():
    with open(fp, 'r', encoding='utf-8') as fh:
      return yaml.safe_load(fh)
  else:
    return deepcopy(init)

def save_yaml(fp:Path, data):
  with open(fp, 'w', encoding='utf-8') as fh:
    yaml.safe_dump(data, fh, sort_keys=False)

def load_json(fp:Path, init=None):
  if init is None: assert fp.exists()

  if fp.exists():
    with open(fp, 'r', encoding='utf-8') as fh:
      return json.load(fh)
  else:
    return deepcopy(init)

def save_json(fp:Path, data):
  def default_fn(x:Any):
    if isinstance(x, (Target, Status, TaskType)): return x.value
    if isinstance(x, Path): return str(x)
    raise TypeError

  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump(data, fh, sort_keys=False, indent=2, ensure_ascii=False, default=default_fn)


def ndarray_to_bytes(x:np.ndarray) -> Tuple[bytes, Tuple[int, ...]]:
  shape = tuple(x.shape)
  bdata = base64.b64encode(x)
  return bdata, shape

def bytes_to_ndarray(s:str, shape:Tuple[int, ...]) -> np.ndarray:
  bdata = base64.decodebytes(s)
  x = np.frombuffer(bdata)
  x = x.reshape(shape)
  return x


def get_metrics(truth, pred, task:TaskType, logger:Logger=None) -> EvalMetrics:
  if   task == TaskType.CLF:
    prec, recall, f1, supp = precision_recall_fscore_support(truth, pred, average='weighted')
    if logger:
      logger.info(f'prec:   {prec:.3%}')
      logger.info(f'recall: {recall:.3%}')
      logger.info(f'f1:     {f1:.3%}')
    return prec, recall, f1
  elif task == TaskType.RGR:
    mae = mean_absolute_error(truth, pred)
    mse = mean_squared_error (truth, pred)
    r2  = r2_score           (truth, pred)
    if logger:
      logger.info(f'mae: {mae:.3f}')
      logger.info(f'mse: {mse:.3f}')
      logger.info(f'r2:  {r2:.3f}')
    return mae, mse, r2

def save_figure(fp:Path, title:str=None, logger:Logger=None):
  if not plt.gcf().axes: return

  #plt.tight_layout()
  plt.suptitle(title or fp.stem)
  plt.savefig(fp, dpi=400)
  if logger: logger.info(f'  save figure to {fp}')


def frame_left_pad(x:Frame, padlen:int) -> Frame:
  xlen = len(x)
  if xlen < padlen:
    x = np.pad(x, ((padlen - xlen, 0), (0, 0)), mode='edge')
  return x

def frame_shift(x:Frame, y:Frame) -> Frame:
  return np.concatenate((x[len(y):, :], y), axis=0)


def make_zip(src:Path, fp: Path):
  fp.parent.mkdir(exist_ok=True, parents=True)
  cmd = f'7z a -tzip "{fp.absolute()}" "{src.absolute()}"'
  #print(f'>> run: {cmd}')
  p = subprocess.Popen(cmd, shell=True, encoding='utf-8')
  p.wait()
