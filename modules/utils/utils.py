#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/15 

import os
import random
import string
import subprocess
from time import time
from threading import RLock
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lock_cuda = RLock()

try:
  from sklearnex import patch_sklearn
  patch_sklearn()
  print('>> extension "sklearn-intelex" enabled for speeding up, comment above line if you found it actually slows down or gets compatiblilty error')
except ImportError:
  print('>> extension "sklearn-intelex" not found, performance may be very slow')

from modules.paths import JOB_PATH
from modules.typing import *

LOG_JOB = os.environ.get('LOG_JOB', False)


def timer(fn:Callable[..., Any]):
  def wrapper(*args, **kwargs):
    t = time()
    r = fn(*args, **kwargs)
    print(f'All things done in {time() - t:.3f}s')
    return r
  return wrapper

def with_lock_cuda(fn:Callable[..., Any]):
  def wrapper(*args, **kwargs):
    with lock_cuda:
      return fn(*args, **kwargs)
  return wrapper


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

def ts_now() -> int:
  return int(time())

def datetime_str() -> str:
  return f'{datetime.now()}'.replace(' ', 'T').replace(':', '-')    # '2023-03-19T18-43-34.485700'

def rand_str(length=4) -> str:
  return ''.join(random.sample(string.ascii_uppercase, length))

def enum_values(enum_cls:Enum) -> List[str]:
  return [e.value for e in enum_cls]


def fix_targets(targets:Union[List[str], str, None]) -> List[str]:
  if not targets: return ['all']
  if isinstance(targets, str): targets = [targets]
  valid_tgts = enum_values(Target)
  for tgt in targets: assert tgt in valid_tgts
  return targets

def fix_jobs(jobs:Union[List[str], None]) -> List[str]:
  valid_jobs = [job.stem for job in JOB_PATH.iterdir() if job.suffix == '.yaml']
  if not jobs: return valid_jobs
  for job in jobs: assert job in valid_jobs
  return jobs


def get_fullname(task:str, job:str) -> str:
  return f'{task}@{job}'

def make_zip(src:Path, fp:Path):
  fp.parent.mkdir(exist_ok=True, parents=True)
  cmd = f'7z a -tzip "{fp.absolute()}" "{src.absolute()}"'
  #print(f'>> run: {cmd}')
  p = subprocess.Popen(cmd, shell=True, encoding='utf-8')
  p.wait()
