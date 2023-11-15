#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/15 

import json
import yaml
import base64
from pathlib import Path
from copy import deepcopy
import pickle as pkl

import numpy as np
import pandas as pd

from modules.typing import *

Data = Union[dict, list]


def read_txt(fp:Path, logger:Logger=None) -> List[str]:
  if logger: logger.info(f'  read txt from {fp}')
  with open(fp, 'r', encoding='utf-8') as fh:
    return fh.read().strip().split('\n')

def save_txt(lines:List[str], fp:Path, logger:Logger=None):
  if logger: logger.info(f'  save txt to {fp}')
  with open(fp, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(lines))

def read_csv(fp:Path, logger:Logger=None) -> DataFrame:
  if logger: logger.info(f'  read csv from {fp}')
  return pd.read_csv(fp, encoding='utf-8')

def save_csv(df:DataFrame, fp:Path, logger:Logger=None):
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

def load_yaml(fp:Path, init:Any=None):
  if init is None: assert fp.exists()

  if fp.exists():
    with open(fp, 'r', encoding='utf-8') as fh:
      return yaml.safe_load(fh)
  else:
    return deepcopy(init)

def save_yaml(data:Data, fp:Path):
  with open(fp, 'w', encoding='utf-8') as fh:
    yaml.safe_dump(data, fh, sort_keys=False)

def _type_cvt_json(x:Any):
  if isinstance(x, (Target, Status, TaskType)): return x.value
  if isinstance(x, Path): return str(x)
  raise TypeError

def load_json(fp:Path, init:Any=None):
  if init is None: assert fp.exists()

  if fp.exists():
    with open(fp, 'r', encoding='utf-8') as fh:
      return json.load(fh)
  else:
    return deepcopy(init)

def save_json(data:Data, fp:Path):
  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump(data, fh, sort_keys=False, indent=2, ensure_ascii=False, default=_type_cvt_json)


def serialize_json(data:Data) -> Any:
  return json.loads(json.dumps(data, sort_keys=False, indent=0, ensure_ascii=False, default=_type_cvt_json))


def ndarray_to_base64(x:ndarray) -> Tuple[str, Tuple[int, ...]]:
  shape = tuple(x.shape)
  bdata = base64.b64encode(x.astype(np.float32))
  sdata = str(bdata, encoding='utf-8')
  return sdata, shape

def base64_to_ndarray(s:str, shape:Tuple[int, ...]) -> ndarray:
  sdata = s.encode('utf-8')
  bdata = base64.decodebytes(sdata)
  x = np.frombuffer(bdata, dtype=np.float32)
  x = x.reshape(shape)
  return x

def ndarray_to_list(x:ndarray) -> List[List[float]]:
  return x.tolist()

def list_to_ndarray(ls:List[List[float]]) -> np.ndarray:
  return np.asarray(ls, dtype=np.float32)
