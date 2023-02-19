#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/19 

import yaml
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime
from time import time
from pprint import pprint as pp
from typing import Dict, Any, Union, Callable
from traceback import print_exc
from importlib import import_module

import pandas as pd
import matplotlib.pyplot as plt

from modules.util import seed_everything
from modules.preprocess import *
from modules.data import load_data, save_data


job: Dict[str, Any] = None
env: Dict[str, Any] = { }

def get(path:str, value=None) -> Union[str, int, float]:
  global job

  r = job
  for s in path.split('/'):
    if s in r:
      r = r[s]
    else:
      return value
  return r

def set(path:str, value=None, overwrite=False) -> Union[str, int, float]:
  global job

  segs = path.split('/')
  paths, key = segs[:-1], segs[-1]

  r = job
  for s in paths:
    if s in r:
      r = r[s]
    else:
      print(f'Error: canot find path {path!r}')
      return

  if r[key] is None or overwrite:
    r[key] = value
  return r[key]

def timestr():
  return f'{datetime.now()}'.replace(' ', 'T').replace(':', '-').split('.')[0]

def task(fn:Callable[..., None]):
  def wrapper(*args, **kwargs):
    global job

    task = fn.__name__.split('_')[-1]
    print(f'>> run task: {task!r}')
    t = time()
    r = fn(*args, **kwargs)
    print(f'<< task done ({time() - t:.3f}s)')

    return r
  return wrapper


@task
def process_data():
  global job, env

  data: pd.DataFrame = None   # 总原始数据
  T = None                    # 唯一时间轴
  for fp in job.get('data', []):
    try:
      df = read_csv(fp)
      cols = list(df.columns)
      T_new, data_new = df[cols[0]], df[cols[1:]]
      if T is None:
        T = T_new
      else:
        if len(T) != len(T_new):
          print(f'>> ignore file {fp!r} due to length mismatch with former ones')
          continue
        if not (T == T_new).all():
          print(f'>> ignore file {fp!r} due to timeline mismatch with former ones')
          continue
      if data is None:
        data = df
      else:
        data = pd.concat([data, data_new], axis='column', ignore_index=True, copy=False)
    except:
      print_exc()

  env['data'] = data          # 第一列为时间戳，其余列为数值特征

  print(f'>> found {len(data)} records')
  print(f'>> column names: {list(data.columns)}')

@task
def process_preprocess():
  global job, env

  if env['data'] is None:
    print('>> no data prepared for preprocess...')
    return

  df: DataFrame = env['data']
  df_r = df.copy(deep=True)

  namespace = globals()
  for proc in job.get('preprocess', []):
    if proc not in namespace:
      print(f'>> Error: processor {proc!r} not found!')
      continue
    
    try:
      print(f'[preprocess] apply {proc}...')

      ret = namespace[proc](df)
      if isinstance(ret, Tuple):
        df, env[proc] = ret
      else:
        df = ret
    except:
      print_exc()
  
  assert isinstance(df, DataFrame)
  env['data'] = df

  if 'plot timeline':
    plt.clf()
    plt.subplot(211) ; plt.title('original')
    for col in df_r.columns[1:]: plt.plot(df_r[col], label=col)
    plt.subplot(212) ; plt.title('preprocessed')
    for col in df.columns: plt.plot(df[col], label=col)
    plt.tight_layout()
    fp = env['log_dp'] / 'timeline.png'
    plt.savefig(fp, dpi=400)
    print(f'savefig to {fp}')

  if 'plot histogram':
    plt.clf()
    plt.subplot(211) ; plt.title('original')
    for col in df_r.columns[1:]: plt.hist(df_r[col], label=col, bins=50)
    plt.subplot(212) ; plt.title('preprocessed')
    for col in df.columns: plt.hist(df[col], label=col, bins=50)
    plt.tight_layout()
    fp = env['log_dp'] / 'hist.png'
    plt.savefig(fp, dpi=400)
    print(f'savefig to {fp}')

@task
def process_dataset():
  global job, env

  df: DataFrame = env['data']
  X = []
  Y = []
  save_data((X, Y), env['log_dp'] / 'data.pkl')

@task
def process_model():
  global job, env

  model_name = get('model/arch')
  manager = import_module(f'modules.models.{model_name}')
  env['manager'] = manager
  model = manager.init(job['model'])
  env['model'] = model

@task
def process_train():
  global job, env

  manager, model = env['manager'], env['model']
  if 'data' not in env: env['data'] = load_data(env['log_dp'] / 'data.pkl')
  manager.train(model, data=env['data'], params=job['train'])
  manager.save(model, fp=env['log_dp'] / 'model.dump')

@task
def process_eval():
  global job, env

  manager, model = env['manager'], env['model']
  manager.load(model, fp=env['log_dp'] / 'model.dump')
  if 'data' not in env: env['data'] = load_data(env['log_dp'] / 'data.pkl')
  manager.eval(model, data=env['data'])


def run(args):
  global job, env

  with open(args.job_file, 'r', encoding='utf-8') as fh:
    job = yaml.safe_load(fh)

  auto_name = f'{get("model/arch")}_{timestr()}'
  name: str = set('misc/name', auto_name, overwrite=False)
  seed_everything(get('misc/seed', 114514))

  print('Job Info:')
  pp(job)

  log_dp = Path('log') / name
  log_dp.mkdir(exist_ok=True)

  env.update({
    'args':   args,
    'data':   None,
    'model':  None,
    'log_dp': log_dp,
  })

  target = get('misc/target', 'all')
  if target == 'all':
    process_data()
    process_preprocess()
    process_dataset()
    process_model()
    process_train()
    process_eval()
  elif target == 'dataset':
    process_data()
    process_preprocess()
    process_dataset()
  elif target == 'train':
    process_model()
    process_train()
  elif target == 'eval':
    process_model()
    process_eval()
  else: raise ValueError(f'unknown target: {target!r}')

  with open(log_dp / 'job.yaml', 'w', encoding='utf-8') as fh:
    yaml.safe_dump(job, fh, sort_keys=False)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-J', '--job_file', required=True, help='path to a *.json file')
  args = parser.parse_args()

  run(args)
