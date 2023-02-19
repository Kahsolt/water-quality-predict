#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/19 

import yaml
from time import time
from pathlib import Path
from argparse import ArgumentParser
from pprint import pprint as pp
from typing import Callable, Any
from traceback import print_exc
from importlib import import_module

import pandas as pd
import matplotlib.pyplot as plt

from modules.util import seed_everything, timestr, save_figure, load_pickle, save_pickle
from modules.data import resample_frame_dataset
from modules.preprocess import *
from modules.typing import *

# log folder caches
JOB_FILE     = 'job.yaml'
SEQ_FILE     = 'seq.pkl'
STATS_FILE   = 'stats.pkl'
DATASET_FILE = 'dataset.pkl'

job: Job = None
env: Env = { }

def get(path:str, value=None) -> Any:
  global job

  r = job
  for s in path.split('/'):
    if s in r:
      r = r[s]
    else:
      return value
  return r

def set(path:str, value=None, overwrite=False) -> Any:
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

def task(fn:Callable[..., Any]):
  def wrapper(*args, **kwargs):
    task = fn.__name__.split('_')[-1]
    print(f'>> run task: {task!r}')
    t = time()
    r = fn(*args, **kwargs)
    print(f'<< task done ({time() - t:.3f}s)')
    return r
  return wrapper

def require_data(fn:Callable[..., Any]):
  def wrapper(*args, **kwargs):
    global env

    if 'seq'     not in env: env['seq']     = load_pickle(env['log_dp'] / SEQ_FILE)
    if 'stats'   not in env: env['stats']   = load_pickle(env['log_dp'] / STATS_FILE)
    if 'dataset' not in env: env['dataset'] = load_pickle(env['log_dp'] / DATASET_FILE)

    if env['stats'] is None: env['stats'] = []

    return fn(*args, **kwargs)
  return wrapper


@task
def process_df():
  global job, env

  df: pd.DataFrame = None     # 总原始数据
  T = None                    # 唯一时间轴
  for fp in get('data', []):
    try:
      df_new = read_csv(fp)
      cols = list(df_new.columns)
      T_new, data_new = df_new[cols[0]], df_new[cols[1:]]
      if T is None:
        T = T_new
      else:
        if len(T) != len(T_new):
          print(f'>> ignore file {fp!r} due to length mismatch with former ones')
          continue
        if not (T == T_new).all():
          print(f'>> ignore file {fp!r} due to timeline mismatch with former ones')
          continue
      if df is None:
        df = df_new
      else:
        df = pd.concat([df_new, data_new], axis='column', ignore_index=True, copy=False)
    except:
      print_exc()

  print(f'>> found {len(df)} records')
  print(f'>> column names: {list(df.columns)}')

  env['df'] = df              # 第一列为时间戳，其余列为数值特征

@task
def process_seq():
  global job, env

  if env['df'] is None:
    print('>> no data frame prepared for preprocess...')
    return

  df: DataFrame = env['df']
  assert isinstance(df, DataFrame)
  df_r = df.copy(deep=True)

  stats = []    # keep ordered
  namespace = globals()
  for proc in get('preprocess', []):
    if proc not in namespace:
      print(f'>> Error: processor {proc!r} not found!')
      continue
    
    try:
      print(f'  apply {proc}...')

      ret = namespace[proc](df)
      if isinstance(ret, Tuple):
        df, st = ret
        stats.append((proc, st))
      else:
        df = ret
    except:
      print_exc()

  if 'plot timeline':
    plt.clf()
    plt.subplot(211) ; plt.title('original')
    for col in df_r.columns[1:]: plt.plot(df_r[col], label=col)   # ignore T
    plt.subplot(212) ; plt.title('preprocessed')
    for col in df.columns: plt.plot(df[col], label=col)
    save_figure(env['log_dp'] / 'timeline.png')

  if 'plot histogram':
    plt.clf()
    plt.subplot(211) ; plt.title('original')
    for col in df_r.columns[1:]: plt.hist(df_r[col], label=col, bins=50)  # ignore T
    plt.subplot(212) ; plt.title('preprocessed')
    for col in df.columns: plt.hist(df[col], label=col, bins=50)
    save_figure(env['log_dp'] / 'hist.png')

  seq: np.ndarray = df.to_numpy()
  assert len(seq.shape) == 2
  save_pickle(seq, env['log_dp'] / SEQ_FILE)
  if stats: save_pickle(stats, env['log_dp'] / STATS_FILE)

  print(f'  seq.shape: {seq.shape}')
  print(f'  stats: {stats}')

  env['seq']   = seq
  env['stats'] = stats

@task
def process_dataset():
  global job, env

  seq: np.ndarray = env['seq']
  assert isinstance(seq, np.ndarray)

  inlen    = get('dataset/in')    ; assert inlen   > 0
  outlen   = get('dataset/out')   ; assert outlen  > 0
  n_train  = get('dataset/train') ; assert n_train > 0
  n_eval   = get('dataset/eval')  ; assert n_eval  > 0
  trainset = resample_frame_dataset(seq, inlen, outlen, n_train)
  evalset  = resample_frame_dataset(seq, inlen, outlen, n_eval)
  dataset  = (trainset, evalset)

  print(f'  train set')
  print(f'    input:  {trainset[0].shape}')
  print(f'    target: {trainset[1].shape}')
  print(f'  eval set')
  print(f'    input:  {evalset[0].shape}')
  print(f'    target: {evalset[1].shape}')

  save_pickle(dataset, env['log_dp'] / DATASET_FILE)

  env['dataset'] = dataset

@task
def process_model():
  global job, env

  model_name = get('model/arch') ; assert model_name
  manager = import_module(f'modules.models.{model_name}')
  env['manager'] = manager
  model = manager.init(get('model/params'))
  env['model'] = model

@require_data
@task
def process_train():
  global job, env

  manager, model = env['manager'], env['model']
  manager.train(model, env['dataset'])
  manager.save(model, env['log_dp'])

@require_data
@task
def process_eval():
  global job, env

  manager, model = env['manager'], env['model']
  model = manager.load(model, env['log_dp'])
  manager.eval(model, env['dataset'], env['log_dp'])

@require_data
@task
def process_infer():
  global job, env

  manager, model = env['manager'], env['model']
  model = manager.load(model, env['log_dp'])
  manager.infer(model, env['seq'], env['stats'])


def run(args):
  global job, env

  with open(args.job_file, 'r', encoding='utf-8') as fh:
    job = yaml.safe_load(fh)

  arch = get('model/arch') ; assert arch
  auto_name = f'{arch}_{timestr()}'
  name: str = set('misc/name', auto_name, overwrite=False)
  seed_everything(get('misc/seed', 114514))

  print('Job Info:')
  pp(job)

  log_dp: Path = args.log_path / name
  log_dp.mkdir(exist_ok=True)

  env.update({
    'args':   args,
    'log_dp': log_dp,
  })

  target: RunTarget = get('misc/target', 'all')
  if   target == 'all':
    process_df()
    process_seq()
    process_dataset()
    process_model()
    process_train()
    process_eval()
    process_infer()
  elif target == 'data':
    process_df()
    process_seq()
    process_dataset()
  elif target == 'train':
    process_model()
    process_train()
  elif target == 'eval':
    process_model()
    process_eval()
  elif target == 'infer':
    process_model()
    process_infer()
  else: raise ValueError(f'unknown target: {target!r}')

  with open(log_dp / JOB_FILE, 'w', encoding='utf-8') as fh:
    yaml.safe_dump(job, fh, sort_keys=False)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-J', '--job_file', required=True, help='path to a *.yaml job file')
  parser.add_argument('--log_path', default=Path('log'), type=Path, help='path to log root folder')
  args = parser.parse_args()

  run(args)
