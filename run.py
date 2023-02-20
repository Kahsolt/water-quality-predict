#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/19 

from time import time
from pathlib import Path
from argparse import ArgumentParser
from pprint import pformat
from typing import Callable, Any
from traceback import format_exc
from importlib import import_module

import pandas as pd
import matplotlib.pyplot as plt

from modules.dataset import resample_frame_dataset
from modules.preprocess import *
from modules.util import *
from modules.typing import *

# log folder caches
JOB_FILE     = 'job.yaml'
SEQ_FILE     = 'seq.pkl'
STATS_FILE   = 'stats.pkl'
DATASET_FILE = 'dataset.pkl'

job: Job = None
env: Env = { }
logger: Logger = None

def job_get(path:str, value=None) -> Any:
  global job

  r = job
  for s in path.split('/'):
    if s in r:
      r = r[s]
    else:
      return value
  return r

def job_set(path:str, value=None, overwrite=False) -> Any:
  global job

  segs = path.split('/')
  paths, key = segs[:-1], segs[-1]

  r = job
  for s in paths:
    if s in r:
      r = r[s]
    else:
      logger.error(f'canot find path {path!r}')
      return

  if r[key] is None or overwrite:
    r[key] = value
  return r[key]

def task(fn:Callable[..., Any]):
  def wrapper(*args, **kwargs):
    task = fn.__name__.split('_')[-1]
    logger.info(f'>> run task: {task!r}')
    t = time()
    r = fn(*args, **kwargs)
    logger.info(f'<< task done ({time() - t:.3f}s)')
    return r
  return wrapper

def require_data(fn:Callable[..., Any]):
  def wrapper(*args, **kwargs):
    global env

    if 'seq' not in env:
      env['seq'] = load_pickle(env['log_dp'] / SEQ_FILE)

    if 'stats' not in env:
      stats = load_pickle(env['log_dp'] / STATS_FILE)
      env['stats'] = stats or []

    if 'dataset' not in env:
      env['dataset'] = load_pickle(env['log_dp'] / DATASET_FILE)

    return fn(*args, **kwargs)
  return wrapper

def require_model(fn:Callable[..., Any]):
  def wrapper(*args, **kwargs):
    global env

    if 'manager' not in env:
      model_name = job_get('model/arch') ; assert model_name
      manager = import_module(f'modules.models.{model_name}')
      env['manager'] = manager
    
    if 'model' not in env:
      model = manager.init(job_get('model/params', {}))
      env['model'] = model

    return fn(*args, **kwargs)
  return wrapper


@task
def process_df():
  global job, env

  df: pd.DataFrame = None     # 总原始数据
  T = None                    # 唯一时间轴
  for fp in job_get('data', []):
    try:
      df_new = read_csv(fp)
      cols = list(df_new.columns)
      T_new, data_new = df_new[cols[0]], df_new[cols[1:]]
      if T is None:
        T = T_new
      else:
        if len(T) != len(T_new):
          logger.info(f'>> ignore file {fp!r} due to length mismatch with former ones')
          continue
        if not (T == T_new).all():
          logger.info(f'>> ignore file {fp!r} due to timeline mismatch with former ones')
          continue
      if df is None:
        df = df_new
      else:
        df = pd.concat([df_new, data_new], axis='column', ignore_index=True, copy=False)
    except:
      logger.error(format_exc())

  logger.info(f'  found {len(df)} records')
  logger.info(f'  column names: {list(df.columns)}')

  env['df'] = df              # 第一列为时间戳，其余列为数值特征

@task
def process_seq():
  global job, env

  if env['df'] is None:
    logger.info('>> no data frame prepared for preprocess...')
    return

  df: DataFrame = env['df']
  assert isinstance(df, DataFrame)
  df_r = df.copy(deep=True)

  stats: Stats = []    # keep ordered
  namespace = globals()
  for proc in job_get('preprocess', []):
    if proc not in namespace:
      logger.info(f'>> Error: processor {proc!r} not found!')
      continue
    
    try:
      logger.info(f'  apply {proc}...')

      ret = namespace[proc](df)
      if isinstance(ret, Tuple):
        df, st = ret
        stats.append((proc, st))
      else:
        df = ret
    except:
      logger.error(format_exc())

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

  logger.info(f'  seq.shape: {seq.shape}')
  logger.info(f'  stats: {stats}')

  env['seq']   = seq
  env['stats'] = stats

@task
def process_dataset():
  global job, env

  seq: np.ndarray = env['seq']
  assert isinstance(seq, np.ndarray)

  inlen    = job_get('dataset/in')    ; assert inlen   > 0
  outlen   = job_get('dataset/out')   ; assert outlen  > 0
  n_train  = job_get('dataset/train') ; assert n_train > 0
  n_eval   = job_get('dataset/eval')  ; assert n_eval  > 0
  trainset = resample_frame_dataset(seq, inlen, outlen, n_train)
  evalset  = resample_frame_dataset(seq, inlen, outlen, n_eval)
  dataset  = (trainset, evalset)

  logger.info(f'  train set')
  logger.info(f'    input:  {trainset[0].shape}')
  logger.info(f'    target: {trainset[1].shape}')
  logger.info(f'  eval set')
  logger.info(f'    input:  {evalset[0].shape}')
  logger.info(f'    target: {evalset[1].shape}')

  save_pickle(dataset, env['log_dp'] / DATASET_FILE)

  env['dataset'] = dataset

def target_data():
  process_df()
  process_seq()
  process_dataset()

@require_data
@require_model
@task
def target_train():
  global job, env

  manager, model = env['manager'], env['model']
  manager.train(model, env['dataset'])
  manager.save(model, env['log_dp'])

@require_data
@require_model
@task
def target_eval():
  global job, env

  manager, model = env['manager'], env['model']
  model = manager.load(model, env['log_dp'])
  manager.eval(model, env['dataset'], env['log_dp'])


def run(args):
  global job, env, logger

  job = load_job(args.job_file)

  if 'job init':
    arch = job_get('model/arch') ; assert arch
    auto_name = f'{arch}_{timestr()}'
    name: str = job_set('misc/name', auto_name, overwrite=False)
    seed_everything(job_get('misc/seed', 114514))

  log_dp: Path = args.log_path / name
  log_dp.mkdir(exist_ok=True)
  logger = get_logger(name, log_dp)   # NOTE: assure no print before logger init

  logger.info('Job Info:')
  logger.info(pformat(job))

  env.update({
    'args':   args,
    'log_dp': log_dp,
  })

  targets: List[RunTarget] = job_get('misc/target', ['all'])
  if 'all' in targets: targets = ['data', 'train', 'eval']
  namespace = globals()
  for tgt in targets:
    namespace[f'target_{tgt}']()

  save_job(job, log_dp / JOB_FILE)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-J', '--job_file', required=True, help='path to a *.yaml job file')
  parser.add_argument('--log_path', default=Path('log'), type=Path, help='path to log root folder')
  args = parser.parse_args()

  run(args)
