#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/19 

from time import time
from copy import deepcopy
from pathlib import Path
from argparse import ArgumentParser
from collections import Counter
from pprint import pformat
from typing import Callable, Any
from traceback import format_exc
from importlib import import_module

import pandas as pd
import matplotlib.pyplot as plt

from modules import preprocess
from modules.dataset import *
from modules.util import *
from modules.typing import *

# log folder caches
JOB_FILE     = 'job.yaml'
SEQ_RAW_FILE = 'seq-raw.pkl'    # seq preprocessed
LABEL_FILE   = 'label.pkl'      # seq label for 'clf'
STATS_FILE   = 'stats.pkl'      # transforming stats for seq
SEQ_FILE     = 'seq.pkl'        # seq transformed
DATASET_FILE = 'dataset.pkl'    # dataset transformed

job: Job = None
env: Env = { }
logger: Logger = logger

def job_get(path:str, value=None) -> Any:
  global job

  r = job
  for s in path.split('/'):
    if s in r:
      r = r[s]
    else:
      return value
  return r or value

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
      seq = load_pickle(env['log_dp'] / SEQ_FILE)
      assert seq is not None
      env['seq'] = seq

    if 'stats' not in env:
      stats = load_pickle(env['log_dp'] / STATS_FILE)
      env['stats'] = stats if stats is not None else []

    if 'dataset' not in env:
      dataset = load_pickle(env['log_dp'] / DATASET_FILE)
      assert dataset is not None
      env['dataset'] = dataset

    if 'label' not in env:
      label = load_pickle(env['log_dp'] / LABEL_FILE)
      env['label'] = label

    return fn(*args, **kwargs)
  return wrapper

def require_model(fn:Callable[..., Any]):
  def wrapper(*args, **kwargs):
    global env

    if 'manager' not in env:
      model_name = job_get('model/name') ; assert model_name
      manager = import_module(f'modules.models.{model_name}')
      env['manager'] = manager
      logger.info('model unit:')
      logger.info(manager)

    if 'model' not in env:
      manager = env['manager']
      model = manager.init(job_get('model/config', {}))
      env['model'] = model
      logger.info('model arch:')
      logger.info(model)

    return fn(*args, **kwargs)
  return wrapper

def defined_preprocessor(proc:str) -> bool:
  found = hasattr(preprocess, proc)
  if not found: logger.error(f'  preprocessor {proc!r} not found!')
  else:         logger.info (f'  apply {proc}...')
  return found


@task
def process_df():
  global job, env

  df: TimeSeq = None       # 总原始数据
  T = None                    # 唯一时间轴
  for fp in job_get('data'):
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

  df: TimeSeq = env['df']
  _, df_r = preprocess.split_time_and_data(df)

  if 'filter T':
    for proc in job_get('preprocess/filter_T', []):
      if not defined_preprocessor(proc): continue
      try:
        df: TimeSeq = getattr(preprocess, proc)(df)
        save_figure(env['log_dp'] / f'filter_T_{proc}.png')
      except: logger.error(format_exc())

  if 'project':   # NOTE: this is required!
    proc = job_get('preprocess/project')
    assert proc and len(proc) == 1
    proc = proc[0]
    assert defined_preprocessor(proc)
    try:    T, df = getattr(preprocess, proc)(df)
    except: logger.error(format_exc())

  if 'filter V':
    for proc in job_get('preprocess/filter_V', []):
      if not defined_preprocessor(proc): continue
      try:
        df: Data = getattr(preprocess, proc)(df)
        save_figure(env['log_dp'] / f'filter_V_{proc}.png')
      except: logger.error(format_exc())

  if 'plot timeline':
    plt.clf()
    plt.subplot(211) ; plt.title('original')
    for col in df_r.columns: plt.plot(df_r[col], label=col)
    plt.subplot(212) ; plt.title('preprocessed')
    for col in df.columns: plt.plot(df[col], label=col)
    save_figure(env['log_dp'] / 'timeline.png')

  if 'plot histogram':
    plt.clf()
    plt.subplot(211) ; plt.title('original')
    for col in df_r.columns: plt.hist(df_r[col], label=col, bins=50)
    plt.subplot(212) ; plt.title('preprocessed')
    for col in df.columns: plt.hist(df[col], label=col, bins=50)
    save_figure(env['log_dp'] / 'hist.png')

  T: Time = T
  seq: Seq = df.to_numpy().astype(np.float32)
  assert len(T) == len(seq)
  save_pickle(seq, env['log_dp'] / SEQ_RAW_FILE)

  logger.info(f'  T.shape: {T.shape}')
  logger.info(f'  seq.shape: {seq.shape}')
  logger.info(f'  seq.dtype: {seq.dtype}')

  env['T']   = T
  env['seq'] = seq

@task
def process_dataset():
  global job, env

  T: Time  = env['T']
  seq: Seq = env['seq']

  encode: Encode = job_get('dataset/encode')
  if encode is not None:    # clf
    label = encode_seq(seq, T, encode)
    freq = Counter(label.flatten())
    logger.info(f'  label freq: {freq}')
  else:
    label = None

  inlen   = job_get('dataset/in')    ; assert inlen   > 0
  outlen  = job_get('dataset/out')   ; assert outlen  > 0
  n_train = job_get('dataset/train') ; assert n_train > 0
  n_eval  = job_get('dataset/eval')  ; assert n_eval  > 0
  trainset = resample_frame_dataset(seq, inlen, outlen, n_train, y=label)
  evalset  = resample_frame_dataset(seq, inlen, outlen, n_eval,  y=label)

  logger.info(f'  train set')
  logger.info(f'    input:  {trainset[0].shape}')
  logger.info(f'    target: {trainset[1].shape}')
  logger.info(f'  eval set')
  logger.info(f'    input:  {evalset[0].shape}')
  logger.info(f'    target: {evalset[1].shape}')

  dataset = (trainset, evalset)
  save_pickle(dataset, env['log_dp'] / DATASET_FILE)
  if label is not None: save_pickle(label, env['log_dp'] / LABEL_FILE)
  
  env['label']   = label
  env['dataset'] = dataset

@task
def process_transform():
  if 'extract stats from seq':
    seq: Seq = env['seq']     # [T, D=1]
    seq_r = deepcopy(seq)

    stats: Stats = []    # keep ordered
    for proc in job_get('preprocess/transform', []):
      if not defined_preprocessor(proc): continue
      try:
        seq, st = getattr(preprocess, proc)(seq)
        stats.append((proc, st))
      except:
        logger.error(format_exc())

    if 'plot timeline T':
      plt.clf()
      plt.subplot(211) ; plt.title('preprocessed')
      for col in range(seq_r.shape[-1]): plt.plot(seq_r[:, col])
      plt.subplot(212) ; plt.title('transformed')
      for col in range(seq.shape[-1]): plt.plot(seq[:, col])
      save_figure(env['log_dp'] / 'timeline_T.png')

    if 'plot histogram T':
      plt.clf()
      plt.subplot(211) ; plt.title('preprocessed')
      for col in range(seq_r.shape[-1]): plt.hist(seq_r[:, col], bins=50)
      plt.subplot(212) ; plt.title('transformed')
      for col in range(seq.shape[-1]): plt.hist(seq[:, col], bins=50)
      save_figure(env['log_dp'] / 'hist_T.png')

    save_pickle(seq, env['log_dp'] / SEQ_FILE)
    if stats: save_pickle(stats, env['log_dp'] / STATS_FILE)
    
    env['seq']   = seq
    env['stats'] = stats

  if 'reapply stats on dataset':
    (X_train, y_train), (X_test, y_test) = env['dataset']

    for (proc, st) in env['stats']:
      logger.info(f'  reapply {proc}...')
      proc_fn = getattr(preprocess, f'{proc}_apply')
      X_train = proc_fn(X_train, *st)
      X_test  = proc_fn(X_test,  *st)
      if env['label'] is None:          # is_task_rgr
        y_train = proc_fn(y_train, *st)
        y_test  = proc_fn(y_test,  *st)

    dataset = (X_train, y_train), (X_test, y_test)
    save_pickle(dataset, env['log_dp'] / DATASET_FILE)

    env['dataset'] = dataset

def target_data():
  process_df()
  process_seq()
  process_dataset()
  process_transform()

@require_data
@require_model
@task
def target_train():
  global job, env

  manager, model = env['manager'], env['model']
  manager.train(model, env['dataset'], job_get('model/config'))
  manager.save(model, env['log_dp'])

@require_data
@require_model
@task
def target_eval():
  global job, env

  manager, model = env['manager'], env['model']
  model = manager.load(model, env['log_dp'])
  manager.eval(model, env['dataset'], job_get('model/config'))

  
def run(args):
  global job, env, logger

  job = load_job(args.job_file)

  if 'job init':
    model_name = job_get('model/name') ; assert model_name
    auto_name = f'{model_name}_{timestr()}'
    name: str = job_set('misc/name', auto_name, overwrite=False)
    seed_everything(job_get('misc/seed', 114514))

  log_dp: Path = args.log_path / name
  if log_dp.exists() and args.no_overwrite: return
  log_dp.mkdir(exist_ok=True, parents=True)
  logger = get_logger(name, log_dp)   # NOTE: assure no print before logger init

  logger.info('Job Info:')
  logger.info(pformat(job))

  env.update({
    'args':   args,
    'log_dp': log_dp,
  })

  targets: List[Target] = job_get('misc/target', ['all'])
  if 'all' in targets:
    targets = ['data', 'train', 'eval']
  for tgt in targets:
    globals()[f'target_{tgt}']()

  save_job(job, log_dp / JOB_FILE)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-J', '--job_file', required=True, help='path to a *.yaml job file')
  parser.add_argument('--log_path', default=Path('log'), type=Path, help='path to log root folder')
  parser.add_argument('--no_overwrite', action='store_true', help='no overwrite if log folder exists')
  args = parser.parse_args()

  run(args)
