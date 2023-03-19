#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/19 

from time import time
from copy import deepcopy
from pathlib import Path
from collections import Counter
from pprint import pformat
from typing import Callable, Any
from traceback import format_exc
from importlib import import_module

import matplotlib.pyplot as plt

from modules import preprocess, transform
from modules.descriptor import *
from modules.dataset import *
from modules.util import *
from modules.typing import *

from config import *


def process(fn:Callable[..., Any]):
  def wrapper(env, *args, **kwargs):
    logger: Logger = env['logger']
    process = fn.__name__.split('_')[-1]
    logger.info(f'>> run process: {process!r}')
    t = time()
    r = fn(env, *args, **kwargs)
    logger.info(f'<< process done ({time() - t:.3f}s)')
    return r
  return wrapper

def require_data_and_model(fn:Callable[..., Any]):
  def wrapper(env, *args, **kwargs):
    job: Descriptor = env['job']
    logger: Logger = env['logger']
    log_dp: Path = env['log_dp']

    # Data
    if 'seq' not in env:
      env['seq'] = load_pickle(log_dp / TRANSFORM_FILE, logger)
      assert env['seq'] is not None

    if 'stats' not in env:
      stats = load_pickle(log_dp / STATS_FILE, logger)
      env['stats'] = stats if stats is not None else []

    if 'dataset' not in env:
      env['dataset'] = load_pickle(log_dp / DATASET_FILE, logger)

    if 'label' not in env:
      env['label'] = load_pickle(log_dp / LABEL_FILE, logger)

    # Model
    if 'manager' not in env:
      model_name = job.get('model/name') ; assert model_name
      manager = import_module(f'modules.models.{model_name}')
      env['manager'] = manager
      logger.info('manager:')
      logger.info(manager)

    if 'model' not in env:
      manager = env['manager']
      model = manager.init(job.get('model/params', {}), logger)
      env['model'] = model
      logger.info('model:')
      logger.info(model)

    return fn(env, *args, **kwargs)
  return wrapper


@process
def process_csv(env:Env):
  logger: Logger = env['logger']
  log_dp: Path = env['log_dp']

  df: TimeSeq = read_csv(env['csv'], logger)   # 总原始数据

  logger.info(f'  found {len(df)} records')
  logger.info(f'  column names({len(df.columns)}): {list(df.columns)}')

  env['df'] = df              # 第一列为时间戳，其余列为数值特征

@process
def process_preprocess(env:Env):
  job: Descriptor = env['job']
  logger: Logger = env['logger']
  log_dp: Path = env['log_dp']

  df: TimeSeq = env['df']
  _, df_r = preprocess.split_time_and_values(df)

  if 'filter T':
    for proc in job.get('preprocess/filter_T', []):
      if not hasattr(preprocess, proc):
        logger.error(f'  preprocessor {proc!r} not found!')
        continue

      try:
        logger.info (f'  apply {proc}...')
        lendf = len(df)
        df: TimeSeq = getattr(preprocess, proc)(df)
        logger.info(f'    {proc}: {lendf} => {len(df)}')
        save_figure(log_dp / f'filter_T_{proc}.png', logger=logger)
      except: logger.error(format_exc())

  if 'project':   # NOTE: this is required!
    proc = job.get('preprocess/project')
    assert proc and isinstance(proc, str)
    assert hasattr(preprocess, proc)
    try:
      logger.info (f'  apply {proc}...')
      T, df = getattr(preprocess, proc)(df)
    except: logger.error(format_exc())

  if 'filter V':
    for proc in job.get('preprocess/filter_V', []):
      if not hasattr(preprocess, proc):
        logger.error(f'  preprocessor {proc!r} not found!')
        continue

      try:
        logger.info (f'  apply {proc}...')
        df: Values = getattr(preprocess, proc)(df)
        save_figure(log_dp / f'filter_V_{proc}.png', logger=logger)
      except: logger.error(format_exc())

  if 'plot timeline':
    plt.clf()
    plt.subplot(211) ; plt.title('original')
    for col in df_r.columns: plt.plot(df_r[col], label=col)
    plt.subplot(212) ; plt.title('preprocessed')
    for col in df.columns: plt.plot(df[col], label=col)
    save_figure(log_dp / 'timeline_preprocess.png', logger=logger)

  if 'plot histogram':
    plt.clf()
    plt.subplot(211) ; plt.title('original')
    for col in df_r.columns: plt.hist(df_r[col], label=col, bins=50)
    plt.subplot(212) ; plt.title('preprocessed')
    for col in df.columns: plt.hist(df[col], label=col, bins=50)
    save_figure(log_dp / 'hist_preprocess.png', logger=logger)

  T: Time = T
  seq: Seq = df.to_numpy().astype(np.float32)
  assert len(T) == len(seq)
  save_pickle(seq, log_dp / PREPROCESS_FILE, logger)

  logger.info(f'  T.shape: {T.shape}')
  logger.info(f'  seq.shape: {seq.shape}')
  logger.info(f'  seq.dtype: {seq.dtype}')

  env['T']   = T
  env['seq'] = seq

@process
def process_dataset(env:Env):
  job: Descriptor = env['job']
  logger: Logger = env['logger']
  log_dp: Path = env['log_dp']

  T: Time  = env['T']
  seq: Seq = env['seq']

  if not job.get('dataset'): return

  # split
  exclusive: bool = job.get('dataset/exclusive', False)
  inputs = seq[:, :-1] if exclusive else seq
  target = seq[:, -1:]
  tgt = target

  # encode tgt
  label = None
  encoder: Encoder = job.get('dataset/encoder')
  if encoder is not None:    # clf
    label = encode_seq(target, T, encoder)
    freq = Counter(label.flatten())
    logger.info(f'  label freq: {freq}')

    # test bad ratio
    tot, bad = 0, 0
    for k, v in freq.items():
      tot += v
      if k > 0: bad += v      # all abnormal cases
    bad_ratio = bad / tot
    freq_min: float = job.get('dataset/freq_min', 0.0) ; assert 0.0 <= freq_min <= 1.0
    if bad_ratio < freq_min:
      logger.info(f'  bad_ratio({bad_ratio}) < freq_min({freq_min}), ignore modeling')
      env['status'] = Status.IGNORED
      return
    else:
      logger.info(f'  bad_ratio: {bad_ratio} (freq_min: {freq_min})')

    tgt = label

  # make slices
  inlen   = job.get('dataset/in')         ; assert inlen   > 0
  outlen  = job.get('dataset/out')        ; assert outlen  > 0
  overlap = job.get('dataset/overlap', 0) ; assert overlap >= 0
  X, Y = slice_frames(inputs, tgt, inlen, outlen, overlap)
  logger.info(f'  dataset')
  logger.info(f'    X: {X.shape}')
  logger.info(f'    Y: {Y.shape}')

  # split dataset
  split   = job.get('dataset/split', 0.2) ; assert 0.0 < split < 1.0
  dataset = ((X_train, Y_train), (X_test, Y_test)) = split_dataset(X, Y)
  logger.info(f'  train split')
  logger.info(f'    input:  {X_train[0].shape}')
  logger.info(f'    target: {Y_train[1].shape}')
  logger.info(f'  test split')
  logger.info(f'    input:  {X_test[0].shape}')
  logger.info(f'    target: {Y_test[1].shape}')

  if label is not None: save_pickle(label, log_dp / LABEL_FILE, logger)

  env['label']   = label
  env['dataset'] = dataset

@process
def process_transform(env:Env):
  job: Descriptor = env['job']
  logger: Logger = env['logger']
  log_dp: Path = env['log_dp']

  if 'extract stats from seq':
    seq: Seq = env['seq']     # [T, D=1]
    seq_r = deepcopy(seq)

    stats: Stats = []    # keep ordered
    for proc in job.get('preprocess/transform', []):
      if not hasattr(transform, proc):
        logger.error(f'  preprocessor {proc!r} not found!')
        continue

      try:
        logger.info (f'  apply {proc}...')
        seq, st = getattr(transform, proc)(seq)
        stats.append((proc, st))
      except:
        logger.error(format_exc())

    if 'plot timeline':
      plt.clf()
      plt.subplot(211) ; plt.title('preprocessed')
      for col in range(seq_r.shape[-1]): plt.plot(seq_r[:, col])
      plt.subplot(212) ; plt.title('transformed')
      for col in range(seq.shape[-1]): plt.plot(seq[:, col])
      save_figure(log_dp / 'timeline_transform.png', logger=logger)

    if 'plot histogram':
      plt.clf()
      plt.subplot(211) ; plt.title('preprocessed')
      for col in range(seq_r.shape[-1]): plt.hist(seq_r[:, col], bins=50)
      plt.subplot(212) ; plt.title('transformed')
      for col in range(seq.shape[-1]): plt.hist(seq[:, col], bins=50)
      save_figure(log_dp / 'hist_transform.png', logger=logger)

    save_pickle(seq, log_dp / TRANSFORM_FILE, logger)
    if stats: save_pickle(stats, log_dp / STATS_FILE, logger)
    
    env['seq']   = seq
    env['stats'] = stats

  if 'reapply stats on dataset' and env.get('dataset'):
    (X_train, y_train), (X_test, y_test) = env['dataset']

    for (proc, st) in env['stats']:
      logger.info(f'  reapply {proc}...')
      proc_fn = getattr(transform, f'{proc}_apply')
      X_train = proc_fn(X_train, *st)
      X_test  = proc_fn(X_test,  *st)
      if env['label'] is None:          # is_task_rgr
        y_train = proc_fn(y_train, *st)
        y_test  = proc_fn(y_test,  *st)

    dataset = (X_train, y_train), (X_test, y_test)
    save_pickle(dataset, env['log_dp'] / DATASET_FILE, logger)

    env['dataset'] = dataset

def target_data(env:Env):
  process_csv(env)
  process_preprocess(env)
  process_dataset(env)
  if env['status'] == Status.IGNORED: return
  process_transform(env)

@require_data_and_model
@process
def target_train(env:Env):
  job: Descriptor = env['job']
  logger: Logger = env['logger']
  log_dp: Path = env['log_dp']

  manager, model = env['manager'], env['model']
  data = env['dataset'] if job.get('dataset') else env['seq']
  manager.train(model, data, job.get('model/params'), logger)
  manager.save(model, log_dp, logger)

@require_data_and_model
@process
def target_eval(env:Env):
  job: Descriptor = env['job']
  logger: Logger = env['logger']
  log_dp: Path = env['log_dp']

  manager, model = env['manager'], env['model']
  model = manager.load(model, log_dp, logger)
  data = env['dataset'] if job.get('dataset') else env['seq']
  stats = manager.eval(model, data, job.get('model/params'), logger)

  task_type = TaskType(job.get('model/name').split('_')[-1])
  if   task_type == TaskType.CLF:
    prec, recall, f1 = stats
    lines = [
      f'prec: {prec}',
      f'recall: {recall}',
      f'f1: {f1}',
    ]
  elif task_type == TaskType.RGR:
    mae, mse, r2 = stats
    lines = [
      f'mae: {mae}',
      f'mse: {mse}',
      f'r2: {r2}',
    ]
  else:
    raise ValueError(f'unknown task type {task_type!r}')

  with open(log_dp / 'scores.txt', 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(lines))


@timer
def run_file(args) -> Status:
  # names
  task_name: str = args.name
  job_name: str = args.job_file.stem
  fullname = f'{task_name}@{job_name}'

  # log_dp
  log_dp = LOG_PATH / task_name / job_name
  logger = None
  if log_dp.exists() and args.no_overwrite:
    logger = get_logger(fullname, log_dp)
    logger.info('ignore due to folder already exists and --no_overwrite enabled')
    return Status.IGNORED

  # job template
  log_dp.mkdir(exist_ok=True, parents=True)
  job = Descriptor.load(args.job_file)
  job.save(log_dp / JOB_FILE)

  # logger
  logger = logger or get_logger(fullname, log_dp)
  logger.info('Job Info:')
  logger.info(pformat(job.cfg))

  seed_everything(fix_seed(job.get('seed', -1)))

  env: Env = {
    'job': job,             # 'job.yaml'
    'logger': logger,       # logger
    'log_dp': log_dp,       # log folder
    'csv': args.csv_file,   # 'data.csv'
    'status': Status.RUNNING,
  }

  job: Descriptor = env['job']
  logger: Logger = env['logger']
  log_dp: Path = env['log_dp']

  targets = args.target.split(',')
  if 'all' in targets: targets = ['data', 'train', 'eval'] 
  for tgt in targets:
    try:
      globals()[f'target_{tgt}'](env)
      if env['status'] == Status.IGNORED:
        return Status.IGNORED
    except:
      logger.error(format_exc())
      return Status.FAILED

  return Status.FINISHED

@timer
def run_folder(args):
  for job_file in args.job_folder.iterdir():
    print(f'>> [run] {job_file}')
    args.job_file = job_file
    run_file(args)


if __name__ == '__main__':
  args = cmd_args()
  assert (args.job_file is None) ^ (args.job_folder is None), 'must specify either --job_file xor --job_folder'

  if args.job_file:   run_file  (args)
  if args.job_folder: run_folder(args)
