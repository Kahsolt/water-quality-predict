#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/11/15 

from time import time
from copy import deepcopy
from pathlib import Path
from argparse import ArgumentParser
from collections import Counter
from pprint import pformat
from traceback import format_exc

from modules import preprocess, transform
from modules.predictor import Env, prepare_for_, predict_with_oracle, predict_with_prediction
from modules.dataset import encode_seq, slice_frames, split_dataset, check_label_presented_cover_expected
from modules.utils import *


def process(fn:Callable[..., Any]):
  def wrapper(env:Env, *args, **kwargs):
    logger: Logger = env.logger
    process = fn.__name__.split('_')[-1]
    logger.info(f'>> run process: {process!r}')
    t = time()
    r = fn(env, *args, **kwargs)
    logger.info(f'<< process done ({time() - t:.3f}s)')
    return r
  return wrapper


@process
def process_csv(env:Env):
  logger: Logger = env.logger

  df: TimeSeq = read_csv(env.csv, logger)   # 总原始数据
  for col in df.columns[1:]: df[col] = df[col].astype('float32')

  logger.info(f'  found {len(df)} records')
  logger.info(f'  column names({len(df.columns)}): {list(df.columns)}')

  env.df = df              # 第一列为时间戳，其余列为数值特征

@process
def process_preprocess(env:Env):
  job: Config = env.job
  logger: Logger = env.logger
  log_dp: Path = env.log_dp

  df: TimeSeq = env.df
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

  if DEBUG_PLOT and 'plot timeline':
    plt.clf()
    plt.subplot(211) ; plt.title('original')
    for col in df_r.columns: plt.plot(df_r[col], label=col)
    plt.subplot(212) ; plt.title('preprocessed')
    for col in df.columns: plt.plot(df[col], label=col)
    save_figure(log_dp / 'timeline_preprocess.png', logger=logger)

  if DEBUG_PLOT and 'plot histogram':
    plt.clf()
    plt.subplot(211) ; plt.title('original')
    for col in df_r.columns: plt.hist(df_r[col], label=col, bins=50)
    plt.subplot(212) ; plt.title('preprocessed')
    for col in df.columns: plt.hist(df[col], label=col, bins=50)
    save_figure(log_dp / 'hist_preprocess.png', logger=logger)

  T: Time = T
  seq: Seq = df.to_numpy().astype(np.float32)
  assert len(T) == len(seq)
  save_pickle(T,   log_dp / TIME_FILE,       logger)
  save_pickle(seq, log_dp / PREPROCESS_FILE, logger)

  logger.info(f'  T.shape: {T.shape}')
  logger.info(f'  seq.shape: {seq.shape}')
  logger.info(f'  seq.dtype: {seq.dtype}')

  env.T   = T
  env.seq = seq

@process
def process_dataset(env:Env):
  job: Config = env.job
  logger: Logger = env.logger
  log_dp: Path = env.log_dp

  T: Time  = env.T
  seq: Seq = env.seq

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
      env.status = Status.IGNORED
      return
    else:
      logger.info(f'  bad_ratio: {bad_ratio} (freq_min: {freq_min})')

    tgt = label

  # make slices
  inlen   = job.get('dataset/inlen')      ; assert inlen   > 0
  outlen  = job.get('dataset/outlen')     ; assert outlen  > 0
  overlap = job.get('dataset/overlap', 0)
  X, Y = slice_frames(inputs, tgt, inlen, outlen, overlap)
  logger.info(f'  dataset')
  logger.info(f'    X: {X.shape}')
  logger.info(f'    Y: {Y.shape}')

  # split dataset
  split   = job.get('dataset/split', 0.2) ; assert 0.0 < split < 1.0
  trainset, testset = split_dataset(X, Y)
  if encoder is not None:    # clf, fix label not present in data
    trainset = check_label_presented_cover_expected(trainset, encoder['name'])
    testset  = check_label_presented_cover_expected(testset,  encoder['name'])
  dataset = (X_train, Y_train), (X_test, Y_test) = trainset, testset
  logger.info(f'  train split')
  logger.info(f'    input:  {X_train.shape}')
  logger.info(f'    target: {Y_train.shape}')
  logger.info(f'  test split')
  logger.info(f'    input:  {X_test.shape}')
  logger.info(f'    target: {Y_test.shape}')

  if label is not None: save_pickle(label, log_dp / LABEL_FILE, logger)

  env.label   = label
  env.dataset = dataset

@process
def process_transform(env:Env):
  job: Config = env.job
  logger: Logger = env.logger
  log_dp: Path = env.log_dp

  if 'extract stats from seq':
    seq: Seq = env.seq     # [T, D=1]
    seq_r = deepcopy(seq)

    stats: Stats = []    # keep ordered
    for proc in job.get('preprocess/transform', []):
      logger.info (f'  apply {proc}...')
      seq, st = getattr(transform, proc)(seq)
      stats.append((proc, st))

    if DEBUG_PLOT and 'plot timeline':
      plt.clf()
      plt.subplot(211) ; plt.title('preprocessed')
      for col in range(seq_r.shape[-1]): plt.plot(seq_r[:, col])
      plt.subplot(212) ; plt.title('transformed')
      for col in range(seq.shape[-1]): plt.plot(seq[:, col])
      save_figure(log_dp / 'timeline_transform.png', logger=logger)

    if DEBUG_PLOT and 'plot histogram':
      plt.clf()
      plt.subplot(211) ; plt.title('preprocessed')
      for col in range(seq_r.shape[-1]): plt.hist(seq_r[:, col], bins=50)
      plt.subplot(212) ; plt.title('transformed')
      for col in range(seq.shape[-1]): plt.hist(seq[:, col], bins=50)
      save_figure(log_dp / 'hist_transform.png', logger=logger)

    save_pickle(seq, log_dp / TRANSFORM_FILE, logger)
    if stats: save_pickle(stats, log_dp / STATS_FILE, logger)
    
    env.seq   = seq
    env.stats = stats

  if 'reapply stats on dataset' and env.dataset is not None:
    (X_train, y_train), (X_test, y_test) = env.dataset

    for (proc, st) in env.stats:
      logger.info(f'  reapply {proc}...')
      proc_fn = getattr(transform, f'{proc}_apply')
      X_train = proc_fn(X_train, *st)
      X_test  = proc_fn(X_test,  *st)
      if env.label is None:          # is_task_rgr
        y_train = proc_fn(y_train, *st)
        y_test  = proc_fn(y_test,  *st)

    dataset = (X_train, y_train), (X_test, y_test)
    save_pickle(dataset, env.log_dp / DATASET_FILE, logger)

    env.dataset = dataset

def target_data(env:Env):
  process_csv(env)
  process_preprocess(env)
  process_dataset(env)
  if env.status == Status.IGNORED: return
  process_transform(env)

@prepare_for_('train')
@process
def target_train(env:Env):
  job: Config = env.job
  logger: Logger = env.logger
  log_dp: Path = env.log_dp

  manager, model = env.manager, env.model
  data = env.dataset if job.get('dataset') else env.seq
  manager.train(model, data, job.get('model/params'), logger)
  manager.save(model, log_dp, logger)

@prepare_for_('train')
@process
def target_eval(env:Env):
  job: Config = env.job
  logger: Logger = env.logger
  log_dp: Path = env.log_dp

  manager, model = env.manager, env.model
  model = manager.load(model, log_dp, logger)
  data = env.dataset if job.get('dataset') else env.seq
  stats = manager.eval(model, data, job.get('model/params'), logger)

  task_type = TaskType(job.get('model/name').split('_')[-1])
  if   task_type == TaskType.CLF:
    acc, prec, recall, f1 = stats
    lines = [
      f'acc: {acc}',
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

  save_txt(lines, log_dp / SCORES_FILE, logger)

@prepare_for_('infer')
@process
def target_infer(env:Env):
  logger: Logger = env.logger
  log_dp: Path = env.log_dp

  preds_o: Seq = predict_with_oracle(env)
  preds_r: Seq = predict_with_prediction(env)
  predicted = (preds_o, preds_r)
  save_pickle(predicted, log_dp / PREDICT_FILE, logger)


def cmd_args():
  parser = ArgumentParser()
  parser.add_argument('-D', '--csv_file',   type=Path,           help='path to a *.csv data file')
  parser.add_argument('-J', '--job_file',   type=Path,           help='path to a *.yaml job file')
  parser.add_argument('-X', '--job_folder', type=Path,           help='path to a folder of *.yaml job file')
  parser.add_argument(      '--name',       default='test',      help='task name')
  parser.add_argument(      '--target',     default='all',       help='job targets, comma seperated string')
  parser.add_argument(      '--overwrite',  action='store_true', help='no overwrite if log folder exists')
  return parser.parse_args()

@timer
def run_file(args, override_cfg:Dict={}) -> JobResult:
  # names
  task_name: str = args.name
  job_name: str = Path(args.job_file).stem
  fullname = get_fullname(task_name, job_name)

  # log_dp
  log_dp = LOG_PATH / task_name / job_name
  logger = None
  if log_dp.exists() and not args.overwrite:
    logger = get_logger(fullname, log_dp)
    logger.info('ignore due to folder already exists and --overwrite enabled')
    close_logger(logger)
    return Status.IGNORED

  # job template
  log_dp.mkdir(exist_ok=True, parents=True)
  job = Config.load(args.job_file)
  for key, val in override_cfg.items():
    try: job[key] = val
    except: print(f'[run_file] override_cfg failed to find key: {key}')
  job.save(log_dp / JOB_FILE)

  # logger
  logger = logger or get_logger(fullname, log_dp)
  if LOG_JOB:
    logger.info('Job Info:')
    logger.info(pformat(job.cfg))

  seed_everything(fix_seed(job.get('seed', -1)))

  env = Env(
    job=job,             # 'job.yaml'
    logger=logger,       # logger
    log_dp=log_dp,       # log folder
    csv=args.csv_file,   # 'data.csv'
  )

  job: Config = env.job
  logger: Logger = env.logger
  log_dp: Path = env.log_dp

  targets: List[Target] = args.target.split(',')
  if 'all' in targets: targets = ['data', 'train', 'eval', 'infer']
  for tgt in targets:
    try:
      globals()[f'target_{tgt}'](env)
      if env.status == Status.IGNORED:
        return Status.IGNORED
    except:
      logger.error(format_exc())
      close_logger(logger)
      return Status.FAILED

  close_logger(logger)
  return Status.FINISHED

@timer
def run_folder(args):
  ok, tot = 0, 0
  for job_file in Path(args.job_folder).iterdir():
    print(f'>> [run] {job_file}')
    args.job_file = job_file
    res = run_file(args)
    ok += res == Status.FINISHED
    tot += 1
  print(f'>> Done (total: {tot}, failed: {tot - ok})')
