#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/19 

from time import time
from copy import deepcopy
from pathlib import Path
from argparse import ArgumentParser
from collections import Counter
from queue import Queue, Empty
from threading import Thread, Event, RLock
from importlib import import_module
from pprint import pformat
from traceback import format_exc

import matplotlib ; matplotlib.use('agg')
import matplotlib.pyplot as plt

from modules import preprocess, transform
from modules.descriptor import *
from modules.dataset import *
from modules.transform import *
from modules.util import *
from modules.typing import *

from config import *


def worker(evt:Event, lock:RLock, queue:Queue):
  while not evt.is_set():
    payload = None
    while payload is None:
      try: payload: Tuple[RunMeta, Trainer] = queue.get(timeout=CHECK_TASK_EVERY)
      except Empty: pass
      if evt.is_set(): return

    with lock:
      run, runtime = payload

      run['info'] += "Setup task log folder\n"
      log_dp: Path = LOG_PATH / run['name']
      log_dp.mkdir(exist_ok=True, parents=True)

      run['info'] += "Load/create task meta file\n"
      task_meta: List[RunMeta] = load_json(log_dp / TASK_FILE, [])
      meta: TaskMeta = new_task_meta()
      task_meta.append(meta)
      save_task_meta = lambda: save_json(log_dp / TASK_FILE, task_meta)
      save_task_meta()    # init save

      args = cmd_args()
      args.name = run['name']
      args.csv_file = log_dp / DATA_FILE

      if run['status'] == Status.QUEUING:
        try: 
          run['info'] += "Unpack task init info\n"
          init_fp = Path(run['task_init_pack'])
          init: TaskInit = load_pickle(init_fp)

          run['info'] += "Check init info\n"
          task_name = init['name'] ; assert task_name == run['name']
          meta['target'] = init['target']
          args.target = ','.join(meta['target'])
          jobs = init['jobs']

          if 'data' in init and init['data'] is not None:
            run['info'] += "Write csv file\n"
            with open(log_dp / DATA_FILE, 'wb') as fh:
              fh.write(init['data'])

          meta['status'] = run['status'] = Status.CREATED
        except:
          run['info'] += format_exc() + '\n'
          meta['status'] = run['status'] = Status.FAILED
        meta['ts_update'] = run['ts_update'] = ts_now()

      if run['status'] == Status.CREATED:
        run['info'] += "Start run jobs\n"
        meta['status'] = run['status'] = Status.RUNNING
        meta['ts_update'] = run['ts_update'] = ts_now()

      save_task_meta()

      if run['status'] == Status.RUNNING:
        try:
          ok, tot = 0, len(jobs)
          for i, job_name in enumerate(jobs):
            run['info'] += f"Running job {job_name!r}\n"
            run['progress'] = f"{i+1} / {tot}"
            meta['ts_update'] = run['ts_update'] = ts_now()

            job_file = JOB_PATH / f'{job_name}.yaml'
            args.job_file = job_file
            res: Status = run_file(args)
            if res != Status.FAILED: ok += 1

            if 'update task meta':
              ttype = job_name.split('_')[0]
              job = Descriptor.load(job_file)
              inlen: int = job.get('dataset/inlen', 1)
              meta['jobs'][job_name]: JobMeta = {
                'type': ttype,
                'status': res,
                'inlen': inlen,
              }

              sc_fp = log_dp / job_name/ SCORES_FILE
              if sc_fp.exists():
                with open(sc_fp, 'r', encoding='utf-8') as fh:
                  lines = fh.read().strip()

                scores = { }
                for line in lines.split('\n'):
                  name, score = line.split(':')
                  scores[name.strip()] = float(score)
                meta['jobs'][job_name]['scores'] = scores

              save_task_meta()

          run['info'] += f"Done all jobs! (total: {tot}, failed: {tot - ok})\n"
          meta['status'] = run['status'] = Status.FINISHED
        except:
          run['info'] += format_exc() + '\n'
          meta['status'] = run['status'] = Status.FAILED
        meta['ts_update'] = run['ts_update'] = ts_now()

      if run['status'] == Status.FINISHED:
        try:
          run['info'] += "Clean up tmp files\n"
          os.unlink(run['task_init_pack'])
          del run['task_init_pack']

          run['info'] += "Done!\n"
        except:
          run['info'] += format_exc() + '\n'
        meta['ts_update'] = run['ts_update'] = ts_now()

      save_task_meta()
      runtime.save_run_meta()

      queue.task_done()


class Trainer:

  def __init__(self):
    self.envs: Dict[str, Env] = { }
    self.queue = Queue()
    self.evt = Event()
    self.lock = RLock()
    self.worker = Thread(target=worker, args=(self.evt, self.lock, self.queue))

    self.run_meta: List[RunMeta] = load_json(LOG_PATH / TASK_FILE, [])
    self._resume()

  def _resume(self):
    for run in self.run_meta:
      if Status(run['status']) in [Status.QUEUING, Status.CREATED, Status.RUNNING]:
        run['status'] = Status.QUEUING    # reset to queuing
        run['ts_update'] = ts_now()
        self.queue.put((run, self))

  def save_run_meta(self):
    save_json(LOG_PATH / TASK_FILE, self.run_meta)

  def add_task(self, name:str, init_fp:Path):
    print(f'>> new task: {name}')
    run = new_run_meta()   # Status.QUEUING
    run['id'] = len(self.run_meta) + 1
    run['name'] = name
    run['task_init_pack'] = init_fp
    self.run_meta.append(run)
    self.queue.put((run, self))

  def start(self):
    self.save_run_meta()
    self.worker.start()

  def stop(self):
    self.evt.set()
    self.worker.join()
    self.save_run_meta()


def predict_with_oracle(env:Env, x:Seq=None) -> Frame:
  job: Descriptor = env['job']
  manager      = env['manager']
  model: Model = env['model']
  stats: Stats = env['stats']

  if x is not None:
    seq: Seq = apply_transforms(x, stats)
  else:
    seq: Seq = env['seq']     # transformed

  inlen:   int = job.get('dataset/inlen')
  overlap: int = job.get('dataset/overlap', 0)

  seq = frame_left_pad(seq, inlen)

  is_task_rgr = env['manager'].TASK_TYPE == TaskType.RGR
  is_model_arima = 'ARIMA' in job['model/name']

  preds: List[Frame] = []
  loc = 10 if is_model_arima else inlen
  while loc < len(seq):
    if is_model_arima:
      y: Frame = manager.infer(model, loc)  # [1]
    else:
      x = seq[loc-inlen:loc, :]
      x = frame_left_pad(x, inlen)          # [I, D]
      y: Frame = manager.infer(model, x)    # [O, 1]
    preds.append(y)
    loc += len(y) - overlap
  preds_o: Seq = np.concatenate(preds, axis=0)    # [T'=R-L+1, 1]

  return inv_transforms(preds_o, stats) if is_task_rgr else preds_o

def predict_with_predicted(env:Env, x:Seq=None) -> Frame:
  job: Descriptor = env['job']
  manager      = env['manager']
  model: Model = env['model']
  stats: Stats = env['stats']

  if x is not None:
    seq: Seq = apply_transforms(x, stats)
  else:
    seq: Seq = env['seq']     # transformed

  inlen:   int = job.get('dataset/inlen')
  overlap: int = job.get('dataset/overlap', 0)

  seq = frame_left_pad(seq, inlen)

  is_task_rgr = env['manager'].TASK_TYPE == TaskType.RGR
  is_model_arima = 'ARIMA' in job['model/name']

  preds: List[Frame] = []
  loc = 10 if is_model_arima else inlen
  x = seq[loc-inlen:loc, :]
  x = frame_left_pad(x, inlen)              # [I, D]
  while loc < len(seq):
    if is_model_arima:
      y: Frame = manager.infer(model, loc)  # [1]
    else:
      y: Frame = manager.infer(model, x)    # [O, 1]
    preds.append(y)
    x = frame_shift(x, y)
    loc += len(y) - overlap
  preds_r: Seq = np.concatenate(preds, axis=0)    # [T'=R-L+1, 1]

  return inv_transforms(preds_r, stats) if is_task_rgr else preds_r


class Predictor:

  def __init__(self) -> None:
    self.envs: Dict[str, Env] = { }

  def predict(self, task:str, job:str, x:Frame) -> Frame:
    fullname = get_fullname(task, job)
    if fullname not in self.envs:
      self.envs[fullname] = load_env(LOG_PATH / task / job / 'job.yaml')

    env = self.envs[fullname]
    y: Frame = predict_with_oracle(env, x)
    return y


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

def prepare_for_(what:EnvKind):
  def wrapper(fn:Callable[..., Any]):
    assert what in ['train', 'infer', 'demo']
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

      if 'dataset' not in env and what == 'train':
        env['dataset'] = load_pickle(log_dp / DATASET_FILE, logger)

      if 'label' not in env and what != 'infer':
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
  return wrapper

def load_env(job_file:Path) -> Env:
  ''' load a pretrained job env '''

  # job
  assert job_file.exists()
  log_dp: Path = job_file.parent
  job = Descriptor.load(job_file)

  # logger
  logger = logging
  logger.info('Job Info:')
  logger.info(pformat(job.cfg))

  seed_everything(fix_seed(job.get('seed', -1)))

  env: Env = {
    'job': job,             # 'job.yaml'
    'logger': logger,       # logger
    'log_dp': log_dp,       # log folder
  }

  @prepare_for_('demo')
  def load_data_and_model(env:Env):
    env['model'] = env['manager'].load(env['model'], log_dp, logger)
  load_data_and_model(env)

  return env


@process
def process_csv(env:Env):
  logger: Logger = env['logger']

  df: TimeSeq = read_csv(env['csv'], logger)   # 总原始数据
  for col in df.columns[1:]: df[col] = df[col].astype('float32')

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
  save_pickle(T,   log_dp / TIME_FILE,       logger)
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
  inlen   = job.get('dataset/inlen')      ; assert inlen   > 0
  outlen  = job.get('dataset/outlen')     ; assert outlen  > 0
  overlap = job.get('dataset/overlap', 0) ; assert overlap >= 0
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
      logger.info (f'  apply {proc}...')
      seq, st = getattr(transform, proc)(seq)
      stats.append((proc, st))

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

@prepare_for_('train')
@process
def target_train(env:Env):
  job: Descriptor = env['job']
  logger: Logger = env['logger']
  log_dp: Path = env['log_dp']

  manager, model = env['manager'], env['model']
  data = env['dataset'] if job.get('dataset') else env['seq']
  manager.train(model, data, job.get('model/params'), logger)
  manager.save(model, log_dp, logger)

@prepare_for_('train')
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

  with open(log_dp / SCORES_FILE, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(lines))

@prepare_for_('infer')
@process
def target_infer(env:Env):
  logger: Logger = env['logger']
  log_dp: Path = env['log_dp']

  preds_o: Seq = predict_with_oracle   (env)
  preds_r: Seq = predict_with_predicted(env)
  predicted = (preds_o, preds_r)
  save_pickle(predicted, log_dp / PREDICT_FILE, logger)


def cmd_args():
  parser = ArgumentParser()
  parser.add_argument('-D', '--csv_file',     type=Path,           help='path to a *.csv data file')
  parser.add_argument('-J', '--job_file',     type=Path,           help='path to a *.yaml job file')
  parser.add_argument('-X', '--job_folder',   type=Path,           help='path to a folder of *.yaml job file')
  parser.add_argument(      '--name',         default='test',      help='task name')
  parser.add_argument(      '--target',       default='all',       help='job targets, comma seperated string')
  parser.add_argument(      '--no_overwrite', action='store_true', help='no overwrite if log folder exists')
  return parser.parse_args()

@timer
def run_file(args) -> JobResult:
  # names
  task_name: str = args.name
  job_name: str = args.job_file.stem
  fullname = get_fullname(task_name, job_name)

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

  targets: List[Target] = args.target.split(',')
  if 'all' in targets: targets = ['data', 'train', 'eval', 'infer'] 
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
  ok, tot = 0, 0
  for job_file in args.job_folder.iterdir():
    print(f'>> [run] {job_file}')
    args.job_file = job_file
    res = run_file(args)
    ok += res == Status.FINISHED
    tot += 1
  print(f'>> Done (total: {tot}, failed: {tot - ok})')


if __name__ == '__main__':
  args = cmd_args()
  assert (args.job_file is None) ^ (args.job_folder is None), 'must specify either --job_file xor --job_folder'

  if args.job_file:   run_file  (args)
  if args.job_folder: run_folder(args)
