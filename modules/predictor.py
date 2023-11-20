#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/11/15 

import logging
from functools import partial
from pprint import pformat
from dataclasses import dataclass
from importlib import import_module

from modules import preprocess, transform
from modules.dataset import *
from modules.utils import *

# 作业运行时环境
@dataclass
class Env:
  job: Config
  logger: Logger
  log_dp: Path
  csv: Path = None
  status: Status = Status.RUNNING    # RUNNING / IGNORED
  df: TimeSeq = None
  T: Time = None
  seq: Seq = None
  label: Seq = None
  dataset: Datasets = None
  stats: Stats = None
  manager: PyModule = None
  model: Model = None

PredictRet = Union[Frames, Tuple[Frames, Frames]]
Predictor = Callable[[Env, Seq, int, bool], PredictRet]


def prepare_for_(what:EnvKind):
  def wrapper(fn:Callable[..., Any]):
    assert what in ['train', 'infer', 'demo']
    def wrapper(env:Env, *args, **kwargs):
      job: Config = env.job
      logger: Logger = env.logger
      log_dp: Path = env.log_dp

      # Data
      if env.seq is None:
        env.seq = load_pickle(log_dp / TRANSFORM_FILE, logger)
        assert env.seq is not None

      if env.stats is None:
        stats = load_pickle(log_dp / STATS_FILE, logger)
        env.stats = stats if stats is not None else []

      if env.dataset is None and what == 'train':
        env.dataset = load_pickle(log_dp / DATASET_FILE, logger)

      if env.label is None and what != 'infer':
        env.label = load_pickle(log_dp / LABEL_FILE, logger)

      # Model
      if env.manager is None:
        model_name = job.get('model/name') ; assert model_name
        manager = import_module(f'modules.models.{model_name}')
        env.manager = manager
        if LOG_JOB:
          logger.info('manager:')
          logger.info(manager)

      if env.model is None:
        manager = env.manager
        model = manager.init(job.get('model/params', {}), logger)
        env.model = model
        if LOG_JOB:
          logger.debug('model:')
          logger.debug(model)

      return fn(env, *args, **kwargs)
    return wrapper
  return wrapper


def load_env(job_or_file:Union[Path, Config]) -> Env:
  ''' load a pretrained job env '''

  # job
  if isinstance(job_or_file, Path):
    assert job_or_file.exists()
    log_dp: Path = job_or_file.parent
    job = Config.load(job_or_file)
  else:
    job = job_or_file

  # logger
  logger = logging
  if LOG_JOB:
    logger.info('Job Info:')
    logger.info(pformat(job.cfg))

  seed_everything(fix_seed(job.get('seed', -1)))

  env = Env(
    job=job,             # 'job.yaml'
    logger=logger,       # logger
    log_dp=log_dp,       # log folder
  )

  @prepare_for_('demo')
  def load_data_and_model(env:Env):
    env.model = env.manager.load(env.model, log_dp, logger)
  load_data_and_model(env)

  return env


def predict_with_(how:PredictKind, env:Env, x:Seq=None, n_roll:int=1, ret_prob:bool=False) -> PredictRet:
  assert how in ['oracle', 'prediction']
  if how == 'oracle': assert n_roll == 1, 'predict_with_ oracle does not use n_roll'

  job: Config  = env.job
  manager      = env.manager
  model: Model = env.model
  stats: Stats = env.stats

  if env.manager.TASK_TYPE == TaskType.CLF:
    assert n_roll == 1, 'for "clf" tasks must set n_roll=1'

  is_task_rgr = env.manager.TASK_TYPE == TaskType.RGR
  is_model_arima = 'ARIMA' in job['model/name']

  if x is not None:
    seq: Seq = transform.apply_transforms(x, stats)
  else:
    seq: Seq = env.seq     # already transformed

  inlen:   int = job.get('dataset/inlen',   1)
  overlap: int = job.get('dataset/overlap', 0)

  seq = frame_left_pad(seq, inlen)

  def predict_rolling(predictor):
    preds: List[Frame] = []
    loc = 10 if is_model_arima else inlen

    ''' predict with oracle: given a seq longer than `inlen`, rolling predict in window '''
    if how == 'oracle':
      while loc < len(seq):
        if is_model_arima:
          y: Frame = predictor(model, loc)  # [NC]
        else:
          x = seq[loc-inlen:loc, :]
          x = frame_left_pad(x, inlen)      # [I, D]
          y: Frame = predictor(model, x)    # [O, 1]
        preds.append(y)
        loc += len(y) - overlap

    ''' predict with oracle: given a seq equal to `inlen`, rolling predict in window '''
    if how == 'prediction':
      x = seq[-inlen:, :]
      x = frame_left_pad(x, inlen)          # [I, D]
      for _ in range(n_roll):
        if is_model_arima:
          y: Frame = predictor(model, loc)  # [1]
        else:
          y: Frame = predictor(model, x)    # [O, 1]
        preds.append(y)
        if n_roll > 1:
          x = frame_shift(x, y)
          loc += len(y) - overlap   # just for arima

    pred: Seq = np.concatenate(preds, axis=0)    # [T'=R-L+1, 1]
    return transform.inv_transforms(pred, stats) if is_task_rgr else pred

  if ret_prob:
    return predict_rolling(manager.infer), predict_rolling(manager.infer_prob)
  else:
    return predict_rolling(manager.infer)

predict_with_oracle:     Predictor = partial(predict_with_, 'oracle')
predict_with_prediction: Predictor = partial(predict_with_, 'prediction')


def predict_from_request(job_file:Path, x:Frame, t:Frame=None, roll:int=1, prob:bool=False) -> PredictRet:
  env = load_env(job_file)
  job = env.job

  # apply preprocess
  if t is not None:
    t: Time   = pd.Series(str(datetime.fromtimestamp(e)) for e in t)
    x: Values = pd.DataFrame(x)
    df = preprocess.combine_time_and_values(t, x) 

    if 'filter T':
      for proc in job.get('preprocess/filter_T', []):
        if not hasattr(preprocess, proc): continue
        if proc in preprocess.IGNORE_INFER: continue

        df: TimeSeq = getattr(preprocess, proc)(df)

    if 'project':   # NOTE: this is required!
      proc = job.get('preprocess/project')
      assert proc and isinstance(proc, str)
      assert hasattr(preprocess, proc)
      _, df = getattr(preprocess, proc)(df)

    if 'filter V':
      for proc in job.get('preprocess/filter_V', []):
        if not hasattr(preprocess, proc): continue
        if proc in preprocess.IGNORE_INFER: continue

        df: Values = getattr(preprocess, proc)(df)

    x = df.to_numpy()

  # model infer
  return predict_with_prediction(env, x, roll, prob)
