#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/17

from pathlib import Path

from modules.util import *
from modules.typing import Status


# => see 'doc/log.md'
def new_task_init() -> TaskInit:
  return {
    'name': None,         # task name
    'data': None,         # *.csv file
    'target': None,       # target
    'jobs': None,         # scheduled jobs
    'thresh': None,       # override `dataset.encoder.params.thresh`
  }

def new_run_meta() -> RunMeta:
  return {
    'id': None,
    'name': None,         # task name
    'status': Status.QUEUING,
    'info': '',           # accumulative
    'progress': None,     # f'{n_job_finished} / {n_job_total}'
    'ts_create': ts_now(),
    'ts_update': None,
    'task_init_pack': None,
  }

def new_task_meta() -> TaskMeta:
  return {
    'status': Status.QUEUING,
    'target': None,
    'jobs': { },
    'ts_create': ts_now(),
    'ts_update': None,
  }


class Descriptor:

  SEP = '/'
    
  def __init__(self, cfg, fp:Path=None):
    self.cfg = cfg
    self.fp = fp

  def __str__(self):
    return str(self.cfg)

  def __repr__(self):
    return repr(self.cfg)

  @classmethod
  def load(cls, fp:Path, init=None):
    if fp.suffix == '.yaml':
      cfg = load_yaml(fp, init)
    elif fp.suffix == '.json':
      cfg = load_json(fp, init)
    else:
      raise ValueError(f'invalid file suffix {fp.suffix}, should be either .yaml or .json')
    return cls(cfg, fp)

  def save(self, fp:Path=None):
    fp = fp or self.fp
    assert fp, 'must specify a file path to save'

    if fp.suffix == '.yaml':
      save_yaml(fp, self.cfg)
    elif fp.suffix == '.json':
      save_json(fp, self.cfg)
    else:
      raise ValueError(f'invalid file suffix {fp.suffix}, should be either .yaml or .json')

  def __getitem__(self, path:str):
    r = self.cfg
    for s in path.split(self.SEP):
      if s in r: r = r[s]
      else: raise KeyError(f'invalid path {path!r}')
    return r
  
  def __setitem__(self, path:str, value):
    paths = path.split(self.SEP)
    segs, key = paths[:-1], paths[-1]
    r = self.cfg
    for s in segs:
      if s in r: r = r[s]
      else: raise KeyError(f'invalid path {path!r}')
    r[key] = value
  
  def get(self, path:str, value=None):
    try: return self[path]
    except: return value


if __name__ == '__main__':
  cfg = Descriptor(new_run_meta())
  print(cfg)

  print(cfg['status'])
  print(cfg['ts_create'])

  cfg['name'] = 'what task'
  cfg['info'] = 'running what job'
  cfg['progress'] = '3 / 8'
  cfg['ts_update'] = ts_now()
  print(cfg)
