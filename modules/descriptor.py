#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/17

import json
import yaml
from pathlib import Path
from copy import deepcopy

from modules.util import ts_now
from modules.typing import Status


# => see 'doc/log.md'
def new_task_init_pack():
  return {
    'name': None,         # task name
    'data': None,         # *.csv file
    'target': None,       # target
    'jobs': None,         # scheduled jobs
  }

def new_runtime_entry():
  return {
    'name': None,         # task name
    'status': Status.CREATED,
    'info': None,
    'progress': None,     # f'{n_job_finished} / {n_job_total}'
    'ts_create': ts_now(),
    'ts_accept': None,
    'task_init_pack': None,
  }

def new_task_entry():
  return {
    'status': Status.CREATED,
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
  def load(cls, fp:Path, init_cfg=None):
    if fp.exists():
      if fp.suffix == '.yaml':
        with open(fp, 'r', encoding='utf-8') as fh:
          cfg = yaml.safe_load(fh)
      elif fp.suffix == '.json':
        with open(fp, 'r', encoding='utf-8') as fh:
          cfg = json.load(fh)
      else:
        raise ValueError(f'invalid file suffix {fp.suffix}, should be either .yaml or .json')
    else:
      assert init_cfg is not None, 'invalid init_cfg, should be serializable for json/yaml, but got None'
      cfg = deepcopy(init_cfg)
    return cls(cfg, fp)

  def save(self, fp:Path=None):
    fp = fp or self.fp
    assert fp, 'must specify a file path to save'

    if fp.suffix == '.yaml':
      with open(fp, 'w', encoding='utf-8') as fh:
        yaml.safe_dump(self.cfg, fh, sort_keys=False)
    elif fp.suffix == '.json':
      with open(fp, 'w', encoding='utf-8') as fh:
        json.dump(self.cfg, fh, sort_keys=False, indent=2, ensure_ascii=False)
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
  cfg = Descriptor(new_runtime_entry())
  print(cfg)

  print(cfg['status'])
  print(cfg['ts_create'])

  cfg['name'] = 'what task'
  cfg['info'] = 'running what job'
  cfg['progress'] = '3 / 8'
  cfg['ts_update'] = ts_now()
  print(cfg)
