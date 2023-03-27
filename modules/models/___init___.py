# every model module must implement these funtions

from pathlib import Path
from modules.typing import *

TASK_TYPE: TaskType = None


def init(params:Params, logger:Logger=None) -> Model:
  ''' init a model '''
  raise NotImplementedError

def train(model:Model, data:Union[Datasets, Seq], params:Params, logger:Logger=None):
  ''' fit model with train dataset '''
  raise NotImplementedError

def eval(model:Model, data:Union[Datasets, Seq], params:Params, logger:Logger=None) -> EvalMetrics:
  ''' get metric scores on eval dataset '''
  raise NotImplementedError

def infer(model:Model, x:Frame, logger:Logger=None) -> Frame:
  ''' predict on one time step '''
  raise NotImplementedError

def infer_prob(model:Model, x:Frame, logger:Logger=None) -> Frame:
  ''' predict on one time step (clf only)'''
  raise NotImplementedError

def save(model:Model, log_dp:Path, logger:Logger=None):
  ''' save model weights/dump '''
  raise NotImplementedError

def load(model:Model, log_dp:Path, logger:Logger=None) -> Model:
  ''' load model weights/dump '''
  raise NotImplementedError
