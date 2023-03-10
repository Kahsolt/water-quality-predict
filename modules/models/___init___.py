# every model module must implement these funtions

from pathlib import Path
from modules.typing import *

TASK_TYPE: ModelTask = None


def init(config:Config) -> Model:
  ''' init a model '''
  raise NotImplementedError

def train(model:Model, data:Union[Datasets, Seq], config:Config):
  ''' fit model with train dataset '''
  raise NotImplementedError

def eval(model:Model, data:Union[Datasets, Seq], config:Config) -> EvalMetrics:
  ''' get metric scores on eval dataset '''
  raise NotImplementedError

def infer(model:Model, x:Frame) -> Frame:
  ''' predict on one time step '''
  raise NotImplementedError

def save(model:Model, log_dp:Path):
  ''' save model weights/dump '''
  raise NotImplementedError

def load(model:Model, log_dp:Path) -> Model:
  ''' load model weights/dump '''
  raise NotImplementedError
