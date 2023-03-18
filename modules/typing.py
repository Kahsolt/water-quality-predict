from pathlib import Path
from enum import Enum
from logging import Logger
from typing import *

from numpy import ndarray
from pandas import DataFrame, Series
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer


# 任务/作业运行目标
class Target(Enum):
  DATA  = 'data'          # 制作数据 := 数值预处理 + 打分类标签 + 划分数据集
  TRAIN = 'train'         # 训练
  EVAL  = 'eval'          # 评估
  ALL   = 'all'           # 全部 := 制作数据 + 训练 + 评估 (糖！)

# 任务/作业进度状态
class Status(Enum):
  CREATED  = 'created'    # 创建任务
  QUEUING  = 'queuing'    # 执行队列中等待
  RUNNING  = 'running'    # 正在执行中
  FINISHED = 'finished'   # 已完成
  IGNORED  = 'ignored'    # 忽略
  FAILED   = 'failed'     # 出错

# 作业类型
class TaskType(Enum):
  CLF = 'clf'       # 分类
  RGR = 'rgr'       # 回归


# 模型
Model = object()
# 模型参数
Params = Dict[str, Any]
# 分类标签编码
Encoder = {
  'name': str,
  'params': Params,
}

# 含时间轴的原始数据
TimeSeq = DataFrame
Time    = Series
Values  = DataFrame
TimeAndValues = Tuple[Time, Values]
# 预处理后的数据帧序列
Seq    = ndarray      # [T, D]
Array  = ndarray      # [T]
Frame  = ndarray      # [I/O, D]
Frames = ndarray      # [N, I/O, D]
# 预处理过程中记录的一些统计量
Stat  = Tuple[Any, ...]
Stats = List[Tuple[str, Stat]]
SeqAndStat = Tuple[Seq, Stat]
# 数据集：(输入X, 输出Y)
Dataset = Tuple[Frames, Frames]
# 数据集组：(训练集, 测试集)
Datasets = Tuple[Dataset, Dataset]
# 所有磁盘缓存的数据
CachedData = Union[Seq, Stats, Datasets]
# 评估结果
EvalMetrics = Tuple[float, ...]


# 作业运行时环境
Env = {
  'fullname': str, 
  'job': 'Descriptor',
  'logger': Logger,
  'log_dp': Path,
  'status': Status,
  'df': TimeSeq,
  'T': Time,
  'seq': Seq,
  'label': Seq,
  'dataset': Datasets,
  'stats': Stats,
  'manager': 'module',
  'model': Model,
}
