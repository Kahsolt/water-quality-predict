import sys
from pathlib import Path
from enum import Enum
from logging import Logger
from typing import *

from numpy import ndarray
from pandas import DataFrame, Series
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

# python module
PyModule = type(sys)

# 任务/作业运行目标
class Target(Enum):
  DATA  = 'data'          # 制作数据 := 数值预处理 + 打分类标签 + 划分数据集
  TRAIN = 'train'         # 训练
  EVAL  = 'eval'          # 评估
  INFER = 'infer'         # 测试 (原地预测)
  ALL   = 'all'           # 全部 := 制作数据 + 训练 + 评估 + 测试 (糖！)

# 任务/作业进度状态
class Status(Enum):
  QUEUING  = 'queuing'    # 执行队列中等待
  RUNNING  = 'running'    # 作业正在执行中
  FINISHED = 'finished'   # 已完成
  FAILED   = 'failed'     # 出错
  IGNORED  = 'ignored'    # 忽略 (作业)

# 作业类型
class TaskType(Enum):
  CLF = 'clf'             # 分类
  RGR = 'rgr'             # 回归


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
Frame  = ndarray      # [I/O, D/1]
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
# 作业运行时环境类型
EnvKind = Union[
  Literal['train'],  # train
  Literal['infer'],  # infer
  Literal['demo'],   # demo inplace infer
]
# 预测类型
PredictKind = Union[
  Literal['oracle'],
  Literal['prediction'],
]
# 任务日志结果
JobResult = Union[
  Literal[Status.FINISHED],
  Literal[Status.FAILED],
  Literal[Status.IGNORED],
]
