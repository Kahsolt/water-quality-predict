from numpy import ndarray
from pandas import DataFrame, Series
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from logging import Logger
from typing import *


# 任务/作业运行目标
Target = Union[
  Literal['data'],      # 制作数据 := 数值预处理 + 打分类标签 + 划分数据集
  Literal['train'],     # 训练
  Literal['eval'],      # 评估
  Literal['all'],       # 全部 := 制作数据 + 训练 + 评估 (糖！)
]
# 任务/作业进度状态
Status = Union[
  Literal['created'],   # 创建任务
  Literal['queuing'],   # 执行队列中等待
  Literal['running'],   # 正在执行中
  Literal['finished'],  # 已完成
  Literal['ignored'],   # 忽略
  Literal['failed'],    # 出错
]

# 作业运行时环境
Env = Dict[str, Any]

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

# 作业类型
TaskType = Union[
  Literal['clf'],       # 分类
  Literal['rgr'],       # 回归
]
# 评估结果
EvalMetrics = Tuple[float, ...]
