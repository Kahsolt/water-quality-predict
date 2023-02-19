import numpy as np
from typing import *

# 任务描述文件
Job = Dict[str, Any]
# 任务运行时环境
Env = Dict[str, Any]

# 运行目标
RunTarget = Union[
  Literal['all'],       # 全部
  Literal['data'],      # 仅制作数据
  Literal['train'],     # 仅训练
  Literal['eval'],      # 仅评估
  Literal['infer'],     # 仅预测
]

# 预处理后的数据帧序列
Seq = np.ndarray
# 预处理过程中记录的一些统计量
Stats = List[Tuple[str, Any]]
# 数据集：(输入X, 输出Y)
Dataset = Tuple[np.ndarray, np.ndarray]
# 数据集组：(训练集, 测试集)
Datasets = Tuple[Dataset, Dataset]
# 所有缓存的数据
CachedData = Union[Seq, Stats, Datasets]

# 模型
Model = object()
# 模型参数
ModelParams = dict
# 模型任务类型
ModelTask = Union[
  Literal['clf'],       # 分类
  Literal['rgr'],       # 回归
]
