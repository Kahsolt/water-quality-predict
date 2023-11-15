# 作业描述文件

    用于描述一个完整的作业流程

----

⚠ 文件名 **必须** 以 `rgr_` 或者 `clf_` 开头，代表任务类型分别为 `回归` 和 `分类`  

=> 参考 [YMAL语法](https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html)


### 作业描述配置

```yaml
preprocess:         # 预处理，可选项为 modules/preprocess.py 文件内节的各函数名
  filter_T:
    - string
  project: string   # 必须在 ['to_hourly', 'to_daily'] 里二选一
  filter_V:
    - string

dataset:            # 数据制作
  exclusive: bool   # 多特征变量建模是否除外自身 (default: False, ie. auto-gressive)
  inlen: int        # 条件序列窗长，知 in 推 out (default: 3)
  outlen: int       # 预测序列窗长，知 in 推 out (default: 1)
  overlap: int      # in/out 窗重叠长度 (default: 0)
  split: float      # 数据集划分，测试集占比 (default: 0.2)
  encoder:          # 分类任务的目标编码 (default: None)
    name: string    # 编码函数，可选项为 modules/dataset.py 文件内各函数名
    params:         # 编码函数的额外参数列表
      [key: value]
  freq_min: float   # 分类任务的非0类占比阈值，小于此阈值将忽略建模 (default: 0.0)

transform:          # 数值变换，可选项为 modules/transform.py 文件内各函数名
  - string

model:
  name: string      # 模型模板，可选项为 modules/models 目录下各文件名
  params:           # 依模型模板不同设置项也不同，详见各模板
    [key: value]

seed: int           # 全局随机数种子, (default: -1, ie. randomized)
```


### 数据制作流程

##### preprocess 数值预处理

```yaml
preprocess:         # 预处理，可选项为 modules/preprocess.py 文件内节的各函数名
  filter_T:
    - string
  project: string   # 必须在 ['to_hourly', 'to_daily'] 里二选一
  filter_V:
    - string
```

- filter_T: 含时处理，挂载需要时间信息的预处理操作
  - ticker_timer: 时间补全并对齐到时间单位 (h) 
  - ltrim_vacant: 抛弃连续一周以上的缺值及之前
- project: 时间刻度投影，并分离时间维度
  - to_hourly: 时间单位固定为小时
  - to_daily: 时间单位固定为日
    - 聚合策略: 当日有效数据 >12h 则取均值，否则置为 NaN
- filter_V: 不含时处理，挂载不需要时间信息的预处理操作
  - remove_outlier: 区间端点线性插值以重定 3σ 界外值
  - wavlet_transform: 小波变换去噪

##### dataset 数据集切片

```yaml
dataset:
  exclusive: bool   # 多特征变量建模是否除外自身 (default: False, ie. auto-gressive)
  inlen: int        # 条件序列窗长，知 in 推 out (default: 3)
  outlen: int       # 预测序列窗长，知 in 推 out (default: 1)
  overlap: int      # in/out 窗重叠长度 (default: 0)
  split: float      # 数据集划分，测试集占比 (default: 0.2)
  encoder:          # 分类任务的目标编码 (default: None)
    name: string    # 编码函数，可选项为 modules/dataset.py 文件内各函数名
    params:         # 编码函数的额外参数列表
      [key: value]
  freq_min: float   # 分类任务的非0类占比阈值，小于此阈值将忽略建模 (default: 0.0)
```

- 对 预处理数据data，分离 特征变量序列seq 和 目标变量序列tgt
- 对于分类任务，对 tgt 打标签得到 lbl；检查异常值是否超出最小建模阈值
- 知 in 推 out，在 seq 上滚动切片作为 X，对应的 tgt/lbl 作为 Y
- 对数据集 (X, Y) 做训练-测试划分

```python
# data: [T, D'], float      # 预处理后的csv数据表
seq = data[:, :-1] if exclusive else data
tgt = data[:, -1]

# seq: [T, D], float        # T 个采样点, D 维特征输入
# tgt: [T, 1], float        # T 个采样点, 1 维目标输出
if task_type == 'clf':
  lbl = encode(tgt, encoder)   # [N, 1], int
  freq = Counter(lbl)
  if freq < freq_min:       # 若异常值太少，则放弃建模 
    return

# X: [N, I, D], float       # N 帧, 帧长 I，维数 D
# Y: [N, O, 1], float/int   # N 帧, 帧长 O，维数 1
X, Y = slice_frames(seq, tgt/lbl, inlen, outlen, overlap)

# trainset: (X_train, Y_train)
# testset: (X_test, Y_test)
trainset, testset = split_dataset(X, Y, split)
```

##### transform 数值转换

```yaml
transform:          # 数值变换，可选项为 modules/transform.py 文件内各函数名
  - string
```

- 数值转换仅面向模型的输入输出界面，必须是可逆计算
  - log: 对数化
  - std_norm: 均值方差归一化
  - minmax_norm: 最大最小值归一化


### 模型训练-推断流程

```yaml
model:
  name: string      # 模型模板，可选项为 modules/models 目录下各文件名
  params:           # 依模型模板不同设置项也不同，详见各模板
    [key: value]
```

⚠ Just read the fucking code!! ⚠

----
by Armit
2023年2月19日
