# 任务描述文件

    用于描述一个完整的任务过程

----

=> 参考 [YMAL语法](https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html)


⚪ 数据类型

```yml
misc:
  name: string      # 唯一任务名，留空会默认分配
  seed: int         # 全局随机数种子
  target:
    - string        # 任务目标列表，可选项为 modules/typing.py 文件内 RunTarget 各选项

data:
  - string          # 数据源，csv文件路径

preprocess:         # 预处理，可选项为 modules/preprocess.py 文件内节的各函数名
  filter_T:
    - string
  project:
    - string        # 第一项必须在 ['to_hourly', 'to_daily'] 里二选一
  filter_V:
    - string
  transform:
    - string

dataset:
  train: int        # 采样出的训练集大小
  eval: int         # 采样出的测试集大小
  in: int           # 已知序列窗长，知 in 推 out
  out: int          # 预测序列窗长，知 in 推 out
  encode:           # 分类任务的目标编码
    name: string    # 编码函数，可选项为 modules/dataset.py 文件内各函数名
    params:         # 编码函数的额外参数列表
      [key: value]

model:
  name: string      # 模型单元，可选项为 modules/models 目录下各文件名
  config:           # 依模型单元不同设置项也不同，详见各模板
    [key: value]
```

----
by Armit
2023年2月19日
