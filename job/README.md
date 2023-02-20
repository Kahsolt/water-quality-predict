# 任务描述文件

    用于描述一个完整的任务过程

----

=> 参考 [YMAL语法](https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html)


⚪ 样例

见 [exmaple](exmaple.yaml)

⚪ 数据类型

```yml
misc:
  name: string      # 唯一任务名，留空会默认分配
  seed: int         # 全局随机数种子
  target:
    - string        # 任务目标列表，可选项为 modules/typing.py 文件内 RunTarget 各选项

data:
  - string          # 数据源，csv文件路径

preprocess:         # 预处理，可选项为 modules/preprocess.py 文件内各函数名
  - string          # 第一项必须在 ['to_hourly', 'to_daily'] 里二选一

dataset:
  train: int        # 采样出的训练集大小
  test: int         # 采样出的测试集大小
  in: int           # 已知窗长，知 in 推 out
  out: int          # 预测窗长，知 in 推 out
  encode: string    # 分类目标编码，可选项为 modules/dataset.py 文件内各函数名

model:
  arch: string      # 模型架构，可选项为 modules/models 目录下各文件名
  [key: value]      # 依模型架构不同超参数也不同，详见各模板

train:
  [key: value]      # 依模型架构不同训练参数也不同，详见各模板
```

----
by Armit
2023年2月19日
