# 作业描述文件

    用于描述一个完整的预测器作业流程

----

=> 参考 [YMAL语法](https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html)

⚠ 文件名 **必须** 以 `rgr_` 或者 `clf_` 开头，代表任务类型分别为 `回归` 和 `分类`


⚪ 数据模式

```yaml
preprocess:         # 预处理，可选项为 modules/preprocess.py 文件内节的各函数名
  filter_T:
    - string
  project:
    - string        # 必须在 ['to_hourly', 'to_daily'] 里二选一
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
  
seed: int           # 全局随机数种子
```

----
by Armit
2023年2月19日
