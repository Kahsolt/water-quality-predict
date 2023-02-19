# water-quality-predict

    江苏常隆农化有限公司水质检测指标的时间序列预测

----

### Quick Start

- extract features: `python preprocess.py`
- inspect into data: `python data.py`
- train model: `python train.py`
- test predict: `python infer.py log\model-100.pt`


### Dataset

三项传感器检测指标 `COD、ph、氨氮`，共 1984 条浮点采样数据  

时间跨度约 3 个月：`2022/6/1 00:00:00 ~ 2022/8/22 15:00:00`，每天 24 个采样点

```
flow：流量
ph：pH值
cod：COD浓度
an：氨氮浓度
tn：总氮浓度
tp：总磷浓度
temp：温度
```

### Model

类似于 WaveNet 的想法，用 CRNN 构建 N-Gram 模型，通过前面 k 个采样点推断下一个采样点、逐点生成

----

by Armit
2022/09/15 
