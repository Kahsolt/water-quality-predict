# water-quality-predict

    江苏常隆农化有限公司水质检测指标的时间序列预测

----

### TODO

[LSTM]
data：是原数据经过处理后方便带入模型的数据
input_preprocess：对输入数据进行预处理的函数
model：LSTM模型结构
mytools：定义了训练和测试的一些函数
train_eval：定义从数据处理到训练与测试过程的函数

PS：缺少代码（1）调参（2）参数保存（3）输入最新数据进行预测

[XGBoost]
data：是原数据经过处理后方便带入模型的数据
3h数值预测分类XGBoost：读取‘w21003_补值.csv’进行计算。
缺少代码（1）能否进行分类（3类＞1%）判断的代码（2）输入最新数据进行预测
24h数值预测分类XGBoost：读取‘w21003_补值.csv’进行计算。
缺少代码（1）能否进行分类（3类＞5%）判断（2）将X转化为每天0:00（3）输入最新数据进行预测
日均值预测分类XGBoost：读取‘w21003_日均值.csv’进行计算。
缺少代码（1）能否进行分类（超标＞5%）判断（2）输入最新数据进行预测

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

----

by Armit
2022/09/15 
