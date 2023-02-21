# water-quality-predict

    水质检测指标的时间序列预测

----

你妈的😠，说起来是个很简单的toy，但是差不多写了一整套批处理作业框架……


![demo](img/demo.png)


### Quick Start

⚪ Train & Eval

- write a job file, see guide => [job/README.md](job/README.md)
- run `python run.py -J path/to/*.yaml`

⚪ Infer

- run `python run_infer.py`


#### Data

- put your `*.csv` files under `data` folder
- each file can contain several columns
  - the first columns is datetime in ISO 8601 format, e.g. `2022-09-27 18:00:00.000`
  - the rest columns are float data from your sensor deivces

----
by Armit
2022/09/15  
2023/02/14  

