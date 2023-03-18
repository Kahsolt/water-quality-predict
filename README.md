# water-quality-predict

    水质检测指标的时间序列预测

----

你妈的😠，说起来是个很简单的toy，但是差不多写了一整套批处理作业框架……  
现在变成一个平台性的作业流了：提交数据集并创建任务 -> 无脑训练若干个预测器 -> 用性能最好的预测器应对新的查询  


### Web API

⚪ Native API

- start server `python app.py`
- point your browser to `http://127.0.0.1:5000/` to see API documentation

Main busisness:

- `POST /merge_csv` to merge data all-in-one if you have multiple data source
- `GET /job/<name>` or `POST /job/<name>` to prepare your job plans
- `POST /task` to create a task and put in processing queue
- `GET /runtime` to see running status
- `GET /task/<name>` to get task status or results 
- `POST /infer` to predict on new data
- `GET /log/<task_name>/<job_name>.log` to see job log
- `GET /log/<task_name>` to download the task log folder

⚪ Proxy API

- start server `python app_proxy.py`
- point your browser to `http://127.0.0.1:5000/` to see API documentation

Main busisness:

- `POST /page2/getFittingCurve`
- `POST /page2/get6hPredictionResult`
- `POST /page2/getModelPerformance`
- `POST /page2/getExceedingPredictionResult`


### Local run

⚪ Data

- prepare your `*.csv` files (suggested to put under `data` folder)
- each file can contain several columns
  - the first columns is datetime in ISO 8601 format, e.g. `2022-09-27 18:00:00.000`
  - the rest columns are float data from your sensor devices
    - the last column is to predict on

⚪ Dataset & Train & Eval

- write a job file, see guide => [doc/job.md](doc/job.md)
- run `python run.py -D path/to/*.csv -J path/to/*.yaml`
  - run test demo: `run_test.cmd`

⚪ Infer

![demo](img/demo.png)

- run `python run_infer.py`


----
by Armit
2022/09/15  
2023/02/14  
