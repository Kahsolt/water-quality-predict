# water-quality-predict

    水质检测指标的时间序列预测平台后端

----

你妈的😠，说起来是个很简单的toy，但是差不多写了一整套批处理作业框架……  
现在变成一个平台性的作业流了：提交数据集并创建任务 -> 无脑训练若干个预测器 -> 用性能最好的预测器应对新的查询  


### Web API

⚪ Native API

- start server `python app.py`
- point your browser to `http://127.0.0.1:5000/` to see API documentation

Main busisness:

- `POST /task` to create a task and put in processing queue
- `GET /task/<name>` to get task results 
- `POST /task/<name>` to retrain a old task with new data
- `POST /infer` to predict on new data

Extras:

- `GET /runtime` to see running queue status and history
- `GET /job/<name>` or `POST /job/<name>` to prepare your job plans
- `GET /log/<task_name>` to download the task log folder
- `GET /log/<task_name>/<job_name>` to download the job log folder
- `GET /log/<task_name>/<job_name>.log` to see job log
- `POST /merge_csv` to merge data all-in-one if you have multiple data source

=> see client demo: [app_test.py](app_test.py)


### Local run

⚪ Data

- prepare your `*.csv` files (suggested to put under `data` folder)
- each file can contain several columns
  - the first columns is datetime in ISO 8601 format, e.g. `2022-09-27 18:00:00.000`
  - the rest columns are float data from your sensor devices
    - the last column is to predict on

⚪ Dataset & Train & Eval

- write a job file, see guide => [doc/job.md](doc/job.md)
- run a single job: `python run.py -D path/to/*.csv -J path/to/*.yaml --target all`
- run folder of jobs: `py run.py -D data\test.csv -X job`
  - run all test demo: `run_test.cmd`

⚪ Infer

![demo](img/demo.png)

- run `python infer.py`


----
by Armit
2022/09/15  
2023/02/14  
