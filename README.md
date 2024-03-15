# water-quality-predict

    水质检测指标的时间序列预测平台后端

----

你妈的😠，说起来是个很简单的toy，但是差不多写了一整套批处理作业框架……  
现在变成一个平台性的作业流了：提交数据集并创建任务 -> 无脑训练若干个预测器 -> 用性能最好的预测器应对新的查询  


### WebApp

> You can launch our flask webapp, train & infer through HTTP requests

Run server:

- start server `python server.py -H <host> -P <port>` (default `port=5000`)
- point your browser to `http://127.0.0.1:5000/` for API documentation
- recognized envvars
  - `DEBUG_PLOT`: save intermediate plots during training for debug
  - `LOG_JOB`: log job setting & model details when loading a pretrained job
- run tests
  - unit test: `python test_ut.py`
  - integral test: `python test_st.py` (require server running)

Run client:

- `python client.py -H <host> -P <port>`

![client](img/client.png)


### Local

> You can also run local command for training, debug inplace-infer results via the demo app

⚪ Data

- prepare your `*.csv` files (suggested to put under `data` folder)
- each file can contain several columns
  - the first columns is datetime in ISO 8601 format, e.g. `2022-09-27 18:00:00.000`
  - the rest columns are float data from your sensor devices
    - the last column is to predict on

⚪ Job & Train

- write a job file, see guide => [doc/job.md](doc/job.md)
- run a single job: `python run.py -D path\to\*.csv -J path\to\*.yaml --target all`
- run folder of jobs: `python run.py -D data\test.csv -X job`
  - see also: `run.cmd`

⚪  Eval

- run demo client app for debug: `python demo.py`

![demo](img/demo.png)


----
by Armit
2024/03/15
