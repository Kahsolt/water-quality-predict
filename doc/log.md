# 工作目录结构

    存储完整的任务结果、系统元数据、临时转储

----

### 文件系统布局

```c
LOG_PATH
├── README.md
├── runtime.json            // system runtime & history
├── <task_name>
│   ├── task.json           // task descriptor
│   ├── data.csv            // data shared by all jobs of the same task
│   ├── data_hist.zip       // history archive of raw data
│   ├── <job_name>
│   │   ├── job.yaml        // job config
│   │   ├── preprocess.pkl  // seq (preprocessed), for visualize
│   │   ├── label.pkl       // encoded target/label, for visualize
│   │   ├── stats.pkl       // stats of transform, for postprocess
│   │   ├── transform.pkl   // seq (transformed), for ARIMA train/eval
│   │   ├── dataset.pkl     // dataset (transformed), for other model train/eval
│   │   ├── scores.txt      // evaluated scores
│   │   ├── predict.pkl     // inplace prediction
│   │   ├── job.log         // job runner logs
│   │   ├── timeline_*.png  // debug curve trends (preprocess & transform)
│   │   ├── hist_*.png      // debug value histogram (preprocess & transform)
│   │   ├── filter_*.png    // debug preprocessors (filter_T & filter_V)
│   │   └── model.*         // model dump / weights
│   └── ...
└── ...

TMP_PATH
├── tip-*.pkl               // task init pack
└── ...
```

### 系统元数据

=> see `modules/descriptor.py`

⚪ /tmp/\<task_init_pack\>.pkl 任务请求临时转储

```python
task_init_pack = {            # => see `POST /task`
  'name': str,                # task name
  'data': bytes,              # files[0].stream().read()
  'target': [str]|str|None,   # target
  'jobs': [str]|None,         # scheduled jobs
}
```

⚪ /log/runtime.json

```json
[                         // => `GET /runtime`
  {                       // Run
    id: int,
    name: str,            // task name
    status: str,          // task status
    info: str,            // cur job
    progress: str,        // f'{n_job_finished} / {n_job_total}'
    ts_create: int,       // task accept time
    ts_update: int,       // last meta info update time
    task_init_pack: str,  // path to tmp task_init_pack.pkl file (will be deleted once task marked finished)
  },
]
```

⚪ /log/\<task_name\>/task.json

```json
[
  {                       // => `GET /task/<name>`
    status: str,
    target: str | str[],
    jobs: {
      "job_name": {
        type: str,
        status: str,
        scores: {
          pres: float,
          recall: float,
          f1: float,
        } | {
          mae: float,
          mse: float,
          r2: float,
        },
      },
    },
    ts_create: int,       // task create time
    ts_update: int,
  },
]
```
