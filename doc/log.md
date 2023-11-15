# 工作目录结构

    存储完整的任务结果、数据记录

----

### 文件系统布局

```c
LOG_PATH
├── runtime.json            // system runtime & history
├── <task_name>
│   ├── task.json           // task descriptor
│   ├── data.csv            // data shared by all jobs of the same task
│   ├── <job_name>
│   │   ├── job.yaml        // job config
│   │   ├── time.pkl        // seq time (preprocessed), for visualize
│   │   ├── preprocess.pkl  // seq values (preprocessed), for visualize
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
├── *.pkl                   // task init pack
└── *.zip                   // download log folders
```

### 数据记录

=> see [`modules/trainer.py`](../modules/trainer.py)

⚪ RunMeta `/log/runtime.json`

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
    init_pack: str,       // path to tmp/<init_pack>.pkl file
  }
]
```

⚪ TaskMeta `/log/<task_name>/task.json`

```json
[
  {                       // => `GET /task/<name>`
    status: str,
    target: str | str[],
    jobs: {
      "job_name": {
        type: str,
        status: str,
        inlen: int,
        scores: {
          acc: float,
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
  }
]
```
