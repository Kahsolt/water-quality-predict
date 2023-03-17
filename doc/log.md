# 任务日志目录结构

    存储完整的任务结果，以及系统元数据

----

### 文件系统布局

```c
LOG_PATH
├── README.md
├── runtime.json            // system runtime & history
├── <task_name>
│   ├── task.json           // task descriptor
│   ├── data.csv            // raw data
│   ├── data_hist.zip       // history archive of raw data
│   ├── <job_name>
│   │   ├── job.yaml        // job config
│   │   ├── seq-raw.pkl     // original timeseq
│   │   ├── stats.pkl       // saved stats of preprocessing
│   │   ├── seq.pkl         // preprocessed timeseq
│   │   ├── label.pkl       // label
│   │   ├── dataset.pkl     // preprocessed (+ label) => train/eval datasets
│   │   ├── timeline*.png   // show timeseq
│   │   ├── hist*.png       // show value histogram
│   │   ├── filter_*.png    // debug preprocessing intermediates
│   │   ├── model.*         // model dump / weights
│   │   ├── scores.txt      // evaluated scores
│   │   └── job.log         // job runner logs
│   └── ...
└── ...
```


### 系统元数据

see `modules/descriptor.py`

⚪ /log/runtime.json

```json
[                         // => `GET /runtime`
  {
    name: str,            // task name
    status: str,          // task status
    info: str,            // cur job
    progress: str,        // f'{n_job_finished} / {n_job_total}'
    ts_create: int,       // task create time
    ts_update: int,       // last meta info update time
    tmp_data_file: str,   // path to tmp stored data.csv file
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
    ts_create: int,
    ts_finish: int,
  },
]
```
