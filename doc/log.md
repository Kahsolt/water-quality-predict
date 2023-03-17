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
  {
    name: str,            // task name
    status: str,          // task status
    info: str,            // cur job
    progress: str,        // f'{n_job_finished} / {n_job_total}'
    ts_accept: int,       // task accept time
    ts_update: int,       // last meta info update time
    task_init_pack: str,  // path to tmp task_init_pack.pkl file (will be deleted once task created successfully)
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