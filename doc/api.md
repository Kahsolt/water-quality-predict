# API documentation

=> see job.yaml doc: [/doc/job](/doc/job) for job config  
=> see log folder doc: [/doc/log](/doc/log) for log folder layout  
=> see encoder: [/doc/encoder](/doc/encoder) for pred class_ids of each ecnoder 

### General 总论

Serving through HTTP protocol, the payload for both requests and responses are JSON and files (optional).  
ℹ All routes are RESTful-like designed.  
ℹ For sending json with files in a single request, refer to [python-requests-post-json-and-file-in-single-request](https://stackoverflow.com/questions/19439961/python-requests-post-json-and-file-in-single-request)  
ℹ <del> For serializing NumPy arrays, refer to [numpy-array-to-base64-and-back-to-numpy-array-python](https://stackoverflow.com/questions/6485790/numpy-array-to-base64-and-back-to-numpy-array-python) </del>  

The common JSON fields for all payloads:

```typescript
// request
interface {

}

// response
interface {
  ok: bool            // success status
  error?: str         // error message
  data?: list|dict    // data payload
}
```

### Data 数据

Time sequence data are stored as a **single** `data.csv` file for each task created.  
=> To pre-merge multiple *.csv files, see [POST /merge_csv](#post-merge_csv-合并多个csv文件)


Can conatin **multiple** columns:

  - First column: datetime string in ISO format
    - ⚠ datetime might be temporal discontinuous
  - Rest columns: float values reading from sensors
    - ⚠ values might contain NaNs
    - ℹ **the last column is to predict on**

```csv
monitorTime,monitor1,monitor2       // column names are arbitary
2021-01-01 00:00:00,0.15,1.25
2021-01-01 01:00:00,0.16,           // value NaN
2021-01-01 02:00:00,,1.3
2021-01-01 04:00:00,0.23,1.2        // time leap
2021-01-01 06:00:00,,
2021-01-01 07:00:00,0.2,0.9
2021-01-01 09:00:00,0.3,1.0
```



### Enums 枚举常量

```cpp
enum Target {
  data        // 数据 := 数值预处理 + 打分类标签 + 划分数据集
  train       // 训练
  eval        // 评估
  infer       // 推断 (原地)
  all         // 全部
};
enum Status {
  created     // 创建任务
  queuing     // 执行队列中等待
  running     // 正在执行中
  finished    // 已完成
  ignored     // 忽略
  failed      // 出错
};
```

----

### POST /task 创建新任务并运行

```typescript
// request
// <= single *.csv file named "csv"
// NOTE: must send jsonify(data) as a file named "json"
interface {
  target?: str|str[]  // task target, choose from `Target`, default to 'all'
  jobs?: str[]        // scheduled jobs, default to all applicable jobs
  name?: str          // custom task name
}

// response
interface {
  name: str           // real task name allocated
}
```

### POST /task/\<name\> 更新任务并重新运行

ℹ 主要用于数据更新，修改作业项，或者想从头制作数据集重新开始 :(

```typescript
// request
// <= single *.csv file (optional) named "csv", **replacing** the old one
// NOTE: must send jsonify(data) as a file named "json"
interface {
  target?: str|str[]  // task target, choose from `Target`, default to 'all'
  jobs?: str[]        // scheduled jobs, default to all applicable jobs
}
```

### GET [/task](/task) 列出任务

```typescript
// response
interface {
  tasks: str[]        // task name
}
```

### GET /task/\<name\> 查询任务

```typescript
// response
interface {
  status: str         // task status, choose from `Status`
  jobs: {
    "job_name": {
      type: str
      statue: str
      inlen: int
      scores: {       // for 'clf' task
        pres: float
        recall: float
        f1: float
      } | {           // for 'rgr' task
        mae: float
        mse: float
        r2: float
      }
    }
  }
  ts_create: int
  ts_update: int
}
```

### DELETE /task/\<name\> 删除任务记录

```typescript
```

----

### GET [/model](/model) 列出模型模板

```typescript
// response
interface {
  models: str[]       // model name
}
```

### GET [/job](/job) 列出作业模板

```typescript
// response
interface {
  jobs: str[]         // job name
}
```

### GET /job/\<name\> 下载/查询作业模板

```typescript
// request
// url params: ?format=
//   format: 'yaml' or 'json', default: 'yaml'

// response
// => <job_name>.yaml file or 
interface {
  job: {
    // config items converted from *.yaml
  }
}
```

### POST /job/\<name\> 上传作业模板

```typescript
// request:
// <= *.yaml file
// NOTE: <name> in url must starts with 'rgr_' or 'clf_'
// url params: ?overwrite=
//   overwrite: 0 or 1, default: 0
```

### DELETE /job/\<name\> 删除任务模板

```typescript
```

----

### POST /infer/\<task_name\>/\<job_name\> 推断

⚪ infer inplace

```typescript
// request
interface {
  inplace: bool             // inplace prediction on original timeseq data
}

// response
interface {
  time: List[str]           // preprocessed time, [T]
  seq: List[List[float]]    // preprocessed values, [T, 1]
  pred: List[List[float]]   // inplace predicted timeseq, [T', 1], T' is shorter than T by `inlen`
}
```

⚪ infer new

```typescript
// request
interface {
  data: List[List[float]]   // input frame, [T, D], length T is arbitary
}

// response
interface {
  pred: List[List[float]]   // output frame, [T', 1], T' is typically smaller than T
}
```

----

### GET /log/\<task_name\> 打包下载任务文件夹

```typescript
// response
// => <task_name>.zip
```

### GET /log/\<task_name\>/\<job_name\> 打包下载作业文件夹

```typescript
// response
// => <task_name>-<job_name>.zip
```

### GET /log/\<task_name\>/\<job_name\>.log 查看作业日志

```typescript
// response
// => job.log file
```

### GET [/runtime](/runtime) 查看系统运行时状态

```typescript
// request
// url params: ?status=
//   status: comma seperated string select from `Status` (default: "queuing,running") or 'all'

// response
interface {
  runtime_hist: [
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
}
```

----

### POST /merge_csv 合并多个csv文件

```typescript
// request
// <= multiple *.csv files
// for each file:
//   first column is datetime in isoformat
//   rest columns are float value
interface {
  target: str         // filename, specify which file is to predict on (as last column)
                      // target file must contain only 1 value column
}

// response
// => data.csv
// multi-dim time-aligned seq as described in '###Data' section
```

### GET [/debug](/debug) 系统调试信息

```typescript
// response
// => plain html page
```

----

<p> by Armit <time> 2023/3/3 </time> </p>
