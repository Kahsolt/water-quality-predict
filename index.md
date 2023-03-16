# API documentation

### General 总论

Serving through HTTP protocol, the payload for both requests and responses are JSON and files (optional).  
ℹ All routes are RESTful-like designed.  
ℹ For sending json with files in a single request, refer to [python-requests-post-json-and-file-in-single-request](https://stackoverflow.com/questions/19439961/python-requests-post-json-and-file-in-single-request)  
ℹ For serializing NumPy arrays, refer to [numpy-array-to-base64-and-back-to-numpy-array-python](https://stackoverflow.com/questions/6485790/numpy-array-to-base64-and-back-to-numpy-array-python)  

The common JSON fields for all payloads:

```typescript
// request
interface {

}

// response
interface {
  ok: bool,           // success
  error?: str,
}
```

### Data 数据

Time sequence data are store as a **single** `data.csv` file for each task.  
=> To pre-merge multiple *.csv files, see [POST /merge_csv](#post-merge_csv-合并多个csv文件)

⚪ Regression

Can conatin **multiple** columns:

  - First column: datetime string in ISO format
  - Rest columns: float values reading from sensors

```csv
monitorTime,monitorAvgValue
2021-01-01 00:00:00,0.17
2021-01-01 01:00:00,0.15
2021-01-01 02:00:00,0.19
2021-01-01 03:00:00,0.23
```

⚪ Classification

```csv
monitorTime,monitorAvgValue,label
2021-01-01 00:00:00,0.17,0
2021-01-01 01:00:00,0.15,3
2021-01-01 02:00:00,0.19,0
2021-01-01 03:00:00,0.23,1
```

----

### Enums 枚举常量

```cpp
enum TaskType {
  rgr,        // 分类
  clf,        // 回归
};
enum TaskTarget {
  data,       // 数据 := 数值预处理 + 打分类标签 + 划分数据集
  train,      // 训练
  eval,       // 评估
  all,        // 全部
};
enum TaskStatus {
  created,    // 创建任务
  queuing,    // 执行队列中等待
  running,    // 正在执行中
  finished,   // 已完成
};
```

----

### POST /merge_csv 合并多个csv文件

```typescript
// request
// <= multiple *.csv files

// response
// => data.csv
```

### POST /task 创建新任务并运行

```typescript
// request
// <= single *.csv file
interface {
  type: str,          // task type, choose from `TaskType`
  target?: str[],     // task target, choose from `TaskTarget`, default to 'all'
  jobs?: str[],       // scheduled jobs, default to all applicable jobs
  name?: str,         // custom task name
}

// response
interface {
  name: str,          // real task name allocated
}
```

### POST /task/\<name\> 更新任务并重新运行

ℹ 主要用于数据更新，修改作业项，或者想从头制作数据集重新开始 :(

```typescript
// request
// <= single *.csv file (optional), **replacing** the old one
interface {
  target?: str[],     // task target, choose from `TaskTarget`
  jobs?: str[],       // scheduled jobs, default to all applicable jobs
}
```

### GET [/task](/task) 列出任务记录

```typescript
// request
interface {
  status: str[],      // filter by task status, choose from `TaskStatus`
}

// response
interface {
  tasks: str[],       // task name
}
```

### GET /task/\<name\> 查询任务记录

```typescript
// response
interface {
  type: str,          // task type, choose from `TaskType`
  status: str,        // task status, choose from `TaskStatus`
  best?: str,         // job name
  jobs?: {
    "job_name": {     // metrics for 'clf' task
      pres: float,
      recall: float,
      f1: float,
    } | {             // metrics for 'rgr' task
      mae: float,
      mse: float,
      r2: float,
    },
  },
  ts_create: int,     // task create time
  ts_finish?: int,    // train finish time
}
```



### DELETE /task/\<name\> 删除任务记录

```typescript
```

----

### GET [/job](/job) 列出作业模板

```typescript
// response
interface {
  jobs: str[],        // job name
}
```

### GET /job/\<name\> 下载/查询作业模板

```typescript
// request
interface {
  type: str,          // 'yaml' or 'json', default: 'yaml'
}

// response
// => <job_name>.yaml file or 
interface {
  job: {
    // config items converted from *.yaml
  },
}
```

### POST /job/\<name\> 上传作业模板

```typescript
// request:
// NOTE: <name> in url must starts with 'rgr_' or 'clf_'
// <= *.yaml file
interface {
  overwrite?: bool,   // overwrite if exists, default to false
}
```

### DELETE /job/\<name\> 删除任务模板

```typescript
```

----

### POST /infer/\<task_name\> 推断

```typescript
// request
interface {
  job?: str,          // default to the best job
  data: str,          // base64 codec of input.flatten(): np.array
  shape: int[],       // shape of input: np.ndarray
}

// response
interface {
  pred: str,          // base64 codec of pred.flatten(): np.array
  shape: int[],       // shape of pred: np.ndarray
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
// => plain html page
```

----

### GET [/debug](/debug) 系统运行时信息

```typescript
// response
// => plain html page
```

----

<p> by Armit <time> 2023/3/3 </time> </p>
