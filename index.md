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

Time sequence data are store as a single *.csv file.

⚪ Regression

Can conatin **multiple** columns:

  - First column: datetime string in ISO format
  - Rest columns: value reading from sensors, usually float numbers

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

### POST /task 创建新任务

see also [Data](#data-数据)

```typescript
// request
// <= *.csv file, see `GET /template.csv`
interface {
  type: str,          // task type, 'rgr' or 'clf'
  name?: str,         // custom task name
  jobs?: str[],       // target jobs, defaults to all available jobs
}

// response
interface {
  name: str,          // real task name allocated
}
```

### GET [/task](/task) 列出任务记录

```typescript
// response
interface {
  tasks: str[],        // name
}
```

### GET /task/\<name\> 查询任务记录

```typescript
// response
interface {
  status: str,        // 'created', 'running', 'finsihed'
  best?: str,         // job name
  jobs?: {
    "job_name": {
      type: str,      // 'clf' or 'rgr'
      metrcis: {      // for 'clf'
        pres: float,
        recall: float,
        f1: float,
      } | {           // for 'rgr'
        mae: float,
        mse: float,
        r2: float,
      },
    },
  },
  ts_create: int,     // task create time
  ts_finish?: int,    // train finish time
}
```

### PUT /task/\<name\> 重新运行任务

### DELETE /task/\<name\> 删除任务记录

```typescript
```

----

### GET [/job](/job) 列出任务模板

```typescript
// response
interface {
  jobs: str[],        // name
}
```

### GET /job/\<name\> 查询任务模板

```typescript
// request
interface {
  type: str,          // 'yaml' or 'json', default: 'yaml'
}

// response
// => *.yaml file or 
interface {
  job: {
    // config items converted from *.yaml
  },
}
```

### POST /job/\<name\> 上传任务模板

```typescript
// request
// <= *.yaml file
interface {
  name?: str,         // job file name, must starts with 'rgr_' or 'clf_'
  overwrite?: bool,   // overwrite if exists
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
  job?: str,          // defaults to the best
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

### GET [/debug](/debug) 系统运行时信息

----

<p> by Armit <time> 2023/3/3 </time> </p>
