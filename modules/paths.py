#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/18 

from pathlib import Path


''' basic path '''
BASE_PATH = Path(__file__).parent.parent
HTML_PATH = BASE_PATH / 'doc'
JOB_PATH  = BASE_PATH / 'job' ; JOB_PATH.mkdir(exist_ok=True)
LOG_PATH  = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)
TMP_PATH  = BASE_PATH / 'tmp' ; TMP_PATH.mkdir(exist_ok=True)
MODEL_PATH = BASE_PATH / 'modules' / 'models'


''' log folder '''
RUNTIME_FILE    = 'runtime.json'      # runtime history for all tasks
TASK_FILE       = 'task.json'         # task descriptor
DATA_FILE       = 'data.csv'          # task resource data, shared by all jobs
JOB_FILE        = 'job.yaml'          # job.yaml copied from template
TIME_FILE       = 'time.pkl'          # seq time (preprocessed), for visualize
PREPROCESS_FILE = 'preprocess.pkl'    # seq values (preprocessed), for visualize
LABEL_FILE      = 'label.pkl'         # encoded target/label, for visualize
STATS_FILE      = 'stats.pkl'         # stats of transform, for inv-transform of model prediction
TRANSFORM_FILE  = 'transform.pkl'     # seq (transformed), for ARIMA train/eval
DATASET_FILE    = 'dataset.pkl'       # dataset (transformed), for other model train/eval
SCORES_FILE     = 'scores.txt'        # job evaluated scores
PREDICT_FILE    = 'predict.pkl'       # inplace prediction results
LOG_FILE        = 'job.log'           # job runner logs
