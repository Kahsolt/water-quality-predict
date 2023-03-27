#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/18 

from pathlib import Path


''' basic path '''
BASE_PATH = Path(__file__).parent
HTML_PATH = BASE_PATH / 'doc'
JOB_PATH  = BASE_PATH / 'job' ; JOB_PATH.mkdir(exist_ok=True)
LOG_PATH  = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)
TMP_PATH  = BASE_PATH / 'tmp' ; TMP_PATH.mkdir(exist_ok=True)


''' log folder '''
RUNTIME_FILE    = 'runtime.json'      # data shared by all tasks

TASK_FILE       = 'task.json'         # task descriptor
DATA_FILE       = 'data.csv'          # data shared by all jobs of the same task
DATA_ZIP_FILE   = 'data_hist.zip'     # history archive of raw data

JOB_FILE        = 'job.yaml'          # job.yaml copied from template
TIME_FILE       = 'time.pkl'          # seq time (preprocessed), for visualize
PREPROCESS_FILE = 'preprocess.pkl'    # seq values (preprocessed), for visualize
LABEL_FILE      = 'label.pkl'         # encoded target/label, for visualize
STATS_FILE      = 'stats.pkl'         # stats of transform, for inv-transform of model prediction
TRANSFORM_FILE  = 'transform.pkl'     # seq (transformed), for ARIMA train/eval
DATASET_FILE    = 'dataset.pkl'       # dataset (transformed), for other model train/eval
SCORES_FILE     = 'scores.txt'        # evaluated scores
PREDICT_FILE    = 'predict.pkl'       # inplace prediction
LOG_FILE        = 'job.log'           # job runner logs

SAVE_META_EVERY  = 300
CHECK_TASK_EVERY = 5


''' HTTP auth '''
AUTH_TOKEN = None
