#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/18 

from pathlib import Path


''' basic path '''
BASE_PATH = Path(__file__).parent
HTML_PATH = BASE_PATH / 'doc'
JOB_PATH  = BASE_PATH / 'job'
LOG_PATH  = BASE_PATH / 'log'
TMP_PATH  = BASE_PATH / 'tmp'


''' log folder layout '''
RUNTIME_FILE    = 'runtime.json'      # data shared by all tasks

TASK_FILE       = 'task.json'         # task descriptor
DATA_FILE       = 'data.csv'          # data shared by all jobs of the same task
DATA_ZIP_FILE   = 'data_hist.zip'     # history archive of raw data

JOB_FILE        = 'job.yaml'          # job.yaml copied from template
PREPROCESS_FILE = 'preprocess.pkl'    # seq (preprocessed), for visualize
LABEL_FILE      = 'label.pkl'         # encoded target/label, for visualize
STATS_FILE      = 'stats.pkl'         # stats of transform, for inv-transform of model prediction
TRANSFORM_FILE  = 'transform.pkl'     # seq (transformed), for ARIMA train/eval
DATASET_FILE    = 'dataset.pkl'       # dataset (transformed), for other model train/eval
SCORES_FILE     = 'scores.txt'        # evaluated scores
LOG_FILE        = 'job.log'           # job runner logs
