#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/18 

from pathlib import Path
from argparse import ArgumentParser


''' basic path '''
BASE_PATH = Path(__file__).parent
HTML_PATH = BASE_PATH / 'doc'
JOB_PATH  = BASE_PATH / 'job'
LOG_PATH  = BASE_PATH / 'log'
TMP_PATH  = BASE_PATH / 'tmp'


''' log folder '''
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
PREDICT_FILE    = 'predict.pkl'       # inplace prediction
LOG_FILE        = 'job.log'           # job runner logs

SAVE_META_EVERY  = 300
CHECK_TASK_EVERY = 5


''' HTTP auth '''
AUTH_TOKEN = None


''' cmdline args '''
def cmd_args():
  parser = ArgumentParser()
  parser.add_argument('-D', '--csv_file',     required=True,     type=Path, help='path to a *.csv data file')
  parser.add_argument('-J', '--job_file',                        type=Path, help='path to a *.yaml job file')
  parser.add_argument('-X', '--job_folder',                      type=Path, help='path to a folder of *.yaml job file')
  parser.add_argument(      '--name',         default='test',               help='task name')
  parser.add_argument(      '--target',       default='all',                help='job targets, comma seperated string')
  parser.add_argument(      '--no_overwrite', action='store_true',          help='no overwrite if log folder exists')
  return parser.parse_args()
