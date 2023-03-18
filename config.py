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
JOB_FILE        = 'job.yaml'
DATA_FILE       = 'data.csv'
PREPROCESS_FILE = 'preprocess.pkl'    # seq preprocessed
LABEL_FILE      = 'label.pkl'         # seq encoded label
STATS_FILE      = 'stats.pkl'         # transforming stats for seq
TRANSFORM_FILE  = 'transform.pkl'     # seq transformed
DATASET_FILE    = 'dataset.pkl'       # dataset transformed
