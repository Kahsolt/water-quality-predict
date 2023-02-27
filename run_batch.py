#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/19 

from argparse import ArgumentParser
from pathlib import Path

from run import run


def run_batch(args):
  for job_file in args.job_folder.iterdir():
    if job_file.suffix not in ['.yaml', '.yml']: continue

    print(f'>> [run] {job_file}')
    args.job_file = job_file
    run(args)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-D', '--job_folder', default=Path('job'), type=Path, help='path to a folder of *.yaml job file')
  parser.add_argument('--log_path', default=Path('log'), type=Path, help='path to log root folder')
  parser.add_argument('--no_overwrite', action='store_true', help='no overwrite if log folder exists')
  args = parser.parse_args()

  run_batch(args)
