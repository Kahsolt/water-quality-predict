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

  clf_tasks, rgr_tasks = [], []
  for log_folder in args.log_path.iterdir():
    if log_folder.is_file(): continue
    with open(log_folder / 'metric.txt', 'r', encoding='utf-8') as fh:
      data = fh.read().strip()

    expname = log_folder.name
    if 'mse' in data:
      mae, mse, r2 = [float(line.split(':')[-1].strip()) for line in data.split('\n')]
      rgr_tasks.append((r2, expname))
    elif 'f1' in data:
      prec, recall, f1 = [float(line.split(':')[-1].strip()) for line in data.split('\n')]
      clf_tasks.append((f1, expname))
    else:
      print(f'Error: cannot decide task_type for {expname}')

  clf_tasks.sort(reverse=True)
  rgr_tasks.sort(reverse=True)

  with open(args.log_path / 'metric-ranklist.txt', 'w', encoding='utf-8') as fh:
    def log(s=''):
      fh.write(s + '\n')
      print(s)
    
    if clf_tasks:
      log('[clf] F1 score:')
      for score, name in clf_tasks:
        log(f'{name}: {score}')
    log()

    if rgr_tasks:
      log('[rgr] R2 score:')
      for score, name in rgr_tasks:
        log(f'{name}: {score}')
    log()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-D', '--job_folder', default=Path('job'), type=Path, help='path to a folder of *.yaml job file')
  parser.add_argument('--log_path', default=Path('log'), type=Path, help='path to log root folder')
  parser.add_argument('--no_overwrite', action='store_true', help='no overwrite if log folder exists')
  args = parser.parse_args()

  run_batch(args)
