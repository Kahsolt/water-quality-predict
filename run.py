#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/19 

# run train jobs (local)

from modules.runner import cmd_args, run_file, run_folder


if __name__ == '__main__':
  args = cmd_args()
  assert any([args.job_file, args.job_folder]), 'must specify either --job_file xor --job_folder'

  if args.job_file:   run_file  (args)
  if args.job_folder: run_folder(args)
