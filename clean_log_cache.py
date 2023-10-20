#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/20

from config import *

def walk(dp:Path):
  for fp in dp.iterdir():
    if fp.is_dir():
      walk(fp)
    else:
      if fp.suffix == '.png':
        print(f'>> delete {fp.relative_to(BASE_PATH)}')
        fp.unlink()

walk(LOG_PATH)
