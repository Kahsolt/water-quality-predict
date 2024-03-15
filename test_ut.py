#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/15

# server unit test

import os

MODULE_LIST = [
  'modules.utils.config',
  'modules.preprocess',
  'modules.dataset',
  'modules.transform',
]

for mod in MODULE_LIST:
  r = os.system(f'python -m {mod}')
  assert r == 0
