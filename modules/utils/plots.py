#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/15 

import os
from pathlib import Path
from logging import Logger

import matplotlib ; matplotlib.use('agg')
import matplotlib.pyplot as plt
#plt.rcParams['font.sans-serif'] = ['SimHei']    # 显示中文
#plt.rcParams['axes.unicode_minus'] = False      # 正常显示负号

DEBUG_PLOT = os.environ.get('DEBUG_PLOT', False)


def save_figure(fp:Path, title:str=None, logger:Logger=None):
  if not DEBUG_PLOT: return
  if not plt.gcf().axes: return

  plt.suptitle(title or fp.stem)
  #plt.tight_layout()
  plt.savefig(fp, dpi=400)
  if logger: logger.info(f'  save figure to {fp}')
