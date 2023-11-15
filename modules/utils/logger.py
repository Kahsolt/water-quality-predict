#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/15 

import sys
from pathlib import Path
import logging
from logging import Logger

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def get_logger(name, log_dp=Path('.')) -> Logger:
  logger = logging.getLogger(name)
  logger.setLevel(logging.WARN)
  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  h = logging.FileHandler(log_dp / 'job.log')
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)
  return logger


def close_logger(logger:Logger):
  for handler in logger.handlers:
    if isinstance(handler, logging.FileHandler):
      handler.flush()
      handler.close()
