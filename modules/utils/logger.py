#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/15 

import sys
from pathlib import Path
import logging
from logging import Logger
from typing import Dict

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

loggers: Dict[str, Logger] = {}


def get_logger(name, log_dp=Path('.')) -> Logger:
  if name not in loggers:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    h = logging.FileHandler(log_dp / 'job.log')
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    loggers[name] = logger
  return loggers[name]


def close_logger(logger:Logger):
  for name, log in loggers.items():
    if log is logger:
      del loggers[name]
      break
  for handler in logger.handlers:
    if isinstance(handler, logging.FileHandler):
      handler.flush()
      handler.close()
