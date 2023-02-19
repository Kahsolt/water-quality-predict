#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/15 

import os
import sys
import logging

import matplotlib.pyplot as plt
import torch

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging


def load_checkpoint(checkpoint_path, model, optimizer=None):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  iteration = checkpoint_dict['iteration']
  learning_rate = checkpoint_dict['learning_rate']
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
  state_dict = checkpoint_dict['model']
  new_state_dict= {}
  for k, v in model.state_dict().items():
    try:
      new_state_dict[k] = state_dict[k]
    except:
      logger.info("%s is not in the checkpoint" % k)
      new_state_dict[k] = v
  model.load_state_dict(new_state_dict)
  logger.info(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration})")
  return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
  logger.info(f"Saving model and optimizer state at iteration {iteration} to {checkpoint_path}")
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save({'model': state_dict,
              'iteration': iteration,
              'optimizer': optimizer.state_dict(),
              'learning_rate': learning_rate}, checkpoint_path)


def get_logger(model_dir, filename="train.log"):
  global logger
  logger = logging.getLogger(os.path.basename(model_dir))
  logger.setLevel(logging.DEBUG)
  
  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  h = logging.FileHandler(os.path.join(model_dir, filename))
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)
  return logger


def plot_predict(ys:list, y_hats:list, save_fp:str='predict.png'):
  n_figs = len(ys)
  for i, (y, y_hat) in enumerate(zip(ys, y_hats), start=1):
    plt.subplot(n_figs, 1, i)
    plt.plot(y_hat, 'r')
    plt.plot(y, 'b')

  print(f'[plot_predict] save to {save_fp}')
  plt.savefig(save_fp)

  plt.show()
