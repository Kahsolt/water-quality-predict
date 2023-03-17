#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/19 

import numpy as np

from modules.typing import *


''' transform: 数值变换，用于训练 (必须是可逆的) '''
def log(seq:Seq) -> SeqAndStat:
  return log_apply(seq), tuple()

def std_norm(seq:Seq) -> SeqAndStat:
  avg = seq.mean(axis=0, keepdims=True)
  std = seq.std (axis=0, keepdims=True)
  seq_n = std_norm_apply(seq, avg, std)
  return seq_n, (avg, std)

def minmax_norm(seq:Seq) -> SeqAndStat:
  vmin = seq.min(axis=0, keepdims=True)
  vmax = seq.max(axis=0, keepdims=True)
  seq_n = minmax_norm_apply(seq, vmin, vmax)
  return seq_n, (vmin, vmax)

# ↑↑↑ above are valid transforms ↑↑↑


''' transform (apply): 数值变换 '''
def log_apply(seq:Seq) -> Seq:
  return np.log(seq + 1e-8)

def std_norm_apply(seq:Seq, avg:ndarray, std:ndarray) -> Seq:
  return (seq - avg) / std

def minmax_norm_apply(seq:Seq, vmin:ndarray, vmax:ndarray) -> Seq:
  return (seq - vmin) / (vmax - vmin)

''' transform (inverse): 数值逆变换 '''
def log_inv(seq:Seq) -> Seq:
  return np.exp(seq)

def std_norm_inv(seq:Seq, avg:ndarray, std:ndarray) -> Seq:
  return seq * std + avg

def minmax_norm_inv(seq:Seq, vmin:ndarray, vmax:ndarray) -> Seq:
  return seq * (vmax - vmin) + vmin
