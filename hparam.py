#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/15 

''' Data '''
TRAIN_DATA_COUNT = 10000   # 1w个重采样序列切片作为一组数据集
TEST_DATA_COUNT  = 1000    # 用1k个重采样序列切片做测试

'''
原始数据总长度1984：每天24个采样点，每周24*7=168个采样点
工业生产的自然周期：天、星期、月、季度、年，这些尺度上要重点考虑
因此作分片训练，考虑某一个覆盖周期

NOTE：按理说，三倍的数据量才能预测一倍数据，如果要预测3天的话你至少得看前9天
实际上由于混沌效应，在输入因素确定后，一定存在某个时间临界点，其之后的值是无论如何无法被预测的……
'''
SEGMENT_SIZE = 168      # 一周
#SEGMENT_SIZE = 360     # 半个月
#SEGMENT_SIZE = 720     # 一个月

# True: 分类
# False: 回归
USE_QT = True

''' Model '''
# 数值特征数目，根据 `preprocess.preprocess()` 里的 `features` 调整
N_FEATURES        = 3
EMBED_WEEKDAY_DIM = 7
EMBED_HOUR_DIM    = 24

if USE_QT:      # 处理为分类问题
  QT_N_BIN    = 100
  FATURE_DIM  = 128
  INPUT_DIM   = EMBED_WEEKDAY_DIM + EMBED_HOUR_DIM + N_FEATURES * FATURE_DIM
  OUTPUT_DIM  = QT_N_BIN    # 处理为分类任务
else:           # 处理为回归问题
  QT_N_BIN      = 100
  FATURE_DIM    = 128
  INPUT_DIM     = EMBED_WEEKDAY_DIM + EMBED_HOUR_DIM + N_FEATURES * FATURE_DIM
  OUTPUT_DIM    = N_FEATURES


''' Train '''
BATCH_SIZE    = 128
EPOCHS        = 1000
LR            = 2e-4
BETAS         = (0.9, 0.999)
WEIGHT_DECAY  = 1e-5

DATA_PATH     = 'data'
if USE_QT:
  DATA_FILE   = 'data/data_qt.pkl'
else:
  DATA_FILE   = 'data/data.pkl'
LOG_PATH      = 'log'
LOG_INTERVAL  = 1000
CKPT_INTERVAL = 100
RANDSEED      = 114514

''' Infer '''
# 知三推一
INFER_STEPS   = SEGMENT_SIZE // 3
