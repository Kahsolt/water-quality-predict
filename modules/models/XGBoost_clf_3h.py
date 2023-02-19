#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/06 


import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
os.chdir('D:/数据F/泰州排口预警/封装算法/XGB_分类')

from preprocess import *
import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from preprocess import *
import random
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

qy= '新港西'
factor_lst = ['w21003']   #igCode
stand_lst = [1]  #标准值
factordic = {
    'w01001':'ph',
    'w01018':'COD',
    'w21001':'tn',
    'w21003':'nh3',
    'w21011':'tp'
    }

''' Commandline '''
for t in range(len(factor_lst)):
    yinzi = factor_lst[t]
    factor = factordic.get(yinzi)
    stand = stand_lst[t]
    parser = ArgumentParser()
    parser.add_argument('--csv_fp', metavar='csv_fp', default='data/'+yinzi+'_补值.csv',help='path to yoru .csv file')
    parser.add_argument('--lag', default = 48, type=int, help='window length')  #前序时间24h/48h/168h
    parser.add_argument('--overwrite', action='store_true', help='force overwrite existing ckpt')
    parser.add_argument('--f_len', default = 3, type=int, help='future length')  #未来3h
    args = parser.parse_args()
    
    
    csv_fp = args.csv_fp
    index = os.path.splitext(os.path.basename(csv_fp))[0]
    lag = args.lag
    f_len = args.f_len
    #save_fp = os.path.join('log', f'xgboost_{index}_lag={lag}_flen={f_len}.pkl')
    #os.makedirs('log', exist_ok=True)
    
    ''' RData'''
    rdata = pd.read_csv(csv_fp, parse_dates=['monitorTime'], index_col=0, date_parser=pd.to_datetime)
    ts1 = rdata['monitorAvgValue'].values
            
    '''对全集进行异常值处理'''
    rdata['monitorAvgValue'] = yichang(ts1) #异常值
    ts = rdata['monitorAvgValue']

    ''' Data'''
        # 知 lag 推 1
    X, Y = [], []
    for i in range(len(ts) - lag - f_len):
      X.append([ts[i + j] for j in range(lag)])
      Y.append([ts[i + j + lag] for j in range(f_len)])
    X = np.asarray(X)
    
    '''分类'''
    Y = pd.DataFrame(Y)
    convert_Y = Y.applymap(lambda x:1 if x>stand else 0)
    y = convert_Y.apply(sum,axis = 1)
    y = np.asarray(y)
    
    ''' 划分训练测试集 '''
    X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1,random_state = 0)   # random_state=50/shuffle = False
    
    
    if (np.sum(y_test==3) != 0)&(np.sum(y_test==0) != 0):
        save_fp = os.path.join('log', f'xgboost_{index}_percentile_lag={lag}_flen={f_len}.pkl')
        os.makedirs('log', exist_ok=True)    
        if not os.path.exists(save_fp) or args.overwrite:
            
            ''' Model '''
            model = XGBClassifier()
            model_gs = GridSearchCV(
            model,   
            {
            'max_depth': [3, 4, 5, 6, 7],
            'n_estimators': [300],
            'learning_rate': [0.05, 0.1, 0.12, 0.125, 0.13, 0.15],
            #'min_child_weight': [1, 3, 5],
            'seed':[42],
            'early_stopping_round': [30],
            'use_label_encoder':[False]
            },
            cv=3,   
            verbose=1,
            n_jobs=-1,  
            scoring='accuracy',
                  )
            
            ''' Train '''
            model_gs.fit(X_train, y_train)
            print('Best: %f using %s' % (model_gs.best_score_, model_gs.best_params_))
            model = model_gs.best_estimator_
            
            ''' Save '''
            joblib.dump(model_gs, save_fp)
        else:
            ''' Load '''
            model_gs = joblib.load(save_fp)
            print('Best: %f using %s' % (model_gs.best_score_, model_gs.best_params_))
            model = model_gs.best_estimator_
        
        '''Infer'''
        pred = model.predict(X_test)
    
        ''' Result '''
        y_pred = np.asanyarray(pred)
        '''模型性能'''
        result = pd.DataFrame(columns=['accuracy','recall','precision','f1'],index=[0])
        accuracy = accuracy_score(y_test, y_pred)
        result.loc[0,'accuracy']=accuracy
        #C = confusion_matrix(y_test, y_pred, labels=[i for i in range(lei)])
        #C0 = C0 + C
        Recall = recall_score(y_test,y_pred,average=None)
        Precision = precision_score(y_test,y_pred,average=None)
        f1=f1_score(y_test, y_pred, average=None)
        #显示3类的召回率、精确率和f1
        result.loc[0,'recall']=Recall[3]
        result.loc[0,'precision']=Precision[3]
        result.loc[0,'f1']=f1[3]
        result.to_csv(f'{qy}_{factor}_lag={lag}_3h分类预测结果.csv')
    else:
        print(qy+factor+'没有正例/反例')