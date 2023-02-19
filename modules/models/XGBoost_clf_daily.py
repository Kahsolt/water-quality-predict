#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/06 

import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
os.chdir('D:/数据F/泰州排口预警/新港西预测/xgboost_日数据_分类')

import joblib
from sklearn.model_selection import GridSearchCV,KFold
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score
import matplotlib.pyplot as plt
from preprocess import *
import random

plt.rcParams['font.sans-serif'] = ['SimHei']
qy= '新港西'
yinzi= 'w21003'  #igCode
factor= 'NH3-N'    #监测因子
stand= 1   #标准值


for lags in [7,14,30]:   #前序时间7d/14d/30d
    ''' Commandline '''
    parser = ArgumentParser()
    parser.add_argument('--csv_fp', metavar='csv_fp', default='data/'+yinzi+'_日均值.csv',help='path to yoru .csv file')
    parser.add_argument('--lag', default = lags, type=int, help='window length')
    parser.add_argument('--overwrite', action='store_true', help='force overwrite existing ckpt')
    parser.add_argument('--f_len', default = 1, type=int, help='future length')
    args = parser.parse_args()
    
    
    csv_fp = args.csv_fp
    index = os.path.splitext(os.path.basename(csv_fp))[0]
    lag = args.lag
    f_len = args.f_len   #往后预测1d
    #save_fp = os.path.join('log', f'xgboost_{index}_lag={lag}_flen={f_len}.pkl')
    #os.makedirs('log', exist_ok=True)
    
    ''' RData'''
    data = pd.read_csv(csv_fp, parse_dates=['date'], index_col=0, date_parser=pd.to_datetime)
    data = pd.DataFrame(data)
    data.columns = ['dayavg']
    data['flag'] = 0
    
    '''对全集进行异常值处理'''
    tx = np.array(data.dayavg)
    tx1 = yichang(tx)  #异常值
    data['dayavg'] = tx1
    
    '''数据分类'''
    data['flag']=0
    data.loc[data[data.dayavg > stand].index.tolist(),'flag'] = 1
    data['flag'].fillna(0,inplace=True)
    ts=np.array(data.dayavg)
    ts1=np.array(data.flag)
    
    ''' Data'''
        # 知 lag 推 1
    X, Y = [], []
    for i in range(len(ts) - lag ):
        X.append([ts[i + j] for j in range(lag)])
        Y.append(ts1[i + lag] )
    X_ = np.asarray(X)
    y_ = np.asarray(Y)
    
    #%%
    
    '''划分交叉验证数据集'''
    kf = KFold(n_splits=5,shuffle=True,random_state=42) 
    
    xtrain=[]
    xtest=[]
    ytrain=[]
    ytest=[]
    for a,b in kf.split(X_):
        x_tr=X_[a,:]
        x_te=X_[b,:]
        y_tr=y_[a]
        y_te=y_[b]
        xtrain.append(x_tr)
        xtest.append(x_te)
        ytrain.append(y_tr)
        ytest.append(y_te)

    #C0 = np.zeros([2,2])
    result = pd.DataFrame(columns=['accuracy','recall','precision','f1'],index=[i for i in range(5)])
    
    '''交叉训练与验证'''
    for count in range(5):
        ''' Hparam '''
        save_fp = os.path.join('log', f'xgboost_{index}_lag={lag}_flen={f_len}_{count}.pkl')
        os.makedirs('log', exist_ok=True)
        X_train = xtrain[count]
        X_test = xtest[count]
        y_train = ytrain[count]
        y_test = ytest[count]
        
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
              'early_stopping_round': [30]
            },
            cv=3,   
            verbose=1,
            n_jobs=-1,  
            scoring='accuracy',
            #scoring='neg_mean_absolute_error',
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
        
                
        
        ''' Result '''
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        
        '''模型性能'''
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        result.loc[count,'accuracy']=accuracy
        #C = confusion_matrix(y_test, y_pred, labels=[i for i in range(lei)])
        #C0 = C0 + C
        Recall = recall_score(y_test,y_pred,average=None)
        Precision = precision_score(y_test,y_pred,average=None)
        f1=f1_score(y_test, y_pred, average=None)
        #显示超标类的召回率、精确率和f1
        result.loc[count,'recall']=Recall[1]
        result.loc[count,'precision']=Precision[1]
        result.loc[count,'f1']=f1[1]

    result.loc['mean'] = result.mean() #五折，最后输出的结果是五次accuracy、recall、precision和f1的平均
    result1=result.iloc[-1,:]
    result1.to_csv(f'{qy}_{factor}_lag={lag}_日均值分类预测结果.csv')
