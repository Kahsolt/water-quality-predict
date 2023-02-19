#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/06 

import os
import numpy as np
import pandas
from argparse import ArgumentParser
os.chdir('D:/数据F/泰州排口预警/水质预测模型_新港西')

import joblib
from sklearn.model_selection import GridSearchCV,KFold
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
    'w21001':'TN',
    'w21003':'NH3-N',
    'w21011':'TP'
    }


''' Commandline '''
for t in range(len(factor_lst)):
    yinzi = factor_lst[t]
    factor = factordic.get(yinzi)
    stand = stand_lst[t]
    parser = ArgumentParser()
    parser.add_argument('--csv_fp', metavar='csv_fp', default='data/'+yinzi+'_补值.csv',help='path to yoru .csv file')
    parser.add_argument('--lag', default = 168, type=int, help='window length')  #前序时间168h/336h/720h
    parser.add_argument('--overwrite', action='store_true', help='force overwrite existing ckpt')
    parser.add_argument('--f_len', default = 24, type=int, help='future length')  #未来24h
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

    '''从每天0点开始为1个X，相当于一天一个'''
    X, Y = [], []
    for i in range(len(ts) - lag - f_len):
      X.append([ts[i + j] for j in range(lag)])
      Y.append([ts[i + j + lag] for j in range(f_len)])
    X = np.asarray(X)
    
    '''分类'''
    Y = pd.DataFrame(Y)
    convert_Y = Y.applymap(lambda x:1 if x>stand else 0)
    convert_Y = convert_Y.apply(sum,axis = 1)
    y = convert_Y.apply(lambda x:0 if x==0 else (1 if 0<x<=3 else (2 if 3<x<=5 else 3)))
    y = np.asarray(y)
    
    
    '''划分交叉验证数据集'''
    kf = KFold(n_splits=5,shuffle=True,random_state=42) 
    
    xtrain=[]
    xtest=[]
    ytrain=[]
    ytest=[]
    for a,b in kf.split(X):
        x_tr=X[a,:]
        x_te=X[b,:]
        y_tr=y[a]
        y_te=y[b]
        xtrain.append(x_tr)
        xtest.append(x_te)
        ytrain.append(y_tr)
        ytest.append(y_te)
        
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
        result.loc[count,'accuracy']=accuracy
        #C = confusion_matrix(y_test, y_pred, labels=[i for i in range(lei)])
        #C0 = C0 + C
        Recall = recall_score(y_test,y_pred,average=None)
        Precision = precision_score(y_test,y_pred,average=None)
        f1=f1_score(y_test, y_pred, average=None)
        #显示3类的召回率、精确率和f1
        result.loc[count,'recall']=Recall[3]
        result.loc[count,'precision']=Precision[3]
        result.loc[count,'f1']=f1[3]
        
    result.loc['mean'] = result.mean() #五折，最后输出的结果是五次accuracy、recall、precision和f1的平均
    result1=result.iloc[-1,:]
    result1.to_csv(f'{qy}_{factor}_lag={lag}_24h分类预测结果.csv')