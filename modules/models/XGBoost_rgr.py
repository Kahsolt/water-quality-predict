#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/06 

import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
os.chdir('D:/数据F/泰州排口预警/封装算法/XGB_回归')
import joblib
import shap
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from preprocess import *

factordic = {
    'w01001':'ph',
    'w01018':'COD',
    'w21001':'TN',
    'w21003':'NH3',
    'w21011':'TP'
    }

lag_flen = [(96,6)]  #前序96小时预测后面6小时

for m in range(len(lag_flen)):
    
    for k in range(len(factor_lst)):
        yinzi = 'w21003'  #igCode
        factor = factordic.get(yinzi)
        ''' Commandline '''
        parser = ArgumentParser()
        parser.add_argument('--csv_fp', metavar='csv_fp', default=qy+'/'+yinzi+'_补值.csv',help='path to your .csv file')
        parser.add_argument('--lag', default = lag_flen[m][0], type=int, help='window length')
        parser.add_argument('--f_len', default = lag_flen[m][1], type=int, help='future length')
        parser.add_argument('--overwrite', action='store_true', help='force overwrite existing ckpt')
        args = parser.parse_args()
        
        
        ''' Hparam '''

        csv_fp = args.csv_fp
        lag = args.lag
        f_len = args.f_len
        index = os.path.splitext(os.path.basename(csv_fp))[0]
        save_fp = os.path.join('log', f'xgboost_{yinzi}_补值_lag={lag}_flen={f_len}.pkl')
        os.makedirs('log', exist_ok=True)
        
        
        ''' RData'''
        rdata = pd.read_csv(csv_fp, parse_dates=['monitorTime'], index_col='monitorTime', date_parser=pd.to_datetime)
        ts = rdata['monitorAvgValue']
        
        '''对全集进行异常值处理'''
        ts1 = np.array(ts)
        ts1 = yichang(ts1)  #异常值
        #ts1, tsmin, tsmax = minmax_norm(ts1)   #归一化
        rdata['monitorAvgValue'] = ts1
        ts = rdata['monitorAvgValue']
        
        ''' Data'''
        # 知 lag 推 1
        traints,testts = train_test_ts(ts,lag = lag)
        X_train,y_train = ts_set_transfer(traints,lag = lag)
        X_test,y_test = ts_set_transfer(testts,lag = lag)
#%%           
        if not os.path.exists(save_fp) or args.overwrite:
        
          ''' Model '''
          model = xgb.XGBRegressor() 
          model_gs = GridSearchCV(
            model,   
            {
              'max_depth': [3, 4, 5, 6, 7],
              'n_estimators': [300],
              'learning_rate': [0.1, 0.12, 0.125, 0.13, 0.15],
              'min_child_weight': [1, 3, 5],
              'seed':[42],
              'early_stopping_round': [30]
            },
            cv=3,   
            verbose=2,
            n_jobs=-1,  
            scoring='neg_median_absolute_error',
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
        
        ''' Shap 
        explainer = shap.Explainer(model)
        shap_values = explainer(X_train)
        shapfig = plt.figure()
        shap.summary_plot(shap_values, X_train, plot_type="bar")
        shapfig.savefig(qy+'//xgb '+qy+factor+' '+str(lag)+'-'+str(f_len)+'shap.jpeg',dpi = 300)'''
        
        ''' Infer '''
        if f_len == 1:
            pred = model.predict(X_train)
        else :
            pred = []
            for i in range(0,len(X_train)):
                res = i % f_len 
                if res == 0:
                    pass
                else:
                    for j in range(1,res+1):
                        X_train[i,-j] = model.predict(X_train[i-j].reshape(1,-1))
                tru_pred = model.predict(X_train[i].reshape(1,-1))
                pred.append(tru_pred)
         
                
        ''' 模型性能 '''
        truth = y_train
        pred = np.asanyarray(pred)
        #truth=renorm(np.array(truth), tsmin, tsmax)
        #pred=renorm(np.array(pred), tsmin, tsmax)
        predict_result = pd.DataFrame()
        predict_result['truth'] = truth
        predict_result['predict'] = pred
        #输出测试集拟合结果
        predict_result.to_csv(qy+'//xgb '+qy+factor+' '+str(lag)+'-'+str(f_len)+' predict.csv',index = False)
        
        mae = mean_absolute_error(truth, pred)
        rmse = np.sqrt(mean_squared_error(truth, pred))
        r2 = r2_score(truth, pred)
        print('mae:', mae)
        print('rmse:', rmse)
        print('r2:', r2)
        result_table = pd.DataFrame()
        result_table['result'] = []
        result_table.loc['factor'] = factor
        result_table.loc['starttime'] = rdata.index[0]
        result_table.loc['endtime'] = rdata.index[-1]
        result_table.loc['mae'] = mae
        result_table.loc['rmse'] = rmse
        result_table.loc['r2'] = r2
        result_table.loc['hparam'] = model_gs.best_estimator_
        #输出指标
        result_table.to_csv(qy+'//xgb '+qy+factor+' '+str(lag)+'-'+str(f_len)+' result.csv')
        
        
        
