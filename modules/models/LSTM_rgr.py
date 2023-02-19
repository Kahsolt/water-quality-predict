import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error as mse,mean_absolute_error as mae,r2_score as r2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


#将数据处理成seq和对应标签
def single_create_sequences(input_data,tw,fl):
    pack = []
    for i in range(len(input_data)-tw-fl):
        train_seq = input_data[i:i+tw] 
        train_label = input_data[i+tw:i+tw+fl] 
        pack.append((train_seq,train_label))
    return pack     

      
def plot(for_vals,fac_vals,factor_name,site_name,future_len,path):
    plt.plot(np.arange(len(for_vals)),for_vals,c='red',label='forcast')#预测集预测值曲线 
    plt.plot(np.arange(len(fac_vals)),fac_vals,c='blue',label='true')#实际值曲线-逐步预测效果
    plt.legend(loc='best')
    plt.xlabel("小时数/h ")
    plt.ylabel(factor_name)
    plt.title( site_name )
    plt.savefig(path+str(factor_name)+str(future_len)+'h new.jpeg',dpi=300)
    plt.show()  


import itertools
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score as r2


class Loss:
    def single_test_loss(self,data_seq,result=False): #用于测试集上返回连续非重复时间段上的真实值与预测值，loss值
        self.eval() #预测模式
        with torch.no_grad():
            fors,facs = [],[]
            for seq, label in data_seq:
                self.hidden_cell = (torch.zeros(1,1,self.hidden_size),
                                    torch.zeros(1,1,self.hidden_size))
                y_pred = self.forward(seq)
                fors.append(y_pred)
                facs.append(label[:,-1].view(-1))
            for_tensor = torch.stack(fors,dim=0)
            fac_tensor = torch.stack(facs,dim=0)
            RMSE = np.sqrt(mse(for_tensor,fac_tensor)) #均方根误差
            MAE = mae(for_tensor,fac_tensor) #平均绝对误差
            R2 = r2(for_tensor,fac_tensor)
            if result == True:
                for_norm_vals = time_continual(for_tensor.numpy())
                return RMSE,MAE,R2,for_norm_vals
            else:
                return RMSE,MAE,R2
    def single_train_loss(self,train_seq,epochs): #训练集上训练并计算最终loss
        #定义损失函数和优化器
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(),lr=0.001)
        #训练
        mse_loss = np.inf
        epoch_loss = np.zeros((epochs,3)) #记录每一轮训练的三类loss计算值
        for i in range(epochs):
            self.train() # 切换到训练模式
            for seq, label in train_seq:
                #初始化模型参数，梯度
                optimizer.zero_grad()
                self.h_init()
                y_pred = self(seq)#传参自动调用__call__()方法->forward()
                single_loss = loss_function(y_pred, label[:,-1].view(-1))
                single_loss.backward()
                optimizer.step()
                mse_loss = single_loss.item()
            
            print(f'epoch:{i:3} last loss:{mse_loss:10.8f}') #打印该轮最后一个seq的均方误差到控制台
            #计算train_set loss:
            RMSE,MAE,R2 = self.single_test_loss(train_seq)
            epoch_loss[i] = [RMSE,MAE,R2]
        return epoch_loss
    

class LSTM(nn.Module, Loss):
    def __init__(self, input_size,output_size,hidden_size=128,num_layers=1):
        super(LSTM,self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers)
        self.linear = nn.Linear(hidden_size,output_size)
        self.h_init()
    def h_init(self):
        self.hidden_cell = (torch.zeros(1,1,self.hidden_size), #batch_size= 1,bidirection=False
                           torch.zeros(1,1,self.hidden_size)) 
    def forward(self, input_seq):
        #lstm_out : (L, N, D * H_out)
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq),1,-1), self.hidden_cell) #batch_size= 1
        prediction = self.linear(lstm_out[-1][-1])
        return prediction


#返回非重复连续时间上的值
def time_continual(tdarray):
    temp = []
    forward_len = len(tdarray[0])
    for i in range(0,len(tdarray),forward_len):
        temp.append(tdarray[i])
    temp[-1] = temp[-1][:len(tdarray)-i]
    temp.append(tdarray[-1])
    return list(itertools.chain(*temp))


#模型的训练预测,batch_size = 1,num_layers = 1,bidirection = 1
#train_window：前多少天对今天有影响
#input_size: 特征数
#hidden_size:隐层数,即为ht的维数
#epochs: 训练轮次
#train_norm/test_norm:张量形式的归一化数据,第二维度为input_size
def single_train_forcast(train_norm,test_norm,train_window,future_len,epochs):
    #准备数据的训练集输入序列、测试机输入序列              
    train_seq = single_create_sequences(train_norm,train_window,future_len)
    #获得输入特征数
    input_size = train_norm.shape[1]
    #生成模型
    model = LSTM(input_size,future_len) 
    #计算train_set loss
    epoch_loss = model.single_train_loss(train_seq,epochs)
    #计算test_set loss并返回连续非重复时间段上预测值：
    test_series = torch.cat((torch.FloatTensor(train_norm[-train_window:]),test_norm),0)
    test_seq = single_create_sequences(test_series,train_window,future_len)                 
    RMSE,MAE,R2,for_vals = model.single_test_loss(test_seq,result=True)
    return epoch_loss,RMSE,MAE,R2,for_vals

def single_model(df,test_rate ,factor_name,site_name,train_window,future_len,epochs,path): #默认预测未来24小时
    for_vals = []
    fac_vals = []   
    test_size=int(len(df)*test_rate)
    #预处理后的训练数据集、每一序列归一化参数[最大值、最小值]的数组
    #astype创建新的内存空间！！！    
    train_norm,norm_param = train_preprocess(df.iloc[:len(df)-test_size,:].values.astype(float))
    test_norm = test_preprocess(df[-test_size:].values.astype(float),norm_param)#获得归一化后的测试集npArray,真实值序列
    fac_vals = preprocess(df.iloc[-test_size:,-1].values.astype(float))#实际值，经补值
    #训练集上每轮次损失函数值，测试集上损失函数值，测试集上预测值
    epoch_loss,RMSE,MAE,R2,for_norm_vals = single_train_forcast(train_norm,test_norm,train_window,future_len,epochs)#future_len即为output_size
    for_vals = re_norm(for_norm_vals,norm_param[-1][0],norm_param[-1][1])#反归一化后的预测值
    #站点预测情况作图
    plot(for_vals,fac_vals,factor_name,site_name,future_len,path)

    return for_vals,fac_vals,RMSE,MAE,R2  #返回测试集拟合值和模型性能

def evaluate(true,forcast):
    RMSE = np.sqrt(mse(true,forcast)) #均方根误差
    MAE = mae(true,forcast) #平均绝对误差
    R2 = r2(true,forcast)
    return RMSE,MAE,R2

if __name__ == "__main__":
    lst=['NH3-N']  #监测因子
    lst1=['w21003']  #igCode
    for y in range(len(lst)):
        path='data//'
        site= '新港西'
        df=pd.read_csv(path+lst1[y]+'_补值.csv',index_col=0).iloc[0:1000,:]
        '''模型输入'''
        for_vals,fac_vals,RMSE,MAE,R2=single_model(df,test_rate=0.1,factor_name=lst[y],site_name =site,train_window=96,future_len = 6,epochs = 30,path=path)
        #训练集：测试集=9：1，前序时间=96h，预测未来时间=6h，训练次数=30，可根据情况调整
        df3=pd.DataFrame(columns=['true_preprocess','forcast'])
        df3['forcast']=for_vals
        df3['true_preprocess']=fac_vals
        '''输出测试集拟合值'''
        df3.to_csv(lst[y]+'_6h预测.csv',index=False)
        
        RMSE,MAE,R2=evaluate(fac_vals, for_vals)
        df4=pd.DataFrame(columns=['RMSE','MAE','R2'],index=[0])
        df4['RMSE']=RMSE
        df4['MAE']=MAE
        df4['R2']=R2
        '''输出模型性能'''
        df4.to_csv(lst[y]+'_result.csv',index=False)
            