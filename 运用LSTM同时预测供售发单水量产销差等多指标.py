# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:05:35 2019

@author: XieJie
"""
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import os
os.getcwd() 
# 构建将间序列转换为监督学习的函数
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all togethe
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
 
# 加载数据集
#def parse(y,m,d,h):
#    return dt.datetime.strptime(' '.join([y,m,d,h]), '%Y %m %d %H')
#
#dataset = pd.read_csv(r'.\mypyworks\StatLedger\数据表\raw.csv', 
#                      parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
#加载温度、湿度日数据
dataset_d = pd.read_excel(r"./mypyworks/自来水数据\20191120刘博士温度湿度降雨量.xlsx",
                         parse_dates = ['日期'])

 
dataset_M = dataset_d.groupby(pd.Grouper(key='日期',freq='MS')).mean()


# 加载售水数据集    
dataset = pd.read_excel(r"./mypyworks/自来水数据/售水相关月数据1999-2019(20191204整合).xlsx",
                         parse_dates = ['日期'],index_col=0)

dataset = pd.merge(dataset,dataset_M,how='outer',left_index=True,right_index=True)
dataset.info()
# 保存到文件中
#dataset.to_csv('处理后数据.csv')
# 加载数据集
#dataset = pd.read_csv('处理后数据.csv', header=0, index_col=0)
#选择样本时期
dataset = dataset['2015':'2019-10']
dataset.info()
#特征选择
dataset = dataset[['公司净水售水量','水厂供水总量','产销差率','发单水量合计',
         '自然增长用户','存量用户','抄表到户改造总表数(增加数量）','平均温度（C）','降雨量','日照时数']]


# 手动更改列名

dataset.index.name = 'date'
# 把所有NA值用0替换
dataset.fillna(0, inplace=True)




values = dataset.values
type(values)
# 指定要绘制的列
groups = range(10)
i = 1
# 绘制每一列
plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
plt.show()  

from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import mean_squared_error
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM 



# 确保所有数据是浮点数
values = values.astype('float32')
# 归一化特征
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# 指定滞后时间大小
n_steps = 1
n_features = 10
# 构建监督学习问题
reframed = series_to_supervised(scaled, n_steps, 1)
# 丢弃不想预测的列
reframed.drop(reframed.columns[[14,15,16,17,18,19]], axis=1, inplace=True)
print(reframed.head())
reframed.to_csv('处理后数据.csv')

# 加载数据集
reframed = pd.read_csv('处理后数据.csv', header=0, index_col=0)
# 分为训练集和测试集
values = reframed.values
n_train_months = 4 * 12 - 1
train = values[:n_train_months, :]
test = values[n_train_months:, :]
# 分为输入和输出
n_obs = n_steps * n_features
train_X, train_y = train[:, :n_obs], train[:, n_obs:]
test_X, test_y = test[:, :n_obs], test[:, n_obs:]
print(train_X.shape, len(train_X), train_y.shape)
# 为了lstm模型将训练数据集重塑为三维形状 [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_steps, n_features))
test_X = test_X.reshape((test_X.shape[0], n_steps, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# 设计神经网络
model = Sequential()
model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(4))
model.compile(loss='mae', optimizer='adam')
# 拟合网络模型
history = model.fit(train_X, train_y, epochs=50, batch_size=10, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# 绘制历史数据

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()



#计算R方
trainyhat = model.predict(train_X)
train_X = train_X.reshape((train_X.shape[0], n_steps*n_features))

from sklearn.metrics import mean_squared_error
for i in range(train_y.shape[1]):
    R_square = 1-(mean_squared_error(train_y[:,i],trainyhat[:,i])/np.var(train_y[:,i]))
    print(R_square)




# 作出预测
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_steps*n_features))
# 反向转换预测值比例
inv_yhat = concatenate((yhat, test_X[:, -6:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,:4]
# 反向转换实际值大小
test_y = test_y.reshape((len(test_y), 4))
inv_y = concatenate((test_y, test_X[:, -6:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,:4]







groups = range(inv_y.shape[1])
#计算预测指标的平均绝对误差（mae），均方根误差(rmse)，平均绝对百分比误差(mape)
def print_rmse_mape(abs_):
    mae=abs_.mean()#Mean Absolute Error ，平均绝对误差
    rmse= sqrt((abs_**2).mean()) #Root Mean Square Error,均方根误差
    mape=(abs_/inv_y[:,group]).mean()# mean absolute percentage error，平均绝对百分比误差
    print("{}的平均绝对误差MAE={}；\n均方根误差RMSE={:.3f}；\n平均绝对百分比误差MAPE={:.2%}。".format(dataset.columns[group],mae,rmse,mape))

for group in groups:
    
    abs_=pd.Series(inv_y[:,group]-inv_yhat[:,group]).abs()
    
    print_rmse_mape(abs_)       


#画出预测结果：蓝色为原数据，红色为测试数据集的预测值
i = 1
# 绘制每一列
plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(inv_y[:, group])
    plt.plot(inv_yhat[:, group])
    plt.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
plt.show()  


