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
dataset_d = pd.read_excel(r"c:/Users/XieJie/mypyworks/自来水数据\20191120刘博士温度湿度降雨量.xlsx",
                         parse_dates = ['日期']) 
dataset_M = dataset_d.groupby(pd.Grouper(key='日期',freq='MS')).mean()
# 加载售水数据集    
dataset = pd.read_excel(r"c:/Users/XieJie/mypyworks/自来水数据/售水相关月数据1999-2019(20191204整合).xlsx",
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
         '自然增长用户','存量用户','抄表到户改造总表数(增加数量）','平均温度（C）','降雨量','日照时数']]#预测指标务必在前面。以方面后面反归一化

#注意！预测指标务必在前面。以方面后面反归一化

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



#正式进入模型部分
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
#StandardScaler()方法应该叫“z-score标准化”，而不是归一化。但有时大家都是混着叫的。
# 分类变量连续化
#encoder = LabelEncoder()
#values[:,4] = encoder.fit_transform(values[:,4])
# 指定滞后时间大小,和指标数量
n_steps = 5 #滞后期
n_pre_steps = 2 #预测期
n_features = 10 # 自变量指标数
n_pre_features = 2 #预测指标数必须摆在矩阵前列

# 构建监督学习问题
reframed = series_to_supervised(scaled, n_steps, 2)#第二个参数是滞后期数用来当自变量，第三各参数是未来期数当预测目标

# 丢弃不是预测的列
reframed.drop(reframed.columns[[69,68,67,66,65,64,63,62,59,58,57,56,55,54,53,52]], axis=1, inplace=True)
print(reframed.head())
#reframed.to_csv('处理后数据.csv')

# 加载数据集
#reframed = pd.read_csv('处理后数据.csv', header=0, index_col=0)
# 分为训练集和测试集
values = reframed.values
n_train_months = 4 * 12 - 1 #取四年的数据训练集，因为预测到未来t+1期，所以48个月-1
train = values[:n_train_months, :]
test = values[n_train_months:, :]
# 分为输入和输出
n_obs = n_steps * n_features  #过去的步长乘以自变量指标来确定X的变量数
train_X, train_y = train[:, :n_obs], train[:, n_obs:]#将训练集分离出目标y，用于fit
test_X, test_y = test[:, :n_obs], test[:, n_obs:]#将测试集分离出目标Y，用于评分
print(train_X.shape, len(train_X), train_y.shape)
# 为了lstm模型将训练数据集重塑为三维形状 [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_steps, n_features))
#第一个参数是层数，采用样本数，第二个是行数，采用了滞后数，第三个是列数，采用自变量指标数
#按照series_to_supervised函数的生产矩阵格式，排成指标为列，滞后为行，样本数量为层三维矩阵
#如果只预测多个指标，只有预测1期，则y可以简化成2维数组，列是指标，行变成样本量。或者预测一个指标的多期，列是期数，行是样本量。
test_X = test_X.reshape((test_X.shape[0], n_steps, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# 设计神经网络
model = Sequential() #生成模型实例
model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2]))) #设置输入样本的形状，二维数组所以是两个值。
#
model.add(Dense(4))#设置输出目标变量  多部预测的神经元怎么设置。
model.compile(loss='mae', optimizer='adam')#设置评估的函数，以及优化器
# 拟合网络模型
history = model.fit(train_X, train_y, epochs=100, batch_size=10, validation_data=(test_X, test_y), verbose=2, shuffle=False)
#epochs整个样本被训练优化50次,重复fit可以再以后模型基础上再训练
#batch_size一次训练所选取的样本数。在小样本数的数据库中，不使用Batch Size是可行的，而且效果也很好。但是一旦是大型的数据库，一次性把所有数据输进网络，肯定会引起内存的爆炸。所以就提出Batch Size的概念。
#iterations（迭代）：每一次迭代都是一次权重更新,训练集有1000个样本，batchsize=10，那么：训练完整个样本集需要： 100次iteration，1次epoch。
# 绘制历史数据

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# 为加快测试，保存及加载模型文件
model.save(r"demo.model")
from keras.models import load_model
model = load_model(r"demo.model")
print(model.summary())

model.summary()

score = model.evaluate(test_X, test_y,batch_size=10, verbose=1)
print(score)

#计算训练数据集的R方
trainyhat = model.predict(train_X)

from sklearn.metrics import mean_squared_error
for i in range(train_y.shape[1]):
    R_square = 1-(mean_squared_error(train_y[:,i],trainyhat[:,i])/np.var(train_y[:,i]))
    print(R_square)

# 作出预测
yhat = model.predict(test_X)

#还原数据结构
#train_X = train_X.reshape((train_X.shape[0], n_steps*n_features))
#test_X = test_X.reshape((test_X.shape[0], n_steps*n_features))


# 反向转换预测值
yhat_tmp = yhat.reshape((-1,n_pre_features))#2为预测指标数
#对其反转矩阵,
inv_yhat = concatenate((yhat_tmp, scaled[:yhat_tmp.shape[0],n_pre_features:]), axis=1)#截取数据集相同的行数对齐
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,:n_pre_features]#得到正式预测值

#评分
# 反向转换实际值大小
test_y = test_y.reshape((-1, n_pre_features))#2为预测指标数
inv_y = concatenate((test_y, scaled[:test_y.shape[0],n_pre_features:]), axis=1)
inv_y = scaler.inverse_transform(inv_y) #因为scaler已经记录了各字段的缩放量。所以肯定能精确还原。
inv_y = inv_y[:,:n_pre_features]

#每个质保评分
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


