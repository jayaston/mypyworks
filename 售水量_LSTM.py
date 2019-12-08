# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:05:35 2019

@author: XieJie
"""
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
 
# 构建将时间序列转换为监督学习特征的函数
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
dataset_d = pd.read_excel(r"C:\Users\Jay\mypyworks\自来水数据\20191120刘博士温度湿度降雨量.xlsx",
                         parse_dates = ['日期'])


dataset_M = dataset_d.groupby(pd.Grouper(key='日期',freq='MS')).mean()


# 加载售水数据集    
dataset = pd.read_excel(r"C:\Users\Jay\mypyworks\自来水数据\售水相关月数据1999-2019(20191204整合).xlsx",
                         parse_dates = ['日期'],index_col=0)

dataset = pd.merge(dataset,dataset_M,how='outer',left_index=True,right_index=True)
dataset.info()

#选择样本时期


#特征选择



# 手动更改列名
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
# 把所有NA值用0替换
dataset.fillna(0, inplace=True)
# 丢弃前24小时
dataset = dataset[24:] 
# 输出前五行
print(dataset.head(5))
# 保存到文件中
#dataset.to_csv('pollution.csv')

# 加载数据集
#dataset = pd.read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
# 指定要绘制的列
groups = [0, 1, 2, 3, 5, 6, 7]
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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM 

#dataset = read_csv('pollution.csv', header=0, index_col=0)
#values = dataset.values
# 整数编码
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# 确保所有数据是浮动的
values = values.astype('float32')
# 归一化特征
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# 指定滞后时间大小
n_hours = 3
n_features = 8
# 构建监督学习问题
reframed = series_to_supervised(scaled, n_hours, 1)
print(reframed.shape)
 
# 分为训练集和测试集
values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# 分为输入和输出
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)
# 重塑为3D形状 [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# 设计网络
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# 拟合网络模型
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# 绘制历史数据
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
 
# 作出预测
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
# 反向转换预测值比例
inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# 反向转换实际值大小
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# 计算RMSE大小
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

#画出预测结果：蓝色为原数据，绿色为训练数据集的预测值，红色为测试数据集的预测值

