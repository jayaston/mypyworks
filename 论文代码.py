# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 20:52:33 2023

@author: Jay
"""
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
import matplotlib.pyplot as plt
from matplotlib import ticker

import pandas as pd
import numpy as np
#np.set_printoptions(suppress=True) 
from scipy import  stats

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF


data_d = data_d.set_index('QUOTA_DATE')
data_m=data_d.resample('MS')[['供水总量','最高温度','平均温度','西村水厂']].agg(
        {'供水总量':['sum'],'最高温度':['mean'],'平均温度':['mean'],'西村水厂':['sum']})
data_m.columns = data_m.columns.get_level_values(0).values




np.std(data_h['广州自来水公司_小时供水量']) / np.mean(data_h['广州自来水公司_小时供水量']) * 100

np.std(data_d['2022']['供水总量']) / np.mean(data_d['2022']['供水总量']) * 100
np.std(data_m['供水总量']) / np.mean(data_m['供水总量']) * 100

#画图--------------------------------------------------------------------------------

data_d = data_d.set_index('QUOTA_DATE')
data_m=data_d.resample('MS')[['供水总量','最高温度','平均温度']].agg(
        {'供水总量':['sum'],'最高温度':['mean'],'平均温度':['mean']})
data_m.columns = data_m.columns.get_level_values(0).values

data_d = data_d['2022'][['供水总量','最高温度','平均温度','西村水厂']]

plt.figure(figsize=(12,10))
plt.subplot(311)
plt.plot(data_h.index.values,data_h['广州自来水公司_小时供水量'])
plt.title('(a)小时供水量序列')
plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%m/%d'))
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(6))
plt.ylabel('供水量（m3）')
plt.subplot(312)
plt.plot(data_d.index.values,data_d['供水总量'])
plt.title('(b)日供水量序列')
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1000000, decimals=0, symbol='万', is_latex=False))
plt.ylabel('供水量（m3）')
plt.subplot(313)
data_m['供水总量'].plot()
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1000000, decimals=0, symbol='万', is_latex=False))
plt.title('(c)月供水量序列')
#plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y%m'))
plt.xlabel('时间')
plt.ylabel('供水量（m3）')
plt.gcf().subplots_adjust(left=0.1,bottom=0.08,right=0.95,top=0.95,hspace=0.3)
plt.show()



xdata = data_h['广州自来水公司_小时供水量']

def diff(timeseries):
    timeseries_diff1 = timeseries.diff(1)
    timeseries_diff2 = timeseries_diff1.diff(1)

    timeseries_diff1 = timeseries_diff1.fillna(0)
    timeseries_diff2 = timeseries_diff2.fillna(0)

    timeseries_adf = ADF(timeseries.tolist())
    timeseries_diff1_adf = ADF(timeseries_diff1.tolist())
    timeseries_diff2_adf = ADF(timeseries_diff2.tolist())

    print('timeseries_adf : ', timeseries_adf)
    print('timeseries_diff1_adf : ', timeseries_diff1_adf)
    print('timeseries_diff2_adf : ', timeseries_diff2_adf)

    plt.figure(figsize=(12, 8))
    plt.plot(timeseries, label='Original', color='blue')
    plt.plot(timeseries_diff1, label='Diff1', color='red')
    plt.plot(timeseries_diff2, label='Diff2', color='purple')
    plt.legend(loc='best')
    plt.show()

diff(xdata)


xdata_diff = xdata.diff(1).dropna()
plot_acf(xdata_diff)
plot_pacf(xdata_diff)
import warnings
import itertools

p = d = q = range(0,6)
pdq = list(itertools.product(p, d, q))
# 忽略ARIMA模型无法估计出结果时的报警信息
import sys
warnings.filterwarnings("ignore")

best_aic = np.inf
best_pdq = None
best_seasonal_pdq = None
temp_model = None

for param in pdq:
    try:
        temp_model = sm.tsa.ARIMA(xdata,param)
        results = temp_model.fit()
        if results.aic < best_aic:
            best_aic = results.aic
            best_pdq = param
    except:
        continue
        
print("Best ARIMA{} model - AIC:{}".format(best_pdq, best_aic))

(p, q) =(sm.tsa.arma_order_select_ic(xdata,max_ar=5,max_ma=5,ic='aic')['aic_min_order'])

model = sm.tsa.ARIMA(xdata,(5,0,2)).fit()
print(model.summary())
model.fittedvalues
pd.DataFrame({'实际值':xdata,'arima':model.fittedvalues})
#每个质保评分
abs_=(xdata- model.fittedvalues).abs()
mae=abs_.mean()#Mean Absolute Error ，平均绝对误差
rmse= np.sqrt((abs_**2).mean()) #Root Mean Square Error,均方根误差
mape=(abs_/xdata).mean()# mean absolute percentage error，平均绝对百分比误差
R2 = r2_score(xdata, model.fittedvalues)
print("平均绝对误差MAE={:.0f}；\n平均绝对百分比误差MAPE={:.2%}；\nR方={:.2%}；\n均方根误差RMSE={:.0f}。".format(mae,mape,R2,rmse))


#预测多期
#k=4
model1 = sm.tsa.ARIMA(xdata[0:672],(5,0,2)).fit()
test = model1.forecast(4)[0]

model1 = sm.tsa.ARIMA(xdata[0:676],(5,0,2)).fit()
test= np.r_[test,model1.forecast(4)[0]]

model1 = sm.tsa.ARIMA(xdata[0:680],(5,0,2)).fit()
test= np.r_[test,model1.forecast(4)[0]]

model1 = sm.tsa.ARIMA(xdata[0:684],(5,0,2)).fit()
test= np.r_[test,model1.forecast(4)[0]]

model1 = sm.tsa.ARIMA(xdata[0:688],(5,0,2)).fit()
test= np.r_[test,model1.forecast(4)[0]]

model1 = sm.tsa.ARIMA(xdata[0:692],(5,0,2)).fit()
test= np.r_[test,model1.forecast(4)[0]]

pd.DataFrame({'实际值':xdata[672:696],'arimak4':test})
#k=24
model1 = sm.tsa.ARIMA(xdata[0:672],(5,0,2)).fit()
test = model1.forecast(24)[0]

#每个评分
abs_=(xdata[672:696]- test).abs()
mae=abs_.mean()#Mean Absolute Error ，平均绝对误差
rmse= np.sqrt((abs_**2).mean()) #Root Mean Square Error,均方根误差
mape=(abs_/xdata[672:696]).mean()# mean absolute percentage error，平均绝对百分比误差
R2 = r2_score(xdata[672:696], test)
print("平均绝对误差MAE={:.0f}；\n平均绝对百分比误差MAPE={:.2%}；\nR方={:.2%}；\n均方根误差RMSE={:.0f}。".format(mae,mape,R2,rmse))

pd.DataFrame({'实际值':xdata[672:696],'arimak24':test})



test = model.predict(720,744)
test.plot()





#----------------------------
#prophet预测
#调用prophet模型
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation 
#修改变量名为prophet标准变量名
sup_water=data_h[['广州自来水公司_小时供水量']]
sup_water=sup_water.reset_index().rename(columns={'QUOTA_DATE':'ds','广州自来水公司_小时供水量':'y'})
#划分训练数据集和测试数据集

sup_water_train = sup_water[:-48] #训练数据集
#sup_water_test = sup_water[-test_len:]  #测试数据集
sup_water_train.info()
#print(sup_water_test)
KeyError: 'metric_file'


m = Prophet(#holidays=holidays,
    growth='logistic', 
    #interval_width = 0.8,           #预测不确定区间宽度
    #n_changepoints = 25,            #识别变点的上限数量             
    #changepoint_range = 0.8 ,       #使用前80%比例数据作为变点识别数据
    #changepoint_prior_scale = 0.05, #changepoint_prior_scale越大，识别为变点越多；越小，识别为变点越少。
    #holidays_prior_scale=10  #holidays_prior_scale越大，假期对目标值的影响越大。越小，假期对目标值的影响越小。
    )        
m = Prophet(yearly_seasonality=False)

m.add_seasonality(name='hourly', period=24, fourier_order=10, prior_scale=10)#prior_scale越大，对于目标变量影响越大。


#按照测试集前两年的数据训练模型
m.fit(sup_water_train)
future = m.make_future_dataframe(periods=48,freq = 'H', 
                                 #include_history = False
                                 )
forecast = m.predict(future)
m.plot_components(forecast)
m.plot(forecast)

#每个评分
y=sup_water_train['y']
yhat=forecast['yhat'][:-48]
abs_=(y- yhat).abs()
mae=abs_.mean()#Mean Absolute Error ，平均绝对误差  
rmse= np.sqrt((abs_**2).mean()) #Root Mean Square Error,均方根误差
mape=(abs_/y).mean()# mean absolute percentage error，平均绝对百分比误差
R2 = r2_score(y, yhat)
print("平均绝对误差MAE={:.0f}；\n平均绝对百分比误差MAPE={:.2%}；\nR方={:.2%}；\n均方根误差RMSE={:.0f}。".format(mae,mape,R2,rmse))

pd.concat([sup_water_train,forecast['yhat'][:-48]],axis=1)



#k=4
test=pd.Series()
for i in range(6):    
    m = Prophet(#holidays=holidays,
    growth='logistic', 
    #interval_width = 0.8,           #预测不确定区间宽度
    #n_changepoints = 25,            #识别变点的上限数量             
    #changepoint_range = 0.8 ,       #使用前80%比例数据作为变点识别数据
    #changepoint_prior_scale = 0.05, #changepoint_prior_scale越大，识别为变点越多；越小，识别为变点越少。
    #holidays_prior_scale=10  #holidays_prior_scale越大，假期对目标值的影响越大。越小，假期对目标值的影响越小。
    )        
    m = Prophet(yearly_seasonality=False)
    
    m.add_seasonality(name='hourly', period=24, fourier_order=10, prior_scale=10)#prior_scale越大，对于目标变量影响越大。

    m.fit(sup_water[:(-48+4*i)])
    future = m.make_future_dataframe(periods=4,freq = 'H', 
                                     #include_history = False
                                     )
    forecast = m.predict(future)
    test = pd.concat([test,forecast['yhat'][-4:]],axis=0)

# m.plot_components(forecast)
# m.plot(forecast)

#每个评分
y=sup_water['y'][-48:-24]
yhat=test
abs_=(y- yhat).abs()
mae=abs_.mean()#Mean Absolute Error ，平均绝对误差  
rmse= np.sqrt((abs_**2).mean()) #Root Mean Square Error,均方根误差
mape=(abs_/y).mean()# mean absolute percentage error，平均绝对百分比误差
R2 = r2_score(y, yhat)
print("平均绝对误差MAE={:.0f}；\n平均绝对百分比误差MAPE={:.2%}；\nR方={:.2%}；\n均方根误差RMSE={:.0f}。".format(mae,mape,R2,rmse))

pd.concat([sup_water[-48:-24],test],axis=1).to_excel(r'd:\BaiduSyncdisk\谢杰\My Documents\谢杰同等学力\论文\论文初稿\小时prophetk=4输出结果.xlsx')


#k=24
test=pd.Series()
  
m = Prophet(#holidays=holidays,
growth='logistic', 
#interval_width = 0.8,           #预测不确定区间宽度
#n_changepoints = 25,            #识别变点的上限数量             
#changepoint_range = 0.8 ,       #使用前80%比例数据作为变点识别数据
#changepoint_prior_scale = 0.05, #changepoint_prior_scale越大，识别为变点越多；越小，识别为变点越少。
#holidays_prior_scale=10  #holidays_prior_scale越大，假期对目标值的影响越大。越小，假期对目标值的影响越小。
)        
m = Prophet(yearly_seasonality=False)

m.add_seasonality(name='hourly', period=24, fourier_order=10, prior_scale=10)

m.fit(sup_water[:-48])
future = m.make_future_dataframe(periods=24,freq = 'H', 
                                 #include_history = False
                                 )
forecast = m.predict(future)
test = forecast['yhat'][-24:]


#每个评分
y=sup_water['y'][-48:-24]
yhat=test
abs_=(y- yhat).abs()
mae=abs_.mean()#Mean Absolute Error ，平均绝对误差  
rmse= np.sqrt((abs_**2).mean()) #Root Mean Square Error,均方根误差
mape=(abs_/y).mean()# mean absolute percentage error，平均绝对百分比误差
R2 = r2_score(y, yhat)
print("平均绝对误差MAE={:.0f}；\n平均绝对百分比误差MAPE={:.2%}；\nR方={:.2%}；\n均方根误差RMSE={:.0f}。".format(mae,mape,R2,rmse))
pd.concat([sup_water[-48:-24],test],axis=1)


#-----------------------------------------------------
#lstm
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
dataset = xdata
dataset.index.name = 'date'
# 把所有NA值用0替换
dataset.fillna(0, inplace=True)
values = dataset.values
type(values)

from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM 

# 确保所有数据是浮点数
values = values.astype('float32')
# 归一化特征
scaler = MinMaxScaler(feature_range=(0, 1))
values = values.reshape(-1,1)
scaled = scaler.fit_transform(values)

n_steps =168 #滞后期
n_pre_steps =24#预测期 注意默认选1期，多步预测选了4期
n_features = 1# 自变量指标数
n_pre_features = 1 #预测指标数必须摆在矩阵前列

# 构建监督学习问题
reframed = series_to_supervised(scaled,n_steps, n_pre_steps)
reframed.shape

values = reframed.values
values.shape
n_train_months = 423 #取数据训练集
train = values[:n_train_months, :]
test = values[n_train_months:, :]
train.shape
# 分为输入和输出
n_obs = n_steps * n_features  
train_X, train_y = train[:, :n_obs], train[:, n_obs:]#将训练集分离出目标y，用于fit

test_X, test_y = test[:, :n_obs], test[:, n_obs:]#将测试集分离出目标Y，用于评分

train_X = train_X.reshape((train_X.shape[0], n_steps, n_features))
test_X = test_X.reshape((test_X.shape[0], n_steps, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# 设计神经网络
model = Sequential() #生成模型实例
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]))) 

model.add(Dense(1))#设置输出目标变量 这里是单步预测， 多部预测的神经元设置请看后面K=4。
model.compile(loss='mae', optimizer='adam')#设置评估的函数，以及优化器
# 拟合网络模型
history = model.fit(train_X, train_y, epochs=100, batch_size=10, validation_data=(test_X, test_y), verbose=2, shuffle=False)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


model.save(r"demo.model")
from keras.models import load_model
model = load_model(r"demo.model")
print(model.summary())

model.summary()

score = model.evaluate(test_X, test_y,batch_size=10, verbose=1)
print(model.metrics_names)
print(score)

#计算训练数据集的R方,rmse,mape
trainyhat = model.predict(train_X)

# 反向转换预测值
yhat_tmp = trainyhat.reshape((-1,n_pre_features))#2为预测指标数
#对其反转矩阵,
inv_yhat = concatenate((yhat_tmp, scaled[:yhat_tmp.shape[0],n_pre_features:]), axis=1)#截取数据集相同的行数对齐
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,:n_pre_features]#得到正式预测值
inv_y = scaler.inverse_transform(train_y)

result = pd.DataFrame( dataset[n_steps:n_steps+n_train_months])
result['预测'] = inv_yhat

#每个评分
def print_rmse_mape(abs_):
    mae=abs_.mean()#Mean Absolute Error ，平均绝对误差
    rmse= sqrt((abs_**2).mean()) #Root Mean Square Error,均方根误差
    mape=(abs_/inv_y.reshape(1,-1)[0,:]).mean()# mean absolute percentage error，平均绝对百分比误差
    R2 = r2_score(inv_y.reshape(1,-1)[0,:],inv_yhat.reshape(1,-1)[0,:])
    print("平均绝对误差MAE={}；\n平均绝对百分比误差MAPE={:.2%}；\nR方={:.2%}；\n均方根误差RMSE={:.3f}。".format(mae,mape,R2,rmse))


abs_=np.abs(inv_yhat.reshape(1,-1)[0,:]-inv_y.reshape(1,-1)[0,:])
print_rmse_mape(abs_)



#多个时间步的预测
#k=4
# 设计神经网络
model = Sequential() #生成模型实例
model.add(LSTM(50,return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(50))
model.add(Dense(4))
model.compile(loss='mae', optimizer='adam')#设置评估的函数，以及优化器
# 拟合网络模型
history = model.fit(train_X, train_y, epochs=100, batch_size=10, validation_data=(test_X, test_y), verbose=2, shuffle=False)

#计算训练数据集的R方,rmse,mape
testyhat = model.predict(test_X)
inv_yhat = scaler.inverse_transform(testyhat)
inv_yhat=inv_yhat[-48:-24,-1:]

result = pd.DataFrame( dataset[-48:-24])
result['预测'] = inv_yhat
result.info()


#每个评分
def print_rmse_mape(abs_):
    mae=abs_.mean()#Mean Absolute Error ，平均绝对误差
    rmse= sqrt((abs_**2).mean()) #Root Mean Square Error,均方根误差
    mape=(abs_/result['广州自来水公司_小时供水量'].values).mean()# mean absolute percentage error，平均绝对百分比误差
    R2 = r2_score(result['广州自来水公司_小时供水量'].values,result['预测'].values)
    print("平均绝对误差MAE={}；\n平均绝对百分比误差MAPE={:.2%}；\nR方={:.2%}；\n均方根误差RMSE={:.3f}。".format(mae,mape,R2,rmse))


abs_=np.abs(result['广州自来水公司_小时供水量']-result['预测']).values
print_rmse_mape(abs_)

#k=24

# 设计神经网络
model = Sequential() #生成模型实例
model.add(LSTM(50,return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2]))) 
model.add(LSTM(50))

model.add(Dense(24))
model.compile(loss='mae', optimizer='adam')#设置评估的函数，以及优化器
# 拟合网络模型
history = model.fit(train_X, train_y, epochs=100, batch_size=10, validation_data=(test_X, test_y), verbose=2, shuffle=False)

testyhat = model.predict(test_X)
inv_yhat = scaler.inverse_transform(testyhat)
inv_yhat=inv_yhat[-48:-24,-1:]

result = pd.DataFrame( dataset[-48:-24])
result['预测'] = inv_yhat
result.info()



#每个质保评分
def print_rmse_mape(abs_):
    mae=abs_.mean()#Mean Absolute Error ，平均绝对误差
    rmse= sqrt((abs_**2).mean()) #Root Mean Square Error,均方根误差
    mape=(abs_/result['广州自来水公司_小时供水量'].values).mean()# mean absolute percentage error，平均绝对百分比误差
    R2 = r2_score(result['广州自来水公司_小时供水量'].values,result['预测'].values)
    print("平均绝对误差MAE={}；\n平均绝对百分比误差MAPE={:.2%}；\nR方={:.2%}；\n均方根误差RMSE={:.3f}。".format(mae,mape,R2,rmse))


abs_=np.abs(result['广州自来水公司_小时供水量']-result['预测']).values
print_rmse_mape(abs_)



#特征选择


dataset = data_m.set_index('QUOTA_DATE')[['供水总量','平均相对湿度']]
#         '自然增长用户','存量用户','抄表到户改造总表数(增加数量）','平均温度（C）','降雨量','日照时数']]#预测指标务必在前面。以方面后面反归一化

#注意！预测指标务必在前面。以方面后面反归一化

dataset.index.name = 'date'
# 把所有NA值用0替换
dataset.fillna(0, inplace=True)
values = dataset.values
type(values)




#正式进入模型部分
from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM 

# 确保所有数据是浮点数
values = values.astype('float32')
# 归一化特征
scaler = MinMaxScaler(feature_range=(0, 1))
#values = values.reshape(-1,1)
scaled = scaler.fit_transform(values)

n_steps =9 #滞后期
n_pre_steps =1 #预测期
n_features = 2# 自变量指标数
n_pre_features = 1 #预测指标数必须摆在矩阵前列

# 构建监督学习问题
reframed = series_to_supervised(scaled,n_steps, n_pre_steps)#第二个参数是滞后期数用来当自变量，第三各参数是未来期数当预测目标
reframed.shape#注意：二维数组表示（行，列）行数是行本数，列数是步数与指标数的乘积

values = reframed.values
values.shape
values = values[:,:-1]##视模型，只留下一个输出变量
n_train_months = 60  #取数据训练集
train = values[:n_train_months, :]
test = values[n_train_months:, :]
train.shape
# 分为输入和输出
n_obs = n_steps * n_features  #过去的步长乘以自变量指标来确定X的变量数
train_X, train_y = train[:, :n_obs], train[:, n_obs:]#将训练集分离出目标y，用于fit

test_X, test_y = test[:, :n_obs], test[:, n_obs:]#将测试集分离出目标Y，用于评分
print(train_X.shape, len(train_X), train_y.shape) 
train_X = train_X.reshape((train_X.shape[0], n_steps, n_features))
test_X = test_X.reshape((test_X.shape[0], n_steps, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# 设计神经网络
model = Sequential() #生成模型实例
model.add(LSTM(50, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2]))) 
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')#设置评估的函数，以及优化器
# 拟合网络模型
history = model.fit(train_X, train_y, epochs=100, batch_size=10, validation_data=(test_X, test_y), verbose=2, shuffle=False)


model.summary()

score = model.evaluate(test_X, test_y,batch_size=10, verbose=1)
print(model.metrics_names)
print(score)

#计算训练数据集的R方,rmse,mape
trainyhat = model.predict(train_X)

# 反向转换预测值
yhat_tmp = trainyhat.reshape((-1,n_pre_features))#2为预测指标数
#对其反转矩阵,
inv_yhat = concatenate((yhat_tmp, scaled[9:9+yhat_tmp.shape[0],1:]), axis=1)

inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,:n_pre_features]#得到正式预测值

inv_y = concatenate((train_y, scaled[9:9+yhat_tmp.shape[0],1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,:n_pre_features]


result = pd.DataFrame( dataset[9:9+yhat_tmp.shape[0]])
result['预测'] = inv_yhat




abs_=np.abs(inv_yhat.reshape(1,-1)[0,:]-inv_y.reshape(1,-1)[0,:])
print_rmse_mape(abs_)


#未来三年月度预测
#----------------------------
#prophet预测
#调用prophet模型
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation 
#修改变量名为prophet标准变量名
sup_water=data_m[['QUOTA_DATE','供水总量']]
sup_water=sup_water.rename(columns={'QUOTA_DATE':'ds','供水总量':'y'})
#划分训练数据集和测试数据集

sup_water_train = sup_water#[:-3] #训练数据集
#sup_water_test = sup_water[-test_len:]  #测试数据集
sup_water_train.info()
#print(sup_water_test)
KeyError: 'metric_file'


m = Prophet(#holidays=holidays,
    growth='logistic', 
    interval_width = 0.8,           #预测不确定区间宽度
    #n_changepoints = 25,            #识别变点的上限数量             
    changepoint_range = 0.8 ,       #使用前80%比例数据作为变点识别数据
    changepoint_prior_scale = 0.02, #changepoint_prior_scale越大，识别为变点越多；越小，识别为变点越少。

    )        
m = Prophet(yearly_seasonality=True,weekly_seasonality=False,daily_seasonality=False)

m.add_country_holidays(country_name="CN")
m.train_holiday_names
#按照测试集前两年的数据训练模型
m.fit(sup_water_train)
future = m.make_future_dataframe(periods=4,freq = 'MS', 
                                 #include_history = False
                                 )


forecast = m.predict(future)
m.plot_components(forecast)
m.plot(forecast)



#每个评分
y=sup_water_train['y']
yhat=forecast['yhat']
abs_=(y- yhat).abs()
mae=abs_.mean()#Mean Absolute Error ，平均绝对误差  
rmse= np.sqrt((abs_**2).mean()) #Root Mean Square Error,均方根误差
mape=(abs_/y).mean()# mean absolute percentage error，平均绝对百分比误差
R2 = r2_score(y, yhat)
print("平均绝对误差MAE={:.0f}；\n平均绝对百分比误差MAPE={:.2%}；\nR方={:.2%}；\n均方根误差RMSE={:.0f}。".format(mae,mape,R2,rmse))

pd.concat([sup_water_train,forecast['yhat']],axis=1).to_excel(r'd:\BaiduSyncdisk\谢杰\My Documents\谢杰同等学力\论文\论文初稿\输出结果\外变量平均湿度prophet.xlsx')





















