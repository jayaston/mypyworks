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


#小时数据读取9月数据
data_H=pd.read_excel(io=r'd:\BaiduSyncdisk\谢杰\My Documents\个人所有填写表格\谢杰同等学力\论文\2021至2022时数据.xlsx')
data_H.info()
data_h=data_H[(data_H['QUOTA_DATE']>='2022/9/1')&(data_H['QUOTA_DATE']<='2022/9/30 23:00:00')].set_index('QUOTA_DATE')
data_h.info()
#data_h.to_excel(r'd:\BaiduSyncdisk\谢杰\My Documents\个人所有填写表格\谢杰同等学力\论文\小时实验数据.xlsx')

#日月数据处理
data_d=pd.read_excel(io=r'd:\BaiduSyncdisk\谢杰\My Documents\个人所有填写表格\谢杰同等学力\论文\2016至2022供水量.xlsx')
data_d = data_d.set_index('QUOTA_DATE')
data_m=data_d.resample('MS')[['供水总量','最高温度','平均温度','西村水厂']].agg(
        {'供水总量':['sum'],'最高温度':['mean'],'平均温度':['mean'],'西村水厂':['sum']})
data_m.columns = data_m.columns.get_level_values(0).values

data_d = data_d['2022-8':'2022-9'][['供水总量','最高温度','平均温度','西村水厂']]

#画图--------------------------------------------------------------------------------
plt.figure(figsize=(12,10))
plt.subplot(311)
plt.plot(data_h.index.values,data_h['广州自来水公司_小时供水量'])
plt.title('小时供水量序列')
#plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%H'))
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(6))
plt.ylabel('供水量（m3）')
plt.subplot(312)
plt.plot(data_d.index.values,data_d['供水总量'])
plt.title('日供水量序列')
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1000000, decimals=0, symbol='万', is_latex=False))
plt.ylabel('供水量（m3）')
plt.subplot(313)
data_m['供水总量'].plot()
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1000000, decimals=0, symbol='万', is_latex=False))
plt.title('月供水量序列')
#plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y%m'))
plt.xlabel('时间')
plt.ylabel('供水量（m3）')
plt.gcf().subplots_adjust(left=0.2,bottom=0.1,right=0.8,top=0.8,hspace=0.3)
plt.show()



data_d=pd.read_excel(io=r'd:\BaiduSyncdisk\谢杰\My Documents\个人所有填写表格\谢杰同等学力\论文\2016至2022供水量.xlsx')
data_d = data_d.set_index('QUOTA_DATE')['2021-7':'2021-9'][['供水总量','最高温度','平均温度','西村水厂']]

data_m = data_m[~(data_m.index.strftime("%m").isin(["01","02","03"]))] 
data_m.info()
#日数据







#--------------------------------------------------------------------
#arima
#确定最佳p、d、q值
#小时水量用三种模型测试
xdata = data_h['广州自来水公司_小时供水量']

import warnings
import itertools
# 设置自相关(AR)、差分(I)、移动平均(MA)的三个参数的取值范围
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


#每个质保评分
abs_=(xdata- model.fittedvalues).abs()
mae=abs_.mean()#Mean Absolute Error ，平均绝对误差
rmse= np.sqrt((abs_**2).mean()) #Root Mean Square Error,均方根误差
mape=(abs_/xdata).mean()# mean absolute percentage error，平均绝对百分比误差
R2 = r2_score(xdata, model.fittedvalues)
print("平均绝对误差MAE={:.0f}；\n平均绝对百分比误差MAPE={:.2%}；\nR方={:.2%}；\n均方根误差RMSE={:.0f}。".format(mae,mape,R2,rmse))


#预测多期
test = model.forecast(24)

test = model.predict(720,744)
test.plot()
xdata['2020-04'].plot()




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
    #growth='logistic', 
    #interval_width = 0.8,           #预测不确定区间宽度
    #n_changepoints = 25,            #识别变点的上限数量             
    #changepoint_range = 0.8 ,       #使用前80%比例数据作为变点识别数据
    #changepoint_prior_scale = 0.05, #changepoint_prior_scale越大，识别为变点越多；越小，识别为变点越少。
    #holidays_prior_scale=10  #holidays_prior_scale越大，假期对目标值的影响越大。越小，假期对目标值的影响越小。
    )        
m = Prophet(yearly_seasonality=False)
#m.add_seasonality(name='weekly', period=168, fourier_order=10, prior_scale=10) #fourier_order越大，对周期变化的拟合越细致也越容易过拟合。
#m.add_seasonality(name='hourly', period=24, fourier_order=10, prior_scale=10)#prior_scale越大，对于目标变量影响越大。
#m.add_regressor('平均温度',prior_scale=10,mode='multiplicative') #回归量采用乘法模型
#时间戳的格式应该是YYYY-MM-DD - HH:MM:SS。当使用子日(日以下)数据时，日季节性将自动匹配

#按照测试集前两年的数据训练模型
m.fit(sup_water_train)
future = m.make_future_dataframe(periods=48,freq = 'H', 
                                 #include_history = False
                                 )
forecast = m.predict(future)
m.plot_components(forecast)
m.plot(forecast)

#每个质保评分
y=sup_water_train['y']
yhat=forecast['yhat'][:-48]
abs_=(y- yhat).abs()
mae=abs_.mean()#Mean Absolute Error ，平均绝对误差  
rmse= np.sqrt((abs_**2).mean()) #Root Mean Square Error,均方根误差
mape=(abs_/y).mean()# mean absolute percentage error，平均绝对百分比误差
R2 = r2_score(y, yhat)
print("平均绝对误差MAE={:.0f}；\n平均绝对百分比误差MAPE={:.2%}；\nR方={:.2%}；\n均方根误差RMSE={:.0f}。".format(mae,mape,R2,rmse))

ma


def fun_mape(df):
    return np.mean(np.abs((df['yhat']-df['y'])/df['y']))
df_cv = cross_validation(m, initial='730 days', period='7 days', horizon = '31 days') 
                mape_ave = df_cv.groupby('cutoff').apply(fun_mape).mean()
                
#计算测试数据集预测值的mape
np.mean(np.abs((forecast['yhat'].values - sup_water_test['y'].values)/sup_water_test['y'].values))
#m如果不能提取出来预测值则在预测完成以后提取
sup_water['cap']= 5100000 #设置增长饱和值
m.fit(sup_water[-730:])
#预测未来30天
future = m.make_future_dataframe(periods=29,freq = 'd', 
                                 include_history = False)

#输入预测期的饱和最大值
future['cap'] = 5100000
#输入天气预报中未来30日的平均温度
future['平均温度'] = [32,32.5,31.5,31.5,31.5,31.5,31.5,28.5,29,29.5,30,30,30.5,30.5,30.5,29.5,30.5,29,28.5,26,27,26.5,28,30.5,29.5,31,32,32.5,32]
future 
forecast = m.predict(future)
fig = m.plot(forecast)
tmp = forecast[['ds','yhat']]














































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

# 保存到文件中
#dataset.to_csv('处理后数据.csv')
# 加载数据集
#dataset = pd.read_csv('处理后数据.csv', header=0, index_col=0)
#选择样本时期

#特征选择

#dataset = dataset[['公司净水售水量','水厂供水总量','产销差率','发单水量合计',
#         '自然增长用户','存量用户','抄表到户改造总表数(增加数量）','平均温度（C）','降雨量','日照时数']]#预测指标务必在前面。以方面后面反归一化

#注意！预测指标务必在前面。以方面后面反归一化

dataset.index.name = 'date'
# 把所有NA值用0替换
dataset.fillna(0, inplace=True)
values = dataset.values
type(values)
# 指定要绘制的列
groups = range(1)
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

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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
n_steps =168 #滞后期
n_pre_steps =1 #预测期
n_features = 1# 自变量指标数
n_pre_features = 1 #预测指标数必须摆在矩阵前列

# 构建监督学习问题
reframed = series_to_supervised(scaled,n_steps, n_pre_steps)#第二个参数是滞后期数用来当自变量，第三各参数是未来期数当预测目标
reframed.shape#注意：二维数组表示（行，列）行数是行本数，列数是步数与指标数的乘积
# 丢弃不是预测的列
#reframed.drop(reframed.columns[[69,68,67,66,65,64,63,62,59,58,57,56,55,54,53,52]], axis=1, inplace=True)
#print(reframed.head())
#reframed.to_csv('处理后数据.csv')

# 加载数据集
#reframed = pd.read_csv('处理后数据.csv', header=0, index_col=0)
# 分为训练集和测试集
values = reframed.values
values.shape
n_train_months = 423 #取四年的数据训练集，因为预测到未来t+1期，所以48个月-1
train = values[:n_train_months, :]
test = values[n_train_months:, :]
train.shape
# 分为输入和输出
n_obs = n_steps * n_features  #过去的步长乘以自变量指标来确定X的变量数
train_X, train_y = train[:, :n_obs], train[:, n_obs:]#将训练集分离出目标y，用于fit

test_X, test_y = test[:, :n_obs], test[:, n_obs:]#将测试集分离出目标Y，用于评分
print(train_X.shape, len(train_X), train_y.shape) #注意：二维数组表示（行，列）行数是行本数，列数是步数与指标数的乘积
# 为了lstm模型将训练数据集重塑为三维形状 [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_steps, n_features))#定义一个三维数组的参数：（层，行，列），层是样本数，行是步长，列是指标数。
#reshape的变化过程是原数组按照第一行从左到右，第二行从左到右边，拆开。重组的时候，按照新定义的规格，第一行从左到右边，填完一层填二层
#将原始数组，指标列数不变，行数不断采用滞后步进行切分，分成不同的层
#按照series_to_supervised函数的生产矩阵格式，排成指标为列，滞后为行，样本数量为层三维矩阵
#如果只预测多个指标，只有预测1期，则y可以简化成2维数组，列是指标，行变成样本量。或者预测一个指标的多期，列是期数，行是样本量。
test_X = test_X.reshape((test_X.shape[0], n_steps, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# 设计神经网络
model = Sequential() #生成模型实例
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]))) #input_shape是2维（Time_step, Input_Sizes), 
#只有第一层需要定义LSTM的参数input_shape或input_dim，官方文档给出的batch_input shape是3维: (Batch_size, Time_step, Input_Sizes), 
#其中Time_step是时间序列的长度, 对应到语句里就是语句的最大长度; Input_Sizes是每个时间点输入x的维度, 对于语句来说,就是一个字的embedding的向量维度.
#在没有定义batch_size的情况下, 可以通过model.fit中的batch_size参数进行定义, 但是这种方法无法再使用model.train_on_batch()函数.  
#如果这里定义了batch_size,那么在test的时候, 也要保证有该batch_size的数据, 否则会出现错误, 这样对只预测一个样本的问题无法兼容.  
#比较好的方法是将Batch_size设置为None, 这样不需要实现设置固定的batch_size,且可以调用train_on_batch.
#在没有定义Input_Sizes的情况下, 我尝试的训练, 会将其默认为1处理.
#另外, input_length指的也是输入句子的长度,即Time_step. 如下两种定义方式,表达的内容相同:
    #model.add(LSTM(32, input_shape=(None，10，64)));model.add(LSTM(32, input_lenth=10, input_dim=64 ))

model.add(Dense(1))#设置输出目标变量  多部预测的神经元怎么设置。
model.compile(loss='mae', optimizer='adam')#设置评估的函数，以及优化器
# 拟合网络模型
history = model.fit(train_X, train_y, epochs=100, batch_size=10, validation_data=(test_X, test_y), verbose=2, shuffle=False)
#epochs整个样本被训练优化多少次,重复fit可以再以后模型基础上再训练
#batch_size整个样本被分成几次塞进模型，每次塞进去的样本量。在小样本数的数据库中，不使用Batch Size是可行的，而且效果也很好。但是一旦是大型的数据库，一次性把所有数据输进网络，肯定会引起内存的爆炸。所以就提出Batch Size的概念。
#iterations（迭代）：每一次迭代都是一次权重更新,训练集有1000个样本，batchsize=10，那么：训练完整个样本集需要： 100次iteration，1次epoch。
#validation的数据并不会被用来调整参数，不会被用于更新权重。对于validation data来说，主要就是为了防止过拟合。
#比如说在训练过程中，查看模型在validation data上的accuracy，如果训练了10个epochs，发现accuracy都没提高，我们就可以及时停止训练，这个技巧被称为Early stopping，可以防止模型过度训练。


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

#每个质保评分
def print_rmse_mape(abs_):
    mae=abs_.mean()#Mean Absolute Error ，平均绝对误差
    rmse= sqrt((abs_**2).mean()) #Root Mean Square Error,均方根误差
    mape=(abs_/inv_y.reshape(1,-1)[0,:]).mean()# mean absolute percentage error，平均绝对百分比误差
    R2 = r2_score(inv_yhat.reshape(1,-1)[0,:], inv_y.reshape(1,-1)[0,:])
    print("平均绝对误差MAE={}；\n平均绝对百分比误差MAPE={:.2%}；\nR方={:.2%}；\n均方根误差RMSE={:.3f}。".format(mae,mape,R2,rmse))


abs_=np.abs(inv_yhat.reshape(1,-1)[0,:]-inv_y.reshape(1,-1)[0,:])
print_rmse_mape(abs_)








#预测期数------------------
groups = range(inv_y.shape[1])
#计算预测指标的平均绝对误差（mae），均方根误差(rmse)，平均绝对百分比误差(mape)

for group in groups:
    
    abs_=pd.Series(inv_y[:,group]-inv_yhat[:,group]).abs()
    
    print_rmse_mape(abs_)       



from sklearn.metrics import mean_squared_error
r_sq=[]
rmse=[]
mape=[]
for i in range(train_y.shape[1]):
    r_sq.append(1-(mean_squared_error(train_y[:,i],trainyhat[:,i])/np.var(train_y[:,i])))
    rmse.append(np.sqrt(mean_squared_error(train_y[:,i],trainyhat[:,i])))
    mape.append((abs(train_y[:,1]- trainyhat[:,1])/train_y[:,1]).mean())
np.array(r_sq).mean()
np.array(rmse).mean()
np.array(mape).mean()


print("均方根误差RMSE:{:.0f}；\n平均绝对百分比误差MAPE:{:.2%}。".format(rmse,mape))

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



from sklearn.metrics import mean_squared_error
r_sq=[]
rmse=[]
mape=[]
for i in range(train_y.shape[1]):
    r_sq.append(1-(mean_squared_error(train_y[:,i],trainyhat[:,i])/np.var(train_y[:,i])))
    rmse.append(np.sqrt(mean_squared_error(train_y[:,i],trainyhat[:,i])))
    mape.append((abs(train_y[:,1]- trainyhat[:,1])/train_y[:,1]).mean())
np.array(r_sq).mean()
np.array(rmse).mean()
np.array(mape).mean()


print("均方根误差RMSE:{:.0f}；\n平均绝对百分比误差MAPE:{:.2%}。".format(rmse,mape))



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






