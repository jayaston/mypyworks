# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 07:46:30 2019

@author: Jay
"""

import sys
import os
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"StatLedger\module")))
except:
    sys.path.append(r'.\mypyworks\StatLedger\module')
import pandas as pd 
import numpy as np 
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
import matplotlib.pyplot as plt
import datetime as dt
import tjfxdata as tjfx
from sklearn.metrics import mean_squared_error
from math import sqrt 
import statsmodels.api as sm
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 移动平均图
def draw_trend(timeSeries, size):
    plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean = timeSeries.ewm(span=size).mean()
 
    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show() 
'''
　　ADF检验，p值小于0.05，则为稳定数据
'''
def testStationarity(ts):
    dftest = sm.tsa.adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput 
# 自相关和偏相关图，默认阶数为31阶
def draw_acf_pacf(ts, lags=20):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    sm.graphics.tsa.plot_acf(ts, lags=lags, ax=ax1)
    ax2 = f.add_subplot(212)
    sm.graphics.tsa.plot_pacf(ts, lags=lags, ax=ax2)
    plt.show()

#startd:str,endd:str,quota:list,method:str,roll:int=3,
#               smoothing_level:float=0.8,seasonal_periods:int=12,splitdate:str,
#               lens:int
    """此函数用于预测时间序列数据。
    method方法有'naive','average','moving_avg','simpleExpSmoothing','Holt_Winter'。
    startd,endd,splitdate需要填入六位日期如‘20190101’，quota是列表如['00','00718','m']。"""

quota = input("请输入预测的指标编码(五位)：")
dept = input("请输入预测的指标所属部门编码：")
typer = input("请输入预测的指标时间类型(m,d)：")

startY = input("请输入历史数据开始年(年四位)：")
startM = input("请输入历史数据开始月：")
startd = "{}{:0>2}01".format(startY,startM)
if typer == "d":
    startD = input("请输入历史数据开始日：")
    startd = "{}{:0>2}{:0>2}".format(startY,startM,startD)
endY = input("请输入历史数据结束年(年四位)：")
endM = input("请输入历史数据结束月：")
endd = "{}{:0>2}01".format(endY,endM)
if typer == "d":
    endD = input("请输入历史数据结束日：")
    endd = "{}{:0>2}{:0>2}".format(endY,endM,endD)


list1 = [[dept,quota,typer]]
shuju_df = tjfx.TjfxData().getdata(startd,endd,list1)
shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)
df = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE')
focastQuota = df.columns.values.tolist()[0][0]+df.columns.values.tolist()[0][1]
df.columns =  [focastQuota]
df[focastQuota].plot( figsize=(12, 8),title= focastQuota)
plt.show()
print("时间序列因素分解图")
sm.tsa.seasonal_decompose(df[focastQuota]).plot() 
print("下面输入一个日期划分训练数据和测试数据集")
splitY = input("请输入年(年四位)：")
splitM = input("请输入月：")
splitdate = "{}{:0>2}01".format(splitY,splitM)
if typer == "d":
    splitD = input("请输入日：")
    splitdate = "{}{:0>2}{:0>2}".format(splitY,splitM,splitD)
#划分数据集
train = df[df.index < dt.datetime.strptime(splitdate,'%Y%m%d')]
test = df[df.index >= dt.datetime.strptime(splitdate,'%Y%m%d')]
#绘制曲线图
train[focastQuota].plot( figsize=(12, 8),title= focastQuota)
test[focastQuota].plot(figsize=(12, 8))
plt.show()

method= input("请输入选用的预测方法的序号：\n\
              1.朴素法，采用历史数据最后一期数据预测下期数据\n\
              2.简单平均法，采用历史数据的平均值预测下期数据\n\
              3.移动平均法，采用最近n期数据的平均值预测下期数据\n\
              4.指数平滑，将历史数据赋予不同权重计算加权平均值预测下期数据\n\
              5.Holt-Winters季节性预测模型\n\
              6.季节性自回归移动平均模型（SARIMAX）:")
if method == "3":
    roll=int(input("请输入移动平均的期数："))
if method == "4":
    smoothing_level=float(input("请输入平滑系数（0-1之间推荐0.6）："))
if method == "5":
    smoothing_level=float(input("请输入平滑系数（0-1之间推荐0.6）："))
    seasonal_periods=int(input("请输入季节周期数（月一般12，日一般为7）："))
if method == "6":
    p=input("请输入p:")
    d=input("请输入d:")
    q=input("请输入q:")
    P=input("请输入P:")
    D=input("请输入D:")
    Q=input("请输入Q:")
    seasonal_periods=int(input("请输入季节周期数（月一般12，日一般为7）："))

lens = int(input("请输入要预测的未来期数"))

if method == '1':
    dd = np.asarray(train.iloc[:,0])
    y_hat = test.copy()
    y_hat['naive'] = dd[len(dd) - 1]
    #画图
    plt.figure(figsize=(12, 8))
    plt.plot(train.index, train[focastQuota], label='Train')
    plt.plot(test.index, test[focastQuota], label='Test')
    plt.plot(y_hat.index, y_hat['naive'], label='Naive Forecast')
    plt.legend(loc='best')
    plt.title("Naive Forecast")
    plt.show()
    #输出根方平均误
    abs_=(test[focastQuota]-y_hat['naive']).abs()
    mae=abs_.mean()#Mean Absolute Error ，平均绝对误差
    rmse= sqrt((abs_**2).mean()) #Root Mean Square Error,均方根误差
    mape=(abs_/test[focastQuota]).mean()# mean absolute percentage error，平均绝对百分比误差
    print("平均绝对误差MAE={mae}；\n均方根误差RMSE={rmse:.3f}；\n平均绝对百分比误差MAPE={mape:.2%}。".format(mae,rmse,mape))
    dd = np.asarray(df.iloc[:,0])
    result = dd[-1]
    print(f"未来{lens}期的预测数据是：{result:s}")
    plt.figure(figsize=(12,8))
    plt.plot(df[focastQuota], label='历史')
    if typer == "m":
        plt.plot(pd.date_range(start= endd ,periods=lens,freq="MS",closed='right'),(result,)*lens,label='预测')
    else :
        plt.plot(pd.date_range(start= endd ,periods=lens,freq="D",closed='right'),(result,)*lens,label='预测')
    
    plt.legend(loc='best')
    plt.show() 
    
    
if method == "2":
    y_hat_avg = test.copy()
    y_hat_avg['avg_forecast'] = train[focastQuota].mean()
    plt.figure(figsize=(12,8))
    plt.plot(train[focastQuota], label='Train')
    plt.plot(test[focastQuota], label='Test')
    plt.plot(y_hat_avg['avg_forecast'], label='Average Forecast')
    plt.legend(loc='best')
    plt.show()
    
    abs_=(test[focastQuota]-y_hat_avg['avg_forecast']).abs()
    mae=abs_.mean()#Mean Absolute Error ，平均绝对误差
    rmse= sqrt((abs_**2).mean()) #Root Mean Square Error,均方根误差
    mape=(abs_/test[focastQuota]).mean()# mean absolute percentage error，平均绝对百分比误差
    print("平均绝对误差MAE={mae}；\n均方根误差RMSE={rmse:.3f}；\n平均绝对百分比误差MAPE={mape:.2%}。".format(mae,rmse,mape))       
    result = df[focastQuota].mean()
    print(f"未来{lens}期的预测数据是：{result:s}")
    plt.figure(figsize=(12,8))
    plt.plot(df[focastQuota], label='历史')
    if typer == "m":
        plt.plot(pd.date_range(start= endd ,periods=lens,freq="MS",closed='right'),(result,)*lens,label='预测')
    else :
        plt.plot(pd.date_range(start= endd ,periods=lens,freq="D",closed='right'),(result,)*lens,label='预测')
    
    plt.legend(loc='best')
    plt.show() 
    
if method == '3':
    y_hat_avg = test.copy()
    y_hat_avg['moving_avg_forecast'] = train[focastQuota].rolling(roll).mean().iloc[-1]
    plt.figure(figsize=(16,8))
    plt.plot(train[focastQuota], label='Train')
    plt.plot(test[focastQuota], label='Test')
    plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast')
    plt.legend(loc='best')
    plt.show() 
    
    abs_=(test[focastQuota]-y_hat_avg['moving_avg_forecast']).abs()
    mae=abs_.mean()#Mean Absolute Error ，平均绝对误差
    rmse= sqrt((abs_**2).mean()) #Root Mean Square Error,均方根误差
    mape=(abs_/test[focastQuota]).mean()# mean absolute percentage error，平均绝对百分比误差
    print("平均绝对误差MAE={mae}；\n均方根误差RMSE={rmse:.3f}；\n平均绝对百分比误差MAPE={mape:.2%}。".format(mae,rmse,mape))       
    result = df[focastQuota].rolling(roll).mean().iloc[-1]
    print(f"未来{lens}期的预测数据是：{result:s}")
    plt.figure(figsize=(12,8))
    plt.plot(df[focastQuota], label='历史')
    if typer == "m":
        plt.plot(pd.date_range(start= endd ,periods=lens,freq="MS",closed='right'),(result,)*lens,label='预测')
    else :
        plt.plot(pd.date_range(start= endd ,periods=lens,freq="D",closed='right'),(result,)*lens,label='预测')
    
    plt.legend(loc='best')
    plt.show() 
    
if method == "4":
    y_hat_avg = test.copy()
    fit = sm.tsa.SimpleExpSmoothing(np.asarray(train[focastQuota])).fit(smoothing_level=smoothing_level, optimized=False)
    y_hat_avg['SES'] = fit.forecast(len(test))
    plt.figure(figsize=(16, 8))
    plt.plot(train[focastQuota], label='Train')
    plt.plot(test[focastQuota], label='Test')
    plt.plot(y_hat_avg['SES'], label='SES')
    plt.legend(loc='best')
    plt.show() 
    
    abs_=(test[focastQuota]-y_hat_avg['SES']).abs()
    mae=abs_.mean()#Mean Absolute Error ，平均绝对误差
    rmse= sqrt((abs_**2).mean()) #Root Mean Square Error,均方根误差
    mape=(abs_/test[focastQuota]).mean()# mean absolute percentage error，平均绝对百分比误差
    print("平均绝对误差MAE={mae}；\n均方根误差RMSE={rmse:.3f}；\n平均绝对百分比误差MAPE={mape:.2%}。".format(mae,rmse,mape))       
    fit1 = sm.tsa.SimpleExpSmoothing(np.asarray(df[focastQuota])).fit(smoothing_level=smoothing_level, optimized=False)
    result = fit1.forecast(lens)
    print(f"未来{lens}期的预测数据是：{result:s}")
    plt.figure(figsize=(12,8))
    plt.plot(df[focastQuota], label='历史')
    if typer == "m":
        plt.plot(pd.date_range(start= endd ,periods=lens,freq="MS",closed='right'),(result,)*lens,label='预测')
    else :
        plt.plot(pd.date_range(start= endd ,periods=lens,freq="D",closed='right'),(result,)*lens,label='预测')
    
    plt.legend(loc='best')
    plt.show() 
    
if method == '5':
    y_hat_avg = test.copy()
    fit = sm.tsa.ExponentialSmoothing(np.asarray(train[focastQuota]),seasonal_periods=seasonal_periods,
                                trend='add',seasonal='add').fit(smoothing_level=smoothing_level)
    y_hat_avg['Holt_Winter'] = fit.forecast(len(test))
    plt.figure(figsize=(16, 8))
    plt.plot(train[focastQuota], label='Train')
    plt.plot(test[focastQuota], label='Test')
    plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
    plt.legend(loc='best')
    plt.show()
    
    abs_=(test[focastQuota]-y_hat_avg['Holt_Winter']).abs()
    mae=abs_.mean()#Mean Absolute Error ，平均绝对误差
    rmse= sqrt((abs_**2).mean()) #Root Mean Square Error,均方根误差
    mape=(abs_/test[focastQuota]).mean()# mean absolute percentage error，平均绝对百分比误差
    print("平均绝对误差MAE={mae}；\n均方根误差RMSE={rmse:.3f}；\n平均绝对百分比误差MAPE={mape:.2%}。".format(mae,rmse,mape))
            
    fit1 = sm.tsa.ExponentialSmoothing(np.asarray(df[focastQuota]),seasonal_periods=seasonal_periods,
                                trend='add',seasonal='add').fit(smoothing_level=smoothing_level)
    result= fit1.forecast(lens)
    print(f"未来{lens}期的预测数据是：{result:s}")
    plt.figure(figsize=(12,8))
    plt.plot(df[focastQuota], label='历史')
    if typer == "m":
        plt.plot(pd.date_range(start= endd ,periods=lens,freq="MS",closed='right'),result,label='预测')
    else :
        plt.plot(pd.date_range(start= endd ,periods=lens,freq="D",closed='right'),result,label='预测')
    
    plt.legend(loc='best')
    plt.show() 
    
if method == "6":
    y_hat_avg = test.copy()
    fit = sm.tsa.SARIMAX(train[focastQuota], order=(p, d, q), seasonal_order=(P, D, Q,seasonal_periods),
                          enforce_stationarity=False,enforce_invertibility=False).fit()
    y_hat_avg['SARIMAX'] = fit.forecast(len(test))
     #ARIMAResults.predict(start=None, end=None, exog=None, typ=‘linear‘, dynamic=False)
     #dynamic，逻辑值，True表样本外预测，默认False样本预测，样本内预测可以通过设置结束值跨样本外预测
     #typ，取值‘linear‘, ‘levels‘表示根据内生变量的差分做线性预测，预测原数据的水平（源数据的模型预测值）
     #ARIMAResults.forecast（step=n）函数可预测未来N期数据
    
    plt.figure(figsize=(16, 8))
    plt.plot(train[focastQuota], label='Train')
    plt.plot(test[focastQuota], label='Test')
    plt.plot(y_hat_avg['SARIMAX'], label='SARIMAX')
    plt.legend(loc='best')
    plt.show()
    
    abs_=(test[focastQuota]-y_hat_avg['SARIMAX']).abs()
    mae=abs_.mean()#Mean Absolute Error ，平均绝对误差
    rmse= sqrt((abs_**2).mean()) #Root Mean Square Error,均方根误差
    mape=(abs_/test[focastQuota]).mean()# mean absolute percentage error，平均绝对百分比误差
    print("平均绝对误差MAE={mae}；\n均方根误差RMSE={rmse:.3f}；\n平均绝对百分比误差MAPE={mape:.2%}。".format(mae,rmse,mape))
    
    fit1 = sm.tsa.SARIMAX(df[focastQuota], order=(p, d, q), seasonal_order=(P, D, Q,seasonal_periods),
                          enforce_stationarity=False,enforce_invertibility=False).fit()
    result= fit1.forecast(lens)
    print(f"未来{lens}期的预测数据是：{result:s}") 
    if typer == "m":
        plt.plot(pd.date_range(start= endd ,periods=lens,freq="MS",closed='right'),result,label='预测')
    else :
        plt.plot(pd.date_range(start= endd ,periods=lens,freq="D",closed='right'),result,label='预测')
    
    plt.legend(loc='best')
    plt.show() 
    
       
        


#获取数据
list1=[['00','00718','m']]
shuju_df = tjfx.TjfxData().getdata('20160101','20191031',list1)
shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)
df = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE')
focastQuota = df.columns.values.tolist()[0][0]+df.columns.values.tolist()[0][1]
df.columns =  [focastQuota] 

##画移动平均和加权平均趋势线
#draw_trend(df[focastQuota],12)
##画时间线
#df[focastQuota].plot()

#划分数据
#df = df[df.index.strftime('%m')>'03']   
train = df[df.index < dt.datetime.strptime('2019-1-1','%Y-%m-%d')]
test = df[df.index >= dt.datetime.strptime('2019-1-1','%Y-%m-%d')]
train[focastQuota].plot( figsize=(12, 8),title= focastQuota)
test[focastQuota].plot(figsize=(12, 8))
plt.show()

#sm.tsa.seasonal_decompose(train[focastQuota]).plot()


#预测方法

#朴素法,采用最近一期的数据预测下一期
dd = np.asarray(train.iloc[:,0])
y_hat = test.copy()
y_hat['naive'] = dd[len(dd) - 1]

plt.figure(figsize=(12, 8))
plt.plot(train.index, train[focastQuota], label='Train')
plt.plot(test.index, test[focastQuota], label='Test')
plt.plot(y_hat.index, y_hat['naive'], label='Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.show()
##计算根方误差
rms = sqrt(mean_squared_error(test[focastQuota],y_hat['naive']))
print(rms)

# 简单平均法，采用有所训练数据的平均数作为预测值
y_hat_avg = test.copy()
y_hat_avg['avg_forecast'] = train[focastQuota].mean()
plt.figure(figsize=(12,8))
plt.plot(train[focastQuota], label='Train')
plt.plot(test[focastQuota], label='Test')
plt.plot(y_hat_avg['avg_forecast'], label='Average Forecast')
plt.legend(loc='best')
plt.show()
rms = sqrt(mean_squared_error(test[focastQuota], y_hat_avg['avg_forecast']))
print(rms)

#移动平均法
y_hat_avg = test.copy()
y_hat_avg['moving_avg_forecast'] = train[focastQuota].rolling(12).mean().iloc[-1]
plt.figure(figsize=(16,8))
plt.plot(train[focastQuota], label='Train')
plt.plot(test[focastQuota], label='Test')
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast')
plt.legend(loc='best')
plt.show() 
rms = sqrt(mean_squared_error(test[focastQuota], y_hat_avg['moving_avg_forecast']))
print(rms)


#加权平均法-简单指数平滑法（无明显线性趋势） 
y_hat_avg = test.copy()
fit = sm.tsa.SimpleExpSmoothing(np.asarray(train[focastQuota])).fit(smoothing_level=0.8, optimized=False)
y_hat_avg['SES'] = fit.forecast(len(test))
plt.figure(figsize=(16, 8))
plt.plot(train[focastQuota], label='Train')
plt.plot(test[focastQuota], label='Test')
plt.plot(y_hat_avg['SES'], label='SES')
plt.legend(loc='best')
plt.show() 
rms = sqrt(mean_squared_error(test[focastQuota], y_hat_avg['SES']))
print(rms)

#加权平均-指数平滑线性趋势季节性预测模型 （Holt-Winters）
y_hat_avg = test.copy()
fit1 = sm.tsa.ExponentialSmoothing(np.asarray(train[focastQuota]),seasonal_periods=12,
                            trend='add',seasonal='add').fit(smoothing_level=0.6)
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
plt.figure(figsize=(16, 8))
plt.plot(train[focastQuota], label='Train')
plt.plot(test[focastQuota], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()
rms = sqrt(mean_squared_error(test[focastQuota], y_hat_avg['Holt_Winter']))
print(rms)
plt.figure(1)

#指数平滑模型参数
#model = ExponentialSmoothing(train, seasonal='additive', seasonal_periods = seasonal_periods).fit()
#pred = model.predict(start=test.index[0], end=test.index[-1])
#trend ({"add", "mul", "additive", "multiplicative", None}, optional) – Type of trend component.
#seasonal ({"add", "mul", "additive", "multiplicative", None}, optional) – Type of seasonal component.

 
## fit(self, smoothing_level=None, smoothing_slope=None, smoothing_seasonal=None,
#       damping_slope=None, optimized=True, use_boxcox=False, remove_bias=False,
#       use_basinhopping=False)
## 若需详细了解 - 建议看源码
## https://www.statsmodels.org/stable/_modules/statsmodels/tsa/holtwinters.html#ExponentialSmoothing
#smoothing_level=None     ## alpha
#smoothing_slope=None     ## beta 
#smoothing_seasonal=None  ## gamma 
#damping_slope=None       ## phi value of the damped method
#optimized=True           ## hould the values that have not been set above be optimized automatically?
#use_boxcox=False         ## {True, False, 'log', float} log->apply log; float->lambda equal to float.
#remove_bias=False        ## 
#use_basinhopping=False   ## Should the opptimser try harder using basinhopping to find optimal values?


#霍尔特(Holt)线性趋势法（等同二次指数平滑） 
#result = sm.tsa.stattools.adfuller(train[focastQuota])
#plt.show()
# 
#y_hat_avg = test.copy() 
#fit = sm.tsa.Holt(np.asarray(train[focastQuota])).fit(smoothing_level=0.6, smoothing_slope=0.1)
#y_hat_avg['Holt_linear'] = fit.forecast(len(test)) 
#plt.figure(figsize=(16, 8))
#plt.plot(train[focastQuota], label='Train')
#plt.plot(test[focastQuota], label='Test')
#plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
#plt.legend(loc='best')
#plt.show()
#
#rms = sqrt(mean_squared_error(test[focastQuota], y_hat_avg['Holt_linear']))
#print(rms)


# 带有季节因素的简单平均法
y_hat_avg = test.copy()
y_train = train.copy()
#y_hat_avg["weekday"]=train[focastQuota].index.weekday#工作日0-6
#y_hat_avg["hour"]= train[focastQuota].index.hour
y_train["month"] =  y_train[focastQuota].index.month
y_hat_avg["month"] =  y_hat_avg[focastQuota].index.month
df_season_avg_forecast = y_train.groupby("month").mean()
df_season_avg_forecast.columns = ['season_avg_forecast']
y_hat_avg = pd.merge(y_hat_avg,df_season_avg_forecast,how="left",left_on="month",right_index=True)
plt.figure(figsize=(12,8))
plt.plot(train[focastQuota], label='Train')
plt.plot(test[focastQuota], label='Test')
plt.plot(y_hat_avg['season_avg_forecast'], label='Season Average Forecast')
plt.legend(loc='best')
plt.show()
rms = sqrt(mean_squared_error(test[focastQuota], y_hat_avg['season_avg_forecast']))
print(rms)


#季节性自回归差分移动平均模型SARIMAX
##稳定性检验
testStationarity(train[focastQuota])
##画自相关图与偏自相关图
draw_acf_pacf(train[focastQuota])
##分解
sm.tsa.seasonal_decompose(train[focastQuota]).plot()

#只做一阶差分，稳定数据
data_diff1 = train[focastQuota].diff(1)
data_diff1.dropna(inplace=True)
testStationarity(data_diff1)
sm.tsa.seasonal_decompose(data_diff1).plot()
draw_acf_pacf(data_diff1,10)

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0,2)
# Generate all different combinations of p, q and q triplets
import itertools
pdq = list(itertools.product(p, d, q))
# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

import warnings
warnings.filterwarnings("ignore") # specify to ignore warning messages
df1 = pd.DataFrame(columns=('param', ' param_seasonal', 'results.aic'))
i = 0
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.SARIMAX(train[focastQuota],order=param,
                                 seasonal_order=param_seasonal,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False)

            results = mod.fit()
            df1.loc[i] = [param,param_seasonal,results.aic]
            i+=1
#            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))        
            
        except:
            continue

##拟合检验
#df1.to_excel("pdq.xls")    
##应用测试数据集（泛化能力测试）

y_hat_avg = test.copy()
fit1 = sm.tsa.SARIMAX(train[focastQuota], order=(1, 1, 1), seasonal_order=(1, 1, 0,12),
                      enforce_stationarity=False,enforce_invertibility=False).fit()
y_hat_avg['SARIMA'] = fit1.predict(start="2019-1-1", end="2019-10-1", dynamic=True)
 #ARIMAResults.predict(start=None, end=None, exog=None, typ=‘linear‘, dynamic=False)
 #dynamic，逻辑值，True表样本外预测，默认False样本预测，样本内预测可以通过设置结束值跨样本外预测
 #typ，取值‘linear‘, ‘levels‘表示根据内生变量的差分做线性预测，预测原数据的水平（源数据的模型预测值）
 #ARIMAResults.forecast（step=n）函数可预测未来N期数据

plt.figure(figsize=(16, 8))
plt.plot(train[focastQuota], label='Train')
plt.plot(test[focastQuota], label='Test')
plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(test[focastQuota], y_hat_avg['SARIMA']))
print(rms)

#自回归差分移动平均模型ARIMA或者ARMA
##一个自动确定P，q的函数
#(p, q) =(sm.tsa.arma_order_select_ic(dta,max_ar=3,max_ma=3,ic='aic')['aic_min_order'])
#这里需要设定自动取阶的 p和q 的最大值，即函数里面的max_ar,和max_ma。ic 参数表示选用的选取标准，这里设置的为aic,当然也可以用bic。然后函数会算出每个 p和q 组合(这里是(0,0)~(3,3)的AIC的值，取其中最小的,这里的结果是(p=0,q=1)。


 

