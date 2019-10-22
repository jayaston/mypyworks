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

list = [['00','00718','d']]
shuju_df = tjfx.TjfxData().getdata('20160101','20191015',list)
shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)
df = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE')
shuju_df.info()
focastQuota = df.columns.values.tolist()[0][0]+df.columns.values.tolist()[0][1]
df.columns =  [focastQuota]
train = df[df.index <= dt.datetime.strptime('2018-12-31','%Y-%m-%d')]
test = df[df.index > dt.datetime.strptime('2018-12-31','%Y-%m-%d')]

train[focastQuota].plot( figsize=(12, 8),title= focastQuota)
test[focastQuota].plot(figsize=(12, 8))
plt.show()

#朴素法

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



rms = sqrt(mean_squared_error(test[focastQuota],y_hat['naive']))
print(rms)


# 简单平均法
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
y_hat_avg['moving_avg_forecast'] = train[focastQuota].rolling(60).mean().iloc[-1]
plt.figure(figsize=(16,8))
plt.plot(train[focastQuota], label='Train')
plt.plot(test[focastQuota], label='Test')
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast')
plt.legend(loc='best')
plt.show()


 
rms = sqrt(mean_squared_error(test[focastQuota], y_hat_avg['moving_avg_forecast']))
print(rms)

#简单指数平滑法
from statsmodels.tsa.api import SimpleExpSmoothing
 
y_hat_avg = test.copy()
fit = SimpleExpSmoothing(np.asarray(train[focastQuota])).fit(smoothing_level=0.6, optimized=False)
y_hat_avg['SES'] = fit.forecast(len(test))
plt.figure(figsize=(16, 8))
plt.plot(train[focastQuota], label='Train')
plt.plot(test[focastQuota], label='Test')
plt.plot(y_hat_avg['SES'], label='SES')
plt.legend(loc='best')
plt.show()

 
rms = sqrt(mean_squared_error(test[focastQuota], y_hat_avg['SES']))
print(rms)

#霍尔特(Holt)线性趋势法

import statsmodels.api as sm
 
sm.tsa.seasonal_decompose(train[focastQuota]).plot()
result = sm.tsa.stattools.adfuller(train[focastQuota])
plt.show()

from statsmodels.tsa.api import Holt
 
y_hat_avg = test.copy()
 
fit = Holt(np.asarray(train[focastQuota])).fit(smoothing_level=0.3, smoothing_slope=0.1)
y_hat_avg['Holt_linear'] = fit.forecast(len(test))
 
plt.figure(figsize=(16, 8))
plt.plot(train[focastQuota], label='Train')
plt.plot(test[focastQuota], label='Test')
plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(test[focastQuota], y_hat_avg['Holt_linear']))
print(rms)

#Holt-Winters季节性预测模型
from statsmodels.tsa.api import ExponentialSmoothing
 
y_hat_avg = test.copy()
fit1 = ExponentialSmoothing(np.asarray(train[focastQuota]), seasonal_periods=7, trend='add', seasonal='add', ).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
plt.figure(figsize=(16, 8))
plt.plot(train[focastQuota], label='Train')
plt.plot(test[focastQuota], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(test[focastQuota], y_hat_avg['Holt_Winter']))
print(rms)

#自回归移动平均模型（ARIMA）
import statsmodels.api as sm
 
y_hat_avg = test.copy()
fit1 = sm.tsa.statespace.SARIMAX(train[focastQuota], order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()
y_hat_avg['SARIMA'] = fit1.predict(start="2013-11-1", end="2013-12-31", dynamic=True)
plt.figure(figsize=(16, 8))
plt.plot(train[focastQuota], label='Train')
plt.plot(test[focastQuota], label='Test')
plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(test[focastQuota], y_hat_avg['SARIMA']))
print(rms)

