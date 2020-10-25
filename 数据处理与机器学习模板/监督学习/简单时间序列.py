# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 22:38:57 2020

@author: Jay
"""

import pandas as pd 
import numpy as np
np.set_printoptions(suppress=True)#不采用科学计数法 
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
import matplotlib.pyplot as plt
import datetime as dt
import statsmodels.api as sm
from math import sqrt 
#朴素法。最近一期作为预测
dd = np.asarray(train.iloc[:,0])
y_hat['naive'] = dd[len(dd) - 1]
rms = sqrt(mean_squared_error(test[focastQuota],y_hat['naive']))
print(rms)

# 简单平均法，采用有所训练数据的平均数作为预测值
y_hat_avg['avg_forecast'] = train[focastQuota].mean()
rms = sqrt(mean_squared_error(test[focastQuota], y_hat_avg['avg_forecast']))

#移动平均法
y_hat_avg['moving_avg_forecast'] = train[focastQuota].rolling(12).mean().iloc[-1]
rms = sqrt(mean_squared_error(test[focastQuota], y_hat_avg['moving_avg_forecast']))
#am = Pd.Series(a).rolling(window=3, min_periods=1缩小前面的NAN, ,center=False 以中心为基点移动平均,on=None对于多列DataFrame，用on来指定使用哪列, axis=0, closed=None)
#也可以指数移动平均
#ewm = Pd.Series(a).ewm(span=10).mean() span是根据范围指定衰减， α=2/(span+1), for span≥1。alpha ：直接指定平滑系数α， 0<α≤1。

#加权平均法-简单指数平滑法（无明显线性趋势） 
fit = sm.tsa.SimpleExpSmoothing(np.asarray(train[focastQuota])).fit(smoothing_level=0.8, optimized=False)
y_hat_avg['SES'] = fit.forecast(len(test))
rms = sqrt(mean_squared_error(test[focastQuota], y_hat_avg['SES']))

#霍尔特(Holt)线性趋势法（等同二次指数平滑） 
fit = sm.tsa.Holt(np.asarray(train[focastQuota])).fit(smoothing_level=0.6, smoothing_slope=0.1)
y_hat_avg['Holt_linear'] = fit.forecast(len(test)) 
rms = sqrt(mean_squared_error(test[focastQuota], y_hat_avg['Holt_linear']))

#加权平均-指数平滑线性趋势季节性预测模型 （Holt-Winters）涵盖上面两种
fit1 = sm.tsa.ExponentialSmoothing(np.asarray(train[focastQuota]),seasonal_periods=12,
                            trend='add',seasonal='add').fit(smoothing_level=0.6)
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
rms = sqrt(mean_squared_error(test[focastQuota], y_hat_avg['Holt_Winter']))
#model = ExponentialSmoothing(train, seasonal='additive', seasonal_periods = seasonal_periods).fit()
#pred = model.predict(start=test.index[0], end=test.index[-1])
#trend ({"add", "mul", "additive", "multiplicative", None}, optional) – Type of trend component.
#seasonal ({"add", "mul", "additive", "multiplicative", None}, optional) – Type of seasonal component.
## fit(self, smoothing_level=None, smoothing_slope=None, smoothing_seasonal=None,
#       damping_slope=None, optimized=True, use_boxcox=False, remove_bias=False,
#       use_basinhopping=False)
#smoothing_level=None     ## alpha
#smoothing_slope=None     ## beta 
#smoothing_seasonal=None  ## gamma 
#damping_slope=None       ## phi value of the damped method
#optimized=True           ## hould the values that have not been set above be optimized automatically?
#use_boxcox=False         ## {True, False, 'log', float} log->apply log; float->lambda equal to float.
#remove_bias=False        ## 
#use_basinhopping=False   ## Should the opptimser try harder using basinhopping to find optimal values?
