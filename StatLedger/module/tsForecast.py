# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 07:46:30 2019

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
import tjfxdata as tjfx
#from sklearn.metrics import mean_squared_error
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
    

    

def tsForecast():       
    """
    此函数用于预测时间序列数据。
    method方法有'naive','average','moving_avg','simpleExpSmoothing','Holt_Winter'。
    """
    def print_rmse_mape(abs_):
        mae=abs_.mean()#Mean Absolute Error ，平均绝对误差
        rmse= sqrt((abs_**2).mean()) #Root Mean Square Error,均方根误差
        mape=(abs_/test[focastQuota]).mean()# mean absolute percentage error，平均绝对百分比误差
        print("平均绝对误差MAE={}；\n均方根误差RMSE={:.3f}；\n平均绝对百分比误差MAPE={:.2%}。".format(mae,rmse,mape))
     
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
    print("正在提取历史数据......")
    list1 = [[dept,quota,typer]]
    shuju_df = tjfx.TjfxData().getdata(startd,endd,list1)
    shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)
    df = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE')
    focastQuota = df.columns.values.tolist()[0][0]+df.columns.values.tolist()[0][1]
    df.columns =  [focastQuota]
    df[focastQuota].plot( figsize=(12, 8),title= focastQuota)
    plt.show()
    print("历史数据时间序列因素分解图")
    sm.tsa.seasonal_decompose(df[focastQuota]).plot()
    plt.show() 
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
    
    lens = int(input("请输入要预测的未来期数："))
    
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
        print_rmse_mape(abs_)
        dd = np.asarray(df.iloc[:,0])
        result = dd[-1]
        print("未来{}期的预测数据是：{}".format(lens,result))
        plt.figure(figsize=(12,8))
        plt.plot(df[focastQuota], label='历史')
        if typer == "m":
            plt.plot(pd.date_range(start= endd ,periods=lens+1,freq="MS",closed='right'),(result,)*lens,label='预测')
        else :
            plt.plot(pd.date_range(start= endd ,periods=lens+1,freq="D",closed='right'),(result,)*lens,label='预测')
        
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
        print_rmse_mape(abs_)
        result = df[focastQuota].mean()
        print("未来{}期的预测数据是：{}".format(lens,result))
        plt.figure(figsize=(12,8))
        plt.plot(df[focastQuota], label='历史')
        if typer == "m":
            plt.plot(pd.date_range(start= endd ,periods=lens+1,freq="MS",closed='right'),(result,)*lens,label='预测')
        else :
            plt.plot(pd.date_range(start= endd ,periods=lens+1,freq="D",closed='right'),(result,)*lens,label='预测')
        
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
        print_rmse_mape(abs_)
        result = df[focastQuota].rolling(roll).mean().iloc[-1]
        print("未来{}期的预测数据是：{}".format(lens,result))
        plt.figure(figsize=(12,8))
        plt.plot(df[focastQuota], label='历史')
        if typer == "m":
            plt.plot(pd.date_range(start= endd ,periods=lens+1,freq="MS",closed='right'),(result,)*lens,label='预测')
        else :
            plt.plot(pd.date_range(start= endd ,periods=lens+1,freq="D",closed='right'),(result,)*lens,label='预测')
        
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
        print_rmse_mape(abs_)
        fit1 = sm.tsa.SimpleExpSmoothing(np.asarray(df[focastQuota])).fit(smoothing_level=smoothing_level, optimized=False)
        result = fit1.forecast(lens)
        print("未来{}期的预测数据是：{}".format(lens,result))
        plt.figure(figsize=(12,8))
        plt.plot(df[focastQuota], label='历史')
        if typer == "m":
            plt.plot(pd.date_range(start= endd ,periods=lens+1,freq="MS",closed='right'),(result,)*lens,label='预测')
        else :
            plt.plot(pd.date_range(start= endd ,periods=lens+1,freq="D",closed='right'),(result,)*lens,label='预测')
        
        plt.legend(loc='best')
        plt.show() 
        
    if method == '5':
        y_hat_avg = test.copy()
        fit = sm.tsa.ExponentialSmoothing(np.asarray(train[focastQuota]),seasonal_periods=seasonal_periods,
                                    trend='add',seasonal='add').fit(smoothing_level=smoothing_level)
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

        y_hat_avg['Holt_Winter'] = fit.forecast(len(test))
        plt.figure(figsize=(16, 8))
        plt.plot(train[focastQuota], label='Train')
        plt.plot(test[focastQuota], label='Test')
        plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
        plt.legend(loc='best')
        plt.show()
        
        abs_=(test[focastQuota]-y_hat_avg['Holt_Winter']).abs()
        print_rmse_mape(abs_)     
        fit1 = sm.tsa.ExponentialSmoothing(np.asarray(df[focastQuota]),seasonal_periods=seasonal_periods,
                                    trend='add',seasonal='add').fit(smoothing_level=smoothing_level)
        result= fit1.forecast(lens)    
        print("未来{}期的预测数据是：{}".format(lens,result))
        plt.figure(figsize=(12,8))
        plt.plot(df[focastQuota], label='历史')
        if typer == "m":
            plt.plot(pd.date_range(start= endd ,periods=lens+1,freq="MS",closed='right'),result,label='预测')
        else :
            plt.plot(pd.date_range(start= endd ,periods=lens+1,freq="D",closed='right'),result,label='预测')
        
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
        print_rmse_mape(abs_)
        fit1 = sm.tsa.SARIMAX(df[focastQuota], order=(p, d, q), seasonal_order=(P, D, Q,seasonal_periods),
                              enforce_stationarity=False,enforce_invertibility=False).fit()
        result= fit1.forecast(lens)
        print("未来{}期的预测数据是：{}".format(lens,result))
        if typer == "m":
            plt.plot(pd.date_range(start= endd ,periods=lens+1,freq="MS",closed='right'),result,label='预测')
        else :
            plt.plot(pd.date_range(start= endd ,periods=lens+1,freq="D",closed='right'),result,label='预测')
        
        plt.legend(loc='best')
        plt.show() 
    
if __name__ == "__main__" :
    tsForecast()
      
        




