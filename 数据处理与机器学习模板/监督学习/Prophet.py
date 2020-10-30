# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:47:27 2020

@author: XieJie
"""
#调入常用包

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import sys
import os
#查看当前目录
os.getcwd()
#获取数据
sup_water = pd.read_excel('./mypyworks/输出/2015-2020日供水总量.xlsx',index_col=0)

#3. 数据选择
#设置模型数据日期范围
end_date = '2019-07-12'#设定模型数据结束日期
test_len = 31          #设定预留测试集长度
horizon = 31           #设定模型预测长度
period = 7             #设定训练集每折滑动期数
initial = 365 *2       #设定模型的训练集长度
cv = 10                 #设定交叉验证折数

#计算模型需要数据跨度
data_len = initial + period *(cv-1) + horizon + test_len
data_len
#根据模型数据时期跨度计算模型数据开始时间
start_date = (dt.datetime.strptime(end_date,'%Y-%m-%d') - dt.timedelta(days = data_len ) ).strftime('%Y-%m-%d')
#提取模型所需数据
sup_water = sup_water.set_index('QUOTA_DATE').sort_index()[start_date:end_date]

#4. 数据清洗
#找出缺失值
sup_water[sup_water.isnull().any(axis=1)]
#统计非数值个数
sup_water['最高温度'][~np.isfinite(sup_water['最高温度'])==1].count()
# print(sup_water['最高温度'].value_counts(dropna=False))
#判断数据框里面所有不是有限数据的数，可以排除null，np.nan,np.inf
sup_water[~(np.isfinite(sup_water).all(axis=1))]
#判断离群值
#计算总体偏离值的标准差
quota = '水厂供水总量' #选择指标名称
window = 3             #选择移动平均长度
std_n = 3              #标准差个数
am = sup_water[quota].rolling(window=window,min_periods=0,center=True).mean()
diff = sup_water[quota].values - am
print(sup_water[np.abs(diff)> std_n * diff.std(ddof=0)])
#提示：出现离群值预警后，需要对数据进行核查，如果无法核实可以设置为缺失值，prophet模型可以自动处理缺失值。

#找出一个相关变量
#添加平均温度自变量
sup_water.eval("平均温度=(最高温度+最低温度)/2",inplace=True)
#多项式处理各自变量观察与目标变量的相关性
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=3,include_bias=1)#最高三次方处理
X_poly = poly_features.fit_transform(sup_water[['最低温度','最高温度','平均温度']])
#将poly数组转成dataframe做相关矩阵
sup_water_poly= pd.DataFrame(data=X_poly,columns=poly_features.get_feature_names())
sup_water_poly['水厂供水总量'] = sup_water['水厂供水总量'].values
sup_water_poly.corrwith(sup_water_poly['水厂供水总量']).sort_values(ascending=False)

#训练模型
#调用prophet模型
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation 
# from sklearn.metrics import mean_squared_error
#修改变量名为prophet标准变量名
sup_water=sup_water.reset_index()[['QUOTA_DATE','水厂供水总量','平均温度']].rename(columns={'QUOTA_DATE':'ds','水厂供水总量':'y'})

#划分训练数据集和测试数据集
sup_water_train = sup_water[:-test_len] #训练数据集
sup_water_test = sup_water[-test_len:]  #测试数据集
#对数化处理
#sup_water['y'] = np.log(sup_water['y'])
#标识对水量影响较大的时段
#标识国庆节
guoqing = pd.DataFrame({
  'holiday': 'guoqing',
  'ds': pd.date_range(start= '2015-10-1',periods=7,freq='AS-OCT'),
  'lower_window': 0,
  'upper_window': 6,
})
#标识农历春节
import sxtwl
lunar = sxtwl.Lunar()
spring_date=[]
for i in list(range(2014,2021)):
    solar_day = lunar.getDayByLunar(i,12,30)
    solar_date = dt.date(solar_day.y, solar_day.m, solar_day.d)
    spring_date.append(solar_date)
    
spring = pd.DataFrame({
  'holiday': 'spring',
  'ds':spring_date,
  'lower_window': -14,
  'upper_window': 14,
})
#标识新冠疫情期
#一级响应期
covid1 = pd.DataFrame({
  'holiday': 'covid1',
  'ds':pd.date_range(start= '2020-1-23',end='2020-2-23'),
  'lower_window': 0,
  'upper_window': 0,
})
#二级响应期
covid2 = pd.DataFrame({
  'holiday': 'covid2',
  'ds':pd.date_range(start= '2020-2-24',end='2020-5-8'),
  'lower_window': 0,
  'upper_window': 0,
})
#三级响应期
covid3 = pd.DataFrame({
  'holiday': 'covid3',
  'ds':pd.date_range(start= '2020-5-9',end='2020-8-13'),
  'lower_window': 0,
  'upper_window': 0,
})
holidays = pd.concat((guoqing, spring,covid1,covid2,covid3))
# 设置需要比较的超参数
growth =["linear","logistic"]

def seq(start,step,n):    
    end = start + step*n
    return list(range(start,end,step))
cap = seq(4900000,50000,10)#设置logistic增长饱和最大值
#通过交叉验证找出评价指标最高的超参数组合。
import warnings
warnings.filterwarnings("ignore") 
# def fun_rmse(df):
#             return np.sqrt(np.mean(np.square(df['y']-df['yhat'])))
def fun_mape(df):
    return np.mean(np.abs((df['yhat']-df['y'])/df['y']))
param_df =pd.DataFrame(columns=('growth',  'cap',  'mape'))
idx = 0
for i in growth:
    if i == "logistic":        
        for j in cap: 
            sup_water_train['cap'] = j
            try:
                m = Prophet(
                    growth='logistic',
                    holidays=holidays,
                    n_changepoints = 25,
                    changepoint_range = 0.8 ,
                    changepoint_prior_scale = 0.05,
                    holidays_prior_scale=10 )
                m.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=10)
                m.add_seasonality(name='yearly', period=365.25, fourier_order=10, prior_scale=10)
                m.add_regressor('平均温度',prior_scale=10,mode='multiplicative')
                m.fit(sup_water_train)

                df_cv = cross_validation(m, initial='730 days', period='7 days', horizon = '31 days') 
                mape_ave = df_cv.groupby('cutoff').apply(fun_mape).mean()
                param_df.loc[idx] = [i,j,mape_ave]   
                idx += 1
                print('第{:d}次运算记录：growth={:s}，cap={:d}，mape={:.5f}'.format(idx,i,j,mape_ave))
            except:
                continue
    else:
        try:
            sup_water_train.drop('cap',axis=1,inplace=True)
        except:
            pass
        try:
            m = Prophet(holidays=holidays,
                n_changepoints = 25,
                changepoint_range = 0.8 ,
                changepoint_prior_scale = 0.05,
                holidays_prior_scale=10 )
            m.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=10)
            m.add_seasonality(name='yearly', period=365.25, fourier_order=10, prior_scale=10)
            m.add_regressor('平均温度',prior_scale=10,mode='multiplicative')
            m.fit(sup_water_train)

            df_cv = cross_validation(m, initial='730 days', period='7 days', horizon = '31 days') 
            mape_ave = df_cv.groupby('cutoff').apply(fun_mape).mean()
            param_df.loc[idx] = [i,np.nan,mape_ave]   
            idx += 1
            print('第{:d}次运算记录：growth={:s}，mape={:.5f}'.format(idx,i,mape_ave))
        except:
            continue

param_df.sort_values(['mape']).head(10)#显示评价指标最高的10条记录
#参数较多可以保存在本地方便查询。
#param_df.sort_values(['mape']).to_excel('param.xls')
#按照交叉验证最佳mape设置超参数设置模型
sup_water_train['cap'] = 5100000 #设置增长饱和值
m = Prophet(holidays=holidays,
    growth='logistic', 
    interval_width = 0.8,           #预测不确定区间宽度
    n_changepoints = 25,            #识别变点的上限数量             
    changepoint_range = 0.8 ,       #使用前80%比例数据作为变点识别数据
    changepoint_prior_scale = 0.05, #changepoint_prior_scale越大，识别为变点越多；越小，识别为变点越少。
    holidays_prior_scale=10)        #holidays_prior_scale越大，假期对目标值的影响越大。越小，假期对目标值的影响越小。
m.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=10) #fourier_order越大，对周期变化的拟合越细致也越容易过拟合。
m.add_seasonality(name='yearly', period=365.25, fourier_order=10, prior_scale=10)#prior_scale越大，对于目标变量影响越大。
m.add_regressor('平均温度',prior_scale=10,mode='multiplicative') #回归量采用乘法模型
#按照测试集前两年的数据训练模型
m.fit(sup_water_train[-730:])
#为测试集输入cap
sup_water_test['cap'] = 5100000
#模型运用测试集进行检验
future = sup_water_test.drop('y',axis=1)
forecast = m.predict(future)
#2019-06-01到2019-07-01测试集实际值与预测值对比
plt.plot(sup_water_test['y'].values, label='实际值')
plt.plot(forecast['yhat'].values,linestyle='--', label='预测值')
plt.legend(loc='best')
plt.show()
#计算测试数据集预测值的mape
np.mean(np.abs((forecast['yhat'].values - sup_water_test['y'].values)/sup_water_test['y'].values))

#模型预测
m = Prophet(holidays=holidays,
    growth='logistic', 
    interval_width = 0.8,           #预测不确定区间宽度
    n_changepoints = 25,            #识别变点的上限数量             
    changepoint_range = 0.8 ,       #使用前80%比例数据作为变点识别数据
    changepoint_prior_scale = 0.05, #changepoint_prior_scale越大，识别为变点越多；越小，识别为变点越少。
    holidays_prior_scale=10)        #holidays_prior_scale越大，假期对目标值的影响越大。越小，假期对目标值的影响越小。
m.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=10) #fourier_order越大，对周期变化的拟合越细致也越容易过拟合。
m.add_seasonality(name='yearly', period=365.25, fourier_order=10, prior_scale=10)#prior_scale越大，对于目标变量影响越大。
m.add_regressor('平均温度',prior_scale=10,mode='multiplicative') #回归量采用乘法模型
#按照测试集前两年的数据训练模型
sup_water['cap']= 5100000 #设置增长饱和值
m.fit(sup_water[-730:])

#预测未来30天
future = m.make_future_dataframe(periods=29,freq = 'd', 
                                 include_history = False)

#输入预测期的饱和最大值
future['cap'] = 5100000
#输入天气预报中未来29日的平均温度
future['平均温度'] = [32,32.5,31.5,31.5,31.5,31.5,31.5,28.5,29,29.5,30,30,30.5,30.5,30.5,29.5,30.5,29,28.5,26,27,26.5,28,30.5,29.5,31,32,32.5,32]
#正式预测
forecast = m.predict(future)
fig = m.plot(forecast)
#未来29天的预测结果
tmp = forecast[['ds','yhat']]
tmp['yhat']=tmp['yhat'].astype('int')
