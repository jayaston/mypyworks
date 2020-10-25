# -*- coding: utf-8 -*-

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
import matplotlib.pyplot as plt

import numpy as np
np.set_printoptions(suppress=True) 
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np
import datetime as dt

import pandas as pd
import numpy as np
from scipy import  stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

import os
os.chdir(r"D:\Python_book\18Timeseries")

# 移动平均图
def draw_trend(timeSeries, size):
    plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size,min_periods=1,center=True).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean = timeSeries.ewm(span=size).mean()
 
    timeSeries.plot(color='blue', label='原曲线',alpha=0.7)
    rol_mean.plot(color='red', label='移动平均',alpha=0.7)
    rol_weighted_mean.plot(color='black', label='指数加权移动平均',alpha=0.7)
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show() 

def testStationarity(ts):
    dftest = sm.tsa.adfuller(ts)
    # 时间序列稳定性检验
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput 
# 自相关和偏相关图，默认阶数为10阶
def draw_acf_pacf(ts, lags=10):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    sm.graphics.tsa.plot_acf(ts, lags=lags, ax=ax1)
    ax2 = f.add_subplot(212)
    sm.graphics.tsa.plot_pacf(ts, lags=lags, ax=ax2)
    plt.show()

#先稳定性考察确定d一阶差分，p值小于0.05，后数据稳定
data_diff1 = train[focastQuota].diff(1)
data_diff1.dropna(inplace=True)
testStationarity(data_diff1)
#因素分解图画自相关偏相关图确定pd
sm.tsa.seasonal_decompose(data_diff1).plot()
draw_acf_pacf(data_diff1,10)

#自相关和偏自相关
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.tsa.plot_acf(dta,lags=20)#lags 表示滞后的阶数
fig = sm.graphics.tsa.plot_pacf(dta,lags=20)
plt.show()



#RMSE参数搜索 确定pdqs，用rmse
# Define the p, d 、 q and s parameters 
p = range(0,10)
d = [1]
q = range(0,2)
s = [6,7,8]

import itertools
pdq = list(itertools.product(p, d, q ))

seasonal_pdq = list(itertools.product(p, d, q ,s))
import warnings
warnings.filterwarnings("ignore") 
df1 = pd.DataFrame(columns=('param', 'param_seasonal', 'rmse'))
i = 0
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            fit = sm.tsa.SARIMAX(train[focastQuota],order=param,
                                 seasonal_order=param_seasonal,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False).fit()

            validation['SARIMA'] = fit.predict(start=validation.index[0], end=validation.index[-1], dynamic=True)
            rmse = np.sqrt(mean_squared_error(validation[focastQuota], validation['SARIMA']))
            df1.loc[i] = [param,param_seasonal,rmse]
            i+=1            
        except:
            continue
df1.sort_values(['rmse']).head(10)



#bic参数搜索
from statsmodels.tsa.arima_model import ARIMA

#定阶
pmax = int(len(xdata)/10) #一般阶数不超过length/10
qmax = int(len(xdata)/10) #一般阶数不超过length/10
bic_matrix = [] #bic矩阵
for p in range(pmax+1):
  tmp = []
  for q in range(qmax+1):
    try: #存在部分报错，所以用try来跳过报错。
      tmp.append(ARIMA(xdata, (p,1,q)).fit().bic)
    except:
      tmp.append(None)
  bic_matrix.append(tmp)
print(bic_matrix)
bic_matrix = pd.DataFrame(bic_matrix) #从中可以找出最小值
print(bic_matrix)
print(bic_matrix.stack())
p,q = bic_matrix.stack().astype('float64').idxmin() #先用stack展平，然后用idxmin找出最小值位置。
print(u'BIC最小的p值和q值为：%s、%s' %(p,q))

#AIC参数搜索
import warnings
import itertools
# 设置自相关(AR)、差分(I)、移动平均(MA)的三个参数的取值范围
p = d = q = range(0, 2)
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
        temp_model = sm.tsa.ARIMA(dta,param)
        results = temp_model.fit()
        if results.aic < best_aic:
            best_aic = results.aic
            best_pdq = param
    except:
        continue
        
print("Best ARIMA{} model - AIC:{}".format(best_pdq, best_aic))



#用测试数据集预测并且评分
fit_best = sm.tsa.SARIMAX(train[focastQuota], order=(2, 1, 1), seasonal_order=(0, 1, 0, 8),
                      enforce_stationarity=False,enforce_invertibility=False).fit()

test['SARIMA'] = pd.Series(fit_best.predict(start=test.index[0], end=test.index[-1], dynamic=True))
print(test)
rmse = np.sqrt(mean_squared_error(test[focastQuota], test['SARIMA']))
print(rmse)
R2 = r2_score(test[focastQuota], test['SARIMA'])
print(R2)
abs_=(test[focastQuota]-test['SARIMA']).abs()
mape=(abs_/test[focastQuota]).mean()# mean absolute percentage error，平均绝对百分比误差
print("均方根误差RMSE:{:.0f}；\n平均绝对百分比误差MAPE:{:.2%}。".format(rmse,mape))


#正式预测未来数据
y_hat_SARIMA = pd.Series(fit_best.predict(start='2020-2-14', end='2020-2-29', dynamic=True),
                          index=pd.date_range('2020-2-14','2020-2-29',freq='D'))

y_hat_SARIMA.to_excel('./数据导出/2月份供水总量预测值.xlsx')

















#确定最佳p、d、q值
import pandas as pd

#参数初始化
discfile = '../data/discdata_processed.xls'

data = pd.read_excel(discfile, index_col = 'COLLECTTIME')
data = data.iloc[: len(data)-5] #不使用最后5个数据
xdata = data['CWXT_DB:184:D:\\']

from statsmodels.tsa.arima_model import ARIMA

#定阶
pmax = int(len(xdata)/10) #一般阶数不超过length/10
qmax = int(len(xdata)/10) #一般阶数不超过length/10
bic_matrix = [] #bic矩阵
for p in range(pmax+1):
  tmp = []
  for q in range(qmax+1):
    try: #存在部分报错，所以用try来跳过报错。
      tmp.append(ARIMA(xdata, (p,1,q)).fit().bic)
    except:
      tmp.append(None)
  bic_matrix.append(tmp)
print(bic_matrix)
#[[1275.6868239439104, 1273.190434524266, 1273.5749982328914, 1274.4669152438114, None], 
#[1276.7491283595593, 1271.8999324285992, None, None, None], 
#[1279.6942963992901, 1277.5553412371614, None, 1280.0924824267408, None],
# [1278.0659994468958, 1278.9885944429066, 1282.782534558853, 1285.943493708969, None],
# [1281.220790614283, 1282.6999920212124, 1286.2975191780365, 1290.1950373803218, None]]
bic_matrix = pd.DataFrame(bic_matrix) #从中可以找出最小值
print(bic_matrix)
#              0            1            2            3     4
# 0  1275.686824  1273.190435  1273.574998  1274.466915  None
# 1  1276.749128  1271.899932          NaN          NaN  None
# 2  1279.694296  1277.555341          NaN  1280.092482  None
# 3  1278.065999  1278.988594  1282.782535  1285.943494  None
# 4  1281.220791  1282.699992  1286.297519  1290.195037  None
print(bic_matrix.stack())
# 0  0    1275.69
#    1    1273.19
#    2    1273.57
#    3    1274.47
# 1  0    1276.75
#    1     1271.9
# 2  0    1279.69
#    1    1277.56
#    3    1280.09
# 3  0    1278.07
#    1    1278.99
#    2    1282.78
#    3    1285.94
# 4  0    1281.22
#    1     1282.7
#    2     1286.3
#    3     1290.2
p,q = bic_matrix.stack().astype('float64').idxmin() #先用stack展平，然后用idxmin找出最小值位置。
print(u'BIC最小的p值和q值为：%s、%s' %(p,q))