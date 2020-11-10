# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 09:47:41 2020

@author: Jay
"""
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
np.set_printoptions(suppress=True) 
from scipy import  stats
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

data = pd.read_excel(r'.\mypyworks\自来水数据\2015-2020日供水总量.xlsx',index_col = 0)
data = data['2020-03':'2020-04']
xdata = data['水厂供水总量']


#确定最佳p、d、q值
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
        temp_model = sm.tsa.ARIMA(xdata,param)
        results = temp_model.fit()
        if results.aic < best_aic:
            best_aic = results.aic
            best_pdq = param
    except:
        continue
        
print("Best ARIMA{} model - AIC:{}".format(best_pdq, best_aic))

(p, q) =(sm.tsa.arma_order_select_ic(xdata,max_ar=5,max_ma=5,ic='aic')['aic_min_order'])

model = sm.tsa.ARIMA(xdata,(1,1,1)).fit()

test = model.predict('2020-04-14','2020-04-21',dynamic=True)
test.plot()
xdata['2020-04'].plot()