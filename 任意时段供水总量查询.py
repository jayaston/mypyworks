# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 08:31:59 2019

@author: XieJie
"""
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
import matplotlib.pyplot as plt
import sys
import os
os.getcwd()
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"StatLedger\module")))
except:
    sys.path.append(r'.\mypyworks\StatLedger\module')
import pandas as pd
import numpy as np
import tjfxdata as tjfx

#import re    
import datetime as dt


list1 = [['00','00718','d']
         ]
shuju_df = tjfx.TjfxData().getdata('20200101','20200802',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce')


test = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],
                      values='QUOTA_VALUE')
#print(test)
test.info()
test.columns=['最低温度','最高温度','水厂供水总量']

#test.plot()
#plt.axhline(y=583357,ls="--")#添加水平直线

##按月份删选
#test['mon'] = test.index.strftime('%m')
#test = test[test.mon<='10']
#test.drop([("mon","")],axis=1,inplace=True)
#按照指定的顺序对列排序
#test = test[['西村水厂','石门水厂','江村水厂','新塘水厂','西洲水厂','南洲水厂']]
#test = test.resample("Y").sum()

test.sort_index().to_excel(r'C:\Users\XieJie\mypyworks\输出\2019-2020日供水总量.xlsx')

